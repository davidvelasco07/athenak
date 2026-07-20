//========================================================================================
// AthenaXXX astrophysical plasma code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file be_sink.cpp
//! \brief Bonnor-Ebert sphere collapse onto a sink particle (GO13 section 3.4 analogue).
//!
//! A marginally supercritical Bonnor-Ebert sphere (Tomida 2011 approximate profile,
//! rho(r) = f * rho_c * (1 + r^2/(rc^2/3))^{-3/2}, clamped at r = rc, enhancement
//! factor f > 1) collapses under gas self-gravity onto a small sink seeded at the
//! centre. Unlike the Shu test there is no exact self-similar accretion rate; the
//! quantitative anchors are:
//!   - the Foster & Chevalier (1993) peak accretion rate ~47 c_s^3/G shortly after
//!     core formation (written to the history for reference),
//!   - exact gas+sink mass conservation (periodic runs) / smooth histories,
//!   - decomposition invariance (single block vs 2^3 blocks, sink at the corner).
//! GO13's version also tests sink CREATION; sinks here are pgen-seeded (creation is a
//! documented follow-up), so this validates the post-formation accretion history.
//!
//! BOUNDARY-FLUX NOTE (outflow BC runs): with the default f=1.2, rc=6.45 setup the
//! ambient medium filling the box (rho~0.15) is itself marginally Jeans-unstable
//! (box/L_J(ambient) ~ 1.6), so under global self-gravity it collapses and draws mass
//! IN through the outflow boundaries. The box-integrated M_tot therefore GROWS with time
//! (~+3400 by t=3.5 here) -- this is genuine boundary flux for an open, accreting cloud,
//! NOT a conservation error in the sink or hydro. Two independent proofs: (i) periodic
//! BCs on the identical setup conserve M_tot to machine precision (2699.270000, constant);
//! (ii) AMR that resolves the core to 256^3-effective leaves the M_tot(t) curve unchanged
//! from the fixed 128^3 grid -- a resolution-independent drift can only be a BC effect.
//! Interpret M_tot(t) as inflow bookkeeping; the accretion physics is in M_sink(t).
//!
//! Code units: c_s = 1, 4 pi G = 1 (G = 1/4pi), central density rho_c = f. The BE
//! sphere with rc = 6.45 is near-critical; t_ff(centre) = pi sqrt(3/(8 f)).
//!
//! History output (problem/user_hist = true): gas mass/momenta, sink mass/velocity,
//! totals, and the F&C reference rate in the Mdot_ana column.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "particles/particles.hpp"
#include "gravity/gravity.hpp"
#include "outputs/outputs.hpp"

namespace {
Real mstar0_, mdot_ref_, msph0_;
Real cs_global_, njeans_global_;   // Jeans-criterion inputs
Real sink_buf_cells_;              // sink-region halo (in cells of the block's own dx)
void BESinkHistory(HistoryData *pdata, Mesh *pm);
void BESinkFinalize(ParameterInput *pin, Mesh *pm);
void BESinkRefinement(MeshBlockPack *pmbp);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &BESinkHistory;
  pgen_final_func = &BESinkFinalize;
  user_ref_func = &BESinkRefinement;    // active only when refinement=adaptive
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->phydro == nullptr || pmbp->pgrav == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
        << "be_sink requires <particles>, <hydro> and <gravity> blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->phydro->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
        << "be_sink requires an isothermal EOS" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real rc  = pin->GetOrAddReal("problem", "cloud_radius", 6.45);
  const Real f   = pin->GetOrAddReal("problem", "f", 1.2);   // supercritical enhancement
  const Real cs  = pin->GetReal("hydro", "iso_sound_speed");
  const Real fpg = pin->GetReal("gravity", "four_pi_G");
  const Real G   = fpg/(4.0*M_PI);
  const Real rcsq = rc*rc/3.0;

  auto &msize = pmy_mesh_->mesh_size;
  auto &indcs = pmy_mesh_->mb_indcs;
  const Real dx = (msize.x1max - msize.x1min)/
                  static_cast<Real>(pmy_mesh_->nmb_rootx1*indcs.nx1);
  // sphere centre offset by dx/2 per axis -> sink at a cell centre (and, with 2^3
  // MeshBlocks, the control volume straddles all 8 blocks: the corner stress test)
  const Real x0 = pin->GetOrAddReal("problem", "x0",
                    0.5*(msize.x1min + msize.x1max) + 0.5*dx);
  const Real y0 = pin->GetOrAddReal("problem", "y0",
                    0.5*(msize.x2min + msize.x2max) + 0.5*dx);
  const Real z0 = pin->GetOrAddReal("problem", "z0",
                    0.5*(msize.x3min + msize.x3max) + 0.5*dx);

  // sink seed: gas mass inside r_ctrl = 1.5 dx (uniform-density approximation is
  // exact to <1% since r_ctrl << rc/sqrt(3))
  const Real rctrl = 1.5*dx;
  const Real mstar = (4.0/3.0)*M_PI*f*rctrl*rctrl*rctrl;
  mstar0_ = mstar;
  mdot_ref_ = 47.0*std::pow(cs, 3)/G;   // Foster & Chevalier (1993) peak rate
  const Real tff = M_PI*std::sqrt(3.0/(8.0*f));

  // solid-body rotation about the z-axis through the sphere centre (as in be_collapse):
  // omega = omegatff/tff; and an optional m=2 azimuthal density perturbation of relative
  // amplitude amp*(r/rc)^2 (the Boss-Bodenheimer fragmentation seed). Both act only inside
  // rc so the ambient stays static.
  const Real omegatff = pin->GetOrAddReal("problem", "omegatff", 0.0);
  const Real omega = omegatff/tff;
  const Real amp = pin->GetOrAddReal("problem", "amp", 0.0);

  // AMR criterion inputs (used only when refinement=adaptive)
  cs_global_ = cs;
  njeans_global_ = pin->GetOrAddReal("problem", "njeans", 16.0);
  // same-level halo (in cells) kept around each sink so its CV never straddles a level
  // jump; rctrl=1 cell + reset stencil reach ~2.5 cells, so 4 gives comfortable margin
  sink_buf_cells_ = pin->GetOrAddReal("problem", "sink_buf_cells", 4.0);

  if (global_variable::my_rank == 0) {
    std::printf("be_sink: rc=%.4g f=%.4g cs=%.4g G=%.6g t_ff=%.4f\n", rc, f, cs, G, tff);
    std::printf("  M_*(0)=%.6e  Mdot_FC_ref=%.6e\n", mstar, mdot_ref_);
  }

  // ---- gas IC: enhanced BE profile, clamped at rc (ambient = edge density) ----
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  par_for("be_gas", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = mbsize.d_view(m).x1min +
                   (static_cast<Real>(i-is) + 0.5)*mbsize.d_view(m).dx1 - x0;
    const Real y = mbsize.d_view(m).x2min +
                   (static_cast<Real>(j-js) + 0.5)*mbsize.d_view(m).dx2 - y0;
    const Real z = mbsize.d_view(m).x3min +
                   (static_cast<Real>(k-ks) + 0.5)*mbsize.d_view(m).dx3 - z0;
    const Real r = Kokkos::sqrt(x*x + y*y + z*z);
    const Real rcl = Kokkos::fmin(r, rc);
    Real rho = f*Kokkos::pow(1.0 + rcl*rcl/rcsq, -1.5);
    if (amp > 0.0 && r < rc) {
      rho *= (1.0 + amp*(r*r)/(rc*rc)*Kokkos::cos(2.0*Kokkos::atan2(y, x)));
    }
    u0(m, IDN, k, j, i) = rho;
    // solid-body rotation about z (v = omega z_hat x r): vx=omega*y, vy=-omega*x, r<rc
    Real mx = 0.0, my = 0.0;
    if (r < rc) {
      mx =  rho*omega*y;
      my = -rho*omega*x;
    }
    u0(m, IM1, k, j, i) = mx;
    u0(m, IM2, k, j, i) = my;
    u0(m, IM3, k, j, i) = 0.0;
  });

  // total sphere gas mass (for the finalize report): host reduce over the IC
  {
    Real msph = 0.0;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmkji = nmb*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
    Kokkos::parallel_reduce("be_msph", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
      const int i = (idx - m*nkji - (k-ks)*nji - (j-js)*nx1) + is;
      const Real dV = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
      sum += u0(m, IDN, k, j, i)*dV;
    }, Kokkos::Sum<Real>(msph));
    msph0_ = msph;
    if (global_variable::my_rank == 0) {
      std::printf("  M_gas(0)=%.6e (box, incl ambient)\n", msph);
    }
  }

  // ---- one sink at the sphere centre (skipped in creation mode, where it is born) ----
  // MPI-robust seed: ppc rounds a single global particle to 0 on every rank under
  // decomposition, so place exactly one sink on the rank whose local block contains
  // (x0,y0,z0) (lower-inclusive -> unique owner, even at a shared block corner) and zero
  // on the rest, resizing the particle arrays to match.
  if (!pmbp->ppart->creation) {
    auto gids = pmbp->gids;
    int mown = -1;
    for (int mm = 0; mm < nmb; ++mm) {
      if (x0 >= mbsize.h_view(mm).x1min && x0 < mbsize.h_view(mm).x1max &&
          y0 >= mbsize.h_view(mm).x2min && y0 < mbsize.h_view(mm).x2max &&
          z0 >= mbsize.h_view(mm).x3min && z0 < mbsize.h_view(mm).x3max) {
        mown = mm; break;
      }
    }
    const int desired = (mown >= 0) ? 1 : 0;
    Kokkos::resize(pmbp->ppart->prtcl_rdata, pmbp->ppart->nrdata, desired);
    Kokkos::resize(pmbp->ppart->prtcl_idata, pmbp->ppart->nidata, desired);
    pmbp->ppart->nprtcl_thispack = desired;
    pmbp->ppart->RefreshMeshParticleCounts();
    if (desired == 1) {
      auto pr = pmbp->ppart->prtcl_rdata;
      auto pi = pmbp->ppart->prtcl_idata;
      const int gidown = gids + mown;
      par_for("be_part", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(int p) {
        pr(IPX, p) = x0;  pr(IPY, p) = y0;  pr(IPZ, p) = z0;
        pr(IPVX, p) = 0.0; pr(IPVY, p) = 0.0; pr(IPVZ, p) = 0.0;
        pr(IPM, p) = mstar;
        pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
        pr(IPX0, p) = x0; pr(IPY0, p) = y0; pr(IPZ0, p) = z0;
        pi(PGID, p) = gidown;
        pi(PTAG, p) = 0;
      });
    }
  }

  pmbp->ppart->dtnew = dx;   // seed; Particles::NewTimeStep refreshes per cycle
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void BESinkHistory()
//! \brief gas mass/momenta + sink mass/velocity + conserved totals (same layout as
//! shu_collapse so the analysis/video tooling is shared; Mdot_ana = F&C peak reference)

void BESinkHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  pdata->nhist = 18;
  pdata->label[0] = "mass";
  pdata->label[1] = "1-mom";
  pdata->label[2] = "2-mom";
  pdata->label[3] = "3-mom";
  pdata->label[4] = "m_sink";
  pdata->label[5] = "vx_sink";
  pdata->label[6] = "x_sink";
  pdata->label[7] = "M_tot";
  pdata->label[8] = "px_tot";
  pdata->label[9] = "Mdot_FC";
  pdata->label[10] = "Lz_gas";   // gas z-angular momentum about the domain axis (rotation)
  pdata->label[11] = "n_sink";   // number of sinks (0 before creation) -> gate video marker
  // per-sink positions of the first two sinks (tag-sorted), for exact video markers
  // incl. the off-centre secondary that forms when the disk fragments (single-rank exact)
  pdata->label[12] = "x_sink0"; pdata->label[13] = "y_sink0"; pdata->label[14] = "z_sink0";
  pdata->label[15] = "x_sink1"; pdata->label[16] = "y_sink1"; pdata->label[17] = "z_sink1";

  auto &u0 = pmbp->phydro->u0;
  auto &mbsz = pmbp->pmb->mb_size;
  auto &ix = pmbp->pmesh->mb_indcs;
  const int is = ix.is, nx1 = ix.nx1;
  const int js = ix.js, nx2 = ix.nx2;
  const int ks = ix.ks, nx3 = ix.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  Real gsum[4] = {0.0, 0.0, 0.0, 0.0};
  for (int v = 0; v < 4; ++v) {
    Real sum = 0.0;
    Kokkos::parallel_reduce("be_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &psum) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
      const int i = (idx - m*nkji - (k-ks)*nji - (j-js)*nx1) + is;
      const Real dV = mbsz.d_view(m).dx1*mbsz.d_view(m).dx2*mbsz.d_view(m).dx3;
      psum += u0(m, v, k, j, i)*dV;
    }, Kokkos::Sum<Real>(sum));
    gsum[v] = sum;
  }

  // gas z-angular momentum about the domain axis: Lz = sum (x*M2 - y*M1) dV
  Real lzsum = 0.0;
  Kokkos::parallel_reduce("be_hist_lz", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &psum) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
    const int i = (idx - m*nkji - (k-ks)*nji - (j-js)*nx1) + is;
    const Real x = mbsz.d_view(m).x1min + (static_cast<Real>(i-is)+0.5)*mbsz.d_view(m).dx1;
    const Real y = mbsz.d_view(m).x2min + (static_cast<Real>(j-js)+0.5)*mbsz.d_view(m).dx2;
    const Real dV = mbsz.d_view(m).dx1*mbsz.d_view(m).dx2*mbsz.d_view(m).dx3;
    psum += (x*u0(m, IM2, k, j, i) - y*u0(m, IM1, k, j, i))*dV;
  }, Kokkos::Sum<Real>(lzsum));

  const int npart = pmbp->ppart->nprtcl_thispack;
  auto pr_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_rdata);
  Kokkos::deep_copy(pr_h, pmbp->ppart->prtcl_rdata);
  auto pi_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_idata);
  Kokkos::deep_copy(pi_h, pmbp->ppart->prtcl_idata);
  Real msink = 0.0, vxsink = 0.0, xsink = 0.0, pxsink = 0.0;
  // track the two lowest-tag sinks (creation order) for exact per-sink video markers
  int t0 = INT_MAX, t1 = INT_MAX, p0 = -1, p1 = -1;
  for (int p = 0; p < npart; ++p) {
    msink  += pr_h(IPM, p);
    vxsink += pr_h(IPVX, p);
    xsink  += pr_h(IPX, p);
    pxsink += pr_h(IPM, p)*pr_h(IPVX, p);
    int tg = pi_h(PTAG, p);
    if (tg < t0)      { t1 = t0; p1 = p0; t0 = tg; p0 = p; }
    else if (tg < t1) { t1 = tg; p1 = p; }
  }

  pdata->hdata[0] = gsum[0];
  pdata->hdata[1] = gsum[1];
  pdata->hdata[2] = gsum[2];
  pdata->hdata[3] = gsum[3];
  pdata->hdata[4] = msink;
  pdata->hdata[5] = vxsink;
  pdata->hdata[6] = xsink;
  pdata->hdata[7] = gsum[0] + msink;
  pdata->hdata[8] = gsum[1] + pxsink;
  // global constant: divide by nranks -- hst user data is MPI_SUM-reduced across ranks
  pdata->hdata[9] = mdot_ref_/static_cast<Real>(global_variable::nranks);
  pdata->hdata[10] = lzsum;
  pdata->hdata[11] = static_cast<Real>(npart);   // n_sink (0 before creation)
  pdata->hdata[12] = (p0 >= 0) ? pr_h(IPX, p0) : 0.0;
  pdata->hdata[13] = (p0 >= 0) ? pr_h(IPY, p0) : 0.0;
  pdata->hdata[14] = (p0 >= 0) ? pr_h(IPZ, p0) : 0.0;
  pdata->hdata[15] = (p1 >= 0) ? pr_h(IPX, p1) : 0.0;
  pdata->hdata[16] = (p1 >= 0) ? pr_h(IPY, p1) : 0.0;
  pdata->hdata[17] = (p1 >= 0) ? pr_h(IPZ, p1) : 0.0;
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void BESinkFinalize()

void BESinkFinalize(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->ppart == nullptr) return;
  const int npart = pmbp->ppart->nprtcl_thispack;
  auto pr_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_rdata);
  Kokkos::deep_copy(pr_h, pmbp->ppart->prtcl_rdata);
  Real msink = 0.0;
  for (int p = 0; p < npart; ++p) msink += pr_h(IPM, p);
  if (global_variable::my_rank == 0) {
    std::printf("\n=== be_sink finalize: t=%.6f ncycle=%d ===\n", pm->time, pm->ncycle);
    std::printf("  M_sink=%.8e (seed %.8e)  accreted=%.6e  M_gas0=%.6e\n",
                msink, mstar0_, msink - mstar0_, msph0_);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BESinkRefinement(MeshBlockPack *pmbp)
//! \brief Combined AMR criterion for BE-collapse-onto-sink.
//!
//! Two logically-OR'd flags per block:
//!   (a) Sink-region: refine any block whose bounds, expanded by sink_buf_cells_ of the
//!       block's own dx, contain a sink position. This guarantees a same-level halo
//!       around every sink so its 27-cell control volume + reset stencil never crosses a
//!       level jump (the accretion kernel's cross-block scatter is same-level only).
//!   (b) Jeans: refine if the local Jeans length is under-resolved
//!       (nJ = cs/sqrt(rho_max) * 2pi/dx < njeans), keeping the collapsing core resolved
//!       so isothermal mass conservation holds (the fixed-grid failure mode).
//! A block derefines only when BOTH criteria agree to derefine.

void BESinkRefinement(MeshBlockPack *pmbp) {
  if (pmbp->ppart == nullptr) return;
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3, ng = indcs.ng;
  const int nkji = (nx3 + 2*ng)*(nx2 + 2*ng)*(nx1 + 2*ng);
  const int nji  = (nx2 + 2*ng)*(nx1 + 2*ng);
  const int ni   = (nx1 + 2*ng);
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  auto pr = pmbp->ppart->prtcl_rdata;
  const int npart = pmbp->ppart->nprtcl_thispack;
  const Real cs = cs_global_;
  const Real njeans = njeans_global_;
  const Real sbuf = sink_buf_cells_;

  par_for_outer("BESinkAMR", DevExeSpace(), 0, 0, 0, (nmb-1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    // (b) Jeans: reduce rho_max over the block (incl. ghosts, as in be_collapse)
    Real team_rhomax;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [&](const int idx, Real &rhomax) {
        const int k = idx/nji;
        const int j = (idx - k*nji)/ni;
        const int i = (idx - k*nji - j*ni);
        rhomax = Kokkos::fmax(u0(m, IDN, k, j, i), rhomax);
      }, Kokkos::Max<Real>(team_rhomax));

    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      const Real dx = size.d_view(m).dx1;
      const Real nj_min = cs/Kokkos::sqrt(team_rhomax)*(2.0*M_PI/dx);
      int fj;
      if (nj_min < njeans)            fj = 1;
      else if (nj_min > njeans*2.5)   fj = -1;
      else                            fj = 0;

      // (a) sink proximity: expand block bounds by sbuf cells of this block's dx
      const Real bx = sbuf*size.d_view(m).dx1;
      const Real by = sbuf*size.d_view(m).dx2;
      const Real bz = sbuf*size.d_view(m).dx3;
      const Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      const Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
      const Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
      int fs = -1;
      for (int p = 0; p < npart; ++p) {
        const Real px = pr(IPX, p), py = pr(IPY, p), pz = pr(IPZ, p);
        if (px > (x1min-bx) && px < (x1max+bx) &&
            py > (x2min-by) && py < (x2max+by) &&
            pz > (x3min-bz) && pz < (x3max+bz)) {
          fs = 1;
        }
      }

      // logical OR: refine if either wants it; derefine only if both agree
      int flag;
      if (fs == 1 || fj == 1)  flag = 1;
      else if (fj == 0)        flag = 0;
      else                     flag = -1;
      refine_flag.d_view(m + mbs) = flag;
    });
  });

  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}
}  // namespace
