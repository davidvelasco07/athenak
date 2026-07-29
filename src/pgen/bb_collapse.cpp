//========================================================================================
// AthenaXXX astrophysical plasma code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bb_collapse.cpp
//! \brief Boss-Bodenheimer rotating-cloud collapse & fragmentation (isothermal).
//!
//! The classic Boss & Bodenheimer (1979) test, as used by Bleuler & Teyssier (2014,
//! MNRAS 445, 4015, figs 7-8) to exercise sink creation + accretion + MERGING. A uniform
//! sphere of radius R and mass M in solid-body rotation Omega about z, with an m=2
//! azimuthal density perturbation rho(phi) = rho0 (1 + a cos 2phi), a = 0.1, embedded in a
//! low-density ambient medium. Isothermal (this pgen). The sphere collapses, the m=2 mode
//! grows into a bar/filament that fragments; sink particles form at the density peaks
//! (creation), accrete, and merge when their control volumes overlap.
//!
//! Dimensionless setup via the standard ratios (scale-free for the isothermal case):
//!   alpha = E_therm/|E_grav| = 5 cs^2 R /(2 G M)   (~0.25)
//!   beta  = E_rot  /|E_grav| = Omega^2 R^3 /(3 G M) (~0.20)
//! Given R, cs, G (=four_pi_G/4pi) and (alpha,beta): M = 5 cs^2 R/(2 alpha G);
//! rho0 = 3M/(4 pi R^3); Omega = sqrt(3 beta G M / R^3). t_ff = sqrt(3 pi/(32 G rho0)).
//! Code units: cs = 1, 4 pi G = 1. Sinks form by creation (<particles>/creation=true),
//! so no sink is seeded here.
//!
//! Reuses the be_sink AMR criterion (Jeans + sink-proximity) and history layout.

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
#include "particles/sink_positions.hpp"
#include "gravity/gravity.hpp"
#include "outputs/outputs.hpp"

namespace {
Real cs_global_, njeans_global_, sink_buf_cells_, tff_global_;
// piecewise-polytropic (barotropic) EOS state (Fig-8 mode; only used when EOS is ideal)
bool poly_mode_ = false;
Real cs0_, rho1_, rho2_, kap1_, kap2_, gam1_, gam2_, gm1_;
void BBHistory(HistoryData *pdata, Mesh *pm);
void BBFinalize(ParameterInput *pin, Mesh *pm);
void BBRefinement(MeshBlockPack *pmbp);
void BBBarotropicReset(Mesh *pm, const Real bdt);

//! Bleuler & Teyssier (2014) eq. 39 piecewise-polytropic pressure:
//!   P = cs0^2 rho                 (rho < rho1, isothermal cold branch)
//!   P = kap1 rho^gam1             (rho1 <= rho < rho2)
//!   P = kap2 rho^gam2             (rho >= rho2)
//! kap1,kap2 fixed by continuity at rho1,rho2.
KOKKOS_INLINE_FUNCTION
Real PBaro(Real rho, Real cs0, Real rho1, Real rho2,
           Real kap1, Real kap2, Real gam1, Real gam2) {
  if (rho < rho1)      return cs0*cs0*rho;
  else if (rho < rho2) return kap1*Kokkos::pow(rho, gam1);
  else                 return kap2*Kokkos::pow(rho, gam2);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &BBHistory;
  pgen_final_func = &BBFinalize;
  user_ref_func = &BBRefinement;     // active only when refinement=adaptive
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->phydro == nullptr || pmbp->pgrav == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
        << "bb_collapse requires <particles>, <hydro> and <gravity> blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // isothermal EOS  -> Fig-7 (scale-free filament fragmentation)
  // ideal gas EOS   -> Fig-8 (piecewise-polytropic / barotropic; heating slows collapse)
  poly_mode_ = pmbp->phydro->peos->eos_data.is_ideal;

  const Real R     = pin->GetOrAddReal("problem", "cloud_radius", 1.0);
  const Real alpha = pin->GetOrAddReal("problem", "alpha", 0.25);   // thermal/grav
  const Real beta  = pin->GetOrAddReal("problem", "beta", 0.20);    // rot/grav
  const Real amp   = pin->GetOrAddReal("problem", "amp", 0.1);      // m=2 amplitude
  const Real contrast = pin->GetOrAddReal("problem", "contrast", 100.0);  // rho0/rho_amb
  // cold (isothermal-branch) sound speed: iso EOS -> iso_sound_speed; ideal -> problem/cs_iso
  const Real cs  = poly_mode_ ? pin->GetOrAddReal("problem", "cs_iso", 1.0)
                              : pin->GetReal("hydro", "iso_sound_speed");
  const Real fpg = pin->GetReal("gravity", "four_pi_G");
  const Real G   = fpg/(4.0*M_PI);

  // derived cloud parameters
  const Real M    = 5.0*cs*cs*R/(2.0*alpha*G);
  const Real rho0 = 3.0*M/(4.0*M_PI*R*R*R);
  const Real omega = std::sqrt(3.0*beta*G*M/(R*R*R));
  const Real rho_amb = rho0/contrast;
  const Real tff = std::sqrt(3.0*M_PI/(32.0*G*rho0));
  tff_global_ = tff;

  auto &msize = pmy_mesh_->mesh_size;
  auto &indcs = pmy_mesh_->mb_indcs;
  const Real x0 = 0.5*(msize.x1min + msize.x1max);
  const Real y0 = 0.5*(msize.x2min + msize.x2max);
  const Real z0 = 0.5*(msize.x3min + msize.x3max);

  cs_global_ = cs;
  njeans_global_ = pin->GetOrAddReal("problem", "njeans", 16.0);
  sink_buf_cells_ = pin->GetOrAddReal("problem", "sink_buf_cells", 4.0);

  // ---- piecewise-polytropic (barotropic) EOS setup, Fig-8 mode (ideal EOS only) ----
  if (poly_mode_) {
    const Real rho1_fac = pin->GetOrAddReal("problem", "rho1_fac", 15.0);   // rho1/rho0
    const Real rho2_fac = pin->GetOrAddReal("problem", "rho2_fac", 300.0);  // rho2/rho0
    cs0_  = cs;
    rho1_ = rho1_fac*rho0;
    rho2_ = rho2_fac*rho0;
    gam1_ = pin->GetOrAddReal("problem", "gam1", 1.1);
    gam2_ = pin->GetOrAddReal("problem", "gam2", 4.0/3.0);
    kap1_ = cs*cs*std::pow(rho1_, 1.0 - gam1_);         // continuity of P at rho1
    kap2_ = kap1_*std::pow(rho2_, gam1_ - gam2_);       // continuity of P at rho2
    gm1_  = pmbp->phydro->peos->eos_data.gamma - 1.0;   // solver (gamma-1)
    // The sink-creation LP threshold reads eos_data.iso_cs (0 for an ideal EOS): set it to
    // the cold-branch sound speed so creation uses the same criterion as the isothermal run.
    pmbp->phydro->peos->eos_data.iso_cs = cs;
    user_srcs = true;                        // enable the per-stage barotropic energy reset
    user_srcs_func = &BBBarotropicReset;
  }

  if (global_variable::my_rank == 0) {
    std::printf("bb_collapse[%s]: R=%.4g alpha=%.3g beta=%.3g amp=%.3g cs=%.3g G=%.6g\n",
                poly_mode_ ? "polytropic" : "isothermal", R, alpha, beta, amp, cs, G);
    std::printf("  M=%.6e rho0=%.6e omega=%.6e rho_amb=%.4e t_ff=%.6f\n",
                M, rho0, omega, rho_amb, tff);
    if (poly_mode_) {
      std::printf("  barotropic: rho1=%.4e (%.1f rho0) rho2=%.4e (%.1f rho0) "
                  "gam1=%.3f gam2=%.4f\n", rho1_, rho1_/rho0, rho2_, rho2_/rho0,
                  gam1_, gam2_);
    }
  }

  // ---- gas IC: uniform sphere (+m=2) in solid-body rotation, low-density ambient ----
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const bool poly = poly_mode_;
  const Real cs0 = cs0_, r1 = rho1_, r2 = rho2_, k1 = kap1_, k2 = kap2_;
  const Real g1 = gam1_, g2 = gam2_, gm1 = gm1_;
  par_for("bb_gas", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = mbsize.d_view(m).x1min +
                   (static_cast<Real>(i-is) + 0.5)*mbsize.d_view(m).dx1 - x0;
    const Real y = mbsize.d_view(m).x2min +
                   (static_cast<Real>(j-js) + 0.5)*mbsize.d_view(m).dx2 - y0;
    const Real z = mbsize.d_view(m).x3min +
                   (static_cast<Real>(k-ks) + 0.5)*mbsize.d_view(m).dx3 - z0;
    const Real r = Kokkos::sqrt(x*x + y*y + z*z);
    Real rho, mx = 0.0, my = 0.0;
    if (r < R) {
      rho = rho0*(1.0 + amp*Kokkos::cos(2.0*Kokkos::atan2(y, x)));
      // solid-body rotation about z: v = omega (z_hat x r) -> vx=omega*y, vy=-omega*x
      mx =  rho*omega*y;
      my = -rho*omega*x;
    } else {
      rho = rho_amb;
    }
    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = mx;
    u0(m, IM2, k, j, i) = my;
    u0(m, IM3, k, j, i) = 0.0;
    if (poly) {  // ideal EOS: seed total energy on the barotropic curve
      const Real p = PBaro(rho, cs0, r1, r2, k1, k2, g1, g2);
      u0(m, IEN, k, j, i) = p/gm1 + 0.5*(mx*mx + my*my)/rho;
    }
  });

  pmbp->ppart->dtnew = (msize.x1max - msize.x1min)/
                       static_cast<Real>(pmy_mesh_->nmb_rootx1*indcs.nx1);
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void BBHistory()
//! \brief gas mass/momenta/Lz + sink count/mass + first-two-sink positions (video markers)

void BBHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  pdata->nhist = 12;
  pdata->label[0] = "mass";
  pdata->label[1] = "1-mom";
  pdata->label[2] = "2-mom";
  pdata->label[3] = "3-mom";
  pdata->label[4] = "m_sink";
  pdata->label[5] = "M_tot";
  pdata->label[6] = "Lz_gas";
  pdata->label[7] = "n_sink";
  pdata->label[8] = "x_sink0"; pdata->label[9] = "y_sink0";
  pdata->label[10] = "x_sink1"; pdata->label[11] = "y_sink1";

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
    Kokkos::parallel_reduce("bb_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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
  Real lzsum = 0.0;
  Kokkos::parallel_reduce("bb_lz", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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
  Real msink = 0.0;
  int t0 = INT_MAX, t1 = INT_MAX, p0 = -1, p1 = -1;
  for (int p = 0; p < npart; ++p) {
    msink += pr_h(IPM, p);
    int tg = pi_h(PTAG, p);
    if (tg < t0)      { t1 = t0; p1 = p0; t0 = tg; p0 = p; }
    else if (tg < t1) { t1 = tg; p1 = p; }
  }
  pdata->hdata[0] = gsum[0];
  pdata->hdata[1] = gsum[1];
  pdata->hdata[2] = gsum[2];
  pdata->hdata[3] = gsum[3];
  pdata->hdata[4] = msink;
  pdata->hdata[5] = gsum[0] + msink;
  pdata->hdata[6] = lzsum;
  pdata->hdata[7] = static_cast<Real>(npart);
  pdata->hdata[8]  = (p0 >= 0) ? pr_h(IPX, p0) : 0.0;
  pdata->hdata[9]  = (p0 >= 0) ? pr_h(IPY, p0) : 0.0;
  pdata->hdata[10] = (p1 >= 0) ? pr_h(IPX, p1) : 0.0;
  pdata->hdata[11] = (p1 >= 0) ? pr_h(IPY, p1) : 0.0;
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void BBFinalize()

void BBFinalize(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->ppart == nullptr) return;
  const int npart = pmbp->ppart->nprtcl_thispack;
  auto pr_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_rdata);
  Kokkos::deep_copy(pr_h, pmbp->ppart->prtcl_rdata);
  Real msink = 0.0;
  for (int p = 0; p < npart; ++p) msink += pr_h(IPM, p);
  if (global_variable::my_rank == 0) {
    std::printf("\n=== bb_collapse finalize: t=%.6f (%.4f t_ff) ncycle=%d ===\n",
                pm->time, pm->time/tff_global_, pm->ncycle);
    std::printf("  N_sink=%d  M_sink=%.6e\n", npart, msink);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BBBarotropicReset(Mesh *pm, const Real bdt)
//! \brief Per-stage barotropic energy projection (Fig-8 mode). Runs as user_srcs_func after
//! RKUpdate and before ConToPrim: overwrite the total energy so that ConToPrim (ideal EOS)
//! recovers the piecewise-polytropic pressure P_baro(rho). Makes the ideal solver behave as
//! a barotropic gas without touching the Riemann solvers. bdt unused (this is a projection,
//! not a rate-based source term).

void BBBarotropicReset(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) return;
  auto &u0 = pmbp->phydro->u0;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const Real cs0 = cs0_, r1 = rho1_, r2 = rho2_, k1 = kap1_, k2 = kap2_;
  const Real g1 = gam1_, g2 = gam2_, gm1 = gm1_;
  par_for("bb_baro_reset", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real rho = u0(m, IDN, k, j, i);
    const Real mx = u0(m, IM1, k, j, i);
    const Real my = u0(m, IM2, k, j, i);
    const Real mz = u0(m, IM3, k, j, i);
    const Real p = PBaro(rho, cs0, r1, r2, k1, k2, g1, g2);
    u0(m, IEN, k, j, i) = p/gm1 + 0.5*(mx*mx + my*my + mz*mz)/rho;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void BBRefinement(MeshBlockPack *pmbp)
//! \brief AMR: Jeans (nJ cells per Jeans length) OR sink-proximity (same-level halo around
//! each sink so its control volume never straddles a level jump). Copied from be_sink.

void BBRefinement(MeshBlockPack *pmbp) {
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
  // sink-proximity test must see EVERY sink, not just this rank's (see sink_positions.hpp):
  // a rank-local test makes the mesh rank-count-dependent and leaves sinks against
  // asymmetric coarse-fine boundaries, corrupting the deposit/gather force.
  int nsink = 0;
  DualArray1D<Real> spos = GatherAllSinkPositions(pmbp, nsink);
  auto sp = spos.d_view;
  const Real cs = cs_global_;
  const Real njeans = njeans_global_;
  const Real sbuf = sink_buf_cells_;

  par_for_outer("BBAMR", DevExeSpace(), 0, 0, 0, (nmb-1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
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

      const Real bx = sbuf*size.d_view(m).dx1;
      const Real by = sbuf*size.d_view(m).dx2;
      const Real bz = sbuf*size.d_view(m).dx3;
      const Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      const Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
      const Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
      int fs = -1;
      for (int p = 0; p < nsink; ++p) {
        const Real px = sp(3*p), py = sp(3*p+1), pz = sp(3*p+2);
        if (px > (x1min-bx) && px < (x1max+bx) &&
            py > (x2min-by) && py < (x2max+by) &&
            pz > (x3min-bz) && pz < (x3max+bz)) {
          fs = 1;
        }
      }
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
