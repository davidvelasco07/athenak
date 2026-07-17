//========================================================================================
// AthenaXXX astrophysical plasma code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shu_collapse.cpp
//! \brief Shu (1977) self-similar isothermal collapse: validation of sink-particle
//! control-volume accretion (Gong & Ostriker 2013 [GO13] section 3.2; Galilean
//! invariance test of GO13 section 3.3 via problem/vadvx..z).
//!
//! Initial condition: the self-similar solution of Shu (1977) eqs. (11)-(12) at time
//! t0 (in units of t_J = L_J/c_s), for overdensity parameter A > 2. The similarity
//! ODEs are integrated ONCE on the host at setup (RK4, inward from xi0 = 10 with the
//! eq. (19) asymptotic series, matching the reference Athena++ pgen
//! athena/src/pgen/particle_accretion.cpp) into a table interpolated per cell:
//!     rho(r) = alpha(xi) / (4 pi G t0^2),   v_r(r) = c_s v(xi),   xi = r/(c_s t0).
//! The profile is truncated at rmax = 0.75*x1max (=1.5 L_J for the standard 4 L_J
//! box): outside, rho = rho(rmax) and v = 0 (GO13). A single sink is seeded at the
//! sphere centre with the profile mass within r_ctrl = 1.5 dx,
//!     M_* = m(xi_ctrl) c_s^3 t0 / G,  m(xi) = xi^2 alpha (xi - v)   [Shu eqs. 8,10],
//! and the analytic accretion rate for comparison is Mdot = m0 c_s^3/G with
//! m0 = lim_{xi->0} m(xi) (A = 2.0004 -> m0 = 0.975).
//!
//! The 3x3x3 control volume around the sink is flattened at t=0 to the profile value
//! just outside it (the reference calls SetControlVolume() at creation) so the seeded
//! sink mass is not re-accreted on the first step.
//!
//! problem/vadvx,vadvy,vadvz (units of c_s) add a uniform bulk flow to gas AND sink
//! for the Galilean-invariance test (use all-periodic BCs there).
//!
//! History output (problem/user_hist = true): gas mass/momenta, sink mass/velocity,
//! and total (gas+sink) mass and x-momentum for conservation checks.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

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

// module state for history/finalize hooks
Real mstar0_, m0_ana_, cs_, gm_, t0_, a_par_;

// ---- Shu (1977) similarity solution -------------------------------------------------
// y[0] = alpha (dimensionless density), y[1] = v (dimensionless velocity)
void shu77_rhs(const Real x, const Real y[2], Real dydx[2]) {
  const Real xv  = x - y[1];
  const Real den = xv*xv - 1.0;
  dydx[0] = y[0]*(y[0] - (2.0/x)*xv)*xv/den;
  dydx[1] = (y[0]*xv - 2.0/x)*xv/den;
}

// asymptotic series for large x; eq. (19) in Shu (1977)
void shu77_asymptotic(const Real x0, const Real A, Real y[2]) {
  y[0] = A/(x0*x0) - A*(A - 2.0)/(2.0*std::pow(x0, 4));
  y[1] = -(A - 2.0)/x0 - (1.0 - A/6.0)*(A - 2.0)/std::pow(x0, 3);
}

// integrate inward from xi0 to xi_min with fixed-step RK4, storing every nskip-th
// point; returns tables in ASCENDING xi order
struct ShuTable {
  std::vector<Real> xi, alpha, v;
  Real m0;  // lim_{xi->0} xi^2 alpha (xi - v)
};

ShuTable IntegrateShu(const Real A, const Real xi0 = 10.0, const Real xi_min = 1.0e-4,
                      const Real dx0 = 1.0e-4, const int nskip = 10) {
  std::vector<Real> txi, tal, tv;
  Real y[2];
  shu77_asymptotic(xi0, A, y);
  Real x = xi0;
  int it = 0;
  txi.push_back(x); tal.push_back(y[0]); tv.push_back(y[1]);
  while (x > xi_min) {
    const Real h = -std::min(dx0*std::max(x, 0.05), x - 0.5*xi_min);  // shrink with x
    Real k1[2], k2[2], k3[2], k4[2], yt[2];
    shu77_rhs(x, y, k1);
    for (int n = 0; n < 2; ++n) yt[n] = y[n] + 0.5*h*k1[n];
    shu77_rhs(x + 0.5*h, yt, k2);
    for (int n = 0; n < 2; ++n) yt[n] = y[n] + 0.5*h*k2[n];
    shu77_rhs(x + 0.5*h, yt, k3);
    for (int n = 0; n < 2; ++n) yt[n] = y[n] + h*k3[n];
    shu77_rhs(x + h, yt, k4);
    for (int n = 0; n < 2; ++n) y[n] += (h/6.0)*(k1[n] + 2.0*k2[n] + 2.0*k3[n] + k4[n]);
    x += h;
    if (++it % nskip == 0 || x <= xi_min) {
      txi.push_back(x); tal.push_back(y[0]); tv.push_back(y[1]);
    }
  }
  ShuTable tab;
  const int N = static_cast<int>(txi.size());
  tab.xi.resize(N); tab.alpha.resize(N); tab.v.resize(N);
  for (int n = 0; n < N; ++n) {   // reverse to ascending xi
    tab.xi[n]    = txi[N-1-n];
    tab.alpha[n] = tal[N-1-n];
    tab.v[n]     = tv[N-1-n];
  }
  tab.m0 = tab.xi[0]*tab.xi[0]*tab.alpha[0]*(tab.xi[0] - tab.v[0]);
  return tab;
}

// linear interpolation in the (ascending) table
void ShuLookup(const ShuTable &tab, const Real xi, Real *alpha, Real *v) {
  const int N = static_cast<int>(tab.xi.size());
  if (xi <= tab.xi[0])   { *alpha = tab.alpha[0];   *v = tab.v[0];   return; }
  if (xi >= tab.xi[N-1]) { *alpha = tab.alpha[N-1]; *v = tab.v[N-1]; return; }
  int lo = 0, hi = N - 1;
  while (hi - lo > 1) { const int mid = (lo + hi)/2;
    if (tab.xi[mid] <= xi) lo = mid; else hi = mid; }
  const Real f = (xi - tab.xi[lo])/(tab.xi[hi] - tab.xi[lo]);
  *alpha = (1.0 - f)*tab.alpha[lo] + f*tab.alpha[hi];
  *v     = (1.0 - f)*tab.v[lo]     + f*tab.v[hi];
}

void ShuHistory(HistoryData *pdata, Mesh *pm);
void ShuFinalize(ParameterInput *pin, Mesh *pm);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &ShuHistory;
  pgen_final_func = &ShuFinalize;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->phydro == nullptr || pmbp->pgrav == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
        << "shu_collapse requires <particles>, <hydro> and <gravity> blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->phydro->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
        << "shu_collapse requires an isothermal EOS" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // parameters (code units: box = 4 L_J -> L_J = x1max/2; c_s and 4 pi G from inputs)
  const Real A     = pin->GetOrAddReal("problem", "A", 2.0004);
  const Real t0    = pin->GetOrAddReal("problem", "t0", 0.43);       // units of t_J
  const Real cs    = pin->GetReal("hydro", "iso_sound_speed");
  const Real fpg   = pin->GetReal("gravity", "four_pi_G");
  const Real G     = fpg/(4.0*M_PI);
  const Real vadvx = pin->GetOrAddReal("problem", "vadvx", 0.0)*cs;
  const Real vadvy = pin->GetOrAddReal("problem", "vadvy", 0.0)*cs;
  const Real vadvz = pin->GetOrAddReal("problem", "vadvz", 0.0)*cs;

  auto &msize = pmy_mesh_->mesh_size;
  auto &indcs = pmy_mesh_->mb_indcs;
  const Real dx = (msize.x1max - msize.x1min)/static_cast<Real>(pmy_mesh_->nmb_rootx1*indcs.nx1);
  // sphere centre: box centre offset by dx/2 per axis so the sink sits at a CELL CENTRE
  // (multigrid needs power-of-two blocks, so an odd grid a la GO13's 129^3 is not
  // possible; the offset achieves the same symmetry). Overridable via problem/x0...
  const Real x0 = pin->GetOrAddReal("problem", "x0",
                    0.5*(msize.x1min + msize.x1max) + 0.5*dx);
  const Real y0 = pin->GetOrAddReal("problem", "y0",
                    0.5*(msize.x2min + msize.x2max) + 0.5*dx);
  const Real z0 = pin->GetOrAddReal("problem", "z0",
                    0.5*(msize.x3min + msize.x3max) + 0.5*dx);
  const Real rmax = 0.75*msize.x1max;                   // 1.5 L_J for the 4 L_J box
  const Real rho_scale = 1.0/(fpg*t0*t0);               // rho = alpha/(4 pi G t0^2)

  // ---- similarity solution table (host, once) ----
  ShuTable tab = IntegrateShu(A);
  const Real rctrl = 1.5*dx;
  Real al_c, v_c;
  const Real xi_ctrl = rctrl/(cs*t0);
  ShuLookup(tab, xi_ctrl, &al_c, &v_c);
  const Real mstar = xi_ctrl*xi_ctrl*al_c*(xi_ctrl - v_c)*std::pow(cs, 3)*t0/G;

  mstar0_ = mstar; m0_ana_ = tab.m0; cs_ = cs; gm_ = G; t0_ = t0; a_par_ = A;

  if (global_variable::my_rank == 0) {
    std::printf("shu_collapse: A=%.6g t0=%.4g cs=%.4g G=%.6g\n", A, t0, cs, G);
    std::printf("  m0(analytic)=%.6f -> Mdot_ana=%.6e ;  M_*(0)=%.6e (xi_ctrl=%.4f)\n",
                tab.m0, tab.m0*std::pow(cs, 3)/G, mstar, xi_ctrl);
  }

  // ---- gas IC (host fill + deep copy; one-time setup cost) ----
  auto &u0 = pmbp->phydro->u0;
  auto u0_h = Kokkos::create_mirror_view(u0);
  auto &mbsize = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;

  for (int m = 0; m < nmb; ++m) {
    const Real dx1 = mbsize.h_view(m).dx1;
    const Real dx2 = mbsize.h_view(m).dx2;
    const Real dx3 = mbsize.h_view(m).dx3;
    for (int k = ks; k <= ke; ++k) {
      const Real z = mbsize.h_view(m).x3min + (k - ks + 0.5)*dx3 - z0;
      for (int j = js; j <= je; ++j) {
        const Real y = mbsize.h_view(m).x2min + (j - js + 0.5)*dx2 - y0;
        for (int i = is; i <= ie; ++i) {
          const Real x = mbsize.h_view(m).x1min + (i - is + 0.5)*dx1 - x0;
          const Real r = std::sqrt(x*x + y*y + z*z);
          // flatten the 3^3 control volume: use the profile just outside it (~2 dx),
          // approximating the reference's SetControlVolume() at particle creation
          const bool in_cv = (std::fabs(x) < 1.5*dx1 && std::fabs(y) < 1.5*dx2 &&
                              std::fabs(z) < 1.5*dx3);
          const Real reff = in_cv ? 2.0*dx1 : std::min(r, rmax);
          Real al, vv;
          ShuLookup(tab, reff/(cs*t0), &al, &vv);
          const Real rho = rho_scale*al;
          const Real vr  = (r <= rmax && !in_cv) ? cs*vv : (in_cv ? cs*vv : 0.0);
          // radial unit vector (regularized at r=0; CV flattening covers the centre)
          const Real rinv = (r > 0.0) ? 1.0/r : 0.0;
          u0_h(m, IDN, k, j, i) = rho;
          u0_h(m, IM1, k, j, i) = rho*(vr*x*rinv + vadvx);
          u0_h(m, IM2, k, j, i) = rho*(vr*y*rinv + vadvy);
          u0_h(m, IM3, k, j, i) = rho*(vr*z*rinv + vadvz);
        }
      }
    }
  }
  Kokkos::deep_copy(u0, u0_h);

  // ---- one sink at the sphere centre ----
  auto &pr = pmbp->ppart->prtcl_rdata;
  auto &pi = pmbp->ppart->prtcl_idata;
  const int npart = pmbp->ppart->nprtcl_thispack;
  auto gids = pmbp->gids;
  if (npart != 1 && global_variable::my_rank == 0) {
    std::printf("shu_collapse WARNING: npart=%d (expected 1; check particles/ppc)\n",
                npart);
  }
  par_for("shu_part", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(int p) {
    pr(IPX, p) = x0;  pr(IPY, p) = y0;  pr(IPZ, p) = z0;
    pr(IPVX, p) = vadvx; pr(IPVY, p) = vadvy; pr(IPVZ, p) = vadvz;
    pr(IPM, p) = mstar;
    pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
    pr(IPX0, p) = x0; pr(IPY0, p) = y0; pr(IPZ0, p) = z0;
    int mown = 0;
    for (int mm = 0; mm < nmb; ++mm) {
      if (x0 >= mbsize.d_view(mm).x1min && x0 < mbsize.d_view(mm).x1max &&
          y0 >= mbsize.d_view(mm).x2min && y0 < mbsize.d_view(mm).x2max &&
          z0 >= mbsize.d_view(mm).x3min && z0 < mbsize.d_view(mm).x3max) {
        mown = mm;
      }
    }
    pi(PGID, p) = gids + mown;
  });

  // seed particle dt for cycle 1 (Particles::NewTimeStep refreshes it every cycle)
  pmbp->ppart->dtnew = dx;

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void ShuHistory()
//! \brief gas mass/momenta + sink mass/velocity + conserved totals

void ShuHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  pdata->nhist = 10;
  pdata->label[0] = "mass";      // gas
  pdata->label[1] = "1-mom";
  pdata->label[2] = "2-mom";
  pdata->label[3] = "3-mom";
  pdata->label[4] = "m_sink";
  pdata->label[5] = "vx_sink";
  pdata->label[6] = "x_sink";
  pdata->label[7] = "M_tot";     // gas + sink (conservation check, periodic runs)
  pdata->label[8] = "px_tot";
  pdata->label[9] = "Mdot_ana";  // constant analytic reference m0*cs^3/G

  // gas sums
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
    Kokkos::parallel_reduce("shu_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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

  // sink quantities (host copy of the small particle array)
  const int npart = pmbp->ppart->nprtcl_thispack;
  auto pr_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_rdata);
  Kokkos::deep_copy(pr_h, pmbp->ppart->prtcl_rdata);
  Real msink = 0.0, vxsink = 0.0, xsink = 0.0, pxsink = 0.0;
  for (int p = 0; p < npart; ++p) {
    msink  += pr_h(IPM, p);
    vxsink += pr_h(IPVX, p);
    xsink  += pr_h(IPX, p);
    pxsink += pr_h(IPM, p)*pr_h(IPVX, p);
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
  pdata->hdata[9] = m0_ana_*std::pow(cs_, 3)/gm_;
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void ShuFinalize()
//! \brief report final sink mass and implied mean accretion rate vs analytic

void ShuFinalize(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->ppart == nullptr) return;
  const int npart = pmbp->ppart->nprtcl_thispack;
  auto pr_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_rdata);
  Kokkos::deep_copy(pr_h, pmbp->ppart->prtcl_rdata);
  Real msink = 0.0;
  for (int p = 0; p < npart; ++p) msink += pr_h(IPM, p);
  if (global_variable::my_rank == 0) {
    const Real mdot_mean = (pm->time > 0.0) ? (msink - mstar0_)/pm->time : 0.0;
    const Real mdot_ana = m0_ana_*std::pow(cs_, 3)/gm_;
    std::printf("\n=== shu_collapse finalize: A=%.6g t=%.6f ncycle=%d ===\n",
                a_par_, pm->time, pm->ncycle);
    std::printf("  M_sink=%.8e (seed %.8e)  <Mdot>=%.6e  Mdot_ana=%.6e  ratio=%.4f\n",
                msink, mstar0_, mdot_mean, mdot_ana, mdot_mean/mdot_ana);
  }
}
}  // namespace
