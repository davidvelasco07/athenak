//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydrostatic.cpp
//! \brief Hydrostatic equilibria for the well-balanced scheme (<hydro>/well_balanced),
//! from Velasco-Romero & Teyssier (2025, MNRAS, arXiv:2405.11063):
//!
//!  iprob=1  Exponential (isothermal-profile) atmosphere in a constant gravity g, with
//!           potential phi = -g.x: rho = rho0 exp(-rho0 phi/p0), p = p0 exp(-rho0 phi/p0)
//!           (Kaeppeli & Mishra 2016; the paper's appendix B for g along x1).  Optional
//!           Gaussian pressure pulse dp = amp*exp(-rho0|g| r^2/(p0*pulse_w2)) centred at
//!           (pulse_x1,pulse_x2,pulse_x3) -- the Xing & Shu (2013) acoustic
//!           perturbation of sec. 3.3 with oblique gravity g = -(1,1).
//!  iprob=2  Three-zone piecewise-polytropic atmosphere with a buoyant underdense bubble
//!           (sec. 3.4), gravity along -x2.  Zone base values are propagated by
//!           continuity from the bottom zone (the paper quotes them rounded).
//!  iprob=3  (MHD) iprob=1 threaded by a uniform, force-free field (b1,b2,b3).
//!  iprob=5  (MHD, 2D) iprob=1 with gravity along -x2 threaded by the current-free arcade
//!           A_z = A0 cos(kx) exp(-ky), k = 2 pi/Lx (B varies in BOTH directions, so the
//!           UCT edge reconstruction of the equilibrium field is exercised).  Face fields
//!           are differences of A_z at cell corners: div B = 0 to round-off.
//!  iprob=4  (MHD) Parker constant-beta atmosphere, gravity along -x2: horizontal field
//!           B_x = sqrt(2 p/beta), rho, p ~ exp(-x2/H), H = (p0/rho0)(1+1/beta)/g.
//!           A true magnetostatic equilibrium with J x B != 0 (magnetic pressure
//!           gradient).
//!  iprob=6  Two-layer Rayleigh-Taylor setup of sec. 3.2 (a modified Liska & Wendroff
//!           2003): rho1 below the sharp interface x2 = yc, rho2 above, p = P0 + g int rho
//!           with g along +x2 (the paper's frame: gravity points up, so the heavy lower
//!           layer is on top in the gravitational sense).  Seeded by
//!           v2 = -rti_amp*dv*cos(k x1), dv = sqrt(gamma (rho1 yc |g| + 1)/rho1), which
//!           does not depend on P0, so P0 sets the Mach number alone.  rti_sigma > 0
//!           replaces the sharp interface by a tanh profile of that width (the
//!           well-balanced scheme needs a smooth equilibrium: at a jump the full face
//!           state W_eq,face + W' can leave the range of the two layers).
//!
//! <problem>/user_hist=true (hydro) adds a history block: the global maximum Mach number
//! (all iprob) and, for iprob=2, the kinetic energy of each of the three zones.
//!
//! The gravity vector is read from <hydro_srcterms> const_accel (dir/val or the dir=0
//! vector form), so the equilibrium and the source term cannot disagree.  Boundaries
//! flagged `user` hold the unperturbed equilibrium in the ghost zones (U' = 0 with the
//! well-balanced scheme, U = U_eq without it).

#include <cmath>
#include <memory>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "hydro/well_balance.hpp"
#include "srcterms/srcterms.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

namespace {

constexpr int kMaxZones = 3;

//----------------------------------------------------------------------------------------
// point functors (primitive state, IEN slot = internal energy density)

struct ExpAtmosphere {
  Real rho0, p0, gm1, g1, g2, g3;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real w[]) const {
    const Real phi = -(g1*x1 + g2*x2 + g3*x3);
    const Real f = exp(-rho0*phi/p0);
    w[IDN] = rho0*f;
    w[IVX] = 0.0; w[IVY] = 0.0; w[IVZ] = 0.0;
    w[IEN] = p0*f/gm1;
  }
};

struct PressurePulse {       // only the internal-energy perturbation dp/(gamma-1)
  Real amp, c1, c2, c3, k2, gm1;
  bool multi_d, three_d;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real w[]) const {
    Real r2 = SQR(x1 - c1);
    if (multi_d) r2 += SQR(x2 - c2);
    if (three_d) r2 += SQR(x3 - c3);
    w[IEN] = amp*exp(-k2*r2)/gm1;
  }
};

struct PolytropeZones {
  int nz;
  Real zb[kMaxZones+1];     // zone boundaries along x2
  Real rho0[kMaxZones], p0[kMaxZones], gam0[kMaxZones];
  Real g;                   // |g| (gravity along -x2)
  Real gm1;                 // gas gamma - 1

  KOKKOS_INLINE_FUNCTION
  void State(const Real z, Real &rho, Real &p) const {
    int iz = 0;
    while (iz < nz-1 && z > zb[iz+1]) {++iz;}
    const Real gz = gam0[iz];
    const Real t = 1.0 - (gz - 1.0)/gz*rho0[iz]*g/p0[iz]*(z - zb[iz]);
    rho = rho0[iz]*pow(t, 1.0/(gz - 1.0));
    p   = p0[iz]*pow(t, gz/(gz - 1.0));
  }
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real w[]) const {
    Real rho, p;
    State(x2, rho, p);
    w[IDN] = rho;
    w[IVX] = 0.0; w[IVY] = 0.0; w[IVZ] = 0.0;
    w[IEN] = p/gm1;
  }
};

struct BubbleDensity {       // only the density perturbation, pressure unperturbed
  PolytropeZones z;
  Real amp, c1, c2, rb;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real w[]) const {
    const Real r2 = SQR(x1 - c1) + SQR(x2 - c2);
    if (r2 < SQR(rb)) {
      Real rho, p;
      z.State(x2, rho, p);
      w[IDN] = amp*rho*exp(-r2/(2.0*SQR(rb)));
    }
  }
};

struct TwoLayer {            // two-layer RTI background (iprob=6), vertical x2
  Real rho1, rho2, yc, p0, g, gm1;
  Real sigma;                // 0: sharp interface; > 0: tanh profile of this width
  KOKKOS_INLINE_FUNCTION
  static Real LnCosh(const Real a) {return fabs(a) + log1p(exp(-2.0*fabs(a))) - log(2.0);}
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real w[]) const {
    Real rho, p;
    if (sigma > 0.0) {     // p = p0 + g int_0^y rho, analytic for the tanh profile
      const Real h = 0.5*(rho1 - rho2);
      rho = rho2 + h*(1.0 - tanh((x2 - yc)/sigma));
      p = p0 + g*(rho2*x2 + h*(x2 - sigma*(LnCosh((x2 - yc)/sigma) - LnCosh(yc/sigma))));
    } else {
      const bool below = (x2 <= yc);
      rho = below ? rho1 : rho2;
      p = below ? (p0 + g*rho1*x2) : (p0 + g*(rho1*yc + rho2*(x2 - yc)));
    }
    w[IDN] = rho;
    w[IVX] = 0.0; w[IVY] = 0.0; w[IVZ] = 0.0;
    w[IEN] = p/gm1;
  }
};

struct RTIVelocity {         // conserved increments of the v2 seed (iprob=6)
  TwoLayer eq;
  Real vamp, kx;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real w[]) const {
    Real weq[5];
    eq(x1, x2, x3, weq);
    const Real v2 = -vamp*cos(kx*x1);
    w[IM2] = weq[IDN]*v2;
    w[IEN] = 0.5*weq[IDN]*v2*v2;
  }
};

struct EqField {             // equilibrium magnetic field at a point
  bool parker;
  Real b1, b2, b3;           // uniform field (iprob=3)
  Real bp0, hscale;          // Parker: B_x = bp0*exp(-x2/(2H))
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real b[3]) const {
    if (parker) {
      b[0] = bp0*exp(-0.5*x2/hscale); b[1] = 0.0; b[2] = 0.0;
    } else {
      b[0] = b1; b[1] = b2; b[2] = b3;
    }
  }
};

// problem parameters shared by the pgen, equilibrium and BC functions
int iprob = 1;
ExpAtmosphere exp_atm;
PolytropeZones poly;
TwoLayer two_layer;
EqField eq_field;
DvceArray5D<Real> ubc_full;   // equilibrium conserved state for non-WB user BCs
// equilibrium face field for user BCs.  A pointer: a file-scope View would be
// constructed before Kokkos::initialize()
std::unique_ptr<DvceFaceFld4D<Real>> bbc_full;

// arcade parameters (iprob=5)
Real arc_a0 = 0.0, arc_k = 1.0;

// face field of the arcade from A_z at cell corners (2D): exactly telescoping, so the
// discrete div B vanishes to round-off.  Global-lattice coordinates.
void FillArcadeField(MeshBlockPack *pmbp, DvceFaceFld4D<Real> &b) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int n1m1 = indcs.nx1 + 2*indcs.ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng - 1) : 0;
  const wellbalance::GlobalGeometry g = wellbalance::MakeGlobalGeometry(pmbp);
  const Real a0 = arc_a0, kk = arc_k;
  auto b1 = b.x1f, b2 = b.x2f, b3 = b.x3f;
  par_for("arcade_b", DevExeSpace(), 0, pmbp->nmb_thispack-1, 0, n3m1,
          0, n2m1+1, 0, n1m1+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real xl, xr, yl, yr;
    g.CellEdges(1, m, i, xl, xr);
    g.CellEdges(2, m, j, yl, yr);
    auto az = [=](const Real x, const Real y) {return a0*cos(kk*x)*exp(-kk*y);};
    if (j <= n2m1) {   // x1-face at (xl, cell j): Bx = dAz/dy
      b1(m,k,j,i) = (az(xl, yr) - az(xl, yl))/(yr - yl);
    }
    if (i <= n1m1) {   // x2-face at (cell i, yl): By = -dAz/dx
      b2(m,k,j,i) = -(az(xr, yl) - az(xl, yl))/(xr - xl);
    }
    if (i <= n1m1 && j <= n2m1) {
      b3(m,k,j,i) = 0.0;
      b3(m,k+1,j,i) = 0.0;
    }
  });
}

// fill w (primitive cell averages) with the equilibrium of the selected problem
void FillEquilibrium(MeshBlockPack *pmbp, DvceArray5D<Real> &w, int nvar) {
  if (iprob == 2) {
    wellbalance::FillEquilibriumCellAverages(pmbp, w, nvar, poly);
  } else if (iprob == 6) {
    wellbalance::FillEquilibriumCellAverages(pmbp, w, nvar, two_layer);
  } else {
    wellbalance::FillEquilibriumCellAverages(pmbp, w, nvar, exp_atm);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HSEEquilibrium()
//! \brief user_eq_func: the well-balanced equilibrium (primitive cell averages).

void HSEEquilibrium(Mesh *pm) {
  auto *phydro = pm->pmb_pack->phydro;
  auto *pmhd = pm->pmb_pack->pmhd;
  if (phydro != nullptr) {
    FillEquilibrium(pm->pmb_pack, phydro->w_eq, phydro->nhydro);
  } else {
    FillEquilibrium(pm->pmb_pack, pmhd->w_eq, pmhd->nmhd);
    if (iprob == 5) {
      FillArcadeField(pm->pmb_pack, pmhd->b_eq);
    } else {
      wellbalance::FillEquilibriumFaceAverages(pm->pmb_pack, pmhd->b_eq, eq_field);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HSEBoundary()
//! \brief user BCs: ghost zones hold the unperturbed equilibrium.  With the well-balanced
//! scheme u0 is the perturbation, so that is U' = 0; otherwise U = U_eq.

void HSEBoundary(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  const bool multi_d = pm->multi_d, three_d = pm->three_d;
  const bool wb = (phydro != nullptr) ? phydro->well_balanced : pmhd->well_balanced;
  const int nvar = (phydro != nullptr) ? phydro->nhydro + phydro->nscalars
                                       : pmhd->nmhd + pmhd->nscalars;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto u0 = (phydro != nullptr) ? phydro->u0 : pmhd->u0;
  auto ubc = ubc_full;   // copy the View handle: a reference to a file-scope View is not captured by device lambdas
  par_for("hse_bc", DevExeSpace(), 0, (pmbp->nmb_thispack-1), 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    auto user = [&](const int f) {return mb_bcs.d_view(m,f) == BoundaryFlag::user;};
    const bool ghost =
      (i < is && user(BoundaryFace::inner_x1)) ||
      (i > ie && user(BoundaryFace::outer_x1)) ||
      (multi_d && j < js && user(BoundaryFace::inner_x2)) ||
      (multi_d && j > je && user(BoundaryFace::outer_x2)) ||
      (three_d && k < ks && user(BoundaryFace::inner_x3)) ||
      (three_d && k > ke && user(BoundaryFace::outer_x3));
    if (ghost) {
      for (int n=0; n<nvar; ++n) {
        u0(m,n,k,j,i) = wb ? 0.0 : ubc(m,n,k,j,i);
      }
    }
  });
  // MHD: ghost faces hold the (full) equilibrium face field.  The faces ON a boundary
  // belong to the active domain (they are evolved by CT) and are left alone.
  if (pmhd != nullptr) {
    auto b0 = pmhd->b0;
    auto bq = *bbc_full;
    // during the well-balanced boundary phase b0 holds B' (mhd.hpp): the equilibrium
    // ghosts are then B' = 0, which also stays valid after AMR regrids (bbc_full is
    // sized for the initial blocks)
    const bool bpert = pmhd->wb_b0_is_pert;
    par_for("hse_bc_b", DevExeSpace(), 0, (pmbp->nmb_thispack-1), 0, n3, 0, n2, 0, n1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      auto user = [&](const int f) {return mb_bcs.d_view(m,f) == BoundaryFlag::user;};
      auto gx = [&](const int ii, const int jj, const int kk, const int iu, const int ju,
                    const int ku) {
        return (ii < is && user(BoundaryFace::inner_x1)) ||
               (ii > iu && user(BoundaryFace::outer_x1)) ||
               (multi_d && jj < js && user(BoundaryFace::inner_x2)) ||
               (multi_d && jj > ju && user(BoundaryFace::outer_x2)) ||
               (three_d && kk < ks && user(BoundaryFace::inner_x3)) ||
               (three_d && kk > ku && user(BoundaryFace::outer_x3));
      };
      if (k < n3 && j < n2 && gx(i, j, k, ie+1, je, ke)) {
        b0.x1f(m,k,j,i) = bpert ? 0.0 : bq.x1f(m,k,j,i);
      }
      if (k < n3 && i < n1 && (multi_d || j < 2) && gx(i, j, k, ie, je+1, ke)) {
        b0.x2f(m,k,j,i) = bpert ? 0.0 : bq.x2f(m,k,j,i);
      }
      if (j < n2 && i < n1 && (three_d || k < 2) && gx(i, j, k, ie, je, ke+1)) {
        b0.x3f(m,k,j,i) = bpert ? 0.0 : bq.x3f(m,k,j,i);
      }
    });
  }
}

//----------------------------------------------------------------------------------------
// Test-only AMR criteria (<amr_criterion0> method = user), chosen by <problem>/amr_test:
//   band  refine every MeshBlock overlapping |x1 - xc(t)| < amr_band_w/2 (periodic in x1),
//         xc = amr_band_x0 + amr_band_v*t, derefine all others.  It ignores the solution, so a
//         static equilibrium is refined and derefined every few steps: the WB exact gate
//         through regrids, load balancing and new-block equilibria.
//   pert  refine where max(|U'_d|/U_eq,d, |U'_E|/U_eq,E) over the block exceeds amr_pert_ref,
//         derefine below amr_pert_ref/4.  U' = u0 with WB, u0 - u_eq without (needs
//         wb_diagnostics).  A WB-aware criterion that follows a perturbation.  Optional window
//         amr_x2min < x2 < amr_x2max: blocks lying entirely outside it are always derefined (e.g.
//         keep the stable zones of the bubble at root resolution; default: no window).
struct AMRTest {
  int mode = 0;                         // 0 none, 1 band, 2 pert
  Real x0 = 0.0, v = 0.0, w = 0.0, ref = 0.0;
  Real x2lo = -1.0e300, x2hi = 1.0e300;  // pert: refinement window along x2
};
AMRTest amr_test;

void HSERefinement(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  auto &refine_flag = pm->pmr->refine_flag;
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  const int nmb = pmbp->nmb_thispack;
  if (amr_test.mode == 1) {
    auto &size = pmbp->pmb->mb_size;
    const Real xmin = pm->mesh_size.x1min, len = pm->mesh_size.x1max - xmin;
    Real xc = std::fmod(amr_test.x0 + amr_test.v*pm->time - xmin, len);
    xc = xmin + ((xc < 0.0) ? xc + len : xc);
    for (int m = 0; m < nmb; ++m) {
      const Real a = size.h_view(m).x1min, b = size.h_view(m).x1max;
      bool in = false;
      for (int s = -1; s <= 1; ++s) {                 // periodic images of the band
        const Real lo = xc - 0.5*amr_test.w + s*len, hi = xc + 0.5*amr_test.w + s*len;
        if (b > lo && a < hi) in = true;
      }
      refine_flag.h_view(m + mbs) = in ? 1 : -1;
    }
    refine_flag.template modify<HostMemSpace>();
    refine_flag.template sync<DevExeSpace>();
    return;
  }
  // pert: device reduction per block (hydro or MHD; both store U' in u0 with WB)
  const bool hyd = (pmbp->phydro != nullptr);
  auto &u0 = hyd ? pmbp->phydro->u0 : pmbp->pmhd->u0;
  auto &ueq = hyd ? pmbp->phydro->u_eq : pmbp->pmhd->u_eq;
  const bool wb = hyd ? pmbp->phydro->well_balanced : pmbp->pmhd->well_balanced;
  const Real ref = amr_test.ref;
  const Real x2lo = amr_test.x2lo, x2hi = amr_test.x2hi;
  auto &msize = pmbp->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  par_for_outer("hse_ref_pert", DevExeSpace(), 0, 0, 0, nmb - 1,
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real qmax = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [=](const int idx, Real &q) {
      const int k = idx/nji + ks;
      const int j = (idx - (k - ks)*nji)/nx1 + js;
      const int i = (idx - (k - ks)*nji - (j - js)*nx1) + is;
      const Real dd = wb ? u0(m,IDN,k,j,i) : u0(m,IDN,k,j,i) - ueq(m,IDN,k,j,i);
      const Real de = wb ? u0(m,IEN,k,j,i) : u0(m,IEN,k,j,i) - ueq(m,IEN,k,j,i);
      q = fmax(q, fmax(fabs(dd)/ueq(m,IDN,k,j,i), fabs(de)/ueq(m,IEN,k,j,i)));
    }, Kokkos::Max<Real>(qmax));
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      const bool outside = (msize.d_view(m).x2max <= x2lo) || (msize.d_view(m).x2min >= x2hi);
      refine_flag.d_view(m + mbs) = outside ? -1 :
                                    ((qmax > ref) ? 1 : ((qmax < 0.25*ref) ? -1 : 0));
    });
  });
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void HSEHistory()
//! \brief user history (hydro): global max Mach number and, for iprob=2, the kinetic
//! energy of each zone.  The history writer SUMS every entry over ranks, so the max is
//! reduced here and stored on rank 0 only.  w0 holds the FULL primitives (also with WB).

void HSEHistory(HistoryData *pdata, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *ph = pmbp->phydro;
  const bool zones = (iprob == 2);
  const bool amrh = (amr_test.mode != 0);
  const int namr = zones ? 4 : 1;                 // first column of the AMR-test block
  pdata->nhist = namr + (amrh ? 2 : 0);
  pdata->label[0] = "Mach-max";
  if (zones) {
    pdata->label[1] = "KE-zone1";
    pdata->label[2] = "KE-zone2";
    pdata->label[3] = "KE-zone3";
  }
  if (amrh) {
    pdata->label[namr] = "nmb-total";
    pdata->label[namr + 1] = "pert-max";
  }
  auto &w0 = ph->w0;
  const Real gamma = ph->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  const Real z1 = poly.zb[1], z2 = poly.zb[2];

  Real mmax = 0.0;
  Kokkos::parallel_reduce("hse_hist_mach", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &lmax) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
    const int i = (idx - m*nkji - (k - ks)*nji - (j - js)*nx1) + is;
    const Real v2 = SQR(w0(m,IVX,k,j,i)) + SQR(w0(m,IVY,k,j,i)) + SQR(w0(m,IVZ,k,j,i));
    const Real cs2 = gamma*gm1*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i);
    lmax = fmax(lmax, sqrt(v2/cs2));
  }, Kokkos::Max<Real>(mmax));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &mmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  pdata->hdata[0] = (global_variable::my_rank == 0) ? mmax : 0.0;

  if (zones) {
    array_sum::GlobalSum ke;
    Kokkos::parallel_reduce("hse_hist_ke", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &lsum) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
      const int i = (idx - m*nkji - (k - ks)*nji - (j - js)*nx1) + is;
      const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      const Real x2v = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                                   size.d_view(m).x2max);
      const int iz = (x2v < z1) ? 0 : ((x2v < z2) ? 1 : 2);
      array_sum::GlobalSum h;
      for (int n=0; n<NHISTORY_VARIABLES; ++n) {h.the_array[n] = 0.0;}
      h.the_array[iz] = vol*0.5*w0(m,IDN,k,j,i)*(SQR(w0(m,IVX,k,j,i)) +
                        SQR(w0(m,IVY,k,j,i)) + SQR(w0(m,IVZ,k,j,i)));
      lsum += h;
    }, Kokkos::Sum<array_sum::GlobalSum>(ke));
    for (int n=0; n<3; ++n) {pdata->hdata[1+n] = ke.the_array[n];}
  }

  if (amrh) {   // block count, and max(|U'_d|/U_eq,d, |U'_E|/U_eq,E) over the whole mesh
    auto &u0 = ph->u0;
    auto &ueq = ph->u_eq;
    const bool wb = ph->well_balanced;
    Real pmax = 0.0;
    Kokkos::parallel_reduce("hse_hist_pert", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &lmax) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k - ks)*nji)/nx1 + js;
      const int i = (idx - m*nkji - (k - ks)*nji - (j - js)*nx1) + is;
      const Real dd = wb ? u0(m,IDN,k,j,i) : u0(m,IDN,k,j,i) - ueq(m,IDN,k,j,i);
      const Real de = wb ? u0(m,IEN,k,j,i) : u0(m,IEN,k,j,i) - ueq(m,IEN,k,j,i);
      lmax = fmax(lmax, fmax(fabs(dd)/ueq(m,IDN,k,j,i), fabs(de)/ueq(m,IEN,k,j,i)));
    }, Kokkos::Max<Real>(pmax));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &pmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    const bool r0 = (global_variable::my_rank == 0);
    pdata->hdata[namr] = r0 ? static_cast<Real>(pm->nmb_total) : 0.0;
    pdata->hdata[namr + 1] = r0 ? pmax : 0.0;
  }
}

// release the file-scope view before Kokkos::finalize
void HSEFinalize(ParameterInput *pin, Mesh *pm) {
  ubc_full = DvceArray5D<Real>();
  bbc_full.reset();
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Hydrostatic()

void ProblemGenerator::Hydrostatic(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *phydro = pmbp->phydro;
  auto *pmhd = pmbp->pmhd;
  EquationOfState *peos_ = (phydro != nullptr) ? phydro->peos
                         : ((pmhd != nullptr) ? pmhd->peos : nullptr);
  SourceTerms *psrc_ = (phydro != nullptr) ? phydro->psrc
                     : ((pmhd != nullptr) ? pmhd->psrc : nullptr);
  if (peos_ == nullptr || !peos_->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "hydrostatic pgen requires <hydro> or <mhd> with an ideal gas EOS"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (psrc_ == nullptr || !psrc_->const_accel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "hydrostatic pgen requires <hydro_srcterms>/<mhd_srcterms> "
              << "const_accel=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real gm1 = peos_->eos_data.gamma - 1.0;
  const Real *gv = psrc_->const_accel_vec;
  iprob = pin->GetOrAddInteger("problem", "iprob", 1);
  if ((iprob >= 3 && iprob <= 5) != (pmhd != nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "hydrostatic: iprob=1,2,6 are hydro problems, iprob=3,4,5 MHD"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  eq_field.parker = false;
  eq_field.b1 = eq_field.b2 = eq_field.b3 = 0.0;

  if (iprob == 1 || iprob == 3 || iprob == 5) {
    exp_atm.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    exp_atm.p0 = pin->GetOrAddReal("problem", "p0", 1.0);
    exp_atm.gm1 = gm1;
    exp_atm.g1 = gv[0]; exp_atm.g2 = gv[1]; exp_atm.g3 = gv[2];
    if (iprob == 5) {
      if (!pmy_mesh_->two_d) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "hydrostatic iprob=5 is 2D only" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      arc_k = 2.0*M_PI/(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
      arc_a0 = pin->GetOrAddReal("problem", "arcade_b0", 0.5)/arc_k;
    }
    if (iprob == 3) {
      eq_field.b1 = pin->GetOrAddReal("problem", "b1", 0.0);
      eq_field.b2 = pin->GetOrAddReal("problem", "b2", 0.0);
      eq_field.b3 = pin->GetOrAddReal("problem", "b3", 0.0);
    }
  } else if (iprob == 4) {
    if (gv[0] != 0.0 || gv[2] != 0.0 || gv[1] >= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "hydrostatic iprob=4 needs gravity along -x2"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real beta = pin->GetOrAddReal("problem", "beta", 1.0);
    exp_atm.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    exp_atm.p0 = pin->GetOrAddReal("problem", "p0", 1.0);
    exp_atm.gm1 = gm1;
    // rho, p ~ exp(-x2/H): the exponential atmosphere with g_eff = g/(1+1/beta)
    exp_atm.g1 = 0.0; exp_atm.g2 = gv[1]/(1.0 + 1.0/beta); exp_atm.g3 = 0.0;
    eq_field.parker = true;
    eq_field.hscale = (exp_atm.p0/exp_atm.rho0)*(1.0 + 1.0/beta)/(-gv[1]);
    eq_field.bp0 = std::sqrt(2.0*exp_atm.p0/beta);
  } else if (iprob == 2) {
    if (gv[0] != 0.0 || gv[2] != 0.0 || gv[1] >= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "hydrostatic iprob=2 needs gravity along -x2"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    poly.nz = 3;
    poly.g = -gv[1];
    poly.gm1 = gm1;
    poly.zb[0] = pin->GetOrAddReal("problem", "z0", 0.0);
    poly.zb[1] = pin->GetOrAddReal("problem", "z1", 1.64);
    poly.zb[2] = pin->GetOrAddReal("problem", "z2", 7.77);
    poly.zb[3] = pin->GetOrAddReal("problem", "z3", 11.0);
    poly.gam0[0] = pin->GetOrAddReal("problem", "gam0_1", 1.20);
    poly.gam0[1] = pin->GetOrAddReal("problem", "gam0_2", 1.67);
    poly.gam0[2] = pin->GetOrAddReal("problem", "gam0_3", 1.01);
    poly.rho0[0] = pin->GetOrAddReal("problem", "rho0_1", 1.81);
    poly.p0[0] = pin->GetOrAddReal("problem", "p0_1", 5.94);
    // continuity: each zone starts from the top of the zone below
    for (int iz=1; iz<poly.nz; ++iz) {
      const Real gz = poly.gam0[iz-1];
      const Real t = 1.0 - (gz - 1.0)/gz*poly.rho0[iz-1]*poly.g/poly.p0[iz-1]
                           *(poly.zb[iz] - poly.zb[iz-1]);
      poly.rho0[iz] = poly.rho0[iz-1]*std::pow(t, 1.0/(gz - 1.0));
      poly.p0[iz] = poly.p0[iz-1]*std::pow(t, gz/(gz - 1.0));
    }
  } else if (iprob == 6) {
    if (gv[0] != 0.0 || gv[2] != 0.0 || gv[1] == 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "hydrostatic iprob=6 needs gravity along x2"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    two_layer.rho1 = pin->GetOrAddReal("problem", "rti_rho1", 2.0);
    two_layer.rho2 = pin->GetOrAddReal("problem", "rti_rho2", 1.0);
    two_layer.yc = pin->GetOrAddReal("problem", "rti_yc", 0.5);
    two_layer.p0 = pin->GetOrAddReal("problem", "rti_p0", 1.0);
    two_layer.sigma = pin->GetOrAddReal("problem", "rti_sigma", 0.0);
    two_layer.g = gv[1];
    two_layer.gm1 = gm1;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "hydrostatic: iprob must be 1..6" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // test-only AMR criteria (see HSERefinement); the deck must use <amr_criterion0> method=user
  {
    const std::string at = pin->GetOrAddString("problem", "amr_test", "none");
    amr_test.mode = (at == "band") ? 1 : ((at == "pert") ? 2 : 0);
    if (at != "none" && amr_test.mode == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "hydrostatic: amr_test must be none, band or pert" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const bool amr_ok = (phydro != nullptr) ?
        (phydro->peos->eos_data.is_ideal &&
         (phydro->well_balanced || phydro->wb_diagnostics)) :
        (pmhd != nullptr && pmhd->peos->eos_data.is_ideal &&
         (pmhd->well_balanced || pmhd->wb_diagnostics));
    if (amr_test.mode != 0 && !amr_ok) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "hydrostatic amr_test needs an ideal gas with well_balanced or "
                << "wb_diagnostics (it measures U' against u_eq)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    amr_test.x0 = pin->GetOrAddReal("problem", "amr_band_x0", 0.0);
    amr_test.v = pin->GetOrAddReal("problem", "amr_band_v", 1.0);
    amr_test.w = pin->GetOrAddReal("problem", "amr_band_w", 0.25);
    amr_test.ref = pin->GetOrAddReal("problem", "amr_pert_ref", 1.0e-6);
    if (pin->DoesParameterExist("problem", "amr_x2min")) {
      amr_test.x2lo = pin->GetReal("problem", "amr_x2min");
    }
    if (pin->DoesParameterExist("problem", "amr_x2max")) {
      amr_test.x2hi = pin->GetReal("problem", "amr_x2max");
    }
    if (amr_test.mode != 0) user_ref_func = HSERefinement;
  }

  // hooks (also needed on restart)
  user_eq_func = HSEEquilibrium;
  if (user_bcs) user_bcs_func = HSEBoundary;
  if (phydro != nullptr) user_hist_func = HSEHistory;
  pgen_final_func = HSEFinalize;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1m1 = indcs.nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
  const int nhyd = (phydro != nullptr) ? phydro->nhydro : pmhd->nmhd;
  const int nvar = nhyd + ((phydro != nullptr) ? phydro->nscalars : pmhd->nscalars);
  auto &w0 = (phydro != nullptr) ? phydro->w0 : pmhd->w0;
  auto &u0 = (phydro != nullptr) ? phydro->u0 : pmhd->u0;

  // full equilibrium state for the user BCs (u0 ghosts without WB; b0 ghosts always)
  if (user_bcs) {
    DvceArray5D<Real> wtmp("hse_wtmp", w0.extent(0), nvar, w0.extent(2), w0.extent(3),
                           w0.extent(4));
    Kokkos::deep_copy(wtmp, 0.0);
    FillEquilibrium(pmbp, wtmp, nhyd);
    Kokkos::realloc(ubc_full, u0.extent(0), nvar, u0.extent(2), u0.extent(3),
                    u0.extent(4));
    if (phydro != nullptr) {
      phydro->peos->PrimToCons(wtmp, ubc_full, 0, n1m1, 0, n2m1, 0, n3m1);
    } else {
      auto &b0 = pmhd->b0;
      bbc_full = std::make_unique<DvceFaceFld4D<Real>>("hse_bbc", b0.x1f.extent(0),
                   b0.x1f.extent(1), b0.x1f.extent(2), b0.x2f.extent(3));
      if (iprob == 5) {
        FillArcadeField(pmbp, *bbc_full);
      } else {
        wellbalance::FillEquilibriumFaceAverages(pmbp, *bbc_full, eq_field);
      }
      DvceArray5D<Real> btmp("hse_btmp", w0.extent(0), 3, w0.extent(2), w0.extent(3),
                             w0.extent(4));
      pmhd->WBBuildFullState(wtmp, *bbc_full, btmp, ubc_full);
    }
  }
  if (restart) return;

  // unperturbed state: the SAME cell averages and the SAME conversion the well-balanced
  // scheme uses for U_eq, so U - U_eq is exactly zero where unperturbed
  Kokkos::deep_copy(w0, 0.0);
  FillEquilibrium(pmbp, w0, nhyd);
  if (phydro != nullptr) {
    phydro->peos->PrimToCons(w0, u0, 0, n1m1, 0, n2m1, 0, n3m1);
  } else {
    if (iprob == 5) {
      FillArcadeField(pmbp, pmhd->b0);
    } else {
      wellbalance::FillEquilibriumFaceAverages(pmbp, pmhd->b0, eq_field);
    }
    pmhd->WBBuildFullState(w0, pmhd->b0, pmhd->bcc0, u0);
  }

  // perturbation, as cell averages of its own functor
  DvceArray5D<Real> dw("hse_dw", w0.extent(0), nvar, w0.extent(2), w0.extent(3),
                       w0.extent(4));
  Kokkos::deep_copy(dw, 0.0);
  bool perturb = false;
  if (iprob == 1 || iprob == 3 || iprob == 4 || iprob == 5) {
    const Real amp = pin->GetOrAddReal("problem", "pulse_amp", 0.0);
    if (amp != 0.0) {
      PressurePulse pp;
      pp.amp = amp;
      pp.c1 = pin->GetOrAddReal("problem", "pulse_x1", 0.3);
      pp.c2 = pin->GetOrAddReal("problem", "pulse_x2", 0.3);
      pp.c3 = pin->GetOrAddReal("problem", "pulse_x3", 0.3);
      const Real w2 = pin->GetOrAddReal("problem", "pulse_w2", 0.01);
      const Real gmag = std::sqrt(SQR(gv[0]) + SQR(gv[1]) + SQR(gv[2]));
      // width rho0|g|/(p0 w2) as in the paper; pulse_k2 overrides it (e.g. for g = 0)
      pp.k2 = pin->GetOrAddReal("problem", "pulse_k2", exp_atm.rho0*gmag/(exp_atm.p0*w2));
      pp.gm1 = gm1;
      pp.multi_d = pmy_mesh_->multi_d;
      pp.three_d = pmy_mesh_->three_d;
      wellbalance::FillEquilibriumCellAverages(pmbp, dw, nhyd, pp);
      perturb = true;
    }
  } else if (iprob == 6) {
    RTIVelocity rv;
    rv.eq = two_layer;
    const Real amp = pin->GetOrAddReal("problem", "rti_amp", 0.025);
    const Real dv = std::sqrt((gm1 + 1.0)*(two_layer.rho1*two_layer.yc*std::abs(gv[1])
                                           + 1.0)/two_layer.rho1);
    rv.vamp = amp*dv;
    rv.kx = 2.0*M_PI*pin->GetOrAddReal("problem", "rti_mode", 1.0)
            /(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
    if (amp != 0.0) {
      wellbalance::FillEquilibriumCellAverages(pmbp, dw, nhyd, rv);
      perturb = true;
    }
  } else {
    BubbleDensity bd;
    bd.z = poly;
    bd.amp = pin->GetOrAddReal("problem", "bubble_amp", -0.012);
    bd.rb = pin->GetOrAddReal("problem", "bubble_r", 0.25);
    const Real xlen = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
    bd.c1 = pin->GetOrAddReal("problem", "bubble_x1",
                              pmy_mesh_->mesh_size.x1min + 0.5*xlen);
    bd.c2 = pin->GetOrAddReal("problem", "bubble_x2", poly.zb[1] + 0.75);
    if (bd.amp != 0.0) {
      wellbalance::FillEquilibriumCellAverages(pmbp, dw, nhyd, bd);
      perturb = true;
    }
  }
  if (perturb) {
    // dw holds conserved increments (the pulse and bubble set only IDN/IEN)
    par_for("hse_perturb", DevExeSpace(), 0, (pmbp->nmb_thispack-1), 0, n3m1, 0, n2m1,
            0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int n=0; n<nhyd; ++n) {u0(m,n,k,j,i) += dw(m,n,k,j,i);}
    });
  }
  return;
}
