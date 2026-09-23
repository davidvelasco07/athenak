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
//!           potential phi = -g.x:  rho = rho0 exp(-rho0 phi/p0), p = p0 exp(-rho0 phi/p0)
//!           (Kaeppeli & Mishra 2016; the paper's appendix B for g along x1).  Optional
//!           Gaussian pressure pulse dp = amp*exp(-rho0|g| r^2/(p0*pulse_w2)) centred at
//!           (pulse_x1,pulse_x2,pulse_x3) -- the Xing & Shu (2013) acoustic perturbation of
//!           sec. 3.3 with oblique gravity g = -(1,1).
//!  iprob=2  Three-zone piecewise-polytropic atmosphere with a buoyant underdense bubble
//!           (sec. 3.4), gravity along -x2.  Zone base values are propagated by
//!           continuity from the bottom zone (the paper quotes them rounded).
//!
//! The gravity vector is read from <hydro_srcterms> const_accel (dir/val or the dir=0
//! vector form), so the equilibrium and the source term cannot disagree.  Boundaries
//! flagged `user` hold the unperturbed equilibrium in the ghost zones (U' = 0 with the
//! well-balanced scheme, U = U_eq without it).

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "hydro/well_balance.hpp"
#include "srcterms/srcterms.hpp"
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

// problem parameters shared by the pgen, equilibrium and BC functions
int iprob = 1;
ExpAtmosphere exp_atm;
PolytropeZones poly;
DvceArray5D<Real> ubc_full;   // equilibrium conserved state for non-WB user BCs

// fill w (primitive cell averages) with the equilibrium of the selected problem
void FillEquilibrium(MeshBlockPack *pmbp, DvceArray5D<Real> &w, int nvar) {
  if (iprob == 1) {
    wellbalance::FillEquilibriumCellAverages(pmbp, w, nvar, exp_atm);
  } else {
    wellbalance::FillEquilibriumCellAverages(pmbp, w, nvar, poly);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HSEEquilibrium()
//! \brief user_eq_func: the well-balanced equilibrium (primitive cell averages).

void HSEEquilibrium(Mesh *pm) {
  auto *phydro = pm->pmb_pack->phydro;
  FillEquilibrium(pm->pmb_pack, phydro->w_eq, phydro->nhydro);
}

//----------------------------------------------------------------------------------------
//! \fn void HSEBoundary()
//! \brief user BCs: ghost zones hold the unperturbed equilibrium.  With the well-balanced
//! scheme u0 is the perturbation, so that is U' = 0; otherwise U = U_eq.

void HSEBoundary(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *phydro = pmbp->phydro;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  const bool multi_d = pm->multi_d, three_d = pm->three_d;
  const bool wb = phydro->well_balanced;
  const int nvar = phydro->nhydro + phydro->nscalars;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = phydro->u0;
  auto &ubc = ubc_full;
  par_for("hse_bc", DevExeSpace(), 0, (pmbp->nmb_thispack-1), 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const bool ghost =
      (i < is && mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) ||
      (i > ie && mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) ||
      (multi_d && j < js && mb_bcs.d_view(m,BoundaryFace::inner_x2)==BoundaryFlag::user) ||
      (multi_d && j > je && mb_bcs.d_view(m,BoundaryFace::outer_x2)==BoundaryFlag::user) ||
      (three_d && k < ks && mb_bcs.d_view(m,BoundaryFace::inner_x3)==BoundaryFlag::user) ||
      (three_d && k > ke && mb_bcs.d_view(m,BoundaryFace::outer_x3)==BoundaryFlag::user);
    if (ghost) {
      for (int n=0; n<nvar; ++n) {
        u0(m,n,k,j,i) = wb ? 0.0 : ubc(m,n,k,j,i);
      }
    }
  });
}

// release the file-scope view before Kokkos::finalize
void HSEFinalize(ParameterInput *pin, Mesh *pm) {
  ubc_full = DvceArray5D<Real>();
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Hydrostatic()

void ProblemGenerator::Hydrostatic(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *phydro = pmbp->phydro;
  if (phydro == nullptr || !phydro->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "hydrostatic pgen requires <hydro> with an ideal gas EOS" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (phydro->psrc == nullptr || !phydro->psrc->const_accel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "hydrostatic pgen requires <hydro_srcterms> const_accel=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real gm1 = phydro->peos->eos_data.gamma - 1.0;
  const Real *gv = phydro->psrc->const_accel_vec;
  iprob = pin->GetOrAddInteger("problem", "iprob", 1);

  if (iprob == 1) {
    exp_atm.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
    exp_atm.p0 = pin->GetOrAddReal("problem", "p0", 1.0);
    exp_atm.gm1 = gm1;
    exp_atm.g1 = gv[0]; exp_atm.g2 = gv[1]; exp_atm.g3 = gv[2];
  } else if (iprob == 2) {
    if (gv[0] != 0.0 || gv[2] != 0.0 || gv[1] >= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "hydrostatic iprob=2 needs gravity along -x2" << std::endl;
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
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "hydrostatic: iprob must be 1 or 2" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // hooks (also needed on restart)
  user_eq_func = HSEEquilibrium;
  if (user_bcs) user_bcs_func = HSEBoundary;
  pgen_final_func = HSEFinalize;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1m1 = indcs.nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
  const int nhyd = phydro->nhydro;
  const int nvar = phydro->nhydro + phydro->nscalars;

  // full equilibrium conserved state for the non-WB user BCs
  if (user_bcs && !phydro->well_balanced) {
    DvceArray5D<Real> wtmp("hse_wtmp", phydro->w0.extent(0), nvar,
                           phydro->w0.extent(2), phydro->w0.extent(3),
                           phydro->w0.extent(4));
    Kokkos::deep_copy(wtmp, 0.0);
    FillEquilibrium(pmbp, wtmp, nhyd);
    Kokkos::realloc(ubc_full, phydro->u0.extent(0), nvar, phydro->u0.extent(2),
                    phydro->u0.extent(3), phydro->u0.extent(4));
    phydro->peos->PrimToCons(wtmp, ubc_full, 0, n1m1, 0, n2m1, 0, n3m1);
  }
  if (restart) return;

  // unperturbed state: the SAME cell averages and the SAME PrimToCons kernel the
  // well-balanced scheme uses for U_eq, so U - U_eq is exactly zero where unperturbed
  auto &w0 = phydro->w0;
  auto &u0 = phydro->u0;
  Kokkos::deep_copy(w0, 0.0);
  FillEquilibrium(pmbp, w0, nhyd);
  phydro->peos->PrimToCons(w0, u0, 0, n1m1, 0, n2m1, 0, n3m1);

  // perturbation, as cell averages of its own functor
  DvceArray5D<Real> dw("hse_dw", w0.extent(0), nvar, w0.extent(2), w0.extent(3),
                       w0.extent(4));
  Kokkos::deep_copy(dw, 0.0);
  bool perturb = false;
  if (iprob == 1) {
    const Real amp = pin->GetOrAddReal("problem", "pulse_amp", 0.0);
    if (amp != 0.0) {
      PressurePulse pp;
      pp.amp = amp;
      pp.c1 = pin->GetOrAddReal("problem", "pulse_x1", 0.3);
      pp.c2 = pin->GetOrAddReal("problem", "pulse_x2", 0.3);
      pp.c3 = pin->GetOrAddReal("problem", "pulse_x3", 0.3);
      const Real w2 = pin->GetOrAddReal("problem", "pulse_w2", 0.01);
      const Real gmag = std::sqrt(SQR(gv[0]) + SQR(gv[1]) + SQR(gv[2]));
      pp.k2 = exp_atm.rho0*gmag/(exp_atm.p0*w2);
      pp.gm1 = gm1;
      pp.multi_d = pmy_mesh_->multi_d;
      pp.three_d = pmy_mesh_->three_d;
      wellbalance::FillEquilibriumCellAverages(pmbp, dw, nhyd, pp);
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
    // dw holds the density (IDN) and internal-energy (IEN) perturbations; v' = 0
    par_for("hse_perturb", DevExeSpace(), 0, (pmbp->nmb_thispack-1), 0, n3m1, 0, n2m1,
            0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u0(m,IDN,k,j,i) += dw(m,IDN,k,j,i);
      u0(m,IEN,k,j,i) += dw(m,IEN,k,j,i);
    });
  }
  return;
}
