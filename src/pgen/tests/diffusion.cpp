//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file diffusion.cpp
//! \brief problem generator for tests of diffusion modules (viscosity, thermal
//! conduction, resistivity).
//!
//! Four test types selected by <problem>/test_type:
//!   "viscosity"    (default) Gaussian transverse velocity profile in x1
//!   "conduction"   Gaussian temperature perturbation in x1
//!   "resistivity"  Sinusoidal B2 mode with periodic BCs (exact decay by exp(-eta*k^2*t))
//!   "taylor_green" 2D viscous Taylor-Green vortex (exact compressible NS solution)
//!
//! This file also contains a function to compute L1 errors, called in Driver::Finalize().

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/resistivity.hpp"

void DiffusionErrors(ParameterInput *pin, Mesh *pm);
void GaussianProfile(Mesh *pm);

namespace {
bool set_initial_conditions = true;
std::string test_type;

struct DiffusionVariables {
  Real d0, amp, t0, x10;
  Real kx;   // wavenumber for resistivity sinusoidal test
  bool use_ho;      // use 4th-order cell-average corrections in IC/BC/reference
  Real diff_coeff;  // diffusivity: nu (viscosity) or kappa*(gamma-1) (conduction)
};
DiffusionVariables dv;
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Diffusion()

void ProblemGenerator::Diffusion(ParameterInput *pin, const bool restart) {
  pgen_final_func = DiffusionErrors;
  user_bcs_func = GaussianProfile;
  if (restart) return;

  test_type = pin->GetOrAddString("problem", "test_type", "viscosity");

  dv.d0  = 1.0;
  dv.amp = pin->GetOrAddReal("problem", "amp", 1.e-6);
  dv.t0  = pin->GetOrAddReal("problem", "t0", 0.5);
  dv.x10 = pin->GetOrAddReal("problem", "x10", 0.0);
  // wavenumber for resistivity test (one full period across domain by default)
  Real Lx = pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min");
  dv.kx  = pin->GetOrAddReal("problem", "kx", 2.0*M_PI/Lx);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto d0_=dv.d0, amp_=dv.amp, x10_=dv.x10;

  //--- viscosity test (Hydro or MHD): Gaussian transverse velocity -----------------------
  if (test_type == "viscosity") {
    Real nu_iso = 0.0;
    if (pmbp->phydro != nullptr) {
      if (pmbp->phydro->pvisc == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "test_type=viscosity requires viscosity in <hydro>"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      nu_iso = pmbp->phydro->pvisc->nu_iso;
      // use_ho: true if fourth_order_diff=true OR mignone=true (both need cell-avg IC)
      dv.use_ho     = pmbp->phydro->pvisc->use_ho || pmbp->phydro->use_mignone;
      dv.diff_coeff = nu_iso;
      EOS_Data &eos = pmbp->phydro->peos->eos_data;
      Real gm1 = eos.gamma - 1.0;
      Real p0 = 1.0/eos.gamma;
      Real t1 = dv.t0;
      if (!(set_initial_conditions)) { t1 += pmbp->pmesh->time; }
      auto &u1 = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;
      bool use_ho_ = dv.use_ho;
      par_for("pgen_visc_hyd", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        Real vy = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))/sqrt(4.*M_PI*nu_iso*t1);
        if (use_ho_) {
          Real dx1 = size.d_view(m).dx1;
          vy *= 1.0 + (dx1*dx1/24.0)*(SQR(x1v-x10_) - 2.0*nu_iso*t1)
                      /(4.0*nu_iso*nu_iso*t1*t1);
        }
        u1(m,IDN,k,j,i) = d0_;
        u1(m,IM1,k,j,i) = 0.0;
        u1(m,IM2,k,j,i) = d0_*vy;
        u1(m,IM3,k,j,i) = d0_*vy;
        if (eos.is_ideal) {
          u1(m,IEN,k,j,i) = p0/gm1 +
                            0.5*(SQR(u1(m,IM2,k,j,i)) + SQR(u1(m,IM3,k,j,i)))/d0_;
        }
      });
    }
    if (pmbp->pmhd != nullptr) {
      if (pmbp->pmhd->pvisc == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "test_type=viscosity requires viscosity in <mhd>"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      nu_iso = pmbp->pmhd->pvisc->nu_iso;
      dv.use_ho     = pmbp->pmhd->pvisc->use_ho;
      dv.diff_coeff = nu_iso;
      EOS_Data &eos = pmbp->pmhd->peos->eos_data;
      Real gm1 = eos.gamma - 1.0;
      Real p0 = 1.0/eos.gamma;
      Real t1 = dv.t0;
      if (!(set_initial_conditions)) { t1 += pmbp->pmesh->time; }
      auto &u1 = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;
      auto &b1 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;
      // For MHD+Mignone IC: InitMignoneIC() handles cell-average correction after pgen.
      // For MHD reference or non-Mignone use_ho: apply correction here.
      bool mignone_ = pmbp->pmhd->use_mignone;
      bool apply_ho_ = dv.use_ho && (!mignone_ || !set_initial_conditions);
      par_for("pgen_visc_mhd", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        Real vy = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))/sqrt(4.*M_PI*nu_iso*t1);
        if (apply_ho_) {
          Real dx1 = size.d_view(m).dx1;
          vy *= 1.0 + (dx1*dx1/24.0)*(SQR(x1v-x10_) - 2.0*nu_iso*t1)
                      /(4.0*nu_iso*nu_iso*t1*t1);
        }
        u1(m,IDN,k,j,i) = d0_;
        u1(m,IM1,k,j,i) = 0.0;
        u1(m,IM2,k,j,i) = d0_*vy;
        u1(m,IM3,k,j,i) = d0_*vy;
        if (eos.is_ideal) {
          u1(m,IEN,k,j,i) = p0/gm1 +
                            0.5*(SQR(u1(m,IM2,k,j,i)) + SQR(u1(m,IM3,k,j,i)))/d0_;
        }
      });
      // zero B field
      auto bx1 = b1.x1f; auto bx2 = b1.x2f; auto bx3 = b1.x3f;
      par_for("pgen_visc_mhd_b", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        bx1(m,k,j,i) = 0.0;
        if (i <= ie) bx2(m,k,j,i) = 0.0;
        if (i <= ie) bx3(m,k,j,i) = 0.0;
      });
    }
    return;
  }

  //--- conduction test (Hydro): Gaussian temperature perturbation -----------------------
  if (test_type == "conduction") {
    if (pmbp->phydro == nullptr || pmbp->phydro->pcond == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "test_type=conduction requires conductivity in <hydro>"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    Real kappa = pmbp->phydro->pcond->kappa;
    // thermal diffusivity: kappa_eff = kappa * (gamma-1) / rho  (for rho=1)
    Real kappa_eff = kappa * gm1;
    dv.use_ho     = pmbp->phydro->pcond->use_ho;
    dv.diff_coeff = kappa_eff;
    Real t1 = dv.t0;
    if (!(set_initial_conditions)) { t1 += pmbp->pmesh->time; }
    auto &u1 = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;
    bool use_ho_ = dv.use_ho;
    par_for("pgen_cond", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      // Gaussian temperature perturbation on top of uniform background
      Real dT = amp_*exp(SQR(x1v-x10_)/(-4.0*kappa_eff*t1))/sqrt(4.*M_PI*kappa_eff*t1);
      if (use_ho_) {
        Real dx1 = size.d_view(m).dx1;
        dT *= 1.0 + (dx1*dx1/24.0)*(SQR(x1v-x10_) - 2.0*kappa_eff*t1)
                    /(4.0*kappa_eff*kappa_eff*t1*t1);
      }
      u1(m,IDN,k,j,i) = d0_;
      u1(m,IM1,k,j,i) = 0.0;
      u1(m,IM2,k,j,i) = 0.0;
      u1(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) {
        Real T_bg = p0;  // background T = p0/rho = p0 (with rho=1)
        u1(m,IEN,k,j,i) = d0_*(T_bg + dT)/gm1;
      }
    });
    return;
  }

  //--- resistivity test (MHD): sinusoidal B2 mode, periodic BCs -----------------------
  if (test_type == "resistivity") {
    if (pmbp->pmhd == nullptr || pmbp->pmhd->presist == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "test_type=resistivity requires ohmic_resistivity in <mhd>"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    Real eta = pmbp->pmhd->presist->eta_ohm;
    Real kx_ = dv.kx;
    Real t1 = dv.t0;
    if (!(set_initial_conditions)) { t1 += pmbp->pmesh->time; }
    auto &u1 = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;
    auto &b1 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;

    // B2 = amp * sin(kx * x) * exp(-eta * kx^2 * t)
    // Use t0 as an initial phase: B2(x,0) = amp * sin(kx*x) * exp(-eta*kx^2*t0)
    Real decay = exp(-eta*kx_*kx_*t1);
    par_for("pgen_resist_u", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u1(m,IDN,k,j,i) = d0_;
      u1(m,IM1,k,j,i) = 0.0;
      u1(m,IM2,k,j,i) = 0.0;
      u1(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) {
        u1(m,IEN,k,j,i) = p0/gm1;  // uniform pressure, B contributes below
      }
    });
    auto bx1 = b1.x1f; auto bx2 = b1.x2f; auto bx3 = b1.x3f;
    par_for("pgen_resist_b", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      // x2-face is at cell center in x1 for x2f
      Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);
      bx1(m,k,j,i) = 0.0;
      // B2 at x2-face (centered at cell i in x1): use face x1-position
      if (i <= ie) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        bx2(m,k,j,i) = amp_*sin(kx_*x1v)*decay;
        bx3(m,k,j,i) = 0.0;
      }
      (void)x1f;
    });
    // add magnetic energy to IEN
    if (eos.is_ideal) {
      par_for("pgen_resist_e", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        Real by = amp_*sin(kx_*x1v)*decay;
        u1(m,IEN,k,j,i) += 0.5*by*by;
      });
    }
    return;
  }

  //--- 2D viscous Taylor-Green vortex (Hydro): exact solution to compressible NS --------
  // vx(x,y,t) =  U0 sin(kx) cos(ky) exp(-2*nu*k^2*t)
  // vy(x,y,t) = -U0 cos(kx) sin(ky) exp(-2*nu*k^2*t)
  //  P(x,y,t) =  P0 - (rho0*U0^2/4)*(cos(2kx)+cos(2ky))*exp(-4*nu*k^2*t)
  //
  // Exact for compressible NS because v.grad(v) = -(1/rho)*grad(P) identically,
  // so the momentum equation reduces to dv/dt = nu*Laplacian(v).
  // Domain: x,y in [-pi,pi], periodic BCs, rho = const = rho0.
  // NOTE: Energy errors do NOT converge. Viscous heating transfers kinetic to internal
  // energy; the reference P(t) does not account for this, causing an ~O(nu*U0^2*T) floor.
  // Assess convergence using M1/M2 errors only. Total energy IS conserved for periodic BCs.
  // L1 error ratios (M1/M2): ~4 (2nd-order) and ~16 (4th-order, with fourth_order_diff).
  if (test_type == "taylor_green") {
    if (pmbp->phydro == nullptr || pmbp->phydro->pvisc == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "test_type=taylor_green requires viscosity in <hydro>"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real kwave = pin->GetOrAddReal("problem", "kwave", 1.0);
    Real U0    = pin->GetOrAddReal("problem", "U0", 0.1);
    Real Mach  = pin->GetOrAddReal("problem", "Mach", 0.1);
    Real rho0  = pin->GetOrAddReal("problem", "rho0", 1.0);
    // background pressure from Mach number: P0 = rho0*(U0/Mach)^2/gamma
    Real P0 = rho0 * SQR(U0/Mach) / eos.gamma;
    Real nu_iso = pmbp->phydro->pvisc->nu_iso;
    dv.use_ho = pmbp->phydro->pvisc->use_ho || pmbp->phydro->use_mignone;
    bool use_ho_ = dv.use_ho;

    Real t1 = 0.0;
    if (!set_initial_conditions) { t1 = pmbp->pmesh->time; }
    Real decay  = exp(-2.0*nu_iso*kwave*kwave*t1);   // D = exp(-2*nu*k^2*t)
    Real decay2 = exp(-4.0*nu_iso*kwave*kwave*t1);   // D2 = D^2 = exp(-4*nu*k^2*t)

    auto &u1 = (set_initial_conditions) ? pmbp->phydro->u0 : pmbp->phydro->u1;
    par_for("pgen_tgv", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx1 = indcs.nx1;
      int nx2 = indcs.nx2;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;

      // velocities
      Real vx =  U0 * sin(kwave*x1v) * cos(kwave*x2v) * decay;
      Real vy = -U0 * cos(kwave*x1v) * sin(kwave*x2v) * decay;
      // cos(2kx) and cos(2ky) — used for pressure and kinetic energy
      Real cos2x = cos(2.0*kwave*x1v);
      Real cos2y = cos(2.0*kwave*x2v);

      if (use_ho_) {
        // Cell-average corrections via sinc functions:
        // <sin(kx)*cos(ky)>_cell = sin(kxc)*cos(kyc)*sinc(k*dx/2)*sinc(k*dy/2)
        Real ax = sin(kwave*dx1*0.5)/(kwave*dx1*0.5);
        Real ay = sin(kwave*dx2*0.5)/(kwave*dx2*0.5);
        vx *= ax * ay;
        vy *= ax * ay;
        // <cos(2kx)>_x = cos(2kxc)*sinc(k*dx);  same sinc factor for cos(2ky)
        Real bx = sin(kwave*dx1)/(kwave*dx1);
        Real by = sin(kwave*dx2)/(kwave*dx2);
        cos2x *= bx;
        cos2y *= by;
      }

      // Exact pressure: P = P0 + (rho0*U0^2/4)*(cos(2kx)+cos(2ky))*D^2
      Real Pval = P0 + rho0*U0*U0/4.0*(cos2x + cos2y)*decay2;
      // Kinetic energy: rho0*U0^2*D^2/4*(1 - cos(2kx)*cos(2ky))
      // Derivation: (vx^2+vy^2)/2 = U0^2*D^2/2*(sin^2kx*cos^2ky+cos^2kx*sin^2ky)
      //   = U0^2*D^2/4*(1-cos2x*cos2y)  [using product-to-sum identities]
      // For use_ho_: replace cos2x->bx*cos(2kxc), cos2y->by*cos(2kyc) above
      Real Ekin = rho0*U0*U0*decay2/4.0*(1.0 - cos2x*cos2y);

      u1(m,IDN,k,j,i) = rho0;
      u1(m,IM1,k,j,i) = rho0 * vx;
      u1(m,IM2,k,j,i) = rho0 * vy;
      u1(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) {
        u1(m,IEN,k,j,i) = Pval/gm1 + Ekin;
      }
    });
    return;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Unknown test_type = '" << test_type << "'" << std::endl;
  exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn void DiffusionErrors()

void DiffusionErrors(ParameterInput *pin, Mesh *pm) {
  set_initial_conditions = false;
  pm->pgen->Diffusion(pin, false);

  Real l1_err[8]{};
  Real linfty_err = 0.0;
  int nvars = 0;

  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // --- Hydro errors (viscosity or conduction) -----------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji  = nx3*nx2*nx1;
    const int nji   = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("diff-err-hyd",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i)));
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i)));
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i)));
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i)));
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i)));
      }
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) evars.the_array[n] = 0.0;
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    for (int n=0; n<nvars; ++n) l1_err[n] = sum_this_mb.the_array[n];
  }

  // --- MHD errors (viscosity, conduction, or resistivity) ----------------------------
  if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    auto &u0_ = pmbp->pmhd->u0;
    auto &u1_ = pmbp->pmhd->u1;
    auto &b0_ = pmbp->pmhd->b0;
    auto &b1_ = pmbp->pmhd->b1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji  = nx3*nx2*nx1;
    const int nji   = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("diff-err-mhd",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i)));
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i)));
      }
      // cell-centered B from face averages
      Real bx = 0.5*(b0_.x1f(m,k,j,i) + b0_.x1f(m,k,j,i+1));
      Real by = 0.5*(b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i));
      Real bz = 0.5*(b0_.x3f(m,k,j,i) + b0_.x3f(m,k+1,j,i));
      Real bx_r = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      Real by_r = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      Real bz_r = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));
      evars.the_array[IBX] = vol*fabs(bx - bx_r);
      evars.the_array[IBY] = vol*fabs(by - by_r);
      evars.the_array[IBZ] = vol*fabs(bz - bz_r);
      max_err = fmax(max_err, fabs(by - by_r));
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) evars.the_array[n] = 0.0;
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    for (int n=0; n<nvars; ++n) l1_err[n] = sum_this_mb.the_array[n];
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nvars, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  Real vol = (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nvars; ++i) l1_err[i] /= vol;

  Real rms_err = 0.0;
  for (int i=0; i<nvars; ++i) rms_err += SQR(l1_err[i]);
  rms_err = std::sqrt(rms_err);

  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1    L-infty       ");
      std::fprintf(pfile, "d_L1         M1_L1         M2_L1         M3_L1         E_L1");
      if (pmbp->pmhd != nullptr) {
        std::fprintf(pfile, "          B1_L1         B2_L1         B3_L1");
      }
      std::fprintf(pfile, "\n");
    }
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int i=0; i<nvars; ++i) std::fprintf(pfile, "  %e", l1_err[i]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn GaussianProfile
//  \brief Sets boundary conditions on x1 faces for viscosity and conduction tests.
//  The resistivity test uses periodic BCs and does not call this function.

void GaussianProfile(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &ie = indcs.ie;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto d0_=dv.d0, amp_=dv.amp, x10_=dv.x10;
  Real t1 = dv.t0 + pm->time;

  // viscosity BC (hydro)
  if (test_type == "viscosity" && pm->pmb_pack->phydro != nullptr) {
    EOS_Data &eos = pm->pmb_pack->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    Real nu_iso = pm->pmb_pack->phydro->pvisc->nu_iso;
    bool use_ho_ = dv.use_ho;
    auto &u0 = pm->pmb_pack->phydro->u0;
    par_for("diffBC_visc", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      auto fill = [&](int cell_i, int li) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(li, nx1, x1min, x1max);
        Real vy = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))/sqrt(4.*M_PI*nu_iso*t1);
        if (use_ho_) {
          Real dx1 = size.d_view(m).dx1;
          vy *= 1.0 + (dx1*dx1/24.0)*(SQR(x1v-x10_) - 2.0*nu_iso*t1)
                      /(4.0*nu_iso*nu_iso*t1*t1);
        }
        u0(m,IDN,k,j,cell_i) = d0_;
        u0(m,IM1,k,j,cell_i) = 0.0;
        u0(m,IM2,k,j,cell_i) = d0_*vy;
        u0(m,IM3,k,j,cell_i) = d0_*vy;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,cell_i) = p0/gm1 +
            0.5*(SQR(u0(m,IM2,k,j,cell_i)) + SQR(u0(m,IM3,k,j,cell_i)))/d0_;
        }
      };
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) fill(is-i-1, -1-i);
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) fill(ie+i+1, ie-is+1+i);
      }
    });
    return;
  }

  // viscosity BC (MHD)
  if (test_type == "viscosity" && pm->pmb_pack->pmhd != nullptr) {
    EOS_Data &eos = pm->pmb_pack->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    Real nu_iso = pm->pmb_pack->pmhd->pvisc->nu_iso;
    bool mignone_ = pm->pmb_pack->pmhd->use_mignone;
    bool use_ho_ = dv.use_ho && (!mignone_ || true);  // always apply correction for BC
    auto &u0 = pm->pmb_pack->pmhd->u0;
    par_for("diffBC_visc_mhd", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      auto fill = [&](int cell_i, int li) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(li, nx1, x1min, x1max);
        Real vy = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))/sqrt(4.*M_PI*nu_iso*t1);
        if (use_ho_) {
          Real dx1 = size.d_view(m).dx1;
          vy *= 1.0 + (dx1*dx1/24.0)*(SQR(x1v-x10_) - 2.0*nu_iso*t1)
                      /(4.0*nu_iso*nu_iso*t1*t1);
        }
        u0(m,IDN,k,j,cell_i) = d0_;
        u0(m,IM1,k,j,cell_i) = 0.0;
        u0(m,IM2,k,j,cell_i) = d0_*vy;
        u0(m,IM3,k,j,cell_i) = d0_*vy;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,cell_i) = p0/gm1 +
            0.5*(SQR(u0(m,IM2,k,j,cell_i)) + SQR(u0(m,IM3,k,j,cell_i)))/d0_;
        }
      };
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) fill(is-i-1, -1-i);
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) fill(ie+i+1, ie-is+1+i);
      }
    });
    return;
  }

  // conduction BC (hydro)
  if (test_type == "conduction" && pm->pmb_pack->phydro != nullptr) {
    EOS_Data &eos = pm->pmb_pack->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    Real kappa = pm->pmb_pack->phydro->pcond->kappa;
    Real kappa_eff = kappa * gm1;
    bool use_ho_ = dv.use_ho;
    auto &u0 = pm->pmb_pack->phydro->u0;
    par_for("diffBC_cond", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      auto fill = [&](int cell_i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        int li = cell_i - is;
        Real x1v = CellCenterX(li, nx1, x1min, x1max);
        Real dT = amp_*exp(SQR(x1v-x10_)/(-4.0*kappa_eff*t1))/sqrt(4.*M_PI*kappa_eff*t1);
        if (use_ho_) {
          Real dx1 = size.d_view(m).dx1;
          dT *= 1.0 + (dx1*dx1/24.0)*(SQR(x1v-x10_) - 2.0*kappa_eff*t1)
                      /(4.0*kappa_eff*kappa_eff*t1*t1);
        }
        Real T_bg = p0;
        u0(m,IDN,k,j,cell_i) = d0_;
        u0(m,IM1,k,j,cell_i) = 0.0;
        u0(m,IM2,k,j,cell_i) = 0.0;
        u0(m,IM3,k,j,cell_i) = 0.0;
        if (eos.is_ideal) u0(m,IEN,k,j,cell_i) = d0_*(T_bg + dT)/gm1;
      };
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) fill(is-i-1);
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) fill(ie+i+1);
      }
    });
    return;
  }

  // resistivity uses periodic BCs — nothing to do here
  return;
}
