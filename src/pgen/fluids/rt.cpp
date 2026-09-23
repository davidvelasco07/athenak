//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rt.cpp
//! \brief Problem generator for RT instability.
//!
//! The gravitational acceleration is read from the <hydro_srcterms> (or <mhd_srcterms>)
//! const_accel block (older decks that set const_accel_val in <hydro>/<mhd> still work).
//! The density ratio is set by `problem/drat` (default 3.0); to reproduce the 2D results
//! of Liska & Wendroff set it to 2.0, while for the 3D results of Dimonte et al use 3.0.
//!
//! Hydro: the unperturbed state is built as Gauss-Legendre cell averages of an exact
//! hydrostatic profile, p = p0 + g*int_0^y rho dy' (for smooth_interface the integral of
//! the tanh profile is analytic).  The same profile is the equilibrium of the
//! well-balanced scheme (<hydro>/well_balanced=true), for which the velocity
//! perturbation is the only departure from equilibrium.
//!
//! FOR 2D HYDRO:
//! Problem domain should be -1/6 < x < 1/6; -0.5 < y < 0.5 with gamma=1.4 to match Liska
//! & Wendroff. Interface is at y=0; perturbation added to Vy. Gravity acts in y-dirn.
//! Use reflecting boundaries in x2 (with the well-balanced scheme they reflect the
//! perturbation, so the hydrostatic walls are exact).  A=(d2-d1)/(d2+d1)=1/3. Options:
//!    - iprob = 1  -- Perturb V2 using single mode
//!    - iprob != 1 -- Perturb V2 using multiple mode
//!
//! FOR 3D:
//! Problem domain should be -0.5 < x < 0.5; -0.5 < y < 0.5, -1. < z < 1., gamma=5/3 to
//! match Dimonte et al.  Interface is at z=0; perturbation added to Vz. Gravity acts in
//! z-dirn. Special reflecting boundary conditions added in x3.  A=1/2.  Options:
//!    - iprob = 1 -- Perturb V3 using single mode
//!    - iprob = 2 -- Perturb V3 using multiple mode
//!    - iprob = 3 -- B rotated by "angle" at interface, multimode perturbation
//!
//! REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)

// C++ headers
#include <cmath>
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/random.hpp"
#include "pgen/pgen.hpp"
#include "hydro/well_balance.hpp"

#include <Kokkos_Random.hpp>

namespace {
//----------------------------------------------------------------------------------------
//! \struct RTEquilibrium
//! \brief Hydrostatic two-layer atmosphere (primitive state at a point), vertical
//! coordinate x2 in 2D and x3 in 3D.  dp/dy = g*rho exactly for both the sharp and the
//! tanh interface.

struct RTEquilibrium {
  Real drat, p0, g, gm1, sigma;
  bool smooth;
  int dir;   // 2 or 3
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real w[]) const {
    const Real y = (dir == 2) ? x2 : x3;
    Real den, pint;   // pint = int_0^y rho dy'
    if (smooth) {
      const Real a = fabs(y/sigma);
      // log(cosh(a)) without overflow
      const Real lncosh = a + log1p(exp(-2.0*a)) - log(2.0);
      den  = 0.5*((drat + 1.0) + (drat - 1.0)*tanh(y/sigma));
      pint = 0.5*(drat + 1.0)*y + 0.5*(drat - 1.0)*sigma*lncosh;
    } else {
      den  = (y > 0.0) ? drat : 1.0;
      pint = den*y;
    }
    w[IDN] = den;
    w[IVX] = 0.0; w[IVY] = 0.0; w[IVZ] = 0.0;
    w[IEN] = (p0 + g*pint)/gm1;
  }
};
RTEquilibrium rt_eq;

void RTHydroEquilibrium(Mesh *pm) {
  auto *ph = pm->pmb_pack->phydro;
  wellbalance::FillEquilibriumCellAverages(pm->pmb_pack, ph->w_eq, ph->nhydro, rt_eq);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Problem Generator for the Rayleigh-Taylor instability test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rti problem generator only works in 2D/3D" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real kx = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
  Real ky = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min);
  Real kz = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min);

  // Read perturbation amplitude, problem switch, density ratio
  Real amp = pin->GetReal("problem","amp");
  int iprob = pin->GetInteger("problem","iprob");
  Real drat = pin->GetOrAddReal("problem","drat",3.0);
  bool smooth_interface = pin->GetOrAddBoolean("problem","smooth_interface",false);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Select either Hydro or MHD.  Gravity comes from the source-term block that actually
  // applies it (it must point along the vertical axis); a const_accel_val given in
  // <hydro>/<mhd> is still accepted for older decks.
  DvceArray5D<Real> u0_;
  Real gm1, p0;
  Real grav_acc;
  SourceTerms *psrc_ = (pmbp->phydro != nullptr) ? pmbp->phydro->psrc
                                                 : pmbp->pmhd->psrc;
  const char *fluid = (pmbp->phydro != nullptr) ? "hydro" : "mhd";
  const int vdir = (pmbp->pmesh->two_d) ? 2 : 3;
  if (psrc_ != nullptr && psrc_->const_accel) {
    grav_acc = psrc_->const_accel_vec[vdir-1];
  } else {
    grav_acc = pin->GetReal(fluid, "const_accel_val");
  }
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
    p0 = 1.0/(pmbp->phydro->peos->eos_data.gamma);
    p0 = pin->GetOrAddReal("problem", "p0", p0);
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    gm1 = (pmbp->pmhd->peos->eos_data.gamma) - 1.0;
    p0 = 1.0/(pmbp->pmhd->peos->eos_data.gamma);
    p0 = pin->GetOrAddReal("problem", "p0", p0);
  }

  // Ensure that p0 is sufficiently large to avoid negative pressures
  if (pmbp->pmesh->two_d) {
    p0 -= grav_acc*pmy_mesh_->mesh_size.x2max;
  } else {
    p0 -= grav_acc*pmy_mesh_->mesh_size.x3max;
  }

  // Hydro: hydrostatic background as cell averages (also the well-balanced equilibrium,
  // whose hook is needed on restart too)
  if (pmbp->phydro != nullptr) {
    rt_eq.drat = drat;
    rt_eq.p0 = p0;
    rt_eq.g = grav_acc;
    rt_eq.gm1 = gm1;
    rt_eq.sigma = 0.01;
    rt_eq.smooth = smooth_interface;
    rt_eq.dir = vdir;
    user_eq_func = RTHydroEquilibrium;
  }
  if (restart) return;

  if (pmbp->phydro != nullptr) {
    auto *ph = pmbp->phydro;
    auto &w0 = ph->w0;
    const int ng = indcs.ng;
    const int n1m1 = indcs.nx1 + 2*ng - 1;
    const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
    const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
    Kokkos::deep_copy(w0, 0.0);
    wellbalance::FillEquilibriumCellAverages(pmbp, w0, ph->nhydro, rt_eq);
    ph->peos->PrimToCons(w0, u0_, 0, n1m1, 0, n2m1, 0, n3m1);
    // velocity perturbation (global-lattice coordinates, so the IC does not depend on
    // the domain decomposition)
    const wellbalance::GlobalGeometry gg = wellbalance::MakeGlobalGeometry(pmbp);
    const bool two_d = pmbp->pmesh->two_d;
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("rt_pert", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real xl, xr;
      gg.CellEdges(1, m, i, xl, xr); const Real x1v = 0.5*(xl + xr);
      gg.CellEdges(2, m, j, xl, xr); const Real x2v = 0.5*(xl + xr);
      gg.CellEdges(3, m, k, xl, xr); const Real x3v = 0.5*(xl + xr);
      const Real den = w0(m,IDN,k,j,i);
      Real f;
      if (two_d) {
        if (iprob == 1) {
          f = (1.0 + cos(kx*x1v))*(1.0 + cos(ky*x2v))/4.0;
        } else {
          auto rand_gen = rand_pool64.get_state();
          f = (rand_gen.frand()-0.5)*(1.0 + cos(ky*x2v))/4.0;
          rand_pool64.free_state(rand_gen);
        }
      } else {
        if (iprob == 1) {
          f = (1.0+cos(kx*x1v))*(1.0+cos(ky*x2v))*(1.0+cos(kz*x3v))/8.0;
        } else {
          auto rand_gen = rand_pool64.get_state();
          Real r = 2.0*static_cast<Real>(rand_gen.frand()) - 1.0;
          f = r*(1.0 + cos(kz*x3v))/2.0;
          rand_pool64.free_state(rand_gen);
        }
      }
      const int iv = two_d ? IM2 : IM3;
      const Real mom = den*amp*f;
      u0_(m,iv,k,j,i) = mom;
      u0_(m,IEN,k,j,i) += 0.5*SQR(mom)/den;
    });
    return;
  }

  // 2D PROBLEM ----------------------------------------------------------------

  if (pmbp->pmesh->two_d) {
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("rt2d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real den=1.0;
      Real sigma = 0.01;
      if (smooth_interface) {
        den = 0.5*((drat + 1.0) + (drat - 1.0)*tanh(x2v/sigma));
      } else {
        if (x2v > 0.0) den *= drat;
      }

      if (iprob == 1) {
        u0_(m,IM2,k,j,i) = (1.0 + cos(kx*x1v))*(1.0 + cos(ky*x2v))/4.0;
      } else {
        auto rand_gen = rand_pool64.get_state();  // get random number state this thread
        u0_(m,IM2,k,j,i) = (rand_gen.frand()-0.5)*(1.0 + cos(ky*x2v))/4.0;
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      }

      u0_(m,IDN,k,j,i) = den;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) *= (den*amp);
      u0_(m,IM3,k,j,i) = 0.0;
      u0_(m,IEN,k,j,i) = (p0 + grav_acc*den*x2v)/gm1 + 0.5*SQR(u0_(m,IM2,k,j,i))/den;
    });

    // initialize magnetic fields if MHD
    if (pmbp->pmhd != nullptr) {
      // Read magnetic field strength
      Real bx = pin->GetReal("problem","b0");
      auto &b0 = pmbp->pmhd->b0;
      par_for("pgen_b0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        b0.x1f(m,k,j,i) = bx;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = 0.0;
        if (i==ie) b0.x1f(m,k,j,i+1) = bx;
        if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
        if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
        u0_(m,IEN,k,j,i) += 0.5*bx*bx;
      });
    }

  // 3D PROBLEM ----------------------------------------------------------------

  } else {
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("rt3d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real den=1.0;
      if (x3v > 0.0) den *= drat;

      if (iprob == 1) {
        u0_(m,IM3,k,j,i) = (1.0+cos(kx*x1v))*(1.0+cos(ky*x2v))*(1.0+cos(kz*x3v))/8.0;
      } else {
        auto rand_gen = rand_pool64.get_state();  // get random number state this thread
        Real r = 2.0*static_cast<Real>(rand_gen.frand()) - 1.0;
        u0_(m,IM3,k,j,i) = r * (1.0 + cos(kz*x3v))/2.0;
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      }

      u0_(m,IDN,k,j,i) = den;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) = 0.0;
      u0_(m,IM3,k,j,i) *= (den*amp);
      u0_(m,IEN,k,j,i) = (p0 + grav_acc*den*x3v)/gm1 + 0.5*SQR(u0_(m,IM3,k,j,i))/den;
    });
  }

  return;
}
