//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_diffusion.cpp
//! \brief Implements isotropic diffusion of passive scalars.
//! The diffusive flux for scalar s_n is:
//!   F_n = -rho_face * nu_scalar * dC_n/dx
//! where C_n = s_n/rho is the primitive concentration, and rho_face is the
//! arithmetic mean of the two adjacent cell densities.

#include <float.h>
#include <algorithm>
#include <limits>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "scalar_diffusion.hpp"
#include "ho_diffusion_stencil.hpp"

//----------------------------------------------------------------------------------------
//! \brief ScalarDiffusion constructor

ScalarDiffusion::ScalarDiffusion(std::string block, MeshBlockPack *pp,
                                 ParameterInput *pin) :
  pmy_pack(pp) {
  nu_scalar = pin->GetReal(block, "nu_scalar");
  use_ho = pin->GetOrAddBoolean(block, "fourth_order_diff", false);
  mignone_ = pin->GetOrAddBoolean(block, "mignone", false);

  // Requires phydro/pmhd on the pack — construct ScalarDiffusion only after those
  // pointers are assigned (see MeshBlockPack::AddPhysics).  Use `block` so ion-neutral
  // runs with both modules get counts from the correct physics.
  if (block == "hydro") {
    if (pmy_pack->phydro == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "ScalarDiffusion(\"hydro\",...) but phydro is null" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    nhydro   = pmy_pack->phydro->nhydro;
    nscalars = pmy_pack->phydro->nscalars;
  } else if (block == "mhd") {
    if (pmy_pack->pmhd == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "ScalarDiffusion(\"mhd\",...) but pmhd is null" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    nhydro   = pmy_pack->pmhd->nmhd;
    nscalars = pmy_pack->pmhd->nscalars;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "ScalarDiffusion: block must be \"hydro\" or \"mhd\"" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (nscalars == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "nu_scalar is set but nscalars = 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \brief ScalarDiffusion destructor

ScalarDiffusion::~ScalarDiffusion() {
}

//----------------------------------------------------------------------------------------
//! \fn void ScalarDiffusion::IsotropicScalarDiffusiveFlux()
//! \brief Adds isotropic scalar diffusion flux to face-centered fluxes.
//! For each scalar n: flx_n -= rho_face * nu_scalar * dC_n/dx

void ScalarDiffusion::IsotropicScalarDiffusiveFlux(const DvceArray5D<Real> &w,
  DvceFaceFld5D<Real> &flx) {
  if (use_ho) {
    FourthOrderIsotropicScalarDiffusiveFlux(w, flx);
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  int nhyd  = nhydro;
  int nscal = nscalars;
  Real nu = nu_scalar;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  auto &flx1 = flx.x1f;
  par_for("scaldiff1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho_face = 0.5*(w(m,IDN,k,j,i) + w(m,IDN,k,j,i-1));
    Real idx1 = 1.0/size.d_view(m).dx1;
    for (int n = 0; n < nscal; ++n) {
      flx1(m, nhyd+n, k, j, i) -= rho_face * nu * (w(m,nhyd+n,k,j,i) -
                                                     w(m,nhyd+n,k,j,i-1)) * idx1;
    }
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto &flx2 = flx.x2f;
  par_for("scaldiff2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho_face = 0.5*(w(m,IDN,k,j,i) + w(m,IDN,k,j-1,i));
    Real idx2 = 1.0/size.d_view(m).dx2;
    for (int n = 0; n < nscal; ++n) {
      flx2(m, nhyd+n, k, j, i) -= rho_face * nu * (w(m,nhyd+n,k,j,i) -
                                                     w(m,nhyd+n,k,j-1,i)) * idx2;
    }
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto &flx3 = flx.x3f;
  par_for("scaldiff3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho_face = 0.5*(w(m,IDN,k,j,i) + w(m,IDN,k-1,j,i));
    Real idx3 = 1.0/size.d_view(m).dx3;
    for (int n = 0; n < nscal; ++n) {
      flx3(m, nhyd+n, k, j, i) -= rho_face * nu * (w(m,nhyd+n,k,j,i) -
                                                     w(m,nhyd+n,k-1,j,i)) * idx3;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ScalarDiffusion::FourthOrderIsotropicScalarDiffusiveFlux()
//! \brief Adds 4th-order isotropic scalar diffusion flux to face-centered fluxes.
//! dC/dn via HoGrad (point vs average from mignone_); rho_face via HoFaceValue.

void ScalarDiffusion::FourthOrderIsotropicScalarDiffusiveFlux(const DvceArray5D<Real> &w,
  DvceFaceFld5D<Real> &flx) {
  const bool use_pt = mignone_;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  int nhyd  = nhydro;
  int nscal = nscalars;
  Real nu = nu_scalar;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  auto &flx1 = flx.x1f;
  par_for("scaldiff4_1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho_face = HoFaceValue(w(m,IDN,k,j,i-2), w(m,IDN,k,j,i-1),
                                      w(m,IDN,k,j,i), w(m,IDN,k,j,i+1));
    Real inv_dx = 1.0/size.d_view(m).dx1;
    for (int n = 0; n < nscal; ++n) {
      Real dCdx = HoGrad(use_pt,
          w(m,nhyd+n,k,j,i-2), w(m,nhyd+n,k,j,i-1),
          w(m,nhyd+n,k,j,i), w(m,nhyd+n,k,j,i+1), inv_dx);
      flx1(m, nhyd+n, k, j, i) -= rho_face * nu * dCdx;
    }
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto &flx2 = flx.x2f;
  par_for("scaldiff4_2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho_face = HoFaceValue(w(m,IDN,k,j-2,i), w(m,IDN,k,j-1,i),
                                      w(m,IDN,k,j,i), w(m,IDN,k,j+1,i));
    Real inv_dy = 1.0/size.d_view(m).dx2;
    for (int n = 0; n < nscal; ++n) {
      Real dCdy = HoGrad(use_pt,
          w(m,nhyd+n,k,j-2,i), w(m,nhyd+n,k,j-1,i),
          w(m,nhyd+n,k,j,i), w(m,nhyd+n,k,j+1,i), inv_dy);
      flx2(m, nhyd+n, k, j, i) -= rho_face * nu * dCdy;
    }
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto &flx3 = flx.x3f;
  par_for("scaldiff4_3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho_face = HoFaceValue(w(m,IDN,k-2,j,i), w(m,IDN,k-1,j,i),
                                      w(m,IDN,k,j,i), w(m,IDN,k+1,j,i));
    Real inv_dz = 1.0/size.d_view(m).dx3;
    for (int n = 0; n < nscal; ++n) {
      Real dCdz = HoGrad(use_pt,
          w(m,nhyd+n,k-2,j,i), w(m,nhyd+n,k-1,j,i),
          w(m,nhyd+n,k,j,i), w(m,nhyd+n,k+1,j,i), inv_dz);
      flx3(m, nhyd+n, k, j, i) -= rho_face * nu * dCdz;
    }
  });

  return;
}
