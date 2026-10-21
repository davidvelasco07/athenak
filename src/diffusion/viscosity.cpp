//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.cpp
//  \brief Implements functions for Viscosity class. This includes isotropic shear
//  viscosity in a Newtonian fluid (in which stress is proportional to shear).
//  Viscosity may be added to Hydro and/or MHD independently.

#include <float.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <string> // string

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "viscosity.hpp"

namespace {

parabolic::DiffusionSelection ParseViscosityIntegrator(const std::string &block,
                                                        ParameterInput *pin) {
  std::string integrator =
      pin->GetOrAddString(block, "viscosity_integrator", "explicit");
  if (integrator == "explicit") {
    return parabolic::DiffusionSelection::explicit_only;
  }
  if (integrator == "sts") {
    return parabolic::DiffusionSelection::sts_only;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<" << block << ">/viscosity_integrator = '" << integrator
            << "' must be 'explicit' or 'sts'" << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
// ctor:
// Note first argument passes string ("hydro" or "mhd") denoting in which class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.

Viscosity::Viscosity(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  // Read parameters for viscosity (if any)
  nu_iso = pin->GetOrAddReal(block,"nu_iso",0.0);
  nu_aniso = pin->GetOrAddReal(block,"nu_aniso",0.0);
  mode = ParseViscosityIntegrator(block, pin);
  if (mode == parabolic::DiffusionSelection::sts_only &&
      (!std::isfinite(nu_iso) || !std::isfinite(nu_aniso) || nu_iso <= 0.0 ||
       nu_aniso != 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "STS requires positive constant isotropic viscosity" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Shearing box: the stored velocity is the perturbation v' = v - v_K, so the stress
  // needs the background shear d(v_K)/dx added explicitly (see AddViscousFluxIso and
  // AddShearingBoxHeating).  In 2D the box is r-z and the azimuth is x3.
  shear_rate = 0.0;
  shear_dir = IVY;
  if (pin->DoesBlockExist("shearing_box")) {
    Real qshear = pin->GetReal("shearing_box", "qshear");
    Real omega0 = pin->GetReal("shearing_box", "omega0");
    shear_rate = -qshear*omega0;
    shear_dir = (pp->pmesh->three_d) ? IVY : IVZ;
    if (nu_aniso != 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << block << ">/nu_aniso is not supported in the "
                << "shearing box" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (mode == parabolic::DiffusionSelection::sts_only) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << block << ">/viscosity_integrator = sts is not "
                << "supported in the shearing box: the parabolic stages do not apply "
                << "shear-periodic boundaries or the shear-heating source. Use explicit."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// Viscosity destructor

Viscosity::~Viscosity() {
}

//----------------------------------------------------------------------------------------
//! \fn void AddViscousFluxes()
//! \brief Wrapper function that adds viscous fluxes for different types of viscosity
//! to face-centered fluxes of conserved variables
//! Currently only isotropic viscosity with constant coefficient implemented

void Viscosity::AddViscousFluxes(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  if (nu_iso != 0.0) {
    AddViscousFluxIso(w0, eos, flx);
  }
  if (nu_aniso != 0.0) {
    AddViscousFluxAniso(w0, eos, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddViscousFluxIso
//  \brief Adds viscous fluxes to face-centered fluxes of conserved variables

ViscFaceParams Viscosity::FaceParams(const EOS_Data &eos) const {
  ViscFaceParams p;
  p.nu = nu_iso;
  p.add_shear = (shear_rate != 0.0);
  p.sr_y = (shear_dir == IVY) ? shear_rate : 0.0;
  p.sr_z = (shear_dir == IVZ) ? shear_rate : 0.0;
  p.multi_d = pmy_pack->pmesh->multi_d;
  p.three_d = pmy_pack->pmesh->three_d;
  p.is_ideal = eos.is_ideal;
  return p;
}

void Viscosity::AddViscousFluxIso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const ViscFaceParams p = FaceParams(eos);

  // fluxes in x1-direction (face-local function shared with the MOOD/FOFC revise passes)
  auto flx1 = flx.x1f;
  par_for("visc1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const ViscFaceFlux f = ViscousFluxX1(p, w0, size.d_view(m), m, k, j, i);
    flx1(m,IVX,k,j,i) -= f.nud*f.fvx;
    flx1(m,IVY,k,j,i) -= f.nud*f.fvy;
    flx1(m,IVZ,k,j,i) -= f.nud*f.fvz;
    if (p.is_ideal) {
      flx1(m,IEN,k,j,i) -= 0.5*f.nud*f.ve;
    }
  });
  if (pmy_pack->pmesh->one_d) {return;}

  // fluxes in x2-direction
  auto flx2 = flx.x2f;
  par_for("visc2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const ViscFaceFlux f = ViscousFluxX2(p, w0, size.d_view(m), m, k, j, i);
    flx2(m,IVX,k,j,i) -= f.nud*f.fvx;
    flx2(m,IVY,k,j,i) -= f.nud*f.fvy;
    flx2(m,IVZ,k,j,i) -= f.nud*f.fvz;
    if (p.is_ideal) {
      flx2(m,IEN,k,j,i) -= 0.5*f.nud*f.ve;
    }
  });
  if (pmy_pack->pmesh->two_d) {return;}

  // fluxes in x3-direction
  auto flx3 = flx.x3f;
  par_for("visc3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const ViscFaceFlux f = ViscousFluxX3(p, w0, size.d_view(m), m, k, j, i);
    flx3(m,IVX,k,j,i) -= f.nud*f.fvx;
    flx3(m,IVY,k,j,i) -= f.nud*f.fvy;
    flx3(m,IVZ,k,j,i) -= f.nud*f.fvz;
    if (p.is_ideal) {
      flx3(m,IEN,k,j,i) -= 0.5*f.nud*f.ve;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddViscousFluxAniso
//  \brief Currently no-op function, to be added later

void Viscosity::AddViscousFluxAniso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Viscosity::AddShearingBoxHeating
//! \brief Shearing box: the evolved energy is E' = rho v'^2/2 + e with v' = v - v_K, so
//! the viscous work of the background shear appears as a source
//!   dE'/dt = sigma : grad(v_K) = (d v_K/dx) sigma_{x,az} = -q Omega sigma_{x,az},
//! where sigma_{x,az} = rho nu (d v'_az/dx + d v'_x/d az + d v_K/dx) is the FULL stress
//! (the flux term -sigma.v' already uses the full stress with the perturbation velocity).
//! For v' = 0 this is rho nu q^2 Omega^2 = (9/4) rho nu Omega^2 for Keplerian shear.
//! Evaluated at cell centres with central differences, like the other shearing-box
//! source terms.  Must be called with primitives of the stage (w0), never with u0.

void Viscosity::AddShearingBoxHeating(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                                      const Real bdt, DvceArray5D<Real> &u0) {
  if (shear_rate == 0.0 || nu_iso == 0.0 || !(eos.is_ideal)) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const Real nu_iso_ = nu_iso;
  const Real sr = shear_rate;
  const int az = shear_dir;
  // in the 2D (r-z) box the azimuth x3 is the invariant direction: d v'_x / d az = 0
  const bool az_is_y = (az == IVY);

  par_for("visc_sbox_heat", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dvaz_dx = (w0(m,az,k,j,i+1) - w0(m,az,k,j,i-1))/(2.0*size.d_view(m).dx1);
    Real dvx_daz = 0.0;
    if (az_is_y) {
      dvx_daz = (w0(m,IVX,k,j+1,i) - w0(m,IVX,k,j-1,i))/(2.0*size.d_view(m).dx2);
    }
    Real sigma = nu_iso_*w0(m,IDN,k,j,i)*(dvaz_dx + dvx_daz + sr);
    u0(m,IEN,k,j,i) += bdt*sr*sigma;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Viscosity::NewTimeStep()
//! \brief Compute new time step for viscosity

void Viscosity::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  // viscous timestep on MeshBlock(s) in this pack for constant isotropic viscosity
  dtnew = std::numeric_limits<float>::max();
  if (nu_iso <= 0.0) return;
  auto size = pmy_pack->pmb->mb_size;
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    Real inv_dx2_sum = 1.0/SQR(size.h_view(m).dx1);
    Real inv_dx2_max = inv_dx2_sum;
    if (pmy_pack->pmesh->multi_d) {
      Real inv_dx2 = 1.0/SQR(size.h_view(m).dx2);
      inv_dx2_sum += inv_dx2;
      inv_dx2_max = std::max(inv_dx2_max, inv_dx2);
    }
    if (pmy_pack->pmesh->three_d) {
      Real inv_dx2 = 1.0/SQR(size.h_view(m).dx3);
      inv_dx2_sum += inv_dx2;
      inv_dx2_max = std::max(inv_dx2_max, inv_dx2);
    }
    Real rate = 2.0*nu_iso*(inv_dx2_sum + inv_dx2_max/3.0);
    dtnew = std::min(dtnew, 1.0/rate);
  }
  return;
}
