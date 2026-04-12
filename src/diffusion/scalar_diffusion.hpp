#ifndef DIFFUSION_SCALAR_DIFFUSION_HPP_
#define DIFFUSION_SCALAR_DIFFUSION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_diffusion.hpp
//! \brief Contains data and functions that implement isotropic diffusion of passive
//!  scalars.  The diffusion flux is: F_s = -rho * nu_scalar * grad(C) where C = s/rho
//!  is the primitive scalar (concentration).

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class ScalarDiffusion
//! \brief data and functions that implement passive scalar diffusion in Hydro and MHD

class ScalarDiffusion {
 public:
  ScalarDiffusion(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~ScalarDiffusion();

  // data
  Real nu_scalar;   // scalar diffusion coefficient
  bool use_ho;      // flag for 4th-order diffusive operators
  //! Matches <hydro|mhd>/mignone; with use_ho, pass w0_c and point stencils, else w0.
  bool mignone_ {false};
  int nhydro;       // number of hydro/mhd variables (offset to first scalar in arrays)
  int nscalars;     // number of passive scalars

  // functions
  void IsotropicScalarDiffusiveFlux(const DvceArray5D<Real> &w,
                                    DvceFaceFld5D<Real> &f);
  void FourthOrderIsotropicScalarDiffusiveFlux(const DvceArray5D<Real> &w,
                                               DvceFaceFld5D<Real> &f);

 private:
  MeshBlockPack* pmy_pack;
};
#endif // DIFFUSION_SCALAR_DIFFUSION_HPP_
