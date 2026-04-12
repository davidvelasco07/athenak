#ifndef DIFFUSION_VISCOSITY_HPP_
#define DIFFUSION_VISCOSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.hpp
//  \brief Contains data and functions that implement various formulations for
//  viscosity. Currently only Navier-Stokes (uniform, isotropic) shear viscosity
//  is implemented. TODO: add Braginskii viscosity

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \class Viscosity
//  \brief data and functions that implement viscosity in Hydro and MHD

class Viscosity {
 public:
  Viscosity(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Viscosity();

  // data
  Real dtnew;
  Real nu_iso;     // coefficient of isotropic kinematic shear viscosity
  bool use_ho;     // flag for 4th-order diffusive operators
  //! Matches input <hydro|mhd>/mignone. When true with use_ho, pass w0_c and use point-value
  //! stencils; when false with use_ho, pass w0 and use cell-average stencils.
  bool mignone_ {false};

  // functions to add viscous fluxes to Hydro and/or MHD fluxes
  //! Pass w0 (2nd order or 4th w/o Mignone) or w0_c (4th with Mignone); see mignone_.
  void IsotropicViscousFlux(const DvceArray5D<Real> &w, const Real nu,
                            const EOS_Data &eos, DvceFaceFld5D<Real> &f);
  void FourthOrderIsotropicViscousFlux(const DvceArray5D<Real> &w, const Real nu,
                                       const EOS_Data &eos, DvceFaceFld5D<Real> &f);

  //! Must be public: Kokkos device lambdas require a public enclosing function (CUDA).
  void FillHoViscFaceVelocity(const DvceArray5D<Real> &w);

 private:
  MeshBlockPack* pmy_pack;

  //! 4th-order face-centered velocity (Vx,Vy,Vz) from HoFaceValue along each face
  //! normal; transverse viscous terms use HoGrad on these face values.
  //! x1f/x2f/x3f use n=0..2. Only allocated when fourth_order_diff is true.
  DvceFaceFld5D<Real> ho_face_vel_;
};

#endif // DIFFUSION_VISCOSITY_HPP_
