#ifndef DIFFUSION_CONDUCTION_HPP_
#define DIFFUSION_CONDUCTION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.hpp
//! \brief Contains data and functions that implement various formulations for conduction.
//  Currently only isotropic conduction implemented

#include <string>

#include "athena.hpp"
#include "diffusion/sts_types.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
// Face-local isotropic heat flux (the quantity to SUBTRACT from the energy flux through
// one face).  Shared by Conduction::AddHeatFluxIso and the MOOD/FOFC revise passes.

KOKKOS_INLINE_FUNCTION
Real HeatFluxIsoX1(const Real alpha, const Real gm1, const DvceArray5D<Real> &w0,
                   const RegionSize &sz, const int m, const int k, const int j,
                   const int i) {
  Real tempr = w0(m,IEN,k,j,i  )/w0(m,IDN,k,j,i  );
  Real templ = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1);
  Real dtempdx = (tempr - templ) * gm1 / sz.dx1;
  Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
  return alpha * densf * dtempdx;
}

KOKKOS_INLINE_FUNCTION
Real HeatFluxIsoX2(const Real alpha, const Real gm1, const DvceArray5D<Real> &w0,
                   const RegionSize &sz, const int m, const int k, const int j,
                   const int i) {
  Real tempr = w0(m,IEN,k,j  ,i)/w0(m,IDN,k,j  ,i);
  Real templ = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i);
  Real dtempdx = (tempr - templ) * gm1 / sz.dx2;
  Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
  return alpha * densf * dtempdx;
}

KOKKOS_INLINE_FUNCTION
Real HeatFluxIsoX3(const Real alpha, const Real gm1, const DvceArray5D<Real> &w0,
                   const RegionSize &sz, const int m, const int k, const int j,
                   const int i) {
  Real tempr = w0(m,IEN,k  ,j,i)/w0(m,IDN,k  ,j,i);
  Real templ = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i);
  Real dtempdx = (tempr - templ) * gm1 / sz.dx3;
  Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
  return alpha * densf * dtempdx;
}

//----------------------------------------------------------------------------------------
//! \class Conduction
//! \brief data and functions that implement thermal conduction in Hydro and MHD

class Conduction {
 public:
  Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Conduction();

  // data
  Real dtnew;
  Real alpha_iso;       // isotropic thermal diffusivity
  Real alpha_aniso;     // anisotropic thermal diffusivity
  bool alpha_spitzer;   // switch to turn on Spitzer conductivity
  Real q_limit;         // saturated heat flux limit
  parabolic::DiffusionSelection mode = parabolic::DiffusionSelection::explicit_only;

  // functions
  void AddHeatFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                     DvceFaceFld5D<Real> &f);
  void AddHeatFluxIso(const DvceArray5D<Real> &w, const EOS_Data &eos,
                      DvceFaceFld5D<Real> &f);
  void AddHeatFluxAniso(const DvceArray5D<Real> &w, const EOS_Data &eos,
                        DvceFaceFld5D<Real> &f);
  void AddHeatFluxSpitzer(const DvceArray5D<Real> &w, const EOS_Data &eos,
                          DvceFaceFld5D<Real> &f);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

 private:
  MeshBlockPack* pmy_pack;
};
#endif // DIFFUSION_CONDUCTION_HPP_
