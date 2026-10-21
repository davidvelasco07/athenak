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
#include "diffusion/sts_types.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// Face-local isotropic viscous flux.  The bulk kernels (Viscosity::AddViscousFluxIso) and
// the MOOD/FOFC revise passes (hydro_mood.cpp, hydro_fofc.cpp, mhd_mood.cpp,
// mhd_fofc.cpp) share these functions, so a revised face receives exactly the viscous
// flux the bulk pass gave it (see mood_candidate_diffusion in hydro.cpp / mhd.cpp).
// Each function returns the quantities to SUBTRACT from the momentum fluxes (always the
// global x,y,z components, whatever the face normal) and from the energy flux.

struct ViscFaceParams {
  Real nu;             // nu_iso
  Real sr_y, sr_z;     // background shear d(v_K)/dx added to dVy/dx (3D) or dVz/dx (2D)
  bool add_shear;      // shearing box: w0 holds the velocity perturbation
  bool multi_d, three_d;
  bool is_ideal;
};

struct ViscFaceFlux {
  Real nud;            // 0.5 nu (rho_L + rho_R)
  Real fvx, fvy, fvz;  // velocity-gradient combinations (global x,y,z components)
  Real ve;             // (vL+vR).fv, so the energy term is 0.5*nud*ve
};
// The caller subtracts nud*fvx, nud*fvy, nud*fvz and 0.5*nud*ve: keeping the products at
// the call site preserves the expression shape (and so the compiler's fused multiply-add
// decisions) of the original bulk kernels, which keeps old and new results bit-identical.

KOKKOS_INLINE_FUNCTION
ViscFaceFlux ViscousFluxX1(const ViscFaceParams &p, const DvceArray5D<Real> &w0,
                           const RegionSize &sz, const int m, const int k, const int j,
                           const int i) {
  // [2(dVx/dx)-(2/3)dVx/dx, dVy/dx, dVz/dx]
  Real fvx = 4.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k,j,i-1))/(3.0*sz.dx1);
  Real fvy =     (w0(m,IVY,k,j,i) - w0(m,IVY,k,j,i-1))/sz.dx1;
  Real fvz =     (w0(m,IVZ,k,j,i) - w0(m,IVZ,k,j,i-1))/sz.dx1;
  // In 2D/3D add [(-2/3)dVy/dy, dVx/dy, 0]
  if (p.multi_d) {
    fvx -= ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k,j+1,i-1)) -
            (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j-1,i-1)))/(6.0*sz.dx2);
    fvy += ((w0(m,IVX,k,j+1,i) + w0(m,IVX,k,j+1,i-1)) -
            (w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j-1,i-1)))/(4.0*sz.dx2);
  }
  // In 3D add [(-2/3)dVz/dz, 0,  dVx/dz]
  if (p.three_d) {
    fvx -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j,i-1)) -
            (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j,i-1)))/(6.0*sz.dx3);
    fvz += ((w0(m,IVX,k+1,j,i) + w0(m,IVX,k+1,j,i-1)) -
            (w0(m,IVX,k-1,j,i) + w0(m,IVX,k-1,j,i-1)))/(4.0*sz.dx3);
  }
  // Shearing box: add background shear d(v_K)/dx to dVy/dx (3D) or dVz/dx (2D r-z)
  if (p.add_shear) {
    fvy += p.sr_y;
    fvz += p.sr_z;
  }
  Real nud = 0.5*p.nu*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
  ViscFaceFlux f;
  f.nud = nud;
  f.fvx = fvx;
  f.fvy = fvy;
  f.fvz = fvz;
  f.ve = 0.0;
  if (p.is_ideal) {
    f.ve = (w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j,i))*fvx +
           (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j,i))*fvy +
           (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k,j,i))*fvz;
  }
  return f;
}

KOKKOS_INLINE_FUNCTION
ViscFaceFlux ViscousFluxX2(const ViscFaceParams &p, const DvceArray5D<Real> &w0,
                           const RegionSize &sz, const int m, const int k, const int j,
                           const int i) {
  // [(dVx/dy+dVy/dx), 2(dVy/dy)-(2/3)(dVx/dx+dVy/dy), dVz/dy]
  Real fvx = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k,j-1,i  ))/sz.dx2 +
            ((w0(m,IVY,k,j,i+1) + w0(m,IVY,k,j-1,i+1)) -
             (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j-1,i-1)))/(4.0*sz.dx1);
  Real fvy = (w0(m,IVY,k,j,i) - w0(m,IVY,k,j-1,i))*4.0/(3.0*sz.dx2) -
            ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k,j-1,i+1)) -
             (w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j-1,i-1)))/(6.0*sz.dx1);
  Real fvz = (w0(m,IVZ,k,j,i  ) - w0(m,IVZ,k,j-1,i  ))/sz.dx2;
  // In 3D add [0, (-2/3)dVz/dz, dVy/dz]
  if (p.three_d) {
    fvy -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j-1,i)) -
            (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j-1,i)))/(6.0*sz.dx3);
    fvz += ((w0(m,IVY,k+1,j,i) + w0(m,IVY,k+1,j-1,i)) -
            (w0(m,IVY,k-1,j,i) + w0(m,IVY,k-1,j-1,i)))/(4.0*sz.dx3);
  }
  // Shearing box (3D): sigma_yx = sigma_xy also carries the background shear d(v_K)/dx
  if (p.add_shear && p.sr_y != 0.0) {
    fvx += p.sr_y;
  }
  Real nud = 0.5*p.nu*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
  ViscFaceFlux f;
  f.nud = nud;
  f.fvx = fvx;
  f.fvy = fvy;
  f.fvz = fvz;
  f.ve = 0.0;
  if (p.is_ideal) {
    f.ve = (w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j,i))*fvx +
           (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j,i))*fvy +
           (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k,j,i))*fvz;
  }
  return f;
}

KOKKOS_INLINE_FUNCTION
ViscFaceFlux ViscousFluxX3(const ViscFaceParams &p, const DvceArray5D<Real> &w0,
                           const RegionSize &sz, const int m, const int k, const int j,
                           const int i) {
  // [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 2(dVz/dz)-(2/3)(dVx/dx+dVy/dy+dVz/dz)]
  Real fvx = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k-1,j,i  ))/sz.dx3 +
            ((w0(m,IVZ,k,j,i+1) + w0(m,IVZ,k-1,j,i+1)) -
             (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k-1,j,i-1)))/(4.0*sz.dx1);
  Real fvy = (w0(m,IVY,k,j,i  ) - w0(m,IVY,k-1,j,i  ))/sz.dx3 +
            ((w0(m,IVZ,k,j+1,i) + w0(m,IVZ,k-1,j+1,i)) -
             (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k-1,j-1,i)))/(4.0*sz.dx2);
  Real fvz = (w0(m,IVZ,k,j,i) - w0(m,IVZ,k-1,j,i))*4.0/(3.0*sz.dx3) -
            ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k-1,j,i+1)) -
             (w0(m,IVX,k,j,i-1) + w0(m,IVX,k-1,j,i-1)))/(6.0*sz.dx1) -
            ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k-1,j+1,i)) -
             (w0(m,IVY,k,j-1,i) + w0(m,IVY,k-1,j-1,i)))/(6.0*sz.dx2);
  Real nud = 0.5*p.nu*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
  ViscFaceFlux f;
  f.nud = nud;
  f.fvx = fvx;
  f.fvy = fvy;
  f.fvz = fvz;
  f.ve = 0.0;
  if (p.is_ideal) {
    f.ve = (w0(m,IVX,k-1,j,i) + w0(m,IVX,k,j,i))*fvx +
           (w0(m,IVY,k-1,j,i) + w0(m,IVY,k,j,i))*fvy +
           (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k,j,i))*fvz;
  }
  return f;
}

//----------------------------------------------------------------------------------------
//! \class Viscosity
//  \brief data and functions that implement viscosity in Hydro and MHD

class Viscosity {
 public:
  Viscosity(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Viscosity();

  // data
  Real dtnew;
  Real nu_iso;      // coefficient of isotropic kinematic shear viscosity
  Real nu_aniso;    // coefficient of anisotropic kinematic shear viscosity
  parabolic::DiffusionSelection mode = parabolic::DiffusionSelection::explicit_only;
  // Shearing box: Hydro/MHD store the velocity PERTURBATION v' = v - v_K with
  // v_K = -qshear*omega0*x in the azimuthal direction, so the background shear
  // d(v_K)/dx must be added to the stress by hand.  Zero outside the shearing box.
  Real shear_rate;  // d(v_K)/dx = -qshear*omega0
  int shear_dir;    // azimuthal velocity index: IVY in 3D, IVZ in the 2D (r-z) box

  // function to add viscous fluxes to Hydro and/or MHD fluxes
  void AddViscousFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                        DvceFaceFld5D<Real> &f);
  void AddViscousFluxIso(const DvceArray5D<Real> &w,const EOS_Data &eos,
                         DvceFaceFld5D<Real> &f);
  void AddViscousFluxAniso(const DvceArray5D<Real> &w,const EOS_Data &eos,
                           DvceFaceFld5D<Real> &f);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);
  // parameters of the face-local flux functions above, for this operator
  ViscFaceParams FaceParams(const EOS_Data &eos) const;
  // shearing box: viscous heating by the background shear, -q*Omega*sigma_xy, added to
  // the (perturbation) total energy in the orbital-advection frame
  void AddShearingBoxHeating(const DvceArray5D<Real> &w, const EOS_Data &eos,
                             const Real bdt, DvceArray5D<Real> &u);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_VISCOSITY_HPP_
