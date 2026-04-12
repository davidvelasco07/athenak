#ifndef DIFFUSION_HO_DIFFUSION_STENCIL_HPP_
#define DIFFUSION_HO_DIFFUSION_STENCIL_HPP_
//========================================================================================
//! \file ho_diffusion_stencil.hpp
//! \brief Shared 4th-order finite-volume stencils for HO diffusion (viscosity, conduction,
//!  scalar diffusion): point-valued cell-center data (Mignone w0_c) vs. volume averages (w0).
//========================================================================================

#include "athena.hpp"

struct HoStencilCoef {
  Real t_m2, t_m1, t_p1, t_p2, t_den;  // transverse: d/dx at cell i, stencil i±1,i±2
  Real n_m2, n_m1, n_0, n_p1, n_den;   // normal: face between i-1|i, stencil i-2..i+1
};

KOKKOS_INLINE_FUNCTION
HoStencilCoef HoStencilCoefFor(const bool point_data) {
  HoStencilCoef c{};
  if (point_data) {
    c.t_m2 = 1.0;  c.t_m1 = -8.0; c.t_p1 = 8.0;  c.t_p2 = -1.0; c.t_den = 12.0;
    c.n_m2 = 1.0;  c.n_m1 = -27.0; c.n_0 = 27.0; c.n_p1 = -1.0; c.n_den = 24.0;
  } else {
    c.t_m2 = 5.0;  c.t_m1 = -34.0; c.t_p1 = 34.0; c.t_p2 = -5.0; c.t_den = 48.0;
    c.n_m2 = 1.0;  c.n_m1 = -15.0; c.n_0 = 15.0; c.n_p1 = -1.0; c.n_den = 12.0;
  }
  return c;
}

//! Normal derivative at face from four collinear samples (same footprint as HoNormDir*).
KOKKOS_INLINE_FUNCTION
Real HoGradT(const bool use_point, const Real v_m2, const Real v_m1,
                      const Real v_p1, const Real v_p2, const Real inv_dx) {
  const HoStencilCoef c = HoStencilCoefFor(use_point);
  return (c.t_m2*v_m2 + c.t_m1*v_m1 + c.t_p1*v_p1 + c.t_p2*v_p2) * (inv_dx/c.t_den);
}

KOKKOS_INLINE_FUNCTION
Real HoGrad(const bool use_point, const Real v_m2, const Real v_m1,
                      const Real v_0, const Real v_p1, const Real inv_dx) {
  const HoStencilCoef c = HoStencilCoefFor(use_point);
  return (c.n_m2*v_m2 + c.n_m1*v_m1 + c.n_0*v_0 + c.n_p1*v_p1) * (inv_dx/c.n_den);
}

//! Face value from four cell-centered samples (same (7/12,-1/12) for point and averages).
KOKKOS_INLINE_FUNCTION
Real HoFaceValue(const Real f_m2, const Real f_m1, const Real f_0,
                      const Real f_p1) {
  return (7.0/12.0)*(f_m1 + f_0) - (1.0/12.0)*(f_m2 + f_p1);
}

KOKKOS_INLINE_FUNCTION
Real HoGradDir1(const bool use_point, const DvceArray5D<Real> &w,
                int m, int n, int k, int j, int i, Real idx1) {
  const HoStencilCoef c = HoStencilCoefFor(use_point);
  return (c.n_m2*w(m,n,k,j,i-2) + c.n_m1*w(m,n,k,j,i-1) + c.n_0*w(m,n,k,j,i)
          + c.n_p1*w(m,n,k,j,i+1)) * (idx1/c.n_den);
}
KOKKOS_INLINE_FUNCTION
Real HoGradDir2(const bool use_point, const DvceArray5D<Real> &w,
                int m, int n, int k, int j, int i, Real idx2) {
  const HoStencilCoef c = HoStencilCoefFor(use_point);
  return (c.n_m2*w(m,n,k,j-2,i) + c.n_m1*w(m,n,k,j-1,i) + c.n_0*w(m,n,k,j,i)
          + c.n_p1*w(m,n,k,j+1,i)) * (idx2/c.n_den);
}
KOKKOS_INLINE_FUNCTION
Real HoGradDir3(const bool use_point, const DvceArray5D<Real> &w,
                int m, int n, int k, int j, int i, Real idx3) {
  const HoStencilCoef c = HoStencilCoefFor(use_point);
  return (c.n_m2*w(m,n,k-2,j,i) + c.n_m1*w(m,n,k-1,j,i) + c.n_0*w(m,n,k,j,i)
          + c.n_p1*w(m,n,k+1,j,i)) * (idx3/c.n_den);
}

#endif // DIFFUSION_HO_DIFFUSION_STENCIL_HPP_
