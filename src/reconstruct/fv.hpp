#ifndef RECONSTRUCT_FV_HPP_
#define RECONSTRUCT_FV_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fv.hpp
//! \brief UNLIMITED central finite-volume face interpolation of cell averages at 4th,
//! 6th and 8th order on a uniform grid (reconstruct = fv4 | fv6 | fv8).
//!
//! Each function returns the point value at the face between cells i and i+1 from the
//! N cell averages symmetric about that face.  The weights are the unique ones exact for
//! polynomials up to degree N-1 (hence order N for smooth data):
//!
//!   fv4:  ( -q_{i-1} +  7 q_i  +  7 q_{i+1} -  q_{i+2}) / 12
//!   fv6:  (  q_{i-2} -  8 q_{i-1} + 37 q_i + 37 q_{i+1} -  8 q_{i+2} +  q_{i+3}) / 60
//!   fv8:  (-3 q_{i-3} + 29 q_{i-2} - 139 q_{i-1} + 533 q_i
//!              + 533 q_{i+1} - 139 q_{i+2} + 29 q_{i+3} - 3 q_{i+4}) / 840
//!
//! The interpolant is symmetric so both cells sharing the face receive the SAME value
//! (wl = wr): the Riemann solver then returns the pure physical flux with no upwind
//! dissipation and no limiting at all.  Stability relies entirely on the MOOD a-posteriori
//! fallback (reconstruct=fvN requires mood=true, enforced in the Hydro/MHD constructors).
//! fv4 is the unlimited base of PPM4/PPMX (CS08 eqn 16); fv6/fv8 are the natural
//! higher-order extensions used by MOOD-type schemes (Clain, Diot & Loubere 2011).
//!
//! The arguments are added in reflection-symmetric pairs (outer to inner) so the result
//! is bit-identical under reversal of the stencil, which keeps exactly symmetric
//! problems exactly symmetric.
//!
//! REFERENCES:
//! (CS) P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth
//! extrema", JCP, 227, 7069 (2008)
//!
//! (CDL) S. Clain, S. Diot & R. Loubere, "A high-order finite volume method for systems
//! of conservation laws -- Multi-dimensional Optimal Order Detection (MOOD)", JCP, 230,
//! 4028 (2011)

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn FV4Face()
//! \brief 4th-order face value at i+1/2 from the 4 cell averages q_{i-1} .. q_{i+2}.

KOKKOS_INLINE_FUNCTION
Real FV4Face(const Real &q_im1, const Real &q_i, const Real &q_ip1, const Real &q_ip2) {
  return (7.0*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;
}

//----------------------------------------------------------------------------------------
//! \fn FV6Face()
//! \brief 6th-order face value at i+1/2 from the 6 cell averages q_{i-2} .. q_{i+3}.

KOKKOS_INLINE_FUNCTION
Real FV6Face(const Real &q_im2, const Real &q_im1, const Real &q_i,
             const Real &q_ip1, const Real &q_ip2, const Real &q_ip3) {
  return ((q_im2 + q_ip3) - 8.0*(q_im1 + q_ip2) + 37.0*(q_i + q_ip1))/60.0;
}

//----------------------------------------------------------------------------------------
//! \fn FV8Face()
//! \brief 8th-order face value at i+1/2 from the 8 cell averages q_{i-3} .. q_{i+4}.

KOKKOS_INLINE_FUNCTION
Real FV8Face(const Real &q_im3, const Real &q_im2, const Real &q_im1, const Real &q_i,
             const Real &q_ip1, const Real &q_ip2, const Real &q_ip3, const Real &q_ip4) {
  return (-3.0*(q_im3 + q_ip4) + 29.0*(q_im2 + q_ip3)
          - 139.0*(q_im1 + q_ip2) + 533.0*(q_i + q_ip1))/840.0;
}

//----------------------------------------------------------------------------------------
//! \fn FV4() / FV6() / FV8()
//! \brief Cell-oriented wrappers with the same interface convention as PLM/PPM4/WENOZ:
//! return the interpolated values at the L/R faces of cell i, i.e. ql(i+1) (right face)
//! and qr(i) (left face).  The stencil is q_{i-r} .. q_{i+r} with r = N/2.

KOKKOS_INLINE_FUNCTION
void FV4(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
         const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  ql_ip1 = FV4Face(q_im1, q_i, q_ip1, q_ip2);
  qr_i   = FV4Face(q_im2, q_im1, q_i, q_ip1);
  return;
}

KOKKOS_INLINE_FUNCTION
void FV6(const Real &q_im3, const Real &q_im2, const Real &q_im1, const Real &q_i,
         const Real &q_ip1, const Real &q_ip2, const Real &q_ip3,
         Real &ql_ip1, Real &qr_i) {
  ql_ip1 = FV6Face(q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3);
  qr_i   = FV6Face(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2);
  return;
}

KOKKOS_INLINE_FUNCTION
void FV8(const Real &q_im4, const Real &q_im3, const Real &q_im2, const Real &q_im1,
         const Real &q_i, const Real &q_ip1, const Real &q_ip2, const Real &q_ip3,
         const Real &q_ip4, Real &ql_ip1, Real &qr_i) {
  ql_ip1 = FV8Face(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3, q_ip4);
  qr_i   = FV8Face(q_im4, q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3);
  return;
}

#endif // RECONSTRUCT_FV_HPP_
