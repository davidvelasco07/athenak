#ifndef RECONSTRUCT_FV_HPP_
#define RECONSTRUCT_FV_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fv.hpp
//! \brief UNLIMITED finite-volume reconstruction of cell averages at 4th, 6th and 8th
//! order on a uniform grid (reconstruct = fv4 | fv6 | fv8).
//!
//! Two families of interpolants live here.
//!
//! 1. Cell-polynomial face states (FV4 / FV6 / FV8, used by the Riemann solver).  Each
//!    cell i carries its own degree-p reconstruction polynomial P_i from the symmetric
//!    kernel of 2r+1 = p+2 cell averages q_{i-r} .. q_{i+r} (the average of the two
//!    degree-p kernels of p+1 cells that straddle the cell, Palafoutas, Velasco-Romero &
//!    Teyssier 2025, JCP 522, Table 1), evaluated at its own two faces:
//!
//!      qr_i   = P_i(x_{i-1/2}),   ql_{i+1} = P_i(x_{i+1/2}).
//!
//!    The two cells sharing a face therefore give DIFFERENT states (the jump is
//!    O(h^{p+1}) for smooth data) and the Riemann solver adds upwind dissipation at that
//!    order.  Left-face weights on q_{i-r} .. q_{i+r} (right face = reversed):
//!
//!      fv4 (p=3):  (  -1,   10,   20,    -6,     1) / 24
//!      fv6 (p=5):  (   1,  -10,   59,    94,   -31,     8,   -1) / 120
//!      fv8 (p=7):  (  -3,   34, -194,   898,  1276,  -446,  142,  -30,  3) / 1680
//!
//!    The right-face value is the SAME function applied to the reversed arguments, so
//!    the pair is bit-identical under reflection of the stencil and exactly symmetric
//!    problems stay exactly symmetric.
//!
//! 2. Single-valued central face interpolants (FV4Face / FV6Face / FV8Face, FVFaceAt):
//!    the point value at the face between cells i and i+1 from the N cell averages
//!    symmetric about that face, the unique weights exact for polynomials up to degree
//!    N-1:
//!
//!      fv4:  ( -q_{i-1} +  7 q_i  +  7 q_{i+1} -  q_{i+2}) / 12
//!      fv6:  (  q_{i-2} -  8 q_{i-1} + 37 q_i + 37 q_{i+1} -  8 q_{i+2} +  q_{i+3}) / 60
//!      fv8:  (-3 q_{i-3} + 29 q_{i-2} - 139 q_{i-1} + 533 q_i
//!                 + 533 q_{i+1} - 139 q_{i+2} + 29 q_{i+3} - 3 q_{i+4}) / 840
//!
//!    The UCT edge value of the face field (mhd_corner_e.cpp ReconstructEdge, where
//!    both sides get the same
//!    value) and the well-balanced face equilibrium (FVFaceAt) use them.  fv4 is the
//!    unlimited base of PPM4/PPMX (CS08 eqn 16).
//!    Magnetic face-to-volume averaging instead integrates the normal polynomial;
//!    its different weights are in eos/mhd_bcc.hpp.
//!
//! Either way there is no limiter: stability relies entirely on the MOOD a-posteriori
//! fallback (reconstruct=fvN requires mood=true, enforced in the Hydro/MHD constructors;
//! Clain, Diot & Loubere 2011).
//!
#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn FV4Face()
//! \brief 4th-order central (single-valued) face value at i+1/2 from q_{i-1} .. q_{i+2}.

KOKKOS_INLINE_FUNCTION
Real FV4Face(const Real &q_im1, const Real &q_i, const Real &q_ip1, const Real &q_ip2) {
  return (7.0*(q_i + q_ip1) - (q_im1 + q_ip2))/12.0;
}

//----------------------------------------------------------------------------------------
//! \fn FV6Face()
//! \brief 6th-order central (single-valued) face value at i+1/2 from q_{i-2} .. q_{i+3}.

KOKKOS_INLINE_FUNCTION
Real FV6Face(const Real &q_im2, const Real &q_im1, const Real &q_i,
             const Real &q_ip1, const Real &q_ip2, const Real &q_ip3) {
  return ((q_im2 + q_ip3) - 8.0*(q_im1 + q_ip2) + 37.0*(q_i + q_ip1))/60.0;
}

//----------------------------------------------------------------------------------------
//! \fn FV8Face()
//! \brief 8th-order central (single-valued) face value at i+1/2 from q_{i-3} .. q_{i+4}.

KOKKOS_INLINE_FUNCTION
Real FV8Face(const Real &q_im3, const Real &q_im2, const Real &q_im1, const Real &q_i,
             const Real &q_ip1, const Real &q_ip2, const Real &q_ip3, const Real &q_ip4) {
  return (-3.0*(q_im3 + q_ip4) + 29.0*(q_im2 + q_ip3)
          - 139.0*(q_im1 + q_ip2) + 533.0*(q_i + q_ip1))/840.0;
}

//----------------------------------------------------------------------------------------
//! \fn FV4Left() / FV6Left() / FV8Left()
//! \brief Value of the cell polynomial P_i at the LEFT face x_{i-1/2}, from the kernel
//! q_{i-r} .. q_{i+r} (r = 2, 3, 4).  The right-face value P_i(x_{i+1/2}) is the same
//! function on the reversed arguments.

KOKKOS_INLINE_FUNCTION
Real FV4Left(const Real &q_im2, const Real &q_im1, const Real &q_i,
             const Real &q_ip1, const Real &q_ip2) {
  return (20.0*q_i + (10.0*q_im1 - 6.0*q_ip1) + (q_ip2 - q_im2))/24.0;
}

KOKKOS_INLINE_FUNCTION
Real FV6Left(const Real &q_im3, const Real &q_im2, const Real &q_im1, const Real &q_i,
             const Real &q_ip1, const Real &q_ip2, const Real &q_ip3) {
  return (94.0*q_i + (59.0*q_im1 - 31.0*q_ip1) + (8.0*q_ip2 - 10.0*q_im2)
          + (q_im3 - q_ip3))/120.0;
}

KOKKOS_INLINE_FUNCTION
Real FV8Left(const Real &q_im4, const Real &q_im3, const Real &q_im2, const Real &q_im1,
             const Real &q_i, const Real &q_ip1, const Real &q_ip2, const Real &q_ip3,
             const Real &q_ip4) {
  return (1276.0*q_i + (898.0*q_im1 - 446.0*q_ip1) + (142.0*q_ip2 - 194.0*q_im2)
          + (34.0*q_im3 - 30.0*q_ip3) + (3.0*q_ip4 - 3.0*q_im4))/1680.0;
}

//----------------------------------------------------------------------------------------
//! \fn FV4() / FV6() / FV8()
//! \brief Cell-oriented wrappers with the same interface convention as PLM/PPM4/WENOZ:
//! return the states of cell i at its two faces, ql(i+1) = P_i(x_{i+1/2}) (right face)
//! and qr(i) = P_i(x_{i-1/2}) (left face), from the kernel q_{i-r} .. q_{i+r}, r = N/2.

KOKKOS_INLINE_FUNCTION
void FV4(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
         const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  qr_i   = FV4Left(q_im2, q_im1, q_i, q_ip1, q_ip2);
  ql_ip1 = FV4Left(q_ip2, q_ip1, q_i, q_im1, q_im2);
  return;
}

KOKKOS_INLINE_FUNCTION
void FV6(const Real &q_im3, const Real &q_im2, const Real &q_im1, const Real &q_i,
         const Real &q_ip1, const Real &q_ip2, const Real &q_ip3,
         Real &ql_ip1, Real &qr_i) {
  qr_i   = FV6Left(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3);
  ql_ip1 = FV6Left(q_ip3, q_ip2, q_ip1, q_i, q_im1, q_im2, q_im3);
  return;
}

KOKKOS_INLINE_FUNCTION
void FV8(const Real &q_im4, const Real &q_im3, const Real &q_im2, const Real &q_im1,
         const Real &q_i, const Real &q_ip1, const Real &q_ip2, const Real &q_ip3,
         const Real &q_ip4, Real &ql_ip1, Real &qr_i) {
  qr_i   = FV8Left(q_im4, q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, q_ip3, q_ip4);
  ql_ip1 = FV8Left(q_ip4, q_ip3, q_ip2, q_ip1, q_i, q_im1, q_im2, q_im3, q_im4);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FVFaceAt()
//! \brief Face value at i-1/2 from the cell averages returned by the accessor q(o)
//! (o = offset from cell i, so the face lies between q(-1) and q(0)): the fv6/fv8
//! interpolant for reconstruct=fv6/fv8 and the fv4 one for every other method.  Reads
//! q(-FVFaceRadius(rm)) .. q(FVFaceRadius(rm)-1) (athena.hpp).  Used by the well-balanced
//! scheme for its single-valued face equilibrium and UCT equilibrium edge value.

template <typename F>
KOKKOS_INLINE_FUNCTION
Real FVFaceAt(const ReconstructionMethod rm, const F &q) {
  if (rm == ReconstructionMethod::fv6) {
    return FV6Face(q(-3), q(-2), q(-1), q(0), q(1), q(2));
  } else if (rm == ReconstructionMethod::fv8) {
    return FV8Face(q(-4), q(-3), q(-2), q(-1), q(0), q(1), q(2), q(3));
  }
  return FV4Face(q(-2), q(-1), q(0), q(1));
}

#endif // RECONSTRUCT_FV_HPP_
