#ifndef UTILS_MOOD_BOUNDS_HPP_
#define UTILS_MOOD_BOUNDS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mood_bounds.hpp
//! \brief Separable neighbourhood min/max for the MOOD relaxed-DMP test
//! (hydro_mood.cpp, mhd_mood.cpp).
//!
//! The NAD test compares a candidate value against the min/max of a stage-input
//! reference quantity over the 3^d-cell neighbourhood.  The reference quantities do
//! not change during the revision loop, so their bounds are computed ONCE per stage
//! here instead of once per tested variable, per cell, per revision inside the
//! detection kernel, and the 27-point scan is replaced by three 3-point sweeps (one per
//! active dimension), each sweep handling every tested quantity ("slot") in one kernel.
//! fmin/fmax are exact and insensitive to association, so the bounds -- and every
//! detector decision -- are bit-identical to the direct scan.  NaN handling is also
//! unchanged: a NaN neighbour is ignored unless every value in the box is NaN, in either
//! formulation.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn MoodNeighborhoodBounds()
//! \brief min/max of the nq slots of the per-cell values `val` over the neighbourhood of
//! every cell in [il,iu]x[jl,ju]x[kl,ku] of MeshBlocks 0..nmb1 (active dimensions only).
//! `val(m,q,...)` must be valid one cell beyond that range in every active dimension.
//! qmn/qmx/tmx are nq-slot work arrays; `val` is overwritten (used as scratch) once the
//! x1 sweep has consumed it.  On return (bmn,bmx) name the arrays holding the bounds:
//! 1D: val -> (qmn,qmx);  2D: val -> (qmn,qmx) -> (val,tmx);
//! 3D: val -> (qmn,qmx) -> (val,tmx) -> (qmn,qmx).

inline void MoodNeighborhoodBounds(const int nmb1, const int nq,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku,
    const bool multi_d, const bool three_d, const DvceArray5D<Real> &val,
    const DvceArray5D<Real> &qmn, const DvceArray5D<Real> &qmx,
    const DvceArray5D<Real> &tmx, DvceArray5D<Real> &bmn, DvceArray5D<Real> &bmx) {
  const int ndj = (multi_d) ? 1 : 0;
  const int ndk = (three_d) ? 1 : 0;
  par_for("mood_bnd_x1", DevExeSpace(), 0, nmb1, kl-ndk, ku+ndk, jl-ndj, ju+ndj, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int q=0; q<nq; ++q) {
      const Real a = val(m,q,k,j,i-1), b = val(m,q,k,j,i), c = val(m,q,k,j,i+1);
      qmn(m,q,k,j,i) = fmin(fmin(a, b), c);
      qmx(m,q,k,j,i) = fmax(fmax(a, b), c);
    }
  });
  bmn = qmn; bmx = qmx;
  if (multi_d) {
    par_for("mood_bnd_x2", DevExeSpace(), 0, nmb1, kl-ndk, ku+ndk, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int q=0; q<nq; ++q) {
        val(m,q,k,j,i) = fmin(fmin(qmn(m,q,k,j-1,i), qmn(m,q,k,j,i)), qmn(m,q,k,j+1,i));
        tmx(m,q,k,j,i) = fmax(fmax(qmx(m,q,k,j-1,i), qmx(m,q,k,j,i)), qmx(m,q,k,j+1,i));
      }
    });
    bmn = val; bmx = tmx;
  }
  if (three_d) {
    par_for("mood_bnd_x3", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int q=0; q<nq; ++q) {
        qmn(m,q,k,j,i) = fmin(fmin(val(m,q,k-1,j,i), val(m,q,k,j,i)), val(m,q,k+1,j,i));
        qmx(m,q,k,j,i) = fmax(fmax(tmx(m,q,k-1,j,i), tmx(m,q,k,j,i)), tmx(m,q,k+1,j,i));
      }
    });
    bmn = qmn; bmx = qmx;
  }
}

#endif // UTILS_MOOD_BOUNDS_HPP_
