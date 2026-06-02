#ifndef RECONSTRUCT_DC_HPP_
#define RECONSTRUCT_DC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dc.hpp
//! \brief piecewise constant (donor cell) reconstruction implemented as inline functions

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn DonorCellX1()
//! \brief For each cell-centered value q(i), returns ql(i+1) and qr(i) over il to iu.
//! Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//!
//! The optional i0 argument shifts the scratch-array (ql/qr) column index by -i0 so the
//! reconstructed states for a tile of cells can be stored in a scratch buffer narrower
//! than the full meshblock. The mesh array q is always indexed by the global i; only the
//! scratch writes are offset. With the default i0=0 the behavior is unchanged.

KOKKOS_INLINE_FUNCTION
void DonorCellX1(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr, const int i0=0) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      ql(n,i+1-i0) = q(m,n,k,j,i);
      qr(n,i  -i0) = q(m,n,k,j,i);
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn DonorCellX2()
//! \brief For each cell-centered value q(j), returns ql(j+1) and qr(j) over il to iu.
//! This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]
//!
//! The optional i0 argument shifts the scratch-array column index by -i0 (see DonorCellX1).

KOKKOS_INLINE_FUNCTION
void DonorCellX2(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j, const int i0=0) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      ql_jp1(n,i-i0) = q(m,n,k,j,i);
      qr_j  (n,i-i0) = q(m,n,k,j,i);
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn DonorCellX3()
//! \brief For each cell-centered value q(k), returns ql(k+1) and qr(k) over il to iu.
//! This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//!
//! The optional i0 argument shifts the scratch-array column index by -i0 (see DonorCellX1).

KOKKOS_INLINE_FUNCTION
void DonorCellX3(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k, const int i0=0) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      ql_kp1(n,i-i0) = q(m,n,k,j,i);
      qr_k  (n,i-i0) = q(m,n,k,j,i);
    });
  }
  return;
}
#endif // RECONSTRUCT_DC_HPP_
