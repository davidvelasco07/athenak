#ifndef MHD_MHD_WB_HPP_
#define MHD_MHD_WB_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_wb.hpp
//! \brief device helpers of the well-balanced MHD scheme shared by the flux path
//! (mhd_wb.cpp) and the MOOD/FOFC revise paths (mhd_mood.cpp, mhd_fofc.cpp).

#include "athena.hpp"
#include "eos/eos.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn AddFaceEquilibriumMHD()
//! \brief Add the face equilibrium (primitives and cell-centred B) to both reconstructed
//! perturbation states of face (m,k,j,i), then floor the full fluid states.
KOKKOS_INLINE_FUNCTION
void AddFaceEquilibriumMHD(const EOS_Data &eos, const bool floors, const int nmhd,
                           const int m, const int k, const int j, const int i,
                           const DvceArray5D<Real> &weqf, const DvceArray5D<Real> &beqf,
                           const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
                           const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br) {
  for (int n=0; n<nmhd; ++n) {
    wl(m,n,k,j,i) += weqf(m,n,k,j,i);
    wr(m,n,k,j,i) += weqf(m,n,k,j,i);
  }
  for (int n=0; n<3; ++n) {
    bl(m,n,k,j,i) += beqf(m,n,k,j,i);
    br(m,n,k,j,i) += beqf(m,n,k,j,i);
  }
  if (floors) {
    wl(m,IDN,k,j,i) = fmax(wl(m,IDN,k,j,i), eos.dfloor);
    wr(m,IDN,k,j,i) = fmax(wr(m,IDN,k,j,i), eos.dfloor);
    const Real efloor = eos.pfloor/(eos.gamma - 1.0);
    wl(m,IEN,k,j,i) = fmax(wl(m,IEN,k,j,i), efloor);
    wr(m,IEN,k,j,i) = fmax(wr(m,IEN,k,j,i), efloor);
  }
}

} // namespace mhd
#endif // MHD_MHD_WB_HPP_
