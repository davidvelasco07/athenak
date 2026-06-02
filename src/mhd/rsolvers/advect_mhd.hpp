#ifndef MHD_RSOLVERS_ADVECT_MHD_HPP_
#define MHD_RSOLVERS_ADVECT_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect_mhd.hpp
//! \brief MHD Riemann solver for pure advection problems (v = constant).  Simply computes
//!  the upwind flux of each variable.  Can only be used for isothermal EOS.

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void Advect
//! \brief An advection Riemann solver for MHD (isothermal)

KOKKOS_INLINE_FUNCTION
void Advect(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez,
     const int i0=0) {
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    const int si = i - i0;  // column index into the (possibly tiled) scratch arrays
    //  Compute upwind fluxes
    if (wl(ivx,si) >= 0.0) {
      flx(m,IDN,k,j,i) = wl(IDN,si)*wl(ivx,si);
      flx(m,ivx,k,j,i) = wl(IDN,si)*wl(ivx,si)*wl(ivx,si);
      flx(m,ivy,k,j,i) = 0.0;
      flx(m,ivz,k,j,i) = 0.0;
      ey(m,k,j,i) = -bl(iby,si)*wl(ivx,si) + bx(m,k,j,i)*wl(ivy,si);
      ez(m,k,j,i) =  bl(ibz,si)*wl(ivx,si) - bx(m,k,j,i)*wl(ivz,si);
    } else {
      flx(m,IDN,k,j,i) = wr(IDN,si)*wr(ivx,si);
      flx(m,ivx,k,j,i) = wr(IDN,si)*wr(ivx,si)*wr(ivx,si);
      flx(m,ivy,k,j,i) = 0.0;
      flx(m,ivz,k,j,i) = 0.0;
      ey(m,k,j,i) = -br(iby,si)*wr(ivx,si) + bx(m,k,j,i)*wr(ivy,si);
      ez(m,k,j,i) =  br(ibz,si)*wr(ivx,si) - bx(m,k,j,i)*wr(ivz,si);
    }
  });

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_ADVECT_MHD_HPP_
