#ifndef MHD_RSOLVERS_LLF_MHD_HPP_
#define MHD_RSOLVERS_LLF_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_mhd.hpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver for non-relativistic MHD.

#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void LLF
//! \brief The LLF Riemann solver for MHD (both ideal gas and isothermal)

KOKKOS_INLINE_FUNCTION
void LLF(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez,
     DvceArray4D<Real> uct_aL = {}, DvceArray4D<Real> uct_dL = {},
     DvceArray4D<Real> uct_dR = {}, DvceArray4D<Real> uct_vt1 = {},
     DvceArray4D<Real> uct_vt2 = {}) {
  bool compute_uct = uct_aL.is_allocated();
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    // Extract left/right primitives
    MHDPrim1D wli, wri;
    wli.d  = wl(IDN,i);
    wli.vx = wl(ivx,i);
    wli.vy = wl(ivy,i);
    wli.vz = wl(ivz,i);
    wli.by = bl(iby,i);
    wli.bz = bl(ibz,i);

    wri.d  = wr(IDN,i);
    wri.vx = wr(ivx,i);
    wri.vy = wr(ivy,i);
    wri.vz = wr(ivz,i);
    wri.by = br(iby,i);
    wri.bz = br(ibz,i);

    if (eos.is_ideal) {
      wli.e = wl(IEN,i);
      wri.e = wr(IEN,i);
    }

    // Extract normal magnetic field
    Real &bxi = bx(m,k,j,i);

    // Call LLF solver on single interface state
    MHDCons1D flux;
    SingleStateLLF_MHD(wli,wri,bxi,eos,flux);

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,ivx,k,j,i) = flux.mx;
    flx(m,ivy,k,j,i) = flux.my;
    flx(m,ivz,k,j,i) = flux.mz;
    if (eos.is_ideal) {flx(m,IEN,k,j,i) = flux.e;}
    ey(m,k,j,i) = flux.by;
    ez(m,k,j,i) = flux.bz;

    // Compute UCT coefficients for LLF (Rusanov): alpha_L = alpha_R = lambda_max
    if (compute_uct) {
      Real qa, qb;
      if (eos.is_ideal) {
        Real pl = eos.IdealGasPressure(wli.e);
        Real pr = eos.IdealGasPressure(wri.e);
        qa = eos.IdealMHDFastSpeed(wli.d, pl, bxi, wli.by, wli.bz);
        qb = eos.IdealMHDFastSpeed(wri.d, pr, bxi, wri.by, wri.bz);
      } else {
        qa = eos.IdealMHDFastSpeed(wli.d, bxi, wli.by, wli.bz);
        qb = eos.IdealMHDFastSpeed(wri.d, bxi, wri.by, wri.bz);
      }
      Real lmax = fmax((fabs(wli.vx) + qa), (fabs(wri.vx) + qb));
      uct_aL(m,k,j,i)  = 0.5;
      uct_dL(m,k,j,i)  = 0.5*lmax;
      uct_dR(m,k,j,i)  = 0.5*lmax;
      uct_vt1(m,k,j,i) = 0.5*(wli.vy + wri.vy);
      uct_vt2(m,k,j,i) = 0.5*(wli.vz + wri.vz);
    }
  });

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_LLF_MHD_HPP_
