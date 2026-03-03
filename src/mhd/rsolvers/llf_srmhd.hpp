#ifndef MHD_RSOLVERS_LLF_SRMHD_HPP_
#define MHD_RSOLVERS_LLF_SRMHD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srmhd.hpp
//! \brief Local Lax-Friedrichs (LLF) Riemann solver for special relativistic MHD.

#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void LLF
//! \brief The LLF Riemann solver for SR MHD

KOKKOS_INLINE_FUNCTION
void LLF_SR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez,
     DvceArray4D<Real> uct_aL = {}, DvceArray4D<Real> uct_dL = {},
     DvceArray4D<Real> uct_dR = {}, DvceArray4D<Real> uct_vt1 = {},
     DvceArray4D<Real> uct_vt2 = {}) {
  bool compute_uct = uct_aL.is_allocated();
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    // Extract left/right primitives
    MHDPrim1D wli,wri;
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

    wli.e = wl(IEN,i);
    wri.e = wr(IEN,i);

    // Extract normal magnetic field
    Real &bxi = bx(m,k,j,i);

    // Call LLF solver on single interface state
    MHDCons1D flux;
    SingleStateLLF_SRMHD(wli,wri,bxi,eos,flux);

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,IEN,k,j,i) = flux.e;
    flx(m,ivx,k,j,i) = flux.mx;
    flx(m,ivy,k,j,i) = flux.my;
    flx(m,ivz,k,j,i) = flux.mz;
    ey(m,k,j,i) = flux.by;
    ez(m,k,j,i) = flux.bz;

    // UCT coefficients for SR LLF (Rusanov): alpha_L = alpha_R = lambda_max
    if (compute_uct) {
      Real gam_l = sqrt(1.0 + SQR(wli.vx) + SQR(wli.vy) + SQR(wli.vz));
      Real gam_r = sqrt(1.0 + SQR(wri.vx) + SQR(wri.vy) + SQR(wri.vz));
      Real b_l0 = bxi*wli.vx + wli.by*wli.vy + wli.bz*wli.vz;
      Real b_l1 = (bxi   + b_l0*wli.vx)/gam_l;
      Real b_l2 = (wli.by + b_l0*wli.vy)/gam_l;
      Real b_l3 = (wli.bz + b_l0*wli.vz)/gam_l;
      Real bsq_l = -SQR(b_l0) + SQR(b_l1) + SQR(b_l2) + SQR(b_l3);
      Real b_r0 = bxi*wri.vx + wri.by*wri.vy + wri.bz*wri.vz;
      Real b_r1 = (bxi   + b_r0*wri.vx)/gam_r;
      Real b_r2 = (wri.by + b_r0*wri.vy)/gam_r;
      Real b_r3 = (wri.bz + b_r0*wri.vz)/gam_r;
      Real bsq_r = -SQR(b_r0) + SQR(b_r1) + SQR(b_r2) + SQR(b_r3);
      Real pl = eos.IdealGasPressure(wli.e);
      Real pr = eos.IdealGasPressure(wri.e);
      Real lm_l, lp_l, lm_r, lp_r;
      eos.IdealSRMHDFastSpeeds(wli.d, pl, wli.vx, gam_l, bsq_l, lp_l, lm_l);
      eos.IdealSRMHDFastSpeeds(wri.d, pr, wri.vx, gam_r, bsq_r, lp_r, lm_r);
      Real lmax = fmax(fmax(lp_l, lp_r), -fmin(lm_l, lm_r));
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
#endif // MHD_RSOLVERS_LLF_SRMHD_HPP_
