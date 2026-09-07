#ifndef MHD_RSOLVERS_LLF_SRMHD_HPP_
#define MHD_RSOLVERS_LLF_SRMHD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srmhd.hpp
//! \brief Local Lax-Friedrichs (LLF) Riemann solver for special relativistic MHD.
//! Per-face implementation for the split-kernel path.

#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn LLF_SR<ivx>()
//! \brief The LLF Riemann solver for SR MHD, single face (m,k,j,i).
template <int ivx>
KOKKOS_INLINE_FUNCTION
void LLF_SR(const EOS_Data &eos,
            const int m, const int k, const int j, const int i,
            const int is, const int js, const int ks,
            const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
            const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
            const DvceArray4D<Real> &bx,
            const DvceArray5D<Real> &flx,
            const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez,
            const int uct_flag = 0,
            const DvceArray4D<Real> &uct_aL = {}, const DvceArray4D<Real> &uct_dL = {},
            const DvceArray4D<Real> &uct_dR = {}, const DvceArray4D<Real> &uct_vt1 = {},
            const DvceArray4D<Real> &uct_vt2 = {}) {
  // NB: gate on the flag, NOT uct_aL.is_allocated().  The UCT arrays are
  // constructed (1,1,1,1) and only realloc'd to full size when emf is
  // uct_hll/uct_hlld, so is_allocated() is TRUE even with ct_contact and
  // every uct_* write would land outside a 1x1x1x1 View (heap corruption).
  const bool compute_uct = (uct_flag > 0);
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;


  // Extract left/right primitives
  MHDPrim1D wli,wri;
  wli.d  = wl(m,IDN,k,j,i);
  wli.vx = wl(m,ivx,k,j,i);
  wli.vy = wl(m,ivy,k,j,i);
  wli.vz = wl(m,ivz,k,j,i);
  wli.by = bl(m,iby,k,j,i);
  wli.bz = bl(m,ibz,k,j,i);

  wri.d  = wr(m,IDN,k,j,i);
  wri.vx = wr(m,ivx,k,j,i);
  wri.vy = wr(m,ivy,k,j,i);
  wri.vz = wr(m,ivz,k,j,i);
  wri.by = br(m,iby,k,j,i);
  wri.bz = br(m,ibz,k,j,i);

  wli.e = wl(m,IEN,k,j,i);
  wri.e = wr(m,IEN,k,j,i);

  // Extract normal magnetic field
  Real bxi = bx(m,k,j,i);

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
    Real b_l1 = (bxi    + b_l0*wli.vx)/gam_l;
    Real b_l2 = (wli.by + b_l0*wli.vy)/gam_l;
    Real b_l3 = (wli.bz + b_l0*wli.vz)/gam_l;
    Real bsq_l = -SQR(b_l0) + SQR(b_l1) + SQR(b_l2) + SQR(b_l3);
    Real b_r0 = bxi*wri.vx + wri.by*wri.vy + wri.bz*wri.vz;
    Real b_r1 = (bxi    + b_r0*wri.vx)/gam_r;
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
    // TRANSPORT VELOCITY, NOT FOUR-VELOCITY.  The UCT composition in mhd_corner_e.cpp
    // forms E = -v x B from these, and the relativistic induction equation
    //     d_t B^i + d_j ( v^j B^i - v^i B^j ) = 0
    // transports with the COORDINATE three-velocity v^i = u^i/u^0 (Mignone & Del Zanna
    // 2021 sec. 5, the RMHD case PLUTO implements).  The primitive w(IVX..IVZ) here is
    // the spatial four-velocity, larger by the Lorentz factor and, in GR, missing the
    // lapse/shift.  Feeding it in unchanged is what made emf=uct_hll drive the whole
    // domain to the density floor; the composition itself was always fine.
    // flat metric: u^0 is exactly gam_l/gam_r, so v^i = u^i/u^0.
    uct_vt1(m,k,j,i) = 0.5*(wli.vy/gam_l + wri.vy/gam_r);
    uct_vt2(m,k,j,i) = 0.5*(wli.vz/gam_l + wri.vz/gam_r);
  }
}
} // namespace mhd
#endif // MHD_RSOLVERS_LLF_SRMHD_HPP_
