#ifndef MHD_RSOLVERS_LLF_GRMHD_HPP_
#define MHD_RSOLVERS_LLF_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_grmhd.hpp
//! \brief LLF Riemann solver for general relativistic MHD.

#include "coordinates/cell_locations.hpp"
#include "llf_mhd_singlestate.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void LLF_GR
//! \brief The LLF Riemann solver for GR MHD

KOKKOS_INLINE_FUNCTION
void LLF_GR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez,
     DvceArray4D<Real> uct_aL = {}, DvceArray4D<Real> uct_dL = {},
     DvceArray4D<Real> uct_dR = {}, DvceArray4D<Real> uct_vt1 = {},
     DvceArray4D<Real> uct_vt2 = {}) {
  bool compute_uct = uct_aL.is_allocated();
  // Cyclic permutation of array indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  par_for_inner(member, il, iu, [&](const int i) {
    // Extract position of interface
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v,x2v,x3v;
    if (ivx == IVX) {
      x1v = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else if (ivx == IVY) {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);
    }

    // Extract left/right primitives.  Note 1/2/3 always refers to x1/2/3 dirs
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
    SingleStateLLF_GRMHD(wli, wri, bxi, x1v, x2v, x3v, ivx, coord, eos, flux);

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,ivx,k,j,i) = flux.mx;
    flx(m,ivy,k,j,i) = flux.my;
    flx(m,ivz,k,j,i) = flux.mz;
    flx(m,IEN,k,j,i) = flux.e;
    ey(m,k,j,i) = flux.by;
    ez(m,k,j,i) = flux.bz;

    // UCT coefficients for GR LLF (Rusanov): alpha_L = alpha_R = lambda_max
    if (compute_uct) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,coord.is_minkowski,coord.bh_spin,glower,gupper);
      Real q = glower[ivx][ivx]*SQR(wli.vx) + glower[ivy][ivy]*SQR(wli.vy) +
               glower[ivz][ivz]*SQR(wli.vz) + 2.0*glower[ivx][ivy]*wli.vx*wli.vy +
           2.0*glower[ivx][ivz]*wli.vx*wli.vz + 2.0*glower[ivy][ivz]*wli.vy*wli.vz;
      Real alpha = std::sqrt(-1.0/gupper[0][0]);
      Real gam_l = sqrt(1.0 + q);
      Real uul0 = gam_l / alpha;
      Real uulx = wli.vx - alpha*gam_l*gupper[0][ivx];
      Real ull_ivx = glower[ivx][0]*uul0 + glower[ivx][ivx]*uulx +
                     glower[ivx][ivy]*(wli.vy - alpha*gam_l*gupper[0][ivy]) +
                     glower[ivx][ivz]*(wli.vz - alpha*gam_l*gupper[0][ivz]);
      Real bul0 = ull_ivx*bxi +
                  (glower[ivy][0]*uul0 + glower[ivy][ivx]*uulx +
                   glower[ivy][ivy]*(wli.vy - alpha*gam_l*gupper[0][ivy]) +
                   glower[ivy][ivz]*(wli.vz - alpha*gam_l*gupper[0][ivz]))*wli.by +
                  (glower[ivz][0]*uul0 + glower[ivz][ivx]*uulx +
                   glower[ivz][ivy]*(wli.vy - alpha*gam_l*gupper[0][ivy]) +
                   glower[ivz][ivz]*(wli.vz - alpha*gam_l*gupper[0][ivz]))*wli.bz;
      Real bul1 = (bxi + bul0*uulx)/uul0;
      Real bul2 = (wli.by + bul0*(wli.vy - alpha*gam_l*gupper[0][ivy]))/uul0;
      Real bul3 = (wli.bz + bul0*(wli.vz - alpha*gam_l*gupper[0][ivz]))/uul0;
      // (approximation - just use squared norm)
      Real bsql = SQR(bul1)+SQR(bul2)+SQR(bul3)-SQR(bul0);
      // Simplify: recompute via same GR fast speed function
      Real pl = eos.IdealGasPressure(wli.e);
      Real lp_l, lm_l;
      eos.IdealGRMHDFastSpeeds(wli.d, pl, uul0, uulx, bsql, gupper[0][0],
                                gupper[0][ivx], gupper[ivx][ivx], lp_l, lm_l);

      q = glower[ivx][ivx]*SQR(wri.vx) + glower[ivy][ivy]*SQR(wri.vy) +
          glower[ivz][ivz]*SQR(wri.vz) + 2.0*glower[ivx][ivy]*wri.vx*wri.vy +
          2.0*glower[ivx][ivz]*wri.vx*wri.vz + 2.0*glower[ivy][ivz]*wri.vy*wri.vz;
      Real gam_r = sqrt(1.0 + q);
      Real uur0 = gam_r / alpha;
      Real uurx = wri.vx - alpha*gam_r*gupper[0][ivx];
      Real ulr_ivx = glower[ivx][0]*uur0 + glower[ivx][ivx]*uurx +
                     glower[ivx][ivy]*(wri.vy - alpha*gam_r*gupper[0][ivy]) +
                     glower[ivx][ivz]*(wri.vz - alpha*gam_r*gupper[0][ivz]);
      Real bur0 = ulr_ivx*bxi +
                  (glower[ivy][0]*uur0 + glower[ivy][ivx]*uurx +
                   glower[ivy][ivy]*(wri.vy - alpha*gam_r*gupper[0][ivy]) +
                   glower[ivy][ivz]*(wri.vz - alpha*gam_r*gupper[0][ivz]))*wri.by +
                  (glower[ivz][0]*uur0 + glower[ivz][ivx]*uurx +
                   glower[ivz][ivy]*(wri.vy - alpha*gam_r*gupper[0][ivy]) +
                   glower[ivz][ivz]*(wri.vz - alpha*gam_r*gupper[0][ivz]))*wri.bz;
      Real bur1r = (bxi + bur0*uurx)/uur0;
      Real bur2r = (wri.by + bur0*(wri.vy - alpha*gam_r*gupper[0][ivy]))/uur0;
      Real bur3r = (wri.bz + bur0*(wri.vz - alpha*gam_r*gupper[0][ivz]))/uur0;
      Real bsqr = SQR(bur1r)+SQR(bur2r)+SQR(bur3r)-SQR(bur0);
      Real pr = eos.IdealGasPressure(wri.e);
      Real lp_r, lm_r;
      eos.IdealGRMHDFastSpeeds(wri.d, pr, uur0, uurx, bsqr, gupper[0][0],
                                gupper[0][ivx], gupper[ivx][ivx], lp_r, lm_r);

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
#endif // MHD_RSOLVERS_LLF_GRMHD_HPP_
