//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fofc.cpp
//! \brief Implements functions for first-order flux correction (FOFC) algorithm.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/rsolvers/llf_mhd_singlestate.hpp"
#include "mhd.hpp"

namespace mhd {
namespace {
//----------------------------------------------------------------------------------------
//! \fn WBFaceLLF_MHD<ivx>()
//! \brief Well-balanced first-order LLF flux at face (m,k,j,i) in direction ivx (the face
//! between cells (k,j,i)-1 and (k,j,i) along ivx): donor-cell perturbation states on both
//! sides plus the face equilibrium, minus the LLF flux of the face equilibrium itself,
//! evaluated in registers in the same kernel so it cancels exactly at U' = 0.  Vector
//! components are permuted so vx/by/bz are the face-normal/transverse ones; the normal
//! field is the full face field bxi.  Returns the conserved fluxes and, in by/bz, the
//! face EMFs (same convention as SingleStateLLF_MHD).
template <int ivx>
KOKKOS_INLINE_FUNCTION
void WBFaceLLF_MHD(const EOS_Data &eos, const int m, const int k, const int j, const int i,
                   const DvceArray5D<Real> &wp, const DvceArray5D<Real> &bp,
                   const DvceArray5D<Real> &weqf, const DvceArray5D<Real> &beqf,
                   const Real bxi, MHDCons1D &flux) {
  constexpr int di = (ivx == IVX) ? 1 : 0;
  constexpr int dj = (ivx == IVY) ? 1 : 0;
  constexpr int dk = (ivx == IVZ) ? 1 : 0;
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;
  constexpr int iby = ((ivx - IVX) + 1) % 3;
  constexpr int ibz = ((ivx - IVX) + 2) % 3;
  MHDPrim1D we, wl, wr;
  we.d  = weqf(m,IDN,k,j,i);
  we.vx = weqf(m,ivx,k,j,i);
  we.vy = weqf(m,ivy,k,j,i);
  we.vz = weqf(m,ivz,k,j,i);
  we.by = beqf(m,iby,k,j,i);
  we.bz = beqf(m,ibz,k,j,i);
  wl.d  = wp(m,IDN,k-dk,j-dj,i-di) + we.d;
  wl.vx = wp(m,ivx,k-dk,j-dj,i-di) + we.vx;
  wl.vy = wp(m,ivy,k-dk,j-dj,i-di) + we.vy;
  wl.vz = wp(m,ivz,k-dk,j-dj,i-di) + we.vz;
  wl.by = bp(m,iby,k-dk,j-dj,i-di) + we.by;
  wl.bz = bp(m,ibz,k-dk,j-dj,i-di) + we.bz;
  wr.d  = wp(m,IDN,k,j,i) + we.d;
  wr.vx = wp(m,ivx,k,j,i) + we.vx;
  wr.vy = wp(m,ivy,k,j,i) + we.vy;
  wr.vz = wp(m,ivz,k,j,i) + we.vz;
  wr.by = bp(m,iby,k,j,i) + we.by;
  wr.bz = bp(m,ibz,k,j,i) + we.bz;
  if (eos.is_ideal) {
    we.e = weqf(m,IEN,k,j,i);
    wl.e = wp(m,IEN,k-dk,j-dj,i-di) + we.e;
    wr.e = wp(m,IEN,k,j,i) + we.e;
  }
  MHDCons1D f, feq;
  SingleStateLLF_MHD(wl, wr, bxi, eos, f);
  SingleStateLLF_MHD(we, we, bxi, eos, feq);
  flux.d  = f.d  - feq.d;
  flux.mx = f.mx - feq.mx;
  flux.my = f.my - feq.my;
  flux.mz = f.mz - feq.mz;
  if (eos.is_ideal) {flux.e = f.e - feq.e;}
  flux.by = f.by - feq.by;
  flux.bz = f.bz - feq.bz;
}
}  // namespace
}  // namespace mhd

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::FOFC
//! \brief Implements first-order flux-correction (FOFC) algorithm for MHD.  First an
//! estimate of the updated conserved variables is made. This estimate is then used to
//! flag any cell where floors will be required during the conversion to primitives. Then
//! the fluxes on the faces of flagged cells are replaced with first-order LLF fluxes.
//! Often this is enough to prevent floors from being needed.  The FOFC infrastructure is
//! also exploited for BH excision.  If a cell is about the horizon, FOFC is automatically
//! triggered (without estimating updated conserved variables).

void MHD::FOFC(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;

  auto &bcc0_ = bcc0;
  auto &e3x1_ = e3x1;
  auto &e2x1_ = e2x1;
  auto &e1x2_ = e1x2;
  auto &e3x2_ = e3x2;
  auto &e2x3_ = e2x3;
  auto &e1x3_ = e1x3;

  if (use_fofc) {
    Real &gam0 = pdriver->gam0[stage-1];
    Real &gam1 = pdriver->gam1[stage-1];
    Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

    int &nmhd_ = nmhd;
    auto &u0_ = u0;
    auto &u1_ = u1;
    auto &utest_ = utest;
    auto &bcctest_ = bcctest;
    auto &b1_ = b1;
    // well-balanced scheme: the candidate B in perturbation form (mhd_mood.cpp)
    const bool wb = well_balanced;
    auto &bcceq_ = bcc_eq;
    auto beq1 = b_eq.x1f;  auto beq2 = b_eq.x2f;  auto beq3 = b_eq.x3f;

    // Index bounds
    int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-1, ju = je+1; }
    if (three_d) { kl = ks-1, ku = ke+1; }

    // Estimate updated conserved variables and cell-centered fields
    par_for("FOFC-newu", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size.d_view(m).dx1;
      Real dtodx2 = beta_dt/size.d_view(m).dx2;
      Real dtodx3 = beta_dt/size.d_view(m).dx3;

      // Estimate conserved variables
      for (int n=0; n<nmhd_; ++n) {
        Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
        if (multi_d) {
          divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
        }
        if (three_d) {
          divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        }
        utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
      }

      // Estimate updated cell-centered fields
      Real b1old = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      Real b2old = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      Real b3old = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));

      if (wb) {
        b1old -= 0.5*(beq1(m,k,j,i) + beq1(m,k,j,i+1));
        b2old -= 0.5*(beq2(m,k,j,i) + beq2(m,k,j+1,i));
        b3old -= 0.5*(beq3(m,k,j,i) + beq3(m,k+1,j,i));
        bcctest_(m,IBX,k,j,i) = bcceq_(m,IBX,k,j,i)
            + gam0*(bcc0_(m,IBX,k,j,i) - bcceq_(m,IBX,k,j,i)) + gam1*b1old;
        bcctest_(m,IBY,k,j,i) = bcceq_(m,IBY,k,j,i)
            + gam0*(bcc0_(m,IBY,k,j,i) - bcceq_(m,IBY,k,j,i)) + gam1*b2old;
        bcctest_(m,IBZ,k,j,i) = bcceq_(m,IBZ,k,j,i)
            + gam0*(bcc0_(m,IBZ,k,j,i) - bcceq_(m,IBZ,k,j,i)) + gam1*b3old;
      } else {
        bcctest_(m,IBX,k,j,i) = gam0*bcc0_(m,IBX,k,j,i) + gam1*b1old;
        bcctest_(m,IBY,k,j,i) = gam0*bcc0_(m,IBY,k,j,i) + gam1*b2old;
        bcctest_(m,IBZ,k,j,i) = gam0*bcc0_(m,IBZ,k,j,i) + gam1*b3old;
      }

      bcctest_(m,IBY,k,j,i) += dtodx1*(e3x1_(m,k,j,i+1) - e3x1_(m,k,j,i));
      bcctest_(m,IBZ,k,j,i) -= dtodx1*(e2x1_(m,k,j,i+1) - e2x1_(m,k,j,i));
      if (multi_d) {
        bcctest_(m,IBX,k,j,i) -= dtodx2*(e3x2_(m,k,j+1,i) - e3x2_(m,k,j,i));
        bcctest_(m,IBZ,k,j,i) += dtodx2*(e1x2_(m,k,j+1,i) - e1x2_(m,k,j,i));
      }
      if (three_d) {
        bcctest_(m,IBX,k,j,i) += dtodx3*(e2x3_(m,k+1,j,i) - e2x3_(m,k,j,i));
        bcctest_(m,IBY,k,j,i) -= dtodx3*(e1x3_(m,k+1,j,i) - e1x3_(m,k,j,i));
      }
    });

    // Test whether conversion to primitives requires floors
    // Note b0 and w0 passed to function, but not used/changed.
    if (wb) {
      peos->ConsToPrimWB(utest_, u_eq, b0, w0, bcctest_, true, il, iu, jl, ju, kl, ku);
    } else {
      peos->ConsToPrim(utest_, b0, w0, bcctest_, true, il, iu, jl, ju, kl, ku);
    }
  }

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = peos->eos_data;
  auto &use_fofc_ = use_fofc;
  auto fofc_ = fofc;
  auto &use_excise_ = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &w0_ = w0;
  auto &b0_ = b0;
  // well-balanced scheme: replacement fluxes through the face equilibrium (WBFaceLLF_MHD)
  const bool wb_ = well_balanced;
  auto wp_ = w_pert;  auto bp_ = bcc_pert;
  auto weqf1 = w_eq_face.x1f;  auto weqf2 = w_eq_face.x2f;  auto weqf3 = w_eq_face.x3f;
  auto beqf1 = bcc_eq_face.x1f;  auto beqf2 = bcc_eq_face.x2f;
  auto beqf3 = bcc_eq_face.x3f;

  // Index bounds
  int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-1, ju = je+1; }
  if (three_d) { kl = ks-1, ku = ke+1; }

  // Replace fluxes with first-order LLF fluxes at i,j,k faces for any cell where FOFC
  // and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }
    }

    // Apply FOFC
    if (fofc_flag || fofc_excision) {
      // load W_{i-1} state
      MHDPrim1D wim1;
      wim1.d  = w0_(m,IDN,k,j,i-1);
      wim1.vx = w0_(m,IVX,k,j,i-1);
      wim1.vy = w0_(m,IVY,k,j,i-1);
      wim1.vz = w0_(m,IVZ,k,j,i-1);
      if (eos.is_ideal) {wim1.e  = w0_(m,IEN,k,j,i-1);}
      wim1.by = bcc0_(m,IBY,k,j,i-1);
      wim1.bz = bcc0_(m,IBZ,k,j,i-1);

      // load W_{i} state
      MHDPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.is_ideal) {wi.e = w0_(m,IEN,k,j,i);}
      wi.by = bcc0_(m,IBY,k,j,i);
      wi.bz = bcc0_(m,IBZ,k,j,i);

      // compute new 1st-order LLF flux at i-face
      {
        Real bxi = b0_.x1f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = LeftEdgeX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wim1, wi, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wim1, wi, bxi, eos, flux);
        } else if (wb_) {
          WBFaceLLF_MHD<IVX>(eos, m, k, j, i, wp_, bp_, weqf1, beqf1, bxi, flux);
        } else {
          SingleStateLLF_MHD(wim1, wi, bxi, eos, flux);
        }

        // store 1st-order fluxes.
        flx1(m,IDN,k,j,i) = flux.d;
        flx1(m,IM1,k,j,i) = flux.mx;
        flx1(m,IM2,k,j,i) = flux.my;
        flx1(m,IM3,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx1(m,IEN,k,j,i) = flux.e;}
        e3x1_(m,k,j,i) = flux.by;
        e2x1_(m,k,j,i) = flux.bz;
      }

      if (multi_d) {
        // load W_{j-1} state, permutting components of vectors
        MHDPrim1D wjm1;
        wjm1.d  = w0_(m,IDN,k,j-1,i);
        wjm1.vx = w0_(m,IVY,k,j-1,i);
        wjm1.vy = w0_(m,IVZ,k,j-1,i);
        wjm1.vz = w0_(m,IVX,k,j-1,i);
        if (eos.is_ideal) {wjm1.e = w0_(m,IEN,k,j-1,i);}
        wjm1.by = bcc0_(m,IBZ,k,j-1,i);
        wjm1.bz = bcc0_(m,IBX,k,j-1,i);

        // load W_{j} state, permutting components of vectors
        MHDPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.is_ideal) {wj.e = w0_(m,IEN,k,j,i);}
        wj.by = bcc0_(m,IBZ,k,j,i);
        wj.bz = bcc0_(m,IBX,k,j,i);

        // compute new first-order flux at j-face
        Real bxi = b0_.x2f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wjm1, wj, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wjm1, wj, bxi, eos, flux);
        } else if (wb_) {
          WBFaceLLF_MHD<IVY>(eos, m, k, j, i, wp_, bp_, weqf2, beqf2, bxi, flux);
        } else {
          SingleStateLLF_MHD(wjm1, wj, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx2(m,IDN,k,j,i) = flux.d;
        flx2(m,IM2,k,j,i) = flux.mx;
        flx2(m,IM3,k,j,i) = flux.my;
        flx2(m,IM1,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j,i) = flux.e;}
        e1x2_(m,k,j,i) = flux.by;
        e3x2_(m,k,j,i) = flux.bz;
      }

      if (three_d) {
        // load W_{k-1} state, permutting components of vectors
        MHDPrim1D wkm1;
        wkm1.d  = w0_(m,IDN,k-1,j,i);
        wkm1.vx = w0_(m,IVZ,k-1,j,i);
        wkm1.vy = w0_(m,IVX,k-1,j,i);
        wkm1.vz = w0_(m,IVY,k-1,j,i);
        if (eos.is_ideal) {wkm1.e = w0_(m,IEN,k-1,j,i);}
        wkm1.by = bcc0_(m,IBX,k-1,j,i);
        wkm1.bz = bcc0_(m,IBY,k-1,j,i);

        // load W_{k} state, permutting components of vectors
        MHDPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.is_ideal) {wk.e = w0_(m,IEN,k,j,i);}
        wk.by = bcc0_(m,IBX,k,j,i);
        wk.bz = bcc0_(m,IBY,k,j,i);

        // compute new first-order flux at k-face
        Real bxi = b0_.x3f(m,k,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wkm1, wk, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wkm1, wk, bxi, eos, flux);
        } else if (wb_) {
          WBFaceLLF_MHD<IVZ>(eos, m, k, j, i, wp_, bp_, weqf3, beqf3, bxi, flux);
        } else {
          SingleStateLLF_MHD(wkm1, wk, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx3(m,IDN,k,j,i) = flux.d;
        flx3(m,IM3,k,j,i) = flux.mx;
        flx3(m,IM1,k,j,i) = flux.my;
        flx3(m,IM2,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k,j,i) = flux.e;}
        e2x3_(m,k,j,i) = flux.by;
        e1x3_(m,k,j,i) = flux.bz;
      }
    }
  });

  // Replace fluxes with first-order LLF fluxes at i+1,j+1,k+1 faces for any cell where
  // FOFC and/or excision is used (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise_) { fofc_excision = excision_flux_(m,k,j,i); }
    }

    // Apply FOFC
    if (fofc_flag || fofc_excision) {
      // load W_{i} state
      MHDPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.is_ideal) {wi.e = w0_(m,IEN,k,j,i);}
      wi.by = bcc0_(m,IBY,k,j,i);
      wi.bz = bcc0_(m,IBZ,k,j,i);

      // load W_{i+1} state
      MHDPrim1D wip1;
      wip1.d  = w0_(m,IDN,k,j,i+1);
      wip1.vx = w0_(m,IVX,k,j,i+1);
      wip1.vy = w0_(m,IVY,k,j,i+1);
      wip1.vz = w0_(m,IVZ,k,j,i+1);
      if (eos.is_ideal) {wip1.e = w0_(m,IEN,k,j,i+1);}
      wip1.by = bcc0_(m,IBY,k,j,i+1);
      wip1.bz = bcc0_(m,IBZ,k,j,i+1);

      // compute new 1st-order LLF flux at (i+1)-face
      {
        Real bxi = b0_.x1f(m,k,j,i+1);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = LeftEdgeX(i+1-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wi, wip1, bxi, x1v, x2v, x3v, IVX, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wi, wip1, bxi, eos, flux);
        } else if (wb_) {
          WBFaceLLF_MHD<IVX>(eos, m, k, j, i+1, wp_, bp_, weqf1, beqf1, bxi, flux);
        } else {
          SingleStateLLF_MHD(wi, wip1, bxi, eos, flux);
        }

        // store 1st-order fluxes.
        flx1(m,IDN,k,j,i+1) = flux.d;
        flx1(m,IM1,k,j,i+1) = flux.mx;
        flx1(m,IM2,k,j,i+1) = flux.my;
        flx1(m,IM3,k,j,i+1) = flux.mz;
        if (eos.is_ideal) {flx1(m,IEN,k,j,i+1) = flux.e;}
        e3x1_(m,k,j,i+1) = flux.by;
        e2x1_(m,k,j,i+1) = flux.bz;
      }

      if (multi_d) {
        // load W_{j} state, permutting components of vectors
        MHDPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.is_ideal) {wj.e = w0_(m,IEN,k,j,i);}
        wj.by = bcc0_(m,IBZ,k,j,i);
        wj.bz = bcc0_(m,IBX,k,j,i);

        // load W_{j+1} state, permutting components of vectors
        MHDPrim1D wjp1;
        wjp1.d  = w0_(m,IDN,k,j+1,i);
        wjp1.vx = w0_(m,IVY,k,j+1,i);
        wjp1.vy = w0_(m,IVZ,k,j+1,i);
        wjp1.vz = w0_(m,IVX,k,j+1,i);
        if (eos.is_ideal) {wjp1.e = w0_(m,IEN,k,j+1,i);}
        wjp1.by = bcc0_(m,IBZ,k,j+1,i);
        wjp1.bz = bcc0_(m,IBX,k,j+1,i);

        // compute new first-order flux at (j+1)-face
        Real bxi = b0_.x2f(m,k,j+1,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j+1-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wj, wjp1, bxi, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wj, wjp1, bxi, eos, flux);
        } else if (wb_) {
          WBFaceLLF_MHD<IVY>(eos, m, k, j+1, i, wp_, bp_, weqf2, beqf2, bxi, flux);
        } else {
          SingleStateLLF_MHD(wj, wjp1, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx2(m,IDN,k,j+1,i) = flux.d;
        flx2(m,IM2,k,j+1,i) = flux.mx;
        flx2(m,IM3,k,j+1,i) = flux.my;
        flx2(m,IM1,k,j+1,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j+1,i) = flux.e;}
        e1x2_(m,k,j+1,i) = flux.by;
        e3x2_(m,k,j+1,i) = flux.bz;
      }

      if (three_d) {
        // load W_{k} state, permutting components of vectors
        MHDPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.is_ideal) {wk.e = w0_(m,IEN,k,j,i);}
        wk.by = bcc0_(m,IBX,k,j,i);
        wk.bz = bcc0_(m,IBY,k,j,i);

        // load W_{k+1} state, permutting components of vectors
        MHDPrim1D wkp1;
        wkp1.d  = w0_(m,IDN,k+1,j,i);
        wkp1.vx = w0_(m,IVZ,k+1,j,i);
        wkp1.vy = w0_(m,IVX,k+1,j,i);
        wkp1.vz = w0_(m,IVY,k+1,j,i);
        if (eos.is_ideal) {wkp1.e = w0_(m,IEN,k+1,j,i);}
        wkp1.by = bcc0_(m,IBX,k+1,j,i);
        wkp1.bz = bcc0_(m,IBY,k+1,j,i);

        // compute new first-order flux at (k+1)-face
        Real bxi = b0_.x3f(m,k+1,j,i);
        MHDCons1D flux;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
          SingleStateLLF_GRMHD(wk, wkp1, bxi, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRMHD(wk, wkp1, bxi, eos, flux);
        } else if (wb_) {
          WBFaceLLF_MHD<IVZ>(eos, m, k+1, j, i, wp_, bp_, weqf3, beqf3, bxi, flux);
        } else {
          SingleStateLLF_MHD(wk, wkp1, bxi, eos, flux);
        }

        // store 1st-order fluxes, permutting indices.
        flx3(m,IDN,k+1,j,i) = flux.d;
        flx3(m,IM3,k+1,j,i) = flux.mx;
        flx3(m,IM1,k+1,j,i) = flux.my;
        flx3(m,IM2,k+1,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k+1,j,i) = flux.e;}
        e2x3_(m,k+1,j,i) = flux.by;
        e1x3_(m,k+1,j,i) = flux.bz;
      }
    }
  });

  // reset FOFC flag (do not reset excision flag)
  if (use_fofc_) {
    Kokkos::deep_copy(fofc, false);
  }

  return;
}

} // namespace mhd
