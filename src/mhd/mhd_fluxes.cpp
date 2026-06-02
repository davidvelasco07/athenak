//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes.cpp
//! \brief Calculate fluxes of the conserved variables, and area-averaged electric fields
//! E = - (v X B) on cell faces for mhd.  Fluxes are stored in face-centered vector
//! 'uflx', while electric fields are stored in individual arrays: e2x1,e3x1 on x1-faces;
//! e1x2,e3x2 on x2-faces; e1x3,e2x3 on x3-faces.

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "mhd/rsolvers/llf_mhd.hpp"
#include "mhd/rsolvers/hlle_mhd.hpp"
#include "mhd/rsolvers/hlld_mhd.hpp"
#include "mhd/rsolvers/llf_srmhd.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"
#include "mhd/rsolvers/llf_grmhd.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"
// #include "mhd/rsolvers/roe_mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::CalculateFlux
//! \brief Calculate fluxes of conserved variables, and face-centered area-averaged EMFs
//! for evolution of magnetic field
//! Note this function is templated over RS for better performance on GPUs.
//!
//! All three directions use a tiled team decomposition (see hydro_fluxes.cpp for the
//! rationale).  The i-tile bounds the per-team scratch width; the x2/x3 kernels also tile
//! the serial transverse walk to lift the league size.  The scratch-column offset i0 maps
//! the global index i onto the narrow tile-local buffer for both the hydro state (wl/wr)
//! and the cell-centered field (bl/br); the longitudinal field bx and the output flux/EMF
//! arrays are always indexed by the global i.

template <MHD_RSolver rsolver_method_>
void MHD::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  int &nmhd_ = nmhd;
  int nvars = nmhd + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  bool extrema = false;
  if (recon_method == ReconstructionMethod::ppmx) {
    extrema = true;
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto &b0_ = bcc0;

  int scr_level = 0;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &flx1_ = uflx.x1f;
  auto &e31_ = e3x1;
  auto &e21_ = e2x1;
  auto &bx_ = b0.x1f;

  // set the loop limits for 1D/2D/3D problems
  int jl,ju,kl,ku;
  if (pmy_pack->pmesh->one_d) {
    jl = js, ju = je, kl = ks, ku = ke;
  } else if (pmy_pack->pmesh->two_d) {
    jl = js-1, ju = je+1, kl = ks, ku = ke;
  } else {
    jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  }
  int il = is, iu = ie+1;
  if (use_fofc) { il = is-1, iu = ie+2; }

  {
    constexpr int TILE_NX1 = 64;                     // i-tile width (bounds scratch)
    const int nflux  = iu - il + 1;                  // # interfaces in flux range [il,iu]
    const int ntiles = (nflux + TILE_NX1 - 1)/TILE_NX1;
    // Scratch spans reconstructed states over [il_t-1, iu_t]; worst-case width TILE_NX1+2.
    size_t scr_size_t = (ScrArray2D<Real>::shmem_size(nvars, TILE_NX1 + 2) +
                         ScrArray2D<Real>::shmem_size(3, TILE_NX1 + 2)) * 2;

    par_for_outer("mhd_flux1",DevExeSpace(), scr_size_t, scr_level,
                  0, nmb1, kl, ku, jl, ju, 0, ntiles-1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                  const int t) {
      const int il_t = il + t*TILE_NX1;
      const int iu_t = (il_t + TILE_NX1 - 1 < iu) ? (il_t + TILE_NX1 - 1) : iu;
      const int i0   = il_t - 1;                     // scratch column = (global i) - i0

      ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, TILE_NX1 + 2);
      ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, TILE_NX1 + 2);
      ScrArray2D<Real> bl(member.team_scratch(scr_level), 3, TILE_NX1 + 2);
      ScrArray2D<Real> br(member.team_scratch(scr_level), 3, TILE_NX1 + 2);

      // Reconstruct qR[i] and qL[i+1], for both W and Bcc
      switch (recon_method_) {
        case ReconstructionMethod::dc:
          DonorCellX1(member, m, k, j, il_t-1, iu_t, w0_, wl, wr, i0);
          DonorCellX1(member, m, k, j, il_t-1, iu_t, b0_, bl, br, i0);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinearX1(member, m, k, j, il_t-1, iu_t, w0_, wl, wr, i0);
          PiecewiseLinearX1(member, m, k, j, il_t-1, iu_t, b0_, bl, br, i0);
          break;
        case ReconstructionMethod::ppm4:
        case ReconstructionMethod::ppmx:
          PiecewiseParabolicX1(member,eos_,extrema,true,  m, k, j, il_t-1, iu_t,
                               w0_, wl, wr, i0);
          PiecewiseParabolicX1(member,eos_,extrema,false, m, k, j, il_t-1, iu_t,
                               b0_, bl, br, i0);
          break;
        case ReconstructionMethod::wenoz:
          WENOZX1(member, eos_, true,  m, k, j, il_t-1, iu_t, w0_, wl, wr, i0);
          WENOZX1(member, eos_, false, m, k, j, il_t-1, iu_t, b0_, bl, br, i0);
          break;
        default:
          break;
      }
      member.team_barrier();

      // compute fluxes over [il_t,iu_t].  MHD RS also computes electric fields, where
      // (IBY) component of flx = E_{z} = -(v x B)_{z} = -(v1*b2 - v2*b1)
      // (IBZ) component of flx = E_{y} = -(v x B)_{y} =  (v1*b3 - v3*b1)
      // NOTE(@pdmullen): Capture variables prior to if constexpr.  Required for cuda 11.6+.
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      auto bx = bx_;
      auto flx1 = flx1_;
      auto e31 = e31_;
      auto e21 = e21_;
      if constexpr (rsolver_method_ == MHD_RSolver::advect) {
        Advect(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
        LLF(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
        HLLE(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
        HLLD(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
        LLF_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
        HLLE_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
        LLF_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
        HLLE_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,bl,br,bx,flx1,e31,e21,i0);
      }
      member.team_barrier();

      // calculate fluxes of scalars (if any), restricted to physical range [is,ie+1]
      if (nvars > nmhd_) {
        const int sil = (il_t > is)   ? il_t : is;
        const int siu = (iu_t < ie+1) ? iu_t : ie+1;
        for (int n=nmhd_; n<nvars; ++n) {
          par_for_inner(member, sil, siu, [&](const int i) {
            if (flx1_(m,IDN,k,j,i) >= 0.0) {
              flx1_(m,n,k,j,i) = flx1_(m,IDN,k,j,i)*wl(n,i-i0);
            } else {
              flx1_(m,n,k,j,i) = flx1_(m,IDN,k,j,i)*wr(n,i-i0);
            }
          });
        }
      }
    });
  }

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto &flx2_ = uflx.x2f;
    auto &by_ = b0.x2f;
    auto &e12_ = e1x2;
    auto &e32_ = e3x2;

    // set the loop limits for 2D/3D problems
    if (pmy_pack->pmesh->two_d) {
      kl = ks, ku = ke;
    } else { // 3D
      kl = ks-1, ku = ke+1;
    }
    jl = js-1, ju = je+1;
    if (use_fofc) { jl = js-2, ju = je+2; }
    // MHD computes the x2 fluxes/EMFs over the i-range [is-1, ie+1].
    const int ilo = is-1, ihi = ie+1;

    {
      constexpr int TILE_NX1 = 64;                   // i-tile width (bounds scratch)
      constexpr int TILE_NX2 = 16;                   // j-tile height (de-serializes walk)
      const int nif      = ihi - ilo + 1;            // # i-cells in flux range [ilo,ihi]
      const int ntiles_i = (nif + TILE_NX1 - 1)/TILE_NX1;
      const int njf      = ju - jl;                  // # j-faces in flux range [jl+1,ju]
      const int ntiles_j = (njf + TILE_NX2 - 1)/TILE_NX2;
      // 3 rolling buffers each for W and Bcc, each only TILE_NX1 wide (was ncells1).
      size_t scr_size_t = (ScrArray2D<Real>::shmem_size(nvars, TILE_NX1) +
                           ScrArray2D<Real>::shmem_size(3, TILE_NX1)) * 3;

      par_for_outer("mhd_flux2",DevExeSpace(), scr_size_t, scr_level,
                    0, nmb1, kl, ku, 0, ntiles_i-1, 0, ntiles_j-1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k,
                    const int ti, const int tj) {
        const int il_t = ilo + ti*TILE_NX1;
        const int iu_t = (il_t + TILE_NX1 - 1 < ihi) ? (il_t + TILE_NX1 - 1) : ihi;
        const int i0   = il_t;                       // scratch column = (global i) - i0
        const int jb_lo = (jl+1) + tj*TILE_NX2;      // first j-face owned by this tile
        const int jb_hi = (jb_lo + TILE_NX2 - 1 < ju) ? (jb_lo + TILE_NX2 - 1) : ju;

        ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr4(member.team_scratch(scr_level), 3, TILE_NX1);
        ScrArray2D<Real> scr5(member.team_scratch(scr_level), 3, TILE_NX1);
        ScrArray2D<Real> scr6(member.team_scratch(scr_level), 3, TILE_NX1);

        // Serial walk over this tile's j-slabs; j = jb_lo-1 is the warmup slab (no flux).
        for (int j=jb_lo-1; j<=jb_hi; ++j) {
          // Permute scratch arrays (parity of the global j keeps the rolling buffer valid).
          auto wl     = scr1;
          auto wl_jp1 = scr2;
          auto wr     = scr3;
          auto bl     = scr4;
          auto bl_jp1 = scr5;
          auto br     = scr6;
          if ((j%2) == 0) {
            wl     = scr2;
            wl_jp1 = scr1;
            bl     = scr5;
            bl_jp1 = scr4;
          }

          // Reconstruct qR[j] and qL[j+1], for both W and Bcc, over the i-tile
          switch (recon_method_) {
            case ReconstructionMethod::dc:
              DonorCellX2(member, m, k, j, il_t, iu_t, w0_, wl_jp1, wr, i0);
              DonorCellX2(member, m, k, j, il_t, iu_t, b0_, bl_jp1, br, i0);
              break;
            case ReconstructionMethod::plm:
              PiecewiseLinearX2(member, m, k, j, il_t, iu_t, w0_, wl_jp1, wr, i0);
              PiecewiseLinearX2(member, m, k, j, il_t, iu_t, b0_, bl_jp1, br, i0);
              break;
            case ReconstructionMethod::ppm4:
            case ReconstructionMethod::ppmx:
              PiecewiseParabolicX2(member,eos_,extrema,true, m,k,j,il_t,iu_t,
                                   w0_,wl_jp1,wr,i0);
              PiecewiseParabolicX2(member,eos_,extrema,false,m,k,j,il_t,iu_t,
                                   b0_,bl_jp1,br,i0);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX2(member, eos_, true,  m, k, j, il_t, iu_t, w0_, wl_jp1, wr, i0);
              WENOZX2(member, eos_, false, m, k, j, il_t, iu_t, b0_, bl_jp1, br, i0);
              break;
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [js,je+1].  MHD RS also computes electric fields, where
          // (IBY) component of flx = E_{x} = -(v x B)_{x} = -(v2*b3 - v3*b2)
          // (IBZ) component of flx = E_{z} = -(v x B)_{z} =  (v2*b1 - v1*b2)
          if (j >= jb_lo) {
            // NOTE(@pdmullen): Capture variables prior to if constexpr.
            auto eos = eos_;
            auto indcs = indcs_;
            auto size = size_;
            auto coord = coord_;
            auto by = by_;
            auto flx2 = flx2_;
            auto e12 = e12_;
            auto e32 = e32_;
            if constexpr (rsolver_method_ == MHD_RSolver::advect) {
              Advect(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
              LLF(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
              HLLE(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
              HLLD(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
              LLF_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
              HLLE_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
              LLF_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
              HLLE_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,bl,br,by,flx2,e12,e32,i0);
            }
            member.team_barrier();

            // calculate fluxes of scalars (if any), restricted to physical range [is,ie]
            if (nvars > nmhd_) {
              const int sil = (il_t > is) ? il_t : is;
              const int siu = (iu_t < ie) ? iu_t : ie;
              for (int n=nmhd_; n<nvars; ++n) {
                par_for_inner(member, sil, siu, [&](const int i) {
                  if (flx2_(m,IDN,k,j,i) >= 0.0) {
                    flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wl(n,i-i0);
                  } else {
                    flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wr(n,i-i0);
                  }
                });
              }
            }
          }
        } // end serial j-walk within tile
      });
    }
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto &flx3_ = uflx.x3f;
    auto &bz_ = b0.x3f;
    auto &e23_ = e2x3;
    auto &e13_ = e1x3;

    // set the loop limits
    kl = ks-1, ku = ke+1;
    if (use_fofc) { kl = ks-2, ku = ke+2; }
    jl = js-1, ju = je+1;
    // MHD computes the x3 fluxes/EMFs over the i-range [is-1, ie+1].
    const int ilo = is-1, ihi = ie+1;

    {
      constexpr int TILE_NX1 = 64;                   // i-tile width (bounds scratch)
      constexpr int TILE_NX3 = 16;                   // k-tile depth (de-serializes walk)
      const int nif      = ihi - ilo + 1;            // # i-cells in flux range [ilo,ihi]
      const int ntiles_i = (nif + TILE_NX1 - 1)/TILE_NX1;
      const int nkf      = ku - kl;                  // # k-faces in flux range [kl+1,ku]
      const int ntiles_k = (nkf + TILE_NX3 - 1)/TILE_NX3;
      // 3 rolling buffers each for W and Bcc, each only TILE_NX1 wide (was ncells1).
      size_t scr_size_t = (ScrArray2D<Real>::shmem_size(nvars, TILE_NX1) +
                           ScrArray2D<Real>::shmem_size(3, TILE_NX1)) * 3;

      par_for_outer("mhd_flux3",DevExeSpace(), scr_size_t, scr_level,
                    0, nmb1, jl, ju, 0, ntiles_i-1, 0, ntiles_k-1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j,
                    const int ti, const int tk) {
        const int il_t = ilo + ti*TILE_NX1;
        const int iu_t = (il_t + TILE_NX1 - 1 < ihi) ? (il_t + TILE_NX1 - 1) : ihi;
        const int i0   = il_t;                       // scratch column = (global i) - i0
        const int kb_lo = (kl+1) + tk*TILE_NX3;      // first k-face owned by this tile
        const int kb_hi = (kb_lo + TILE_NX3 - 1 < ku) ? (kb_lo + TILE_NX3 - 1) : ku;

        ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr4(member.team_scratch(scr_level), 3, TILE_NX1);
        ScrArray2D<Real> scr5(member.team_scratch(scr_level), 3, TILE_NX1);
        ScrArray2D<Real> scr6(member.team_scratch(scr_level), 3, TILE_NX1);

        // Serial walk over this tile's k-slabs; k = kb_lo-1 is the warmup slab (no flux).
        for (int k=kb_lo-1; k<=kb_hi; ++k) {
          // Permute scratch arrays (parity of the global k keeps the rolling buffer valid).
          auto wl     = scr1;
          auto wl_kp1 = scr2;
          auto wr     = scr3;
          auto bl     = scr4;
          auto bl_kp1 = scr5;
          auto br     = scr6;
          if ((k%2) == 0) {
            wl     = scr2;
            wl_kp1 = scr1;
            bl     = scr5;
            bl_kp1 = scr4;
          }

          // Reconstruct qR[k] and qL[k+1], for both W and Bcc, over the i-tile
          switch (recon_method_) {
            case ReconstructionMethod::dc:
              DonorCellX3(member, m, k, j, il_t, iu_t, w0_, wl_kp1, wr, i0);
              DonorCellX3(member, m, k, j, il_t, iu_t, b0_, bl_kp1, br, i0);
              break;
            case ReconstructionMethod::plm:
              PiecewiseLinearX3(member, m, k, j, il_t, iu_t, w0_, wl_kp1, wr, i0);
              PiecewiseLinearX3(member, m, k, j, il_t, iu_t, b0_, bl_kp1, br, i0);
              break;
            case ReconstructionMethod::ppm4:
            case ReconstructionMethod::ppmx:
              PiecewiseParabolicX3(member,eos_,extrema,true, m,k,j,il_t,iu_t,
                                   w0_,wl_kp1,wr,i0);
              PiecewiseParabolicX3(member,eos_,extrema,false,m,k,j,il_t,iu_t,
                                   b0_,bl_kp1,br,i0);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX3(member, eos_, true,  m, k, j, il_t, iu_t, w0_, wl_kp1, wr, i0);
              WENOZX3(member, eos_, false, m, k, j, il_t, iu_t, b0_, bl_kp1, br, i0);
              break;
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [ks,ke+1].  MHD RS also computes electric fields, where
          // (IBY) component of flx = E_{y} = -(v x B)_{y} = -(v3*b1 - v1*b3)
          // (IBZ) component of flx = E_{x} = -(v x B)_{x} =  (v3*b2 - v2*b3)
          if (k >= kb_lo) {
            // NOTE(@pdmullen): Capture variables prior to if constexpr.
            auto eos = eos_;
            auto indcs = indcs_;
            auto size = size_;
            auto coord = coord_;
            auto bz = bz_;
            auto flx3 = flx3_;
            auto e23 = e23_;
            auto e13 = e13_;
            if constexpr (rsolver_method_ == MHD_RSolver::advect) {
              Advect(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
              LLF(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
              HLLE(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
              HLLD(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
              LLF_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
              HLLE_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
              LLF_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
              HLLE_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,bl,br,bz,flx3,e23,e13,i0);
            }
            member.team_barrier();

            // calculate fluxes of scalars (if any), restricted to physical range [is,ie]
            if (nvars > nmhd_) {
              const int sil = (il_t > is) ? il_t : is;
              const int siu = (iu_t < ie) ? iu_t : ie;
              for (int n=nmhd_; n<nvars; ++n) {
                par_for_inner(member, sil, siu, [&](const int i) {
                  if (flx3_(m,IDN,k,j,i) >= 0.0) {
                    flx3_(m,n,k,j,i) = flx3_(m,IDN,k,j,i)*wl(n,i-i0);
                  } else {
                    flx3_(m,n,k,j,i) = flx3_(m,IDN,k,j,i)*wr(n,i-i0);
                  }
                });
              }
            }
          }
        } // end serial k-walk within tile
      });
    }
  }

  return;
}

// function definitions for each template parameter
template void MHD::CalculateFluxes<MHD_RSolver::advect>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlld>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_gr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace mhd
