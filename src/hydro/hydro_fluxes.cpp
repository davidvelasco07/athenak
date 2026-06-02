//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//! \brief Calculate 3D fluxes for hydro

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "hydro/rsolvers/advect_hyd.hpp"
#include "hydro/rsolvers/llf_hyd.hpp"
#include "hydro/rsolvers/hlle_hyd.hpp"
#include "hydro/rsolvers/hllc_hyd.hpp"
#include "hydro/rsolvers/roe_hyd.hpp"
#include "hydro/rsolvers/llf_srhyd.hpp"
#include "hydro/rsolvers/hlle_srhyd.hpp"
#include "hydro/rsolvers/hllc_srhyd.hpp"
#include "hydro/rsolvers/llf_grhyd.hpp"
#include "hydro/rsolvers/hlle_grhyd.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void Hydro::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes
//! Note this function is templated over RS for better performance on GPUs.
//!
//! All three directions use a tiled team decomposition.  Tiling along x1 decouples the
//! per-team scratch-array width from the meshblock x1 size (so large meshblocks no longer
//! require oversized scratch, and tiny ones can be spread across more teams).  The x2/x3
//! kernels additionally tile the serial transverse walk into many short walks, lifting the
//! league size and fixing the occupancy collapse seen for large single meshblocks.  The
//! scratch-column offset i0 maps a global index i onto the narrow tile-local buffer; the
//! reconstruction wrappers and Riemann solvers all accept this i0 (default 0).

template <Hydro_RSolver rsolver_method_>
void Hydro::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  int &nhyd_  = nhydro;
  int nvars = nhydro + nscalars;
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

  int scr_level = 0;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &flx1_ = uflx.x1f;

  // set the loop limits for 1D/2D/3D problems
  int il = is, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (use_fofc) {
    il = is-1, iu = ie+2;
    if (pmy_pack->pmesh->two_d) {
      jl = js-1, ju = je+1, kl = ks, ku = ke;
    } else {
      jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    }
  }

  {
    constexpr int TILE_NX1 = 64;                     // tile width (compile-time constant)
    const int nflux  = iu - il + 1;                  // # interfaces in flux range [il,iu]
    const int ntiles = (nflux + TILE_NX1 - 1)/TILE_NX1;
    // Scratch spans reconstructed states over [il_t-1, iu_t]; worst-case width TILE_NX1+2.
    size_t scr_size_t = ScrArray2D<Real>::shmem_size(nvars, TILE_NX1 + 2) * 2;

    // Reuse the 4D par_for_outer overload, mapping its (m,n,k,j) league dims onto our
    // (m, k, j, tile) decomposition.
    par_for_outer("hflux_x1",DevExeSpace(), scr_size_t, scr_level,
                  0, nmb1, kl, ku, jl, ju, 0, ntiles-1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                  const int t) {
      const int il_t = il + t*TILE_NX1;
      const int iu_t = (il_t + TILE_NX1 - 1 < iu) ? (il_t + TILE_NX1 - 1) : iu;
      const int i0   = il_t - 1;                     // scratch column = (global i) - i0

      ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, TILE_NX1 + 2);
      ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, TILE_NX1 + 2);

      // Reconstruct over [il_t-1, iu_t] to obtain BOTH L/R states over [il_t, iu_t]
      switch (recon_method_) {
        case ReconstructionMethod::dc:
          DonorCellX1(member, m, k, j, il_t-1, iu_t, w0_, wl, wr, i0);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinearX1(member, m, k, j, il_t-1, iu_t, w0_, wl, wr, i0);
          break;
        case ReconstructionMethod::ppm4:
        case ReconstructionMethod::ppmx:
          PiecewiseParabolicX1(member,eos_,extrema,true, m, k, j, il_t-1, iu_t,
                               w0_, wl, wr, i0);
          break;
        case ReconstructionMethod::wenoz:
          WENOZX1(member, eos_, true, m, k, j, il_t-1, iu_t, w0_, wl, wr, i0);
          break;
        default:
          break;
      }
      member.team_barrier();

      // compute fluxes over [il_t,iu_t]
      // NOTE(@pdmullen): Capture variables prior to if constexpr.  Required for cuda 11.6+.
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      auto flx1 = flx1_;
      if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
        Advect(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
        LLF(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
        HLLE(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
        HLLC(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
        Roe(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
        LLF_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
        HLLE_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
        HLLC_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
        LLF_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
        HLLE_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVX,wl,wr,flx1,i0);
      }
      member.team_barrier();

      // calculate fluxes of scalars (if any), restricted to physical range [is,ie+1]
      if (nvars > nhyd_) {
        const int sil = (il_t > is)   ? il_t : is;
        const int siu = (iu_t < ie+1) ? iu_t : ie+1;
        for (int n=nhyd_; n<nvars; ++n) {
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

    // set the loop limits for 1D/2D/3D problems
    il = is, iu = ie, jl = js-1, ju = je+1, kl = ks, ku = ke;
    if (use_fofc) {
      jl = js-2, ju = je+2;
      if (pmy_pack->pmesh->two_d) {
        il = is-1, iu = ie+1, kl = ks, ku = ke;
      } else {
        il = is-1, iu = ie+1, kl = ks-1, ku = ke+1;
      }
    }

    // 2D tiling (i-tile x j-tile).  The i-tile bounds the scratch width (fixing the
    // meshblock x1-size constraint), while the j-tile breaks the previously fully-serial
    // j-walk into many short walks -- lifting the league from nmb*nk to
    // nmb*nk*ntiles_i*ntiles_j and so fixing the occupancy collapse.  The rolling buffer is
    // preserved *within* each j-tile (3 small buffers + a 1-slab warmup per tile).
    {
      constexpr int TILE_NX1 = 64;                   // i-tile width (bounds scratch)
      constexpr int TILE_NX2 = 16;                   // j-tile height (de-serializes walk)
      const int nif      = iu - il + 1;              // # i-cells in flux range [il,iu]
      const int ntiles_i = (nif + TILE_NX1 - 1)/TILE_NX1;
      const int njf      = ju - jl;                  // # j-faces in flux range [jl+1,ju]
      const int ntiles_j = (njf + TILE_NX2 - 1)/TILE_NX2;
      // 3 rolling buffers, each only TILE_NX1 wide (was ncells1).
      size_t scr_size_t = ScrArray2D<Real>::shmem_size(nvars, TILE_NX1) * 3;

      // Reuse the 4D par_for_outer overload, mapping (m,n,k,j) -> (m, k, i-tile, j-tile).
      par_for_outer("hflux_x2",DevExeSpace(), scr_size_t, scr_level,
                    0, nmb1, kl, ku, 0, ntiles_i-1, 0, ntiles_j-1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k,
                    const int ti, const int tj) {
        const int il_t = il + ti*TILE_NX1;
        const int iu_t = (il_t + TILE_NX1 - 1 < iu) ? (il_t + TILE_NX1 - 1) : iu;
        const int i0   = il_t;                       // scratch column = (global i) - i0
        const int jb_lo = (jl+1) + tj*TILE_NX2;      // first j-face owned by this tile
        const int jb_hi = (jb_lo + TILE_NX2 - 1 < ju) ? (jb_lo + TILE_NX2 - 1) : ju;

        ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, TILE_NX1);

        // Serial walk over this tile's j-slabs; j = jb_lo-1 is the warmup slab (no flux).
        for (int j=jb_lo-1; j<=jb_hi; ++j) {
          // Permute scratch arrays (parity of the global j keeps the rolling buffer valid).
          auto wl     = scr1;
          auto wl_jp1 = scr2;
          auto wr     = scr3;
          if ((j%2) == 0) {
            wl     = scr2;
            wl_jp1 = scr1;
          }

          // Reconstruct qR[j] and qL[j+1] over the i-tile
          switch (recon_method_) {
            case ReconstructionMethod::dc:
              DonorCellX2(member, m, k, j, il_t, iu_t, w0_, wl_jp1, wr, i0);
              break;
            case ReconstructionMethod::plm:
              PiecewiseLinearX2(member, m, k, j, il_t, iu_t, w0_, wl_jp1, wr, i0);
              break;
            case ReconstructionMethod::ppm4:
            case ReconstructionMethod::ppmx:
              PiecewiseParabolicX2(member,eos_,extrema,true, m, k, j, il_t, iu_t,
                                   w0_, wl_jp1, wr, i0);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX2(member, eos_, true, m, k, j, il_t, iu_t, w0_, wl_jp1, wr, i0);
              break;
            default:
              break;
          }
          member.team_barrier();

          if (j >= jb_lo) {
            // NOTE: Capture variables prior to if constexpr.  Required for cuda 11.6+.
            auto eos = eos_;
            auto indcs = indcs_;
            auto size = size_;
            auto coord = coord_;
            auto flx2 = flx2_;
            if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
              Advect(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
              LLF(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
              HLLE(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
              HLLC(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
              Roe(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
              LLF_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
              HLLE_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
              HLLC_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
              LLF_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
              HLLE_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVY,wl,wr,flx2,i0);
            }
            member.team_barrier();

            // calculate fluxes of scalars (if any), restricted to physical range [is,ie]
            if (nvars > nhyd_) {
              const int sil = (il_t > is) ? il_t : is;
              const int siu = (iu_t < ie) ? iu_t : ie;
              for (int n=nhyd_; n<nvars; ++n) {
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

    // set the loop limits
    il = is, iu = ie, jl = js, ju = je, kl = ks-1, ku = ke+1;
    if (use_fofc) { il = is-1, iu = ie+1, jl = js-1, ju = je+1, kl = ks-2, ku = ke+2; }

    // 2D tiling (i-tile x k-tile), mirroring the x2 tiling.  The i-tile bounds the scratch
    // width (removing the meshblock x1-size constraint), while the k-tile breaks the
    // previously fully-serial k-walk into many short walks.  The rolling buffer is
    // preserved *within* each k-tile (3 small buffers + a 1-slab warmup per tile).
    {
      constexpr int TILE_NX1 = 64;                   // i-tile width (bounds scratch)
      constexpr int TILE_NX3 = 16;                   // k-tile depth (de-serializes walk)
      const int nif      = iu - il + 1;              // # i-cells in flux range [il,iu]
      const int ntiles_i = (nif + TILE_NX1 - 1)/TILE_NX1;
      const int nkf      = ku - kl;                  // # k-faces in flux range [kl+1,ku]
      const int ntiles_k = (nkf + TILE_NX3 - 1)/TILE_NX3;
      // 3 rolling buffers, each only TILE_NX1 wide (was ncells1).
      size_t scr_size_t = ScrArray2D<Real>::shmem_size(nvars, TILE_NX1) * 3;

      // Reuse the 4D par_for_outer overload, mapping (m,n,k,j) -> (m, j, i-tile, k-tile).
      par_for_outer("hflux_x3",DevExeSpace(), scr_size_t, scr_level,
                    0, nmb1, jl, ju, 0, ntiles_i-1, 0, ntiles_k-1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j,
                    const int ti, const int tk) {
        const int il_t = il + ti*TILE_NX1;
        const int iu_t = (il_t + TILE_NX1 - 1 < iu) ? (il_t + TILE_NX1 - 1) : iu;
        const int i0   = il_t;                       // scratch column = (global i) - i0
        const int kb_lo = (kl+1) + tk*TILE_NX3;      // first k-face owned by this tile
        const int kb_hi = (kb_lo + TILE_NX3 - 1 < ku) ? (kb_lo + TILE_NX3 - 1) : ku;

        ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, TILE_NX1);
        ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, TILE_NX1);

        // Serial walk over this tile's k-slabs; k = kb_lo-1 is the warmup slab (no flux).
        for (int k=kb_lo-1; k<=kb_hi; ++k) {
          // Permute scratch arrays (parity of the global k keeps the rolling buffer valid).
          auto wl     = scr1;
          auto wl_kp1 = scr2;
          auto wr     = scr3;
          if ((k%2) == 0) {
            wl     = scr2;
            wl_kp1 = scr1;
          }

          // Reconstruct qR[k] and qL[k+1] over the i-tile
          switch (recon_method_) {
            case ReconstructionMethod::dc:
              DonorCellX3(member, m, k, j, il_t, iu_t, w0_, wl_kp1, wr, i0);
              break;
            case ReconstructionMethod::plm:
              PiecewiseLinearX3(member, m, k, j, il_t, iu_t, w0_, wl_kp1, wr, i0);
              break;
            case ReconstructionMethod::ppm4:
            case ReconstructionMethod::ppmx:
              PiecewiseParabolicX3(member,eos_,extrema,true, m, k, j, il_t, iu_t,
                                   w0_, wl_kp1, wr, i0);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX3(member, eos_, true, m, k, j, il_t, iu_t, w0_, wl_kp1, wr, i0);
              break;
            default:
              break;
          }
          member.team_barrier();

          if (k >= kb_lo) {
            // NOTE: Capture variables prior to if constexpr.  Required for cuda 11.6+.
            auto eos = eos_;
            auto indcs = indcs_;
            auto size = size_;
            auto coord = coord_;
            auto flx3 = flx3_;
            if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
              Advect(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
              LLF(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
              HLLE(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
              HLLC(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
              Roe(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
              LLF_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
              HLLE_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
              HLLC_SR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
              LLF_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
              HLLE_GR(member,eos,indcs,size,coord,m,k,j,il_t,iu_t,IVZ,wl,wr,flx3,i0);
            }
            member.team_barrier();

            // calculate fluxes of scalars (if any), restricted to physical range [is,ie]
            if (nvars > nhyd_) {
              const int sil = (il_t > is) ? il_t : is;
              const int siu = (iu_t < ie) ? iu_t : ie;
              for (int n=nhyd_; n<nvars; ++n) {
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
template void Hydro::CalculateFluxes<Hydro_RSolver::advect>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::roe>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_gr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace hydro
