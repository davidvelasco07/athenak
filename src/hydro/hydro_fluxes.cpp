//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//! \brief Calculate 3D fluxes for hydro.
//!
//! Fluxes are computed with two 1D-RangePolicy kernels per direction: (1) a per-cell
//! reconstruction kernel that materializes the L/R primitive states in the global
//! wl3d/wr3d buffers, followed by (2) a per-face Riemann solve that reads those
//! buffers and writes the interface flux.  All reconstruction methods (DC/PLM/PPM4/
//! PPMX/WENOZ/TENO) and non-relativistic Riemann solvers (Advect/LLF/HLLE/HLLC/Roe) are
//! supported; the reconstruction method is chosen at runtime, the solver at compile time
//! via the rsolver template parameter.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "hydro/rsolvers/solve_face_hyd.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Hydro::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes.
//! Templated over the Riemann solver for better performance on GPUs.

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

  // Face-normal and transverse flux ranges.  With FOFC/MOOD the correction algorithms
  // read the main fluxes beyond the active domain in every dimension (to build the
  // candidate update in a ghost halo for cross-block detection consistency), so the
  // reconstruction/solve ranges are extended on both sides: by one cell for FOFC, and
  // by mood_max_revs cells for MOOD (each revision iteration shrinks the halo of ghost
  // cells whose candidate is consistent with the neighbor block by one — the
  // parallel-MOOD "light cone").
  int il1 = is, iu1 = ie+1, jl2 = js, ju2 = je+1, kl3 = ks, ku3 = ke+1;
  int itl = is, itu = ie, jtl = js, jtu = je, ktl = ks, ktu = ke;
  if (use_fofc || use_mood) {
    const int ext = (use_mood) ? mood_max_revs : 1;
    il1 = is-ext; iu1 = ie+1+ext;
    jl2 = js-ext; ju2 = je+1+ext;
    kl3 = ks-ext; ku3 = ke+1+ext;
    itl = is-ext; itu = ie+ext;
    if (pmy_pack->pmesh->multi_d) { jtl = js-ext; jtu = je+ext; }
    if (pmy_pack->pmesh->three_d) { ktl = ks-ext; ktu = ke+ext; }
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto wl_ = wl3d;
  auto wr_ = wr3d;

  //------------------------------------------------------------------------------------
  // x1 direction
  {
    auto &flx1 = uflx.x1f;
    // Reconstruction over cells i in [il1-1, iu1], j in [jtl, jtu], k in [ktl, ktu]
    ReconDispatch<IVX>(recon_method_, "hflux_x1_recon", nmb1,
        ktl, ktu, jtl, jtu, il1-1, iu1, eos_, true, nvars, w0_, wl_, wr_);

    // Riemann solve over faces i in [il1, iu1]
    par_for("hflux_x1_rsolve", DevExeSpace(),
      0, nmb1, ktl, ktu, jtl, jtu, il1, iu1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;
        auto wr = wr_;
        auto flx = flx1;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFace<rsolver_method_, IVX>(eos, indcs, size, coord,
                                        m, k, j, i, is_, js_, ks_, wl, wr, flx);
      });

    // Scalar fluxes (upwind from sign of mass flux)
    if (nvars > nhyd_) {
      par_for("hflux_x1_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je, is, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd_; n < nvars; ++n) {
            if (flx1(m, IDN, k, j, i) >= 0.0) {
              flx1(m, n, k, j, i) = flx1(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx1(m, n, k, j, i) = flx1(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  //------------------------------------------------------------------------------------
  // x2 direction
  if (pmy_pack->pmesh->multi_d) {
    auto &flx2 = uflx.x2f;
    // Reconstruction over cells j in [jl2-1, ju2], i in [itl, itu], k in [ktl, ktu]
    ReconDispatch<IVY>(recon_method_, "hflux_x2_recon", nmb1,
        ktl, ktu, jl2-1, ju2, itl, itu, eos_, true, nvars, w0_, wl_, wr_);

    // Riemann solve over faces j in [jl2, ju2]
    par_for("hflux_x2_rsolve", DevExeSpace(),
      0, nmb1, ktl, ktu, jl2, ju2, itl, itu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;
        auto wr = wr_;
        auto flx = flx2;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFace<rsolver_method_, IVY>(eos, indcs, size, coord,
                                        m, k, j, i, is_, js_, ks_, wl, wr, flx);
      });

    if (nvars > nhyd_) {
      par_for("hflux_x2_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je+1, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd_; n < nvars; ++n) {
            if (flx2(m, IDN, k, j, i) >= 0.0) {
              flx2(m, n, k, j, i) = flx2(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx2(m, n, k, j, i) = flx2(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  //------------------------------------------------------------------------------------
  // x3 direction
  if (pmy_pack->pmesh->three_d) {
    auto &flx3 = uflx.x3f;
    // Reconstruction over cells k in [kl3-1, ku3], j in [jtl, jtu], i in [itl, itu]
    ReconDispatch<IVZ>(recon_method_, "hflux_x3_recon", nmb1,
        kl3-1, ku3, jtl, jtu, itl, itu, eos_, true, nvars, w0_, wl_, wr_);

    // Riemann solve over faces k in [kl3, ku3]
    par_for("hflux_x3_rsolve", DevExeSpace(),
      0, nmb1, kl3, ku3, jtl, jtu, itl, itu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;
        auto wr = wr_;
        auto flx = flx3;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFace<rsolver_method_, IVZ>(eos, indcs, size, coord,
                                        m, k, j, i, is_, js_, ks_, wl, wr, flx);
      });

    if (nvars > nhyd_) {
      par_for("hflux_x3_scalars", DevExeSpace(),
        0, nmb1, ks, ke+1, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd_; n < nvars; ++n) {
            if (flx3(m, IDN, k, j, i) >= 0.0) {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
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
