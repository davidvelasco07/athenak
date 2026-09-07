//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes.cpp
//! \brief Calculate 3D fluxes of the conserved variables, and area-averaged electric
//! fields E = - (v X B) on cell faces for mhd.
//!
//! Fluxes are computed with two 1D-RangePolicy kernels per direction: (1) a per-cell
//! reconstruction kernel that materializes the L/R primitive states (w0) in the global
//! wl3d/wr3d buffers and the L/R cell-centered magnetic field (bcc0) in the
//! bl3d/br3d buffers, followed by (2) a per-face Riemann solve that reads those
//! buffers and writes both the interface flux and the two area-averaged EMF components.
//! All reconstruction methods (DC/PLM/PPM4/PPMX/WENOZ/TENO/PPM) and Riemann solvers
//! (Advect/LLF/HLLE/HLLD and the SR/GR variants) are supported; the reconstruction method
//! is chosen at runtime, the solver at compile time via the rsolver template parameter.
//!
//! Fluxes are stored in face-centered vector 'uflx', while electric fields are stored in
//! individual arrays: e2x1,e3x1 on x1-faces; e1x2,e3x2 on x2-faces; e1x3,e2x3 on x3-faces.
//! Because constrained transport needs EMFs at every transverse cell edge, the flux/EMF
//! kernels run over a transverse range extended beyond the active domain (wider for UCT).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "mhd/rsolvers/solve_face_mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void MHD::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute MHD fluxes and
//! face-centered area-averaged EMFs.  Templated over the Riemann solver for GPU perf.

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

  // Face-normal flux range. MOOD needs base fluxes over mood_halo0; FOFC over 1.
  const int mext = use_mood ? mood_halo0 : (use_fofc ? 1 : 0);
  int il1 = is-mext, iu1 = ie+1+mext;
  int jl2 = js-mext, ju2 = je+1+mext;
  int kl3 = ks-mext, ku3 = ke+1+mext;

  // For UCT the corner-EMF composition needs face-stored UCT quantities on a wider
  // transverse range: edge recon reads faces up to 3 cells away with HO methods.
  const auto emf_method_ = emf_method;
  const bool use_uct = (emf_method_ != MHD_EMF::ct_contact);
  const bool ho_uct = use_uct &&
                      !(recon_method == ReconstructionMethod::dc ||
                        recon_method == ReconstructionMethod::plm);
  const int ext = std::max(ho_uct ? 3 : 1, mext);
  const int uct_fl = (emf_method_ == MHD_EMF::uct_hlld) ? 2 : (use_uct ? 1 : 0);

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto &bcc0_ = bcc0;
  auto wl_ = wl3d;
  auto wr_ = wr3d;
  auto bl_ = bl3d;
  auto br_ = br3d;

  //------------------------------------------------------------------------------------
  // x1 direction
  {
    auto &flx1 = uflx.x1f;
    auto &bx_ = b0.x1f;
    auto &e31 = e3x1;
    auto &e21 = e2x1;
    auto aL1 = aL_x1f;  auto dl1 = dL_x1f;  auto dr1 = dR_x1f;
    auto vt1_1 = vy_x1f;  auto vt2_1 = vz_x1f;

    int jl = js, ju = je, kl = ks, ku = ke;
    if (pmy_pack->pmesh->multi_d) { jl = js-ext; ju = je+ext; }
    if (pmy_pack->pmesh->three_d) { kl = ks-ext; ku = ke+ext; }

    ReconDispatch<IVX>(recon_method_, "mflux_x1_recon_w", nmb1,
        kl, ku, jl, ju, il1-1, iu1, eos_, true, nvars, w0_, wl_, wr_);
    ReconDispatch<IVX>(recon_method_, "mflux_x1_recon_b", nmb1,
        kl, ku, jl, ju, il1-1, iu1, eos_, false, 3, bcc0_, bl_, br_);

    par_for("mflux_x1_rsolve", DevExeSpace(),
      0, nmb1, kl, ku, jl, ju, il1, iu1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = bx_;  auto flx = flx1;  auto eyl = e31;  auto ezl = e21;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFaceMHD<rsolver_method_, IVX>(eos, indcs, size, coord,
                                           m, k, j, i, is_, js_, ks_,
                                           wl, wr, bl, br, bx, flx, eyl, ezl,
                                           uct_fl, aL1, dl1, dr1, vt1_1, vt2_1);
      });

    if (nvars > nmhd_) {
      par_for("mflux_x1_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je, is, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nmhd_; n < nvars; ++n) {
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
    auto &by_ = b0.x2f;
    auto &e12 = e1x2;
    auto &e32 = e3x2;
    auto aL2 = aL_x2f;  auto dl2 = dL_x2f;  auto dr2 = dR_x2f;
    auto vt1_2 = vz_x2f;  auto vt2_2 = vx_x2f;

    int kl = ks, ku = ke;
    if (pmy_pack->pmesh->three_d) { kl = ks-ext; ku = ke+ext; }

    ReconDispatch<IVY>(recon_method_, "mflux_x2_recon_w", nmb1,
        kl, ku, jl2-1, ju2, is-ext, ie+ext, eos_, true, nvars, w0_, wl_, wr_);
    ReconDispatch<IVY>(recon_method_, "mflux_x2_recon_b", nmb1,
        kl, ku, jl2-1, ju2, is-ext, ie+ext, eos_, false, 3, bcc0_, bl_, br_);

    par_for("mflux_x2_rsolve", DevExeSpace(),
      0, nmb1, kl, ku, jl2, ju2, is-ext, ie+ext,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = by_;  auto flx = flx2;  auto eyl = e12;  auto ezl = e32;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFaceMHD<rsolver_method_, IVY>(eos, indcs, size, coord,
                                           m, k, j, i, is_, js_, ks_,
                                           wl, wr, bl, br, bx, flx, eyl, ezl,
                                           uct_fl, aL2, dl2, dr2, vt1_2, vt2_2);
      });

    if (nvars > nmhd_) {
      par_for("mflux_x2_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je+1, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nmhd_; n < nvars; ++n) {
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
    auto &bz_ = b0.x3f;
    auto &e23 = e2x3;
    auto &e13 = e1x3;
    auto aL3 = aL_x3f;  auto dl3 = dL_x3f;  auto dr3 = dR_x3f;
    auto vt1_3 = vx_x3f;  auto vt2_3 = vy_x3f;

    ReconDispatch<IVZ>(recon_method_, "mflux_x3_recon_w", nmb1,
        kl3-1, ku3, js-ext, je+ext, is-ext, ie+ext, eos_, true, nvars, w0_, wl_, wr_);
    ReconDispatch<IVZ>(recon_method_, "mflux_x3_recon_b", nmb1,
        kl3-1, ku3, js-ext, je+ext, is-ext, ie+ext, eos_, false, 3, bcc0_, bl_, br_);

    par_for("mflux_x3_rsolve", DevExeSpace(),
      0, nmb1, kl3, ku3, js-ext, je+ext, is-ext, ie+ext,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = bz_;  auto flx = flx3;  auto eyl = e23;  auto ezl = e13;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFaceMHD<rsolver_method_, IVZ>(eos, indcs, size, coord,
                                           m, k, j, i, is_, js_, ks_,
                                           wl, wr, bl, br, bx, flx, eyl, ezl,
                                           uct_fl, aL3, dl3, dr3, vt1_3, vt2_3);
      });

    if (nvars > nmhd_) {
      par_for("mflux_x3_scalars", DevExeSpace(),
        0, nmb1, ks, ke+1, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nmhd_; n < nvars; ++n) {
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
template void MHD::CalculateFluxes<MHD_RSolver::advect>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlld>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_gr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace mhd
