#ifndef MHD_RSOLVERS_SOLVE_FACE_MHD_HPP_
#define MHD_RSOLVERS_SOLVE_FACE_MHD_HPP_
//========================================================================================
//! \file solve_face_mhd.hpp
//! \brief SolveFaceMHD shared by mhd_fluxes.cpp and mhd_mood.cpp (ct_contact path).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "mhd/rsolvers/llf_mhd.hpp"
#include "mhd/rsolvers/hlle_mhd.hpp"
#include "mhd/rsolvers/hlld_mhd.hpp"
#include "mhd/rsolvers/llf_srmhd.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"
#include "mhd/rsolvers/llf_grmhd.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"

namespace mhd {

template <MHD_RSolver rsolver_method_, int ivx>
KOKKOS_INLINE_FUNCTION
void SolveFaceMHD(const EOS_Data &eos, const RegionIndcs &indcs,
                  const DualArray1D<RegionSize> &size, const CoordData &coord,
                  const int m, const int k, const int j, const int i,
                  const int is, const int js, const int ks,
                  const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
                  const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
                  const DvceArray4D<Real> &bx,
                  const DvceArray5D<Real> &flx,
                  const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez,
                  const int uct_flag = 0,
                  const DvceArray4D<Real> &uct_aL = {},
                  const DvceArray4D<Real> &uct_dL = {},
                  const DvceArray4D<Real> &uct_dR = {},
                  const DvceArray4D<Real> &uct_vt1 = {},
                  const DvceArray4D<Real> &uct_vt2 = {}) {
  if constexpr (rsolver_method_ == MHD_RSolver::advect) {
    Advect<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
    LLF<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez,
             uct_flag, uct_aL, uct_dL, uct_dR, uct_vt1, uct_vt2);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
    HLLE<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez,
              uct_flag, uct_aL, uct_dL, uct_dR, uct_vt1, uct_vt2);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
    HLLD<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez,
              uct_flag, uct_aL, uct_dL, uct_dR, uct_vt1, uct_vt2);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
    LLF_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez,
                uct_flag, uct_aL, uct_dL, uct_dR, uct_vt1, uct_vt2);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
    HLLE_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez,
                 uct_flag, uct_aL, uct_dL, uct_dR, uct_vt1, uct_vt2);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
    LLF_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks,
                wl, wr, bl, br, bx, flx, ey, ez,
                uct_flag, uct_aL, uct_dL, uct_dR, uct_vt1, uct_vt2);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
    HLLE_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks,
                 wl, wr, bl, br, bx, flx, ey, ez,
                 uct_flag, uct_aL, uct_dL, uct_dR, uct_vt1, uct_vt2);
  }
}

} // namespace mhd
#endif // MHD_RSOLVERS_SOLVE_FACE_MHD_HPP_
