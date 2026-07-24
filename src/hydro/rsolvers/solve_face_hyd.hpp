#ifndef HYDRO_RSOLVERS_SOLVE_FACE_HYD_HPP_
#define HYDRO_RSOLVERS_SOLVE_FACE_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file solve_face_hyd.hpp
//! \brief Compile-time dispatch of the hydro Riemann solver for a single face, reading
//! the L/R primitive states from the global wl/wr buffers and writing the interface
//! flux.  Shared by the split-kernel flux path (hydro_fluxes.cpp) and the MOOD
//! a-posteriori fallback (hydro_mood.cpp).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
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
//! \fn SolveFace<ivx>()
//! \brief Dispatch the (compile-time) Riemann solver for a single face.  Capturing the
//! solver inputs into locals before the constexpr-if is required for CUDA 11.6+.
template <Hydro_RSolver rsolver_method_, int ivx>
KOKKOS_INLINE_FUNCTION
void SolveFace(const EOS_Data &eos, const RegionIndcs &indcs,
               const DualArray1D<RegionSize> &size, const CoordData &coord,
               const int m, const int k, const int j, const int i,
               const int is, const int js, const int ks,
               const DvceArray5D<Real> &wl,
               const DvceArray5D<Real> &wr,
               const DvceArray5D<Real> &flx) {
  if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
    Advect<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
    LLF<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
    HLLE<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
    HLLC<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
    Roe<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
    LLF_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
    HLLE_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
    HLLC_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
    LLF_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks, wl, wr, flx);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
    HLLE_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks, wl, wr, flx);
  }
}

} // namespace hydro
#endif  // HYDRO_RSOLVERS_SOLVE_FACE_HYD_HPP_
