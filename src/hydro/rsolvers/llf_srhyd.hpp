#ifndef HYDRO_RSOLVERS_LLF_SRHYD_HPP_
#define HYDRO_RSOLVERS_LLF_SRHYD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srhyd.cpp
//! \brief Local Lax Friedrichs (LLF) Riemann solver for special relativistic hydro

#include "llf_hyd_singlestate.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void LLF
//! \brief Wrapper function for the LLF Riemann solver for SR hydrodynamics which calls
//! the single state LLF solver

//! The optional i0 argument shifts the scratch-array (wl/wr) column index by -i0 so a tile
//! of interfaces narrower than the full meshblock can be solved from a small scratch
//! buffer. The output flux array flx is always indexed by the global i; only the scratch
//! reads are offset. With the default i0=0 the behavior is unchanged.

KOKKOS_INLINE_FUNCTION
void LLF_SR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx,
     const int i0=0) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    const int si = i - i0;  // column index into the (possibly tiled) scratch arrays
    // Extract left/right primitives
    HydPrim1D wli,wri;
    wli.d  = wl(IDN,si);
    wli.vx = wl(ivx,si);
    wli.vy = wl(ivy,si);
    wli.vz = wl(ivz,si);

    wri.d  = wr(IDN,si);
    wri.vx = wr(ivx,si);
    wri.vy = wr(ivy,si);
    wri.vz = wr(ivz,si);

    wli.e = wl(IEN,si);
    wri.e = wr(IEN,si);

    // Call LLF solver on single interface state
    HydCons1D flux;
    SingleStateLLF_SRHyd(wli,wri,eos,flux);

    // Store results into 3D array of fluxes
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,ivx,k,j,i) = flux.mx;
    flx(m,ivy,k,j,i) = flux.my;
    flx(m,ivz,k,j,i) = flux.mz;
    flx(m,IEN,k,j,i) = flux.e;
  });

  return;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_LLF_SRHYD_HPP_
