#ifndef HYDRO_RSOLVERS_ADVECT_HYD_HPP_
#define HYDRO_RSOLVERS_ADVECT_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect_hyd.hpp
//! \brief Riemann solver for pure advection problems (v = constant).  Simply computes the
//! upwind flux of each variable.  Can only be used for isothermal EOS.

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void Advect
//! \brief An advection Riemann solver for hydrodynamics (isothermal)

//! The optional i0 argument shifts the scratch-array (wl/wr) column index by -i0 so a tile
//! of interfaces narrower than the full meshblock can be solved from a small scratch
//! buffer. The output flux array flx is always indexed by the global i; only the scratch
//! reads are offset. With the default i0=0 the behavior is unchanged.

KOKKOS_INLINE_FUNCTION
void Advect(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx,
     const int i0=0) {
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;

  par_for_inner(member, il, iu, [&](const int i) {
    const int si = i - i0;  // column index into the (possibly tiled) scratch arrays
    //  Compute upwind fluxes
    if (wl(ivx,si) >= 0.0) {
      flx(m,IDN,k,j,i) = wl(IDN,si)*wl(ivx,si);
      flx(m,ivx,k,j,i) = wl(IDN,si)*wl(ivx,si)*wl(ivx,si);
      flx(m,ivy,k,j,i) = 0.0;
      flx(m,ivz,k,j,i) = 0.0;
    } else {
      flx(m,IDN,k,j,i) = wr(IDN,si)*wr(ivx,si);
      flx(m,ivx,k,j,i) = wr(IDN,si)*wr(ivx,si)*wr(ivx,si);
      flx(m,ivy,k,j,i) = 0.0;
      flx(m,ivz,k,j,i) = 0.0;
    }
  });

  return;
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_ADVECT_HYD_HPP_
