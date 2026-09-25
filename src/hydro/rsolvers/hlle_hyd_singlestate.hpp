#ifndef HYDRO_RSOLVERS_HLLE_HYD_SINGLESTATE_HPP_
#define HYDRO_RSOLVERS_HLLE_HYD_SINGLESTATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_hyd_singlestate.hpp
//! \brief HLLE Riemann solver for a single L/R hydro state.  Face-normal velocity is
//! already stored in HydPrim1D::vx (same convention as SingleStateLLF_Hyd).

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "eos/eos.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void SingleStateHLLE_Hyd
//! \brief HLLE flux for one L/R pair (ideal gas or isothermal)

KOKKOS_INLINE_FUNCTION
void SingleStateHLLE_Hyd(const HydPrim1D &wl, const HydPrim1D &wr, const EOS_Data &eos,
                         HydCons1D &flux) {
  Real gm1 = eos.gamma - 1.0;
  Real igm1 = 1.0/gm1;
  Real iso_cs = eos.iso_cs;

  const Real wl_idn = wl.d, wr_idn = wr.d;
  const Real wl_ivx = wl.vx, wr_ivx = wr.vx;
  const Real wl_ivy = wl.vy, wr_ivy = wr.vy;
  const Real wl_ivz = wl.vz, wr_ivz = wr.vz;

  Real wl_ipr = 0.0, wr_ipr = 0.0;
  if (eos.is_ideal) {
    wl_ipr = eos.IdealGasPressure(wl.e);
    wr_ipr = eos.IdealGasPressure(wr.e);
  }

  Real sqrtdl = sqrt(wl_idn);
  Real sqrtdr = sqrt(wr_idn);
  Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

  Real wroe_ivx = (sqrtdl*wl_ivx + sqrtdr*wr_ivx)*isdlpdr;
  Real wroe_ivy = (sqrtdl*wl_ivy + sqrtdr*wr_ivy)*isdlpdr;
  Real wroe_ivz = (sqrtdl*wl_ivz + sqrtdr*wr_ivz)*isdlpdr;

  Real el = 0.0, er = 0.0, hroe = 0.0;
  if (eos.is_ideal) {
    el = wl_ipr*igm1 + 0.5*wl_idn*(SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
    er = wr_ipr*igm1 + 0.5*wr_idn*(SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
    hroe = ((el + wl_ipr)/sqrtdl + (er + wr_ipr)/sqrtdr)*isdlpdr;
  }

  Real qa, qb;
  Real a = iso_cs;
  if (eos.is_ideal) {
    qa = eos.IdealHydroSoundSpeed(wl_idn, wl_ipr);
    qb = eos.IdealHydroSoundSpeed(wr_idn, wr_ipr);
    a = hroe - 0.5*(SQR(wroe_ivx) + SQR(wroe_ivy) + SQR(wroe_ivz));
    a = (a < 0.0) ? 0.0 : sqrt(gm1*a);
  } else {
    qa = iso_cs;
    qb = iso_cs;
  }

  Real al = fmin((wroe_ivx - a), (wl_ivx - qa));
  Real ar = fmax((wroe_ivx + a), (wr_ivx + qb));
  Real bp = (ar > 0.0) ? ar : 1.0e-20;
  Real bm = (al < 0.0) ? al : -1.0e-20;

  qa = wl_ivx - bm;
  qb = wr_ivx - bp;

  HydCons1D fl, fr;
  fl.d  = wl_idn*qa;
  fr.d  = wr_idn*qb;
  fl.mx = wl_idn*wl_ivx*qa;
  fr.mx = wr_idn*wr_ivx*qb;
  fl.my = wl_idn*wl_ivy*qa;
  fr.my = wr_idn*wr_ivy*qb;
  fl.mz = wl_idn*wl_ivz*qa;
  fr.mz = wr_idn*wr_ivz*qb;
  if (eos.is_ideal) {
    fl.mx += wl_ipr;
    fr.mx += wr_ipr;
    fl.e  = el*qa + wl_ipr*wl_ivx;
    fr.e  = er*qb + wr_ipr*wr_ivx;
  } else {
    fl.mx += (iso_cs*iso_cs)*wl_idn;
    fr.mx += (iso_cs*iso_cs)*wr_idn;
  }

  qa = 0.0;
  if (bp != bm) qa = 0.5*(bp + bm)/(bp - bm);

  flux.d  = 0.5*(fl.d  + fr.d ) + qa*(fl.d  - fr.d );
  flux.mx = 0.5*(fl.mx + fr.mx) + qa*(fl.mx - fr.mx);
  flux.my = 0.5*(fl.my + fr.my) + qa*(fl.my - fr.my);
  flux.mz = 0.5*(fl.mz + fr.mz) + qa*(fl.mz - fr.mz);
  if (eos.is_ideal) flux.e = 0.5*(fl.e + fr.e) + qa*(fl.e - fr.e);
  return;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLE_HYD_SINGLESTATE_HPP_
