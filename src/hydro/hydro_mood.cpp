//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_mood.cpp
//! \brief Implements the MOOD (Multidimensional Optimal Order Detection) a-posteriori
//! fallback scheme for Hydro, following Clain, Diot & Loubere and the superfv/spd
//! design used on feature/fallback-validation.
//!
//! Hydro-only surgical port onto the sink-particles hydro path: detection is the
//! same (PAD / NaN / NAD + SED), but face revision uses this branch's 1D PLM/DC
//! reconstruction plus SingleState HLLE (Newtonian) or SingleState LLF (SR/GR).
//! Unlimited reconstruct=ppm is the intended base; the cascade is base -> PLM -> DC.

#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "reconstruct/plm.hpp"
#include "hydro/rsolvers/llf_hyd_singlestate.hpp"
#include "hydro/rsolvers/hlle_hyd_singlestate.hpp"
#include "hydro.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace hydro {

namespace {

//----------------------------------------------------------------------------------------
//! \fn SEDAlpha1D()
//! \brief One-dimensional smooth-extrema detector of Vilar-type at cell offset `o`
//! (in {-1,0,1}) along direction (dk,dj,di) from cell (m,k,j,i) of array q, variable n.
//! Returns alpha in [0,1]; alpha >= 1 means the data look like a smooth extremum.
KOKKOS_INLINE_FUNCTION
Real SEDAlpha1D(const DvceArray5D<Real> &q, const int m, const int n,
                const int k, const int j, const int i,
                const int dk, const int dj, const int di, const int o) {
  const int kc = k + o*dk, jc = j + o*dj, ic = i + o*di;
  Real du_m = 0.5*(q(m,n,kc,     jc,     ic     ) - q(m,n,kc-2*dk,jc-2*dj,ic-2*di));
  Real du_c = 0.5*(q(m,n,kc+dk,  jc+dj,  ic+di  ) - q(m,n,kc-dk,  jc-dj,  ic-di  ));
  Real du_p = 0.5*(q(m,n,kc+2*dk,jc+2*dj,ic+2*di) - q(m,n,kc,     jc,     ic     ));
  Real dv = 0.25*(du_p - du_m);
  if (dv == 0.0) return 1.0;
  Real vl = du_m - du_c;
  Real vr = du_p - du_c;
  Real alpha_l = -((dv < 0.0) ? fmax(vl, 0.0) : fmin(vl, 0.0))/dv;
  Real alpha_r =  ((dv > 0.0) ? fmax(vr, 0.0) : fmin(vr, 0.0))/dv;
  return fmin(1.0, fmin(alpha_l, alpha_r));
}

//----------------------------------------------------------------------------------------
//! \fn SEDAlpha()
//! \brief Full smooth-extrema detector for variable n at cell (m,k,j,i).
KOKKOS_INLINE_FUNCTION
Real SEDAlpha(const DvceArray5D<Real> &q, const int m, const int n,
              const int k, const int j, const int i,
              const bool multi_d, const bool three_d) {
  Real alpha = 1.0;
  for (int o=-1; o<=1; ++o) {
    alpha = fmin(alpha, SEDAlpha1D(q, m, n, k, j, i, 0, 0, 1, o));
  }
  if (multi_d) {
    for (int o=-1; o<=1; ++o) {
      alpha = fmin(alpha, SEDAlpha1D(q, m, n, k, j, i, 0, 1, 0, o));
    }
  }
  if (three_d) {
    for (int o=-1; o<=1; ++o) {
      alpha = fmin(alpha, SEDAlpha1D(q, m, n, k, j, i, 1, 0, 0, o));
    }
  }
  return alpha;
}

//----------------------------------------------------------------------------------------
//! Reconstruct L/R of one variable at the face whose right cell is (k,j,i) and whose
//! normal is (dk,dj,di).  PLM uses the two adjacent cells; DC is piecewise constant.
KOKKOS_INLINE_FUNCTION
void ReconFaceVar(const ReconstructionMethod recon,
                  const DvceArray5D<Real> &q,
                  const int m, const int n,
                  const int k, const int j, const int i,
                  const int dk, const int dj, const int di,
                  Real &ql, Real &qr) {
  if (recon == ReconstructionMethod::plm) {
    Real dummy;
    PLM(q(m,n,k-2*dk,j-2*dj,i-2*di),
        q(m,n,k-  dk,j-  dj,i-  di),
        q(m,n,k,     j,     i     ),
        ql, dummy);
    PLM(q(m,n,k-  dk,j-  dj,i-  di),
        q(m,n,k,     j,     i     ),
        q(m,n,k+  dk,j+  dj,i+  di),
        dummy, qr);
  } else {
    ql = q(m,n,k-dk,j-dj,i-di);
    qr = q(m,n,k,j,i);
  }
}

//----------------------------------------------------------------------------------------
//! Load a permuted HydPrim1D L/R pair at a face.  Face-normal velocity is always vx.
KOKKOS_INLINE_FUNCTION
void ReconFaceHyd(const ReconstructionMethod recon,
                  const DvceArray5D<Real> &w,
                  const int m, const int k, const int j, const int i,
                  const int ivx, const bool is_ideal,
                  HydPrim1D &wl, HydPrim1D &wr) {
  int dk = 0, dj = 0, di = 0;
  int n_vx = IVX, n_vy = IVY, n_vz = IVZ;
  if (ivx == IVX) {
    di = 1; n_vx = IVX; n_vy = IVY; n_vz = IVZ;
  } else if (ivx == IVY) {
    dj = 1; n_vx = IVY; n_vy = IVZ; n_vz = IVX;
  } else {
    dk = 1; n_vx = IVZ; n_vy = IVX; n_vz = IVY;
  }
  ReconFaceVar(recon, w, m, IDN, k, j, i, dk, dj, di, wl.d,  wr.d);
  ReconFaceVar(recon, w, m, n_vx, k, j, i, dk, dj, di, wl.vx, wr.vx);
  ReconFaceVar(recon, w, m, n_vy, k, j, i, dk, dj, di, wl.vy, wr.vy);
  ReconFaceVar(recon, w, m, n_vz, k, j, i, dk, dj, di, wl.vz, wr.vz);
  if (is_ideal) {
    ReconFaceVar(recon, w, m, IEN, k, j, i, dk, dj, di, wl.e, wr.e);
  }
}

//----------------------------------------------------------------------------------------
//! Store a HydCons1D flux, un-permuting momentum components back to IM1/IM2/IM3.
KOKKOS_INLINE_FUNCTION
void StoreFaceFlux(DvceArray5D<Real> flx, const int m, const int k, const int j,
                   const int i, const int ivx, const bool is_ideal,
                   const HydCons1D &flux) {
  flx(m,IDN,k,j,i) = flux.d;
  if (ivx == IVX) {
    flx(m,IM1,k,j,i) = flux.mx;
    flx(m,IM2,k,j,i) = flux.my;
    flx(m,IM3,k,j,i) = flux.mz;
  } else if (ivx == IVY) {
    flx(m,IM2,k,j,i) = flux.mx;
    flx(m,IM3,k,j,i) = flux.my;
    flx(m,IM1,k,j,i) = flux.mz;
  } else {
    flx(m,IM3,k,j,i) = flux.mx;
    flx(m,IM1,k,j,i) = flux.my;
    flx(m,IM2,k,j,i) = flux.mz;
  }
  if (is_ideal) { flx(m,IEN,k,j,i) = flux.e; }
}

//----------------------------------------------------------------------------------------
//! Riemann solve for one revised face using the selected base solver.
//! Only matching adapters are admitted by Hydro; no solver substitution is allowed.
template <Hydro_RSolver rsolver_method_>
KOKKOS_INLINE_FUNCTION
void MoodSolveFace(const HydPrim1D &wl, const HydPrim1D &wr, const EOS_Data &eos,
                   const bool is_sr, const bool is_gr,
                   const Real x1v, const Real x2v, const Real x3v, const int ivx,
                   const CoordData &coord, HydCons1D &flux) {
  if (is_gr) {
    SingleStateLLF_GRHyd(wl, wr, x1v, x2v, x3v, ivx, coord, eos, flux);
  } else if (is_sr) {
    SingleStateLLF_SRHyd(wl, wr, eos, flux);
  } else if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
    const HydPrim1D &w = (wl.vx >= 0.0) ? wl : wr;
    flux.d = w.d*w.vx;
    flux.mx = w.d*w.vx*w.vx;
    flux.my = w.d*w.vy*w.vx;
    flux.mz = w.d*w.vz*w.vx;
    if (eos.is_ideal) flux.e = w.e*w.vx;
  } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
    SingleStateLLF_Hyd(wl, wr, eos, flux);
  } else {
    SingleStateHLLE_Hyd(wl, wr, eos, flux);
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void Hydro::MOODLoop
//! \brief detect/demote/revise loop.

template <Hydro_RSolver rsolver_method_>
void Hydro::MOODLoop(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;
  int nx1 = indcs_.nx1, nx2 = indcs_.nx2, nx3 = indcs_.nx3;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  int nmb1 = nmb - 1;
  int &nhyd_ = nhydro;
  int nvars = nhydro + nscalars;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  const bool is_sr = pmy_pack->pcoord->is_special_relativistic;
  const bool is_gr = pmy_pack->pcoord->is_general_relativistic;
  const bool newtonian = !(is_sr || is_gr);

  auto &u0_ = u0;
  auto &u1_ = u1;
  auto &w0_ = w0;
  auto &utest_ = utest;
  auto &fofc_ = fofc;
  auto &fb_level_ = fb_level;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  const bool use_sed = mood_sed;
  const Real eps0  = mood_eps0;
  const Real rtol = mood_rtol;
  const Real atol = mood_atol;
  const Real theta = mood_nad_theta;
  const bool nad_energy = mood_nad_energy;
  // Velocity NAD is an opt-in port of fallback-validation's MHD detector.
  // Hydro's density/energy-only default is retained for validated wave problems.
  const bool nad_von = (mood_nad_v > 0);
  const bool nad_vmag = (mood_nad_v == 1);
  const bool nad_scalars = mood_nad_scalars && (nscalars > 0);
  const int ntest = (nad_scalars) ? nvars : nhyd_;
  const int n_fb = n_fb_tiers;
  const int nrevs = mood_detect ? mood_max_revs : 0;
  const bool is_ideal_ = eos_.is_ideal;
  const int scale_mode = mood_nad_scale;

  Real gscale0 = 0.0, gscale1 = 0.0, gscale_s = 0.0;
  Real gscalev[3] = {0.0, 0.0, 0.0};
  if (scale_mode == 1 || scale_mode == 3) {
    Real gdmn = std::numeric_limits<Real>::max(), gdmx = -gdmn;
    Real gemn = gdmn, gemx = gdmx;
    Real gvmx = 0.0;
    Real cfl_fac = 1.0;
    const int ni = ie-is+1, nji = (je-js+1)*ni, nkji = (ke-ks+1)*nji;
    const int nmkji = nmb*nkji;
    const bool ideal = is_ideal_;
    const bool want_v = (scale_mode == 3);
    Kokkos::parallel_reduce("mood_grange",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &dmn, Real &dmx, Real &emn, Real &emx,
                  Real &vmx) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/ni;
      int i = (idx - m*nkji - k*nji - j*ni) + is;
      j += js; k += ks;
      Real d = w0_(m,IDN,k,j,i);
      dmn = fmin(dmn, d); dmx = fmax(dmx, d);
      if (ideal) { Real e = w0_(m,IEN,k,j,i); emn = fmin(emn,e); emx = fmax(emx,e); }
      if (want_v) {
        Real v = fmax(fabs(w0_(m,IVX,k,j,i)),
                 fmax(fabs(w0_(m,IVY,k,j,i)), fabs(w0_(m,IVZ,k,j,i))));
        vmx = fmax(vmx, v);
      }
    }, Kokkos::Min<Real>(gdmn), Kokkos::Max<Real>(gdmx),
       Kokkos::Min<Real>(gemn), Kokkos::Max<Real>(gemx), Kokkos::Max<Real>(gvmx));
#if MPI_PARALLEL_ENABLED
    Real rmin[2] = {gdmn, gemn}, rmax[3] = {gdmx, gemx, gvmx};
    MPI_Allreduce(MPI_IN_PLACE, rmin, 2, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rmax, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    gdmn = rmin[0]; gemn = rmin[1]; gdmx = rmax[0]; gemx = rmax[1]; gvmx = rmax[2];
#endif
    gscale0 = fmax(gdmx - gdmn, 0.0);
    gscale1 = fmax(gemx - gemn, 0.0);
    // velocity ranges (stage-input primitive) for the NAD scale, as in fallback-validation
    if (nad_von) {
      Real gv0mn = std::numeric_limits<Real>::max(), gv0mx = -gv0mn;
      Real gv1mn = gv0mn, gv1mx = gv0mx;
      Real gv2mn = gv0mn, gv2mx = gv0mx;
      const bool vmag_mode = nad_vmag;
      Kokkos::parallel_reduce("mood_grangev",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &v0mn, Real &v0mx, Real &v1mn, Real &v1mx,
                    Real &v2mn, Real &v2mx) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/ni;
        int i = (idx - m*nkji - k*nji - j*ni) + is;
        j += js; k += ks;
        if (vmag_mode) {
          Real v = sqrt(SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i))
                      + SQR(w0_(m,IVZ,k,j,i)));
          v0mn = fmin(v0mn, v); v0mx = fmax(v0mx, v);
        } else {
          Real vx = w0_(m,IVX,k,j,i), vy = w0_(m,IVY,k,j,i), vz = w0_(m,IVZ,k,j,i);
          v0mn = fmin(v0mn, vx); v0mx = fmax(v0mx, vx);
          v1mn = fmin(v1mn, vy); v1mx = fmax(v1mx, vy);
          v2mn = fmin(v2mn, vz); v2mx = fmax(v2mx, vz);
        }
      }, Kokkos::Min<Real>(gv0mn), Kokkos::Max<Real>(gv0mx),
         Kokkos::Min<Real>(gv1mn), Kokkos::Max<Real>(gv1mx),
         Kokkos::Min<Real>(gv2mn), Kokkos::Max<Real>(gv2mx));
#if MPI_PARALLEL_ENABLED
      Real vmn[3] = {gv0mn, gv1mn, gv2mn};
      Real vmx[3] = {gv0mx, gv1mx, gv2mx};
      MPI_Allreduce(MPI_IN_PLACE, vmn, 3, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, vmx, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      gv0mn = vmn[0]; gv1mn = vmn[1]; gv2mn = vmn[2];
      gv0mx = vmx[0]; gv1mx = vmx[1]; gv2mx = vmx[2];
#endif
      gscalev[0] = fmax(gv0mx - gv0mn, 0.0);
      gscalev[1] = fmax(gv1mx - gv1mn, 0.0);
      gscalev[2] = fmax(gv2mx - gv2mn, 0.0);
    }
    if (scale_mode == 3) {
      Real dxmin = std::numeric_limits<Real>::max();
      auto &msize = pmy_pack->pmb->mb_size;
      for (int m=0; m<nmb; ++m) {
        dxmin = std::min(dxmin, static_cast<Real>(msize.h_view(m).dx1));
        if (multi_d) dxmin = std::min(dxmin, static_cast<Real>(msize.h_view(m).dx2));
        if (three_d) dxmin = std::min(dxmin, static_cast<Real>(msize.h_view(m).dx3));
      }
      Real cfl_adv = (pmy_pack->pmesh->dt)*gvmx/dxmin;
      cfl_fac = fmin(1.0, cfl_adv);
      gscale0 *= cfl_fac;
      gscale1 *= cfl_fac;
      for (int v=0; v<3; ++v) gscalev[v] *= cfl_fac;
    }
    if (nad_scalars) {
      Real gsmn = std::numeric_limits<Real>::max(), gsmx = -gsmn;
      const int nlo = nhyd_, nhi = nvars - 1;
      Kokkos::parallel_reduce("mood_grange_s",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &smn, Real &smx) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/ni;
        int i = (idx - m*nkji - k*nji - j*ni) + is;
        j += js; k += ks;
        for (int n=nlo; n<=nhi; ++n) {
          Real s = w0_(m,n,k,j,i);
          smn = fmin(smn, s); smx = fmax(smx, s);
        }
      }, Kokkos::Min<Real>(gsmn), Kokkos::Max<Real>(gsmx));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &gsmn, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &gsmx, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
      gscale_s = fmax(gsmx - gsmn, 0.0)*cfl_fac;
    }
  }

  const int hmax = nrevs;
  int il = is-hmax, iu = ie+hmax, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-hmax, ju = je+hmax; }
  if (three_d) { kl = ks-hmax, ku = ke+hmax; }

  const int ndi = 1;
  const int ndj = (multi_d) ? 1 : 0;
  const int ndk = (three_d) ? 1 : 0;

  {
    auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
    const bool excising = (pmy_pack->pcoord->is_general_relativistic &&
                           pmy_pack->pcoord->coord_data.bh_excise);
    const int nfb_seed = n_fb;
    if (excising) {
      const int e3 = static_cast<int>(fb_level_.extent(1)) - 1;
      const int e2 = static_cast<int>(fb_level_.extent(2)) - 1;
      const int e1 = static_cast<int>(fb_level_.extent(3)) - 1;
      par_for("mood_seed_excision", DevExeSpace(), 0, nmb1,
              0, e3, 0, e2, 0, e1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        fb_level_(m,k,j,i) = excision_flux_(m,k,j,i) ? nfb_seed : 0;
      });
    } else {
      Kokkos::deep_copy(fb_level_, 0);
    }
  }

  int ndemoted_total = 0;
  for (int rev=1; rev<=nrevs; ++rev) {
    const int dh = 1 + (nrevs - rev);
    il = is-dh; iu = ie+dh;
    if (multi_d) { jl = js-dh; ju = je+dh; }
    if (three_d) { kl = ks-dh; ku = ke+dh; }

    const int ni   = (iu - il + 1);
    const int nji  = (ju - jl + 1)*ni;
    const int nkji = (ku - kl + 1)*nji;
    const int nmkji = nmb*nkji;

    Kokkos::deep_copy(fofc_, false);
    par_for("mood_newu", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size_.d_view(m).dx1;
      Real dtodx2 = beta_dt/size_.d_view(m).dx2;
      Real dtodx3 = beta_dt/size_.d_view(m).dx3;
      for (int n=0; n<ntest; ++n) {
        Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
        if (multi_d) {
          divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
        }
        if (three_d) {
          divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        }
        utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
      }
    });

    if (scale_mode == 2 && rev == 1) {
      Real gdd = 0.0, gde = 0.0, gds = 0.0;
      const int nia = ie-is+1, njia = (je-js+1)*nia, nkji_a = (ke-ks+1)*njia;
      const int nmkji_a = nmb*nkji_a;
      const bool ideal = is_ideal_;
      const bool newt = newtonian;
      const bool do_s = nad_scalars;
      const int nlo = nhyd_, nhi = nvars - 1;
      Kokkos::parallel_reduce("mood_gdu",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji_a),
      KOKKOS_LAMBDA(const int &idx, Real &dd, Real &de, Real &ds) {
        int m = idx/nkji_a;
        int k = (idx - m*nkji_a)/njia;
        int j = (idx - m*nkji_a - k*njia)/nia;
        int i = (idx - m*nkji_a - k*njia - j*nia) + is;
        j += js; k += ks;
        Real d = fabs(utest_(m,IDN,k,j,i) - u0_(m,IDN,k,j,i));
        if (isfinite(d)) dd = fmax(dd, d);
        if (ideal) {
          Real enew, eold;
          if (newt) {
            Real dtst = utest_(m,IDN,k,j,i);
            enew = (dtst > 0.0) ?
                   utest_(m,IEN,k,j,i) - 0.5*(SQR(utest_(m,IM1,k,j,i))
                   + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dtst : 0.0;
            eold = (dtst > 0.0) ? w0_(m,IEN,k,j,i) : 0.0;
          } else {
            enew = utest_(m,IEN,k,j,i);
            eold = u0_(m,IEN,k,j,i);
          }
          Real e = fabs(enew - eold);
          if (isfinite(e)) de = fmax(de, e);
        }
        if (do_s) {
          const Real dtst = utest_(m,IDN,k,j,i);
          if (dtst > 0.0) {
            for (int n=nlo; n<=nhi; ++n) {
              Real s = fabs(utest_(m,n,k,j,i)/dtst - w0_(m,n,k,j,i));
              if (isfinite(s)) ds = fmax(ds, s);
            }
          }
        }
      }, Kokkos::Max<Real>(gdd), Kokkos::Max<Real>(gde), Kokkos::Max<Real>(gds));
#if MPI_PARALLEL_ENABLED
      Real rmx[3] = {gdd, gde, gds};
      MPI_Allreduce(MPI_IN_PLACE, rmx, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      gdd = rmx[0]; gde = rmx[1]; gds = rmx[2];
#endif
      gscale0 = gdd;
      gscale1 = gde;
      gscale_s = gds;
      if (nad_von) {
        Real gvd0 = 0.0, gvd1 = 0.0, gvd2 = 0.0;
        const bool vmag_mode2 = nad_vmag;
        Kokkos::parallel_reduce("mood_gduv",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji_a),
        KOKKOS_LAMBDA(const int &idx, Real &vd0, Real &vd1, Real &vd2) {
          int m = idx/nkji_a;
          int k = (idx - m*nkji_a)/njia;
          int j = (idx - m*nkji_a - k*njia)/nia;
          int i = (idx - m*nkji_a - k*njia - j*nia) + is;
          j += js; k += ks;
          Real dtst = utest_(m,IDN,k,j,i);
          if (dtst > 0.0) {
            if (vmag_mode2) {
              Real vn = sqrt(SQR(utest_(m,IM1,k,j,i)) + SQR(utest_(m,IM2,k,j,i))
                           + SQR(utest_(m,IM3,k,j,i)))/dtst;
              Real vo = sqrt(SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i))
                           + SQR(w0_(m,IVZ,k,j,i)));
              Real dv = fabs(vn - vo);
              if (isfinite(dv)) vd0 = fmax(vd0, dv);
            } else {
              Real dvx = fabs(utest_(m,IM1,k,j,i)/dtst - w0_(m,IVX,k,j,i));
              Real dvy = fabs(utest_(m,IM2,k,j,i)/dtst - w0_(m,IVY,k,j,i));
              Real dvz = fabs(utest_(m,IM3,k,j,i)/dtst - w0_(m,IVZ,k,j,i));
              if (isfinite(dvx)) vd0 = fmax(vd0, dvx);
              if (isfinite(dvy)) vd1 = fmax(vd1, dvy);
              if (isfinite(dvz)) vd2 = fmax(vd2, dvz);
            }
          }
        }, Kokkos::Max<Real>(gvd0), Kokkos::Max<Real>(gvd1), Kokkos::Max<Real>(gvd2));
#if MPI_PARALLEL_ENABLED
        Real vd[3] = {gvd0, gvd1, gvd2};
        MPI_Allreduce(MPI_IN_PLACE, vd, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
        gvd0 = vd[0]; gvd1 = vd[1]; gvd2 = vd[2];
#endif
        gscalev[0] = gvd0; gscalev[1] = gvd1; gscalev[2] = gvd2;
      }
    }

    peos->ConsToPrim(utest_, w0_, true, il, iu, jl, ju, kl, ku);

    int ndemoted = 0, ndemoted_int = 0;
    const Real gsv0 = gscalev[0], gsv1 = gscalev[1], gsv2 = gscalev[2];
    Kokkos::parallel_reduce("mood_detect",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, int &sum, int &sum_int) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/ni;
      int i = (idx - m*nkji - k*nji - j*ni) + il;
      j += jl;
      k += kl;

      const int lv = fb_level_(m,k,j,i);
      if (lv >= n_fb) {
        fofc_(m,k,j,i) = false;
        return;
      }

      bool danger = fofc_(m,k,j,i);
      if (!danger) {
        for (int n=0; n<ntest; ++n) {
          if (!isfinite(utest_(m,n,k,j,i))) { danger = true; }
        }
      }

      bool trouble = danger;

      if (!trouble && lv < 1) {
        const Real dtst = utest_(m,IDN,k,j,i);
        const int npass = (is_ideal_ && nad_energy) ? 2 : 1;
        for (int pass=0; pass<npass && !trouble; ++pass) {
          const int n = (pass == 0) ? IDN : IEN;
          Real q_new;
          if (!newtonian) {
            q_new = utest_(m,n,k,j,i);
          } else if (n == IDN) {
            q_new = dtst;
          } else {
            q_new = utest_(m,IEN,k,j,i) - 0.5*(SQR(utest_(m,IM1,k,j,i))
                  + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dtst;
          }
          const DvceArray5D<Real> &qref = (newtonian) ? w0_ : u0_;
          Real qmn = qref(m,n,k,j,i), qmx = qmn;
          for (int ddk=-ndk; ddk<=ndk; ++ddk) {
          for (int ddj=-ndj; ddj<=ndj; ++ddj) {
          for (int ddi=-ndi; ddi<=ndi; ++ddi) {
            Real q = qref(m,n,k+ddk,j+ddj,i+ddi);
            qmn = fmin(qmn, q);
            qmx = fmax(qmx, q);
          }}}
          Real eps_m, eps_p;
          if (scale_mode == 0) {
            eps_m = fmax(rtol*fabs(qmn), atol);
            eps_p = fmax(rtol*fabs(qmx), atol);
          } else if (scale_mode == 1) {
            const Real gsc = (pass == 0) ? gscale0 : gscale1;
            if (theta >= 1.0) {
              Real eps = fmax(rtol*gsc, atol);
              eps_m = eps;
              eps_p = eps;
            } else {
              const Real gth = pow(gsc, theta);
              eps_m = fmax(rtol*gth*pow(fabs(qmn), 1.0-theta), atol);
              eps_p = fmax(rtol*gth*pow(fabs(qmx), 1.0-theta), atol);
            }
          } else {
            const Real gsc = (pass == 0) ? gscale0 : gscale1;
            Real eps = fmax(rtol*gsc, atol);
            eps_m = eps;
            eps_p = eps;
          }
          eps_m = fmax(eps_m, eps0*fabs(qmn));
          eps_p = fmax(eps_p, eps0*fabs(qmx));
          bool nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
          if (nad && use_sed) {
            nad = (SEDAlpha(qref, m, n, k, j, i, multi_d, three_d) < 1.0);
          }
          if (nad) { trouble = true; }
        }
      }

      // Match fallback-validation/src/mhd/mhd_mood.cpp velocity NAD: same
      // candidate, 3^ndim bounds, global scale modes and component-wise SED.
      // As there, |v| mode has no smooth-extrema exemption.
      if (!trouble && lv < 1 && nad_von) {
        const Real dtst = utest_(m,IDN,k,j,i);
        const int nvpass = nad_vmag ? 1 : 3;
        for (int vidx=0; vidx<nvpass && !trouble; ++vidx) {
          Real q_new;
          // candidate velocity from the trial conserved state (newtonian: mom/rho)
          if (nad_vmag) {
            q_new = sqrt(SQR(utest_(m,IM1,k,j,i)) + SQR(utest_(m,IM2,k,j,i))
                       + SQR(utest_(m,IM3,k,j,i))) / dtst;
          } else {
            q_new = utest_(m,IM1+vidx,k,j,i) / dtst;
          }
          Real qmn, qmx;
          if (nad_vmag) {
            Real v0 = sqrt(SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i))
                         + SQR(w0_(m,IVZ,k,j,i)));
            qmn = v0; qmx = v0;
            for (int dk=-ndk; dk<=ndk; ++dk) {
            for (int dj=-ndj; dj<=ndj; ++dj) {
            for (int di=-ndi; di<=ndi; ++di) {
              Real q = sqrt(SQR(w0_(m,IVX,k+dk,j+dj,i+di))
                          + SQR(w0_(m,IVY,k+dk,j+dj,i+di))
                          + SQR(w0_(m,IVZ,k+dk,j+dj,i+di)));
              qmn = fmin(qmn, q);
              qmx = fmax(qmx, q);
            }}}
          } else {
            const int n = IVX + vidx;
            qmn = w0_(m,n,k,j,i); qmx = qmn;
            for (int dk=-ndk; dk<=ndk; ++dk) {
            for (int dj=-ndj; dj<=ndj; ++dj) {
            for (int di=-ndi; di<=ndi; ++di) {
              Real q = w0_(m,n,k+dk,j+dj,i+di);
              qmn = fmin(qmn, q);
              qmx = fmax(qmx, q);
            }}}
          }
          const Real gsc = (vidx == 0) ? gsv0 : ((vidx == 1) ? gsv1 : gsv2);
          Real eps_m, eps_p;
          if (scale_mode == 0) {          // spd-style relative
            eps_m = fmax(rtol*fabs(qmn), atol);
            eps_p = fmax(rtol*fabs(qmx), atol);
          } else if (scale_mode == 1) {   // grange with Mach-softening exponent theta
            if (theta >= 1.0) {
              Real eps = fmax(rtol*gsc, atol);
              eps_m = eps;
              eps_p = eps;
            } else {
              const Real gth = pow(gsc, theta);
              eps_m = fmax(rtol*gth*pow(fabs(qmn), 1.0-theta), atol);
              eps_p = fmax(rtol*gth*pow(fabs(qmx), 1.0-theta), atol);
            }
          } else {                        // gdu & gcfl: flat global scale
            Real eps = fmax(rtol*gsc, atol);
            eps_m = eps;
            eps_p = eps;
          }
          eps_m = fmax(eps_m, eps0*fabs(qmn));
          eps_p = fmax(eps_p, eps0*fabs(qmx));
          bool nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
          if (nad && use_sed) {
            const Real alpha = nad_vmag ? 0.0
                : SEDAlpha(w0_, m, IVX+vidx, k, j, i, multi_d, three_d);
            nad = (alpha < 1.0);
          }
          if (nad) trouble = true;
        }
      }

      if (!trouble && lv < 1 && nad_scalars) {
        const Real dtst = utest_(m,IDN,k,j,i);
        for (int n=nhyd_; n<nvars && !trouble; ++n) {
          if (!(dtst > 0.0)) break;
          const Real q_new = utest_(m,n,k,j,i)/dtst;
          Real qmn = w0_(m,n,k,j,i), qmx = qmn;
          for (int ddk=-ndk; ddk<=ndk; ++ddk) {
          for (int ddj=-ndj; ddj<=ndj; ++ddj) {
          for (int ddi=-ndi; ddi<=ndi; ++ddi) {
            Real q = w0_(m,n,k+ddk,j+ddj,i+ddi);
            qmn = fmin(qmn, q);
            qmx = fmax(qmx, q);
          }}}
          Real eps_m, eps_p;
          if (scale_mode == 0) {
            eps_m = fmax(rtol*fabs(qmn), atol);
            eps_p = fmax(rtol*fabs(qmx), atol);
          } else {
            Real eps = fmax(rtol*gscale_s, atol);
            eps_m = eps;
            eps_p = eps;
          }
          eps_m = fmax(eps_m, eps0*fabs(qmn));
          eps_p = fmax(eps_p, eps0*fabs(qmx));
          bool nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
          if (nad && use_sed) {
            nad = (SEDAlpha(w0_, m, n, k, j, i, multi_d, three_d) < 1.0);
          }
          if (nad) { trouble = true; }
        }
      }

      fofc_(m,k,j,i) = trouble;
      if (trouble) {
        fb_level_(m,k,j,i) += 1;
        sum++;
        if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
          sum_int++;
        }
      }
    }, Kokkos::Sum<int>(ndemoted), Kokkos::Sum<int>(ndemoted_int));

    if (ndemoted == 0) break;
    ndemoted_total += ndemoted_int;

    //------------------------------------------------------------------------------------
    // Revise fluxes on every face touching a newly-demoted cell.
    par_for("mood_rev_x1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, is-dh, ie+1+dh,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (fofc_(m,k,j,i-1) || fofc_(m,k,j,i)) {
        int lv_l = fb_level_(m,k,j,i-1), lv_r = fb_level_(m,k,j,i);
        int tier = (lv_l > lv_r) ? lv_l : lv_r;
        ReconstructionMethod frecon = (tier >= n_fb) ?
            ReconstructionMethod::dc : ReconstructionMethod::plm;
        HydPrim1D wl, wr;
        ReconFaceHyd(frecon, w0_, m, k, j, i, IVX, is_ideal_, wl, wr);
        Real x1v = 0.0, x2v = 0.0, x3v = 0.0;
        if (is_gr) {
          x1v = LeftEdgeX(i-is, nx1, size_.d_view(m).x1min, size_.d_view(m).x1max);
          x2v = CellCenterX(j-js, nx2, size_.d_view(m).x2min, size_.d_view(m).x2max);
          x3v = CellCenterX(k-ks, nx3, size_.d_view(m).x3min, size_.d_view(m).x3max);
        }
        HydCons1D flux;
        MoodSolveFace<rsolver_method_>(wl, wr, eos_, is_sr, is_gr,
                                       x1v, x2v, x3v, IVX, coord_, flux);
        StoreFaceFlux(flx1, m, k, j, i, IVX, is_ideal_, flux);
        for (int n=nhyd_; n<nvars; ++n) {
          Real sl, sr;
          ReconFaceVar(frecon, w0_, m, n, k, j, i, 0, 0, 1, sl, sr);
          if (flx1(m,IDN,k,j,i) >= 0.0) {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*sl;
          } else {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*sr;
          }
        }
      }
    });

    if (multi_d) {
      par_for("mood_rev_x2", DevExeSpace(), 0, nmb1, kl, ku, js-dh, je+1+dh, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (fofc_(m,k,j-1,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k,j-1,i), lv_r = fb_level_(m,k,j,i);
          int tier = (lv_l > lv_r) ? lv_l : lv_r;
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          HydPrim1D wl, wr;
          ReconFaceHyd(frecon, w0_, m, k, j, i, IVY, is_ideal_, wl, wr);
          Real x1v = 0.0, x2v = 0.0, x3v = 0.0;
          if (is_gr) {
            x1v = CellCenterX(i-is, nx1, size_.d_view(m).x1min, size_.d_view(m).x1max);
            x2v = LeftEdgeX(j-js, nx2, size_.d_view(m).x2min, size_.d_view(m).x2max);
            x3v = CellCenterX(k-ks, nx3, size_.d_view(m).x3min, size_.d_view(m).x3max);
          }
          HydCons1D flux;
          MoodSolveFace<rsolver_method_>(wl, wr, eos_, is_sr, is_gr,
                                         x1v, x2v, x3v, IVY, coord_, flux);
          StoreFaceFlux(flx2, m, k, j, i, IVY, is_ideal_, flux);
          for (int n=nhyd_; n<nvars; ++n) {
            Real sl, sr;
            ReconFaceVar(frecon, w0_, m, n, k, j, i, 0, 1, 0, sl, sr);
            if (flx2(m,IDN,k,j,i) >= 0.0) {
              flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*sl;
            } else {
              flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*sr;
            }
          }
        }
      });
    }

    if (three_d) {
      par_for("mood_rev_x3", DevExeSpace(), 0, nmb1, ks-dh, ke+1+dh, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (fofc_(m,k-1,j,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k-1,j,i), lv_r = fb_level_(m,k,j,i);
          int tier = (lv_l > lv_r) ? lv_l : lv_r;
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          HydPrim1D wl, wr;
          ReconFaceHyd(frecon, w0_, m, k, j, i, IVZ, is_ideal_, wl, wr);
          Real x1v = 0.0, x2v = 0.0, x3v = 0.0;
          if (is_gr) {
            x1v = CellCenterX(i-is, nx1, size_.d_view(m).x1min, size_.d_view(m).x1max);
            x2v = CellCenterX(j-js, nx2, size_.d_view(m).x2min, size_.d_view(m).x2max);
            x3v = LeftEdgeX(k-ks, nx3, size_.d_view(m).x3min, size_.d_view(m).x3max);
          }
          HydCons1D flux;
          MoodSolveFace<rsolver_method_>(wl, wr, eos_, is_sr, is_gr,
                                         x1v, x2v, x3v, IVZ, coord_, flux);
          StoreFaceFlux(flx3, m, k, j, i, IVZ, is_ideal_, flux);
          for (int n=nhyd_; n<nvars; ++n) {
            Real sl, sr;
            ReconFaceVar(frecon, w0_, m, n, k, j, i, 1, 0, 0, sl, sr);
            if (flx3(m,IDN,k,j,i) >= 0.0) {
              flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*sl;
            } else {
              flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*sr;
            }
          }
        }
      });
    }
  }

  // A revision can make a previously accepted neighbor inadmissible. The
  // reference loop's fixed revision budget does not by itself certify its final
  // fluxes. Check the active-cell isothermal update before EOS flooring can turn
  // a negative density into enormous velocities. This is a failure guard only;
  // it does not change the validated detector, tier selection, or fluxes.
  if (newtonian && !is_ideal_ && nrevs > 0 &&
      pdriver->time_evolution == TimeEvolution::dynamic) {
    const int ni = ie-is+1, nji = (je-js+1)*ni, nkji = (ke-ks+1)*nji;
    const int nmkji = nmb*nkji;
    const Real dfloor = eos_.dfloor;
    int nbad = 0;
    Kokkos::parallel_reduce("mood_final_pad", Kokkos::RangePolicy<>(0, nmkji),
    KOKKOS_LAMBDA(const int idx, int &sum) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/ni;
      int i = (idx - m*nkji - k*nji - j*ni) + is;
      j += js; k += ks;
      bool bad = false;
      for (int n=0; n<nhyd_; ++n) {
        Real divf = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/size_.d_view(m).dx1;
        if (multi_d) {
          divf += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/size_.d_view(m).dx2;
        }
        if (three_d) {
          divf += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/size_.d_view(m).dx3;
        }
        Real u = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf;
        bad = bad || !isfinite(u) || (n == IDN && u < dfloor);
      }
      if (bad) ++sum;
    }, Kokkos::Sum<int>(nbad));
    if (nbad > 0) {
      std::cerr << "### FATAL ERROR: MOOD revision budget exhausted with " << nbad
                << " inadmissible active cells on rank " << global_variable::my_rank
                << ", cycle=" << pmy_pack->pmesh->ncycle << ", stage=" << stage
                << ". Increase mood_max_revs (and nghost) or reduce CFL."
                << std::endl;
#if MPI_PARALLEL_ENABLED
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
      std::exit(EXIT_FAILURE);
    }
  }

  pmy_pack->pmesh->ecounter.nmood += ndemoted_total;
  return;
}

template void Hydro::MOODLoop<Hydro_RSolver::advect>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::llf>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hlle>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hllc>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::roe>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::llf_sr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hllc_sr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::llf_gr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace hydro
