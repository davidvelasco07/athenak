//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_mood.cpp
//! \brief Implements the MOOD (Multidimensional Optimal Order Detection) a-posteriori
//! fallback scheme for Hydro, following the design of Clain, Diot & Loubere and the
//! superfv reference implementation (github.com/jpalafou/superfv).
//!
//! Each RK stage, after the base high-order fluxes are computed, a candidate update is
//! built with the exact stage combination.  Cells where the candidate is inadmissible
//! are detected using:
//!   PAD:  physical admissibility — conversion to primitives would require floors
//!         (existing EOS ConsToPrim only_testfloors path, same as FOFC),
//!   NaN:  any non-finite candidate value,
//!   NAD:  numerical admissibility — a relaxed discrete-maximum-principle check on the
//!         candidate vs the stage-input state over the 3^ndim neighborhood, optionally
//!         exempted at smooth extrema (SED detector) so smooth flow retains full order.
//! Flagged cells descend a per-cell fallback cascade (base -> PLM -> DC); the fluxes on
//! all faces of a demoted cell are recomputed at the demoted tier with the SAME Riemann
//! solver as the base scheme.  The detect/demote/revise loop runs up to mood_max_revs
//! iterations, exiting early once no new cell is flagged.
//!
//! The loop runs inside the Fluxes task, before SendFlux/RecvFlux, so revised fluxes
//! are restricted at fine/coarse AMR boundaries like any other flux (conservative).
//! Detection is evaluated on the interior plus one ghost layer so neighboring
//! MeshBlocks make identical decisions about their shared faces (as FOFC does).
//! Like FOFC, revised faces lose any viscous/conduction flux contribution.

#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "hydro/rsolvers/solve_face_hyd.hpp"
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
//! Follows the spd (jpalafou/spd, modular-gpu) convention: a vanishing second
//! derivative means locally linear data, which is treated as smooth (alpha = 1) —
//! clipping dv away from zero instead would return alpha = 0 there and disable the
//! smooth-extrema exemption exactly where the DMP bounds are tightest.
KOKKOS_INLINE_FUNCTION
Real SEDAlpha1D(const DvceArray5D<Real> &q, const int m, const int n,
                const int k, const int j, const int i,
                const int dk, const int dj, const int di, const int o) {
  const int kc = k + o*dk, jc = j + o*dj, ic = i + o*di;
  // central first derivatives at cell offsets -1, 0, +1 (relative to o)
  Real du_m = 0.5*(q(m,n,kc,     jc,     ic     ) - q(m,n,kc-2*dk,jc-2*dj,ic-2*di));
  Real du_c = 0.5*(q(m,n,kc+dk,  jc+dj,  ic+di  ) - q(m,n,kc-dk,  jc-dj,  ic-di  ));
  Real du_p = 0.5*(q(m,n,kc+2*dk,jc+2*dj,ic+2*di) - q(m,n,kc,     jc,     ic     ));
  // second derivative
  Real dv = 0.25*(du_p - du_m);
  if (dv == 0.0) return 1.0;
  // left/right detectors
  Real vl = du_m - du_c;
  Real vr = du_p - du_c;
  Real alpha_l = -((dv < 0.0) ? fmax(vl, 0.0) : fmin(vl, 0.0))/dv;
  Real alpha_r =  ((dv > 0.0) ? fmax(vr, 0.0) : fmin(vr, 0.0))/dv;
  return fmin(1.0, fmin(alpha_l, alpha_r));
}

//----------------------------------------------------------------------------------------
//! \fn SEDAlpha()
//! \brief Full smooth-extrema detector for variable n at cell (m,k,j,i): per active
//! dimension the 1D detector is minimized over the cell and its two 1D neighbors, then
//! minimized across dimensions.  Total stencil +/-3.
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

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void Hydro::MOODLoop
//! \brief detect/demote/revise loop described in the file header.

template <Hydro_RSolver rsolver_method_>
void Hydro::MOODLoop(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

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
  const bool newtonian = !(pmy_pack->pcoord->is_special_relativistic ||
                           pmy_pack->pcoord->is_general_relativistic);

  auto &u0_ = u0;
  auto &u1_ = u1;
  auto &w0_ = w0;
  auto &utest_ = utest;
  auto &fofc_ = fofc;
  auto &fb_level_ = fb_level;
  auto wl_ = wl3d;
  auto wr_ = wr3d;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  const bool use_sed = mood_sed;
  const Real eps0  = mood_eps0;
  const Real rtol = mood_rtol;
  const Real atol = mood_atol;
  const Real theta = mood_nad_theta;
  const bool nad_energy = mood_nad_energy;
  // Passive scalars carry their own discontinuities, so they are tested as well.  In
  // kinematic/uniform-background advection they are the ONLY variable that varies, and
  // without them the detector sees a constant state and never fires.
  const bool nad_scalars = mood_nad_scalars && (nscalars > 0);
  // candidate variables: scalars are built only when they are checked
  const int ntest = (nad_scalars) ? nvars : nhyd_;
  const int n_fb = n_fb_tiers;
  const int nrevs = mood_max_revs;
  const bool is_ideal_ = eos_.is_ideal;

  const int scale_mode = mood_nad_scale;

  // GLOBAL tolerance scale per NAD variable (index 0 = density, 1 = energy).  A single
  // global scalar keeps the tolerance uniform across cells — decomposition-invariant
  // and free of fence-sitting cells — while making it dynamics-aware:
  //   grange: domain-wide dynamic range of the stage-input state (collapses ~M^2 at
  //           low Mach where a relative |bound| tolerance goes blind);
  //   gdu:    domain-wide max per-stage change |u*-u| of the first (pre-revision)
  //           candidate — additionally timestep-aware, and immune to stratified
  //           backgrounds since the statics cancel in the per-step difference
  //           (computed after the first candidate below).
  // Both reduce over ACTIVE cells only (never ghosts) for layout independence.
  Real gscale0 = 0.0, gscale1 = 0.0, gscale_s = 0.0;
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
    if (scale_mode == 3) {
      // global advective Courant number of the full step (min over blocks of dx)
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
    }
    if (nad_scalars) {
      // Global dynamic range of the passive-scalar CONCENTRATION, minimized/maximized
      // over every scalar component so one uniform tolerance covers them all (same
      // decomposition-invariance argument as the density/energy scales above).
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

  // Detection halo: candidate/detection are evaluated on the interior plus a halo of
  // ghost cells so that both MeshBlocks sharing a face flag and demote it identically
  // from identical ghost data.  The first revision iteration uses a halo of nrevs
  // cells, shrinking by one per iteration (the parallel-MOOD "light cone"): a cell's
  // iteration-r candidate depends on flux revisions one cell away in iteration r-1, so
  // only cells at least one cell inside the previous halo stay consistent with the
  // neighbor block.  The final iteration retains the one-cell halo needed for the
  // shared faces.  The base fluxes are computed over the full nrevs-wide halo
  // (hydro_fluxes.cpp).
  const int hmax = nrevs;
  int il = is-hmax, iu = ie+hmax, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-hmax, ju = je+hmax; }
  if (three_d) { kl = ks-hmax, ku = ke+hmax; }

  // neighborhood extents for DMP/SED (only active dimensions)
  const int ndi = 1;
  const int ndj = (multi_d) ? 1 : 0;
  const int ndk = (three_d) ? 1 : 0;

  // reset per-cell cascade level at the start of every RK stage
  Kokkos::deep_copy(fb_level_, 0);

  // detect / demote / revise loop
  int ndemoted_total = 0;
  for (int rev=1; rev<=nrevs; ++rev) {
    // detection halo for this iteration (shrinks toward 1 at the final iteration)
    const int dh = 1 + (nrevs - rev);
    il = is-dh; iu = ie+dh;
    if (multi_d) { jl = js-dh; ju = je+dh; }
    if (three_d) { kl = ks-dh; ku = ke+dh; }

    // flattened-index helpers for the detection parallel_reduce
    const int ni   = (iu - il + 1);
    const int nji  = (ju - jl + 1)*ni;
    const int nkji = (ku - kl + 1)*nji;
    const int nmkji = nmb*nkji;

    //------------------------------------------------------------------------------------
    // (1) candidate update with the exact RK stage combination, over this iteration's
    // detection halo.
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

    //------------------------------------------------------------------------------------
    // (1b) gdu scale: on the first iteration, the global max per-stage change of the
    // NAD variables over the ACTIVE cells of the pre-revision candidate.  Frozen for
    // the remaining iterations (the first candidate depends only on the base fluxes,
    // so this scalar is identical on every rank/block layout).
    if (scale_mode == 2 && rev == 1) {
      Real gdd = 0.0, gde = 0.0, gds = 0.0;
      const int ni = ie-is+1, nji = (je-js+1)*ni, nkji_a = (ke-ks+1)*nji;
      const int nmkji_a = nmb*nkji_a;
      const bool ideal = is_ideal_;
      const bool newt = newtonian;
      const bool do_s = nad_scalars;
      const int nlo = nhyd_, nhi = nvars - 1;
      Kokkos::parallel_reduce("mood_gdu",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji_a),
      KOKKOS_LAMBDA(const int &idx, Real &dd, Real &de, Real &ds) {
        int m = idx/nkji_a;
        int k = (idx - m*nkji_a)/nji;
        int j = (idx - m*nkji_a - k*nji)/ni;
        int i = (idx - m*nkji_a - k*nji - j*ni) + is;
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
    }

    //------------------------------------------------------------------------------------
    // (2) PAD: flag cells where conversion to primitives would require floors
    // (sets fofc; b0/w0 passed but not used/changed)
    peos->ConsToPrim(utest_, w0_, true, il, iu, jl, ju, kl, ku);

    //------------------------------------------------------------------------------------
    // (3) NaN + NAD detection; combine with PAD; filter out cells that cannot descend
    // further; demote survivors and count them
    int ndemoted = 0, ndemoted_int = 0;
    Kokkos::parallel_reduce("mood_detect",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, int &sum, int &sum_int) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/ni;
      int i = (idx - m*nkji - k*nji - j*ni) + il;
      j += jl;
      k += kl;

      // cells already at the bottom tier cannot be revised further
      const int lv = fb_level_(m,k,j,i);
      if (lv >= n_fb) {
        fofc_(m,k,j,i) = false;
        return;
      }

      // TIER-FLOOR SPLIT: PAD/NaN flags (imminent physical failure — floors would
      // fire, or non-finite data) may descend the full cascade to first-order DC,
      // where maximum dissipation is genuinely wanted.  NAD flags (accuracy trouble:
      // relaxed-DMP violations) demote only to the PLM tier, so no matter how
      // stringent the NAD tolerance is, the scheme is never less accurate than PLM
      // in NAD-flagged regions.
      bool danger = fofc_(m,k,j,i);  // PAD

      // NaN/Inf anywhere in the candidate
      if (!danger) {
        for (int n=0; n<ntest; ++n) {
          if (!isfinite(utest_(m,n,k,j,i))) { danger = true; }
        }
      }

      bool trouble = danger;

      // NAD: relaxed DMP of the candidate vs stage-input state over the neighborhood,
      // with a per-variable smooth-extrema exemption (computed lazily, only for a
      // variable whose DMP check fails — the spd shortcut).
      //
      // Variables checked: mass density and, for an ideal gas, the internal-energy
      // density (Newtonian) / conserved energy (SR/GR).  These two Galilean-invariant
      // scalars are DECOMPOSITION-INVARIANT: the trouble set is bit-identical regardless
      // of how the domain is split into MeshBlocks, because the light-cone halo makes
      // the candidate scalars bit-identical across block boundaries.  Adding the
      // individual velocity components (as spd optionally can via limiting_variables)
      // breaks that: the per-component switch amplifies layout-dependent round-off in
      // the momenta, which the zero-dissipation ppm base then grows, and results
      // stop being bit-reproducible across decompositions.  Density+energy detects both
      // shocks (density jumps) and thermal/contact troubles (energy) while preserving
      // reproducibility.  spd's own default is density only (limiting_variables=[0]);
      // pressure is additionally guarded by the PAD floor check above.
      //
      // NOTE: spd evaluates SED on the candidate field; in block-parallel AthenaK that
      // would require the candidate 3 cells beyond every detection cell (much larger
      // ghost halo or a mid-loop utest exchange), so the stage-input state — whose
      // ghosts are already valid from the regular comms — is used here.
      // NAD is only evaluated for cells still above its PLM floor (level 0).
      if (!trouble && lv < 1) {
        const Real dtst = utest_(m,IDN,k,j,i);
        const int npass = (is_ideal_ && nad_energy) ? 2 : 1;
        for (int pass=0; pass<npass && !trouble; ++pass) {
          const int n = (pass == 0) ? IDN : IEN;
          // candidate value of the checked variable
          Real q_new;
          if (!newtonian) {
            q_new = utest_(m,n,k,j,i);            // conserved density / energy
          } else if (n == IDN) {
            q_new = dtst;                         // density
          } else {                                // internal energy density
            q_new = utest_(m,IEN,k,j,i) - 0.5*(SQR(utest_(m,IM1,k,j,i))
                  + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dtst;
          }
          // DMP bounds from the stage-input state over the neighborhood
          const DvceArray5D<Real> &qref = (newtonian) ? w0_ : u0_;
          Real qmn = qref(m,n,k,j,i), qmx = qmn;
          for (int dk=-ndk; dk<=ndk; ++dk) {
          for (int dj=-ndj; dj<=ndj; ++dj) {
          for (int di=-ndi; di<=ndi; ++di) {
            Real q = qref(m,n,k+dk,j+dj,i+di);
            qmn = fmin(qmn, q);
            qmx = fmax(qmx, q);
          }}}
          // Coarse, scale-aware relaxation: a fraction rtol of the selected GLOBAL
          // scale (grange/gdu; or the spd-style per-bound |m|,|M| in relative mode),
          // floored by round-off (eps0*|bound|) and an absolute atol.  A global scale
          // stays uniform across cells — decomposition-invariant, no fence-sitters.
          Real eps_m, eps_p;
          if (scale_mode == 0) {          // spd-style relative
            eps_m = fmax(rtol*fabs(qmn), atol);
            eps_p = fmax(rtol*fabs(qmx), atol);
          } else if (scale_mode == 1) {   // grange with Mach-softening exponent theta:
            // eps = rtol * G^theta * |bound|^(1-theta); theta=1 -> pure global range
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
          } else {                        // gdu
            const Real gsc = (pass == 0) ? gscale0 : gscale1;
            Real eps = fmax(rtol*gsc, atol);
            eps_m = eps;
            eps_p = eps;
          }
          eps_m = fmax(eps_m, eps0*fabs(qmn));
          eps_p = fmax(eps_p, eps0*fabs(qmx));
          bool nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
          // exempt smooth extrema of this variable from NAD (never from PAD/NaN)
          if (nad && use_sed) {
            nad = (SEDAlpha(qref, m, n, k, j, i, multi_d, three_d) < 1.0);
          }
          if (nad) { trouble = true; }
        }
      }

      // NAD for passive scalars.  The advected CONCENTRATION r = (d*r)/d is the bounded
      // quantity — a tracer cannot leave the range it is advected from — and it is also
      // what the revise pass reconstructs, so the DMP is applied to r rather than to the
      // conserved scalar density (whose bounds would move with the density even where
      // the tracer itself is perfectly smooth).  r is a passively advected scalar, so it
      // is Galilean-invariant and decomposition-invariant on the same grounds as density.
      if (!trouble && lv < 1 && nad_scalars) {
        const Real dtst = utest_(m,IDN,k,j,i);
        for (int n=nhyd_; n<nvars && !trouble; ++n) {
          // a non-positive candidate density is already PAD's business; skip the ratio
          if (!(dtst > 0.0)) break;
          const Real q_new = utest_(m,n,k,j,i)/dtst;
          Real qmn = w0_(m,n,k,j,i), qmx = qmn;
          for (int dk=-ndk; dk<=ndk; ++dk) {
          for (int dj=-ndj; dj<=ndj; ++dj) {
          for (int di=-ndi; di<=ndi; ++di) {
            Real q = w0_(m,n,k+dk,j+dj,i+di);
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

      // demote flagged cells one tier; fofc marks "newly demoted" for the revise pass.
      // sum (incl. the ghost halo) controls the loop exit and must be consistent with
      // the neighbor blocks' decisions; sum_int (interior only) is the diagnostic
      // counter, so it is decomposition-invariant.
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
    // (4) revise fluxes on every face touching a newly-demoted cell, at the tier given
    // by the max cascade level of the two adjacent cells (PLM, or DC at the bottom).
    // Each face kernel writes only its own wl/wr slots (ReconFace), so there are no
    // write races, and unrevised faces keep their base-scheme states.

    // x1 faces
    par_for("mood_rev_x1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, is-dh, ie+1+dh,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (fofc_(m,k,j,i-1) || fofc_(m,k,j,i)) {
        int lv_l = fb_level_(m,k,j,i-1), lv_r = fb_level_(m,k,j,i);
        int tier = (lv_l > lv_r) ? lv_l : lv_r;
        ReconstructionMethod frecon = (tier >= n_fb) ?
            ReconstructionMethod::dc : ReconstructionMethod::plm;
        ReconFace<IVX>(frecon, m, k, j, i, nvars, w0_, wl_, wr_);
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
        for (int n=nhyd_; n<nvars; ++n) {
          if (flx1(m,IDN,k,j,i) >= 0.0) {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wl_(m,n,k,j,i);
          } else {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wr_(m,n,k,j,i);
          }
        }
      }
    });

    // x2 faces
    if (multi_d) {
      par_for("mood_rev_x2", DevExeSpace(), 0, nmb1, kl, ku, js-dh, je+1+dh, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (fofc_(m,k,j-1,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k,j-1,i), lv_r = fb_level_(m,k,j,i);
          int tier = (lv_l > lv_r) ? lv_l : lv_r;
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          ReconFace<IVY>(frecon, m, k, j, i, nvars, w0_, wl_, wr_);
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
          for (int n=nhyd_; n<nvars; ++n) {
            if (flx2(m,IDN,k,j,i) >= 0.0) {
              flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wl_(m,n,k,j,i);
            } else {
              flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wr_(m,n,k,j,i);
            }
          }
        }
      });
    }

    // x3 faces
    if (three_d) {
      par_for("mood_rev_x3", DevExeSpace(), 0, nmb1, ks-dh, ke+1+dh, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (fofc_(m,k-1,j,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k-1,j,i), lv_r = fb_level_(m,k,j,i);
          int tier = (lv_l > lv_r) ? lv_l : lv_r;
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          ReconFace<IVZ>(frecon, m, k, j, i, nvars, w0_, wl_, wr_);
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
          for (int n=nhyd_; n<nvars; ++n) {
            if (flx3(m,IDN,k,j,i) >= 0.0) {
              flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wl_(m,n,k,j,i);
            } else {
              flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wr_(m,n,k,j,i);
            }
          }
        }
      });
    }
  }

  pmy_pack->pmesh->ecounter.nmood += ndemoted_total;

  return;
}

// function definitions for each template parameter
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
