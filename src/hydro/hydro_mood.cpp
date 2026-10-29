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
//! built with the RK stage combination (mood_predictor=stage, default) or with a
//! full-step forward-Euler predictor from the stage input (mood_predictor=euler).
//! Cells where the candidate is inadmissible are detected using:
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
//! The loop runs before SendFlux/RecvFlux (after the immutable candidate exchange
//! when mood_frozen_flux is enabled), so revised fluxes
//! are restricted at fine/coarse AMR boundaries like any other flux (conservative).
//! Detection is evaluated on the interior plus one ghost layer so neighboring
//! MeshBlocks make identical decisions about their shared faces (as FOFC does).
//! Explicit viscous/conduction fluxes: with mood_candidate_diffusion (default) they are
//! in the candidate and every revised face gets its face-local diffusive flux re-added
//! after the replacement; with it off they are added after this loop (detector sees the
//! purely hyperbolic candidate).

#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "reconstruct/recon.hpp"
#include "hydro/rsolvers/solve_face_hyd.hpp"
#include "hydro/well_balance.hpp"
#include "utils/mood_bounds.hpp"
#include "hydro.hpp"

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

  // Trial-state coefficients (mood_predictor): the RK stage combination (default), or a
  // full-step forward Euler from the stage input.  Only the trial state uses them; the
  // accepted RK update is unchanged.
  const bool euler_pred = mood_euler_predictor;
  const Real gam0 = euler_pred ? 1.0 : pdriver->gam0[stage-1];
  const Real gam1 = euler_pred ? 0.0 : pdriver->gam1[stage-1];
  const Real beta_dt = (euler_pred ? 1.0 : pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

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

  // Well-balanced scheme (hydro_wb.cpp): u0/u1/utest and the fluxes are PERTURBATION
  // quantities.  Following the paper (sec. 2.4.4) the NAD/SED test acts on the
  // perturbation (rho', e' = e - e_eq), PAD on the full state, and revised faces go
  // through the face equilibrium exactly like the base fluxes.  The TOLERANCE of the
  // relaxed DMP (relative band, round-off floor) is however taken from
  // the FULL state w_eq + w', exactly as the same run without WB would set it: the band
  // expresses the discretization noise the scheme may add relative to the variable's
  // physical scale, and a band relative to |w'| alone is |w'|/|w| tighter, which demoted
  // a fixed fraction of cell-steps on smooth, well-resolved pulses (2026-10-13).
  const bool wb = well_balanced;
  auto &ueq_ = u_eq;
  auto &weq_ = w_eq;
  const DvceArray5D<Real> wnad_ = wb ? w_pert : w0;   // NAD reference primitives
  auto weqf1 = w_eq_face.x1f, weqf2 = w_eq_face.x2f, weqf3 = w_eq_face.x3f;
  auto feq1 = f_eq.x1f, feq2 = f_eq.x2f, feq3 = f_eq.x3f;
  const Real eps0_eq = mood_eps0_eq;

  const bool use_sed = mood_sed;
  const Real eps0  = mood_eps0;
  const Real rtol = mood_rtol;
  const bool local_rcfl = mood_nad_rcfl;
  const Real step_dt = pmy_pack->pmesh->dt;
  const Real atol = mood_atol;
  const bool nad_energy = mood_nad_energy;
  // Passive scalars carry their own discontinuities, so they are tested as well.  In
  // kinematic/uniform-background advection they are the ONLY variable that varies, and
  // without them the detector sees a constant state and never fires.
  const bool nad_scalars = mood_nad_scalars && (nscalars > 0);
  // candidate variables: scalars are built only when they are checked
  const int ntest = (nad_scalars) ? nvars : nhyd_;
  const int n_fb = n_fb_tiers;
  // mood_detect=false: no detection at all, so the cascade never runs and the scheme
  // reduces to its unlimited base.  Kept as a first-class option because it is the
  // only way to attribute a property to the detector rather than to the base scheme.
  const int nrevs = mood_detect ? mood_max_revs : 0;
  const bool is_ideal_ = eos_.is_ideal;
  const Real gm_ = eos_.gamma, cs_iso_ = eos_.iso_cs;

  // Detection halo: candidate/detection are evaluated on the interior plus a halo of
  // ghost cells so that both MeshBlocks sharing a face flag and demote it identically
  // from identical ghost data.  The first revision iteration uses a halo of nrevs
  // cells, shrinking by one per iteration (the parallel-MOOD "light cone"): a cell's
  // iteration-r candidate depends on flux revisions one cell away in iteration r-1, so
  // only cells at least one cell inside the previous halo stay consistent with the
  // neighbor block.  The final iteration retains the one-cell halo needed for the
  // shared faces. The base fluxes over this halo are exchanged from owners when
  // mood_frozen_flux is enabled, or computed locally on the legacy path.
  const int hmax = nrevs;
  int il = is-hmax, iu = ie+hmax, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-hmax, ju = je+hmax; }
  if (three_d) { kl = ks-hmax, ku = ke+hmax; }

  // neighborhood extents for DMP/SED (only active dimensions)
  const int ndj = (multi_d) ? 1 : 0;
  const int ndk = (three_d) ? 1 : 0;

  // Cells at and around the excised region are pre-set to the BOTTOM tier every stage
  // rather than reset to zero.  Inside the excision the primitives are not inverted at
  // all -- ConsToPrim overwrites them with (dexcise, pexcise, 0) -- so the relaxed-DMP
  // bounds built from those neighbours describe a held-constant state and mean nothing,
  // and letting the detector reason about them produces permanent spurious demotion
  // around the horizon.  FOFC already solves this the same way: it forces first-order
  // LLF fluxes for every cell in excision_flux (see hydro_fofc.cpp), so the region
  // always takes the most robust flux.  Seeding fb_level to n_fb_tiers reproduces that
  // exactly -- the revise pass maps t >= n_fb_tiers to donor-cell reconstruction -- and
  // is what allows mood=true to run with BH excision at all.
  {
    auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
    const bool excising = (pmy_pack->pcoord->is_general_relativistic &&
                           pmy_pack->pcoord->coord_data.bh_excise);
    const int nfb_seed = n_fb;
    if (excising) {
      // bounds from the View itself, so this does not depend on local index names
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

  // face-local re-add of the explicit diffusion fluxes on revised faces (see
  // mood_candidate_diffusion in hydro.cpp); the bulk pass used the same functions
  const bool readd_visc = mood_candidate_diffusion && has_explicit_viscosity;
  const bool readd_cond = mood_candidate_diffusion && has_explicit_conduction;
  const ViscFaceParams vfp =
      (pvisc != nullptr) ? pvisc->FaceParams(eos_) : ViscFaceParams{};
  const Real cond_alpha = (pcond != nullptr) ? pcond->alpha_iso : 0.0;
  const Real cond_gm1 = eos_.gamma - 1.0;

  // debug/validation: forced demotion of the region x1 < x1max (see hydro.cpp)
  const bool dbg_demote = mood_debug_demote;
  const Real dbg_x1max = mood_debug_demote_x1max;

  //--------------------------------------------------------------------------------------
  // Relaxed-DMP bounds of the NAD reference quantities over the 3^d-cell neighbourhood,
  // once per stage (the reference state is immutable through the revision loop):
  // per-cell values gathered into mood_val, then separable 3-point sweeps over all
  // slots at once (utils/mood_bounds.hpp; bmn/bmx name the arrays holding the result)
  // -- bit-identical to the direct scan the detection kernel used to repeat per
  // variable, per cell, per revision.
  // Slots: [0,nhe) density, energy; WB: [nhe,2*nhe) the same in FULL state (the
  // tolerance scale, see the note above); then the scalar concentrations.
  const int nhe = (is_ideal_ && nad_energy) ? 2 : 1;
  const int s_full = wb ? nhe : 0;
  const int s_sc = nhe*(wb ? 2 : 1);
  auto bmn = mood_qmn;
  auto bmx = mood_qmx;
  if (nrevs > 0) {
    auto val = mood_val;
    const int nsc = nad_scalars ? (nvars - nhyd_) : 0;
    // values one cell beyond the first-revision detection halo (il..ku at hmax)
    par_for("mood_bnd_val", DevExeSpace(), 0, nmb1, kl-ndk, ku+ndk, jl-ndj, ju+ndj,
            il-1, iu+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int p=0; p<nhe; ++p) {
        const int n = (p == 0) ? IDN : IEN;
        const Real q = (newtonian) ? wnad_(m,n,k,j,i) : u0_(m,n,k,j,i);
        val(m,p,k,j,i) = q;
        if (wb) { val(m,nhe+p,k,j,i) = q + weq_(m,n,k,j,i); }
      }
      for (int sc=0; sc<nsc; ++sc) { val(m,s_sc+sc,k,j,i) = w0_(m,nhyd_+sc,k,j,i); }
    });
    MoodNeighborhoodBounds(nmb1, s_sc+nsc, il, iu, jl, ju, kl, ku, multi_d, three_d,
                           mood_val, mood_qmn, mood_qmx, mood_tmx, bmn, bmx);
  }

  // detect / demote / revise loop
  int ndemoted_total = 0;
  for (int rev=1; rev<=nrevs; ++rev) {
    // detection halo for this iteration (shrinks toward 1 at the final iteration)
    const int dh = 1 + (nrevs - rev);
    il = is-dh; iu = ie+dh;
    if (multi_d) { jl = js-dh; ju = je+dh; }
    if (three_d) { kl = ks-dh; ku = ke+dh; }

    //------------------------------------------------------------------------------------
    // (1) candidate update (mood_predictor: the RK stage combination, or forward Euler
    // over the full timestep), over this iteration's detection halo.
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
    // (2) PAD: flag cells where conversion to primitives would require floors
    // (sets fofc; b0/w0 passed but not used/changed)
    if (wb) {
      peos->ConsToPrimWB(utest_, u_eq, w0_, true, il, iu, jl, ju, kl, ku);
    } else {
      peos->ConsToPrim(utest_, w0_, true, il, iu, jl, ju, kl, ku);
    }

    //------------------------------------------------------------------------------------
    // (3) NaN + NAD detection; combine with PAD; filter out cells that cannot descend
    // further; demote survivors and count them.  A plain par_for with atomic counters
    // (incremented only by flagged cells) instead of a multi-value parallel_reduce: the
    // counts are exact integers either way, the reduction epilogue is gone, and the few
    // flagged cells make atomic contention negligible.  cnt(0) = demoted incl. the halo
    // (loop exit), cnt(1) = demoted interior cells (event counter).
    auto cnt_ = mood_cnt;
    Kokkos::deep_copy(cnt_, 0);
    par_for("mood_detect", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

      // rcfl uses only this cell's immutable stage-input state and local spacing.
      // Retain the max-component/min-spacing convention and full-step dt of rcfl;
      // no pack/domain extrema or additional stencil are needed.  WB retains its
      // signal-speed contribution so a static equilibrium has a nonzero band.
      Real cell_rtol = rtol;
      if (local_rcfl && !danger && lv < 1) {
        Real speed = fmax(fabs(w0_(m,IVX,k,j,i)),
                     fmax(fabs(w0_(m,IVY,k,j,i)), fabs(w0_(m,IVZ,k,j,i))));
        if (wb) {
          const Real d = w0_(m,IDN,k,j,i);
          const Real cs2 = is_ideal_ ? gm_*(gm_-1.0)*w0_(m,IEN,k,j,i)/d
                                     : SQR(cs_iso_);
          speed += sqrt(fmax(cs2, 0.0));
        }
        Real dx = size_.d_view(m).dx1;
        if (multi_d) dx = fmin(dx, size_.d_view(m).dx2);
        if (three_d) dx = fmin(dx, size_.d_view(m).dx3);
        cell_rtol *= fmin(1.0, step_dt*speed/dx);
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
            q_new = dtst;                         // density (WB: rho' = U'_rho)
          } else if (wb) {                        // WB: e' = e(U'+U_eq) - e_eq
            const Real dfull = dtst + ueq_(m,IDN,k,j,i);
            const Real efull = utest_(m,IEN,k,j,i) + ueq_(m,IEN,k,j,i);
            q_new = (efull - 0.5*(SQR(utest_(m,IM1,k,j,i))
                  + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dfull)
                  - weq_(m,IEN,k,j,i);
          } else {                                // internal energy density
            q_new = utest_(m,IEN,k,j,i) - 0.5*(SQR(utest_(m,IM1,k,j,i))
                  + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dtst;
          }
          // DMP bounds from the stage-input state over the neighborhood, precomputed
          // per stage (mood_bnd_val above).  WB: DMP bounds on the perturbation; the
          // tolerance scale smn/smx from the full-state bounds (note at the top).
          const DvceArray5D<Real> &qref = (newtonian) ? wnad_ : u0_;   // for SED
          const Real qmn = bmn(m,pass,k,j,i), qmx = bmx(m,pass,k,j,i);
          const Real smn = wb ? bmn(m,s_full+pass,k,j,i) : qmn;
          const Real smx = wb ? bmx(m,s_full+pass,k,j,i) : qmx;
          // Local relative relaxation of the neighboring lower/upper bounds.
          Real eps_m = fmax(cell_rtol*fabs(smn), atol);
          Real eps_p = fmax(cell_rtol*fabs(smx), atol);
          eps_m = fmax(eps_m, eps0*fabs(smn));
          eps_p = fmax(eps_p, eps0*fabs(smx));
          if (wb) {
            // Optional extra round-off floor tied to the EQUILIBRIUM (mood_eps0_eq): the
            // perturbation carries the cancellation error of F - F_eq (~eps*|F_eq|).
            // With the full-state scales above it is normally redundant (eps0*|smn|
            // already contains |q_eq|); kept for runs that set it explicitly.
            const Real eq_floor = eps0_eq*fabs(weq_(m,n,k,j,i));
            eps_m = fmax(eps_m, eq_floor);
            eps_p = fmax(eps_p, eq_floor);
          }
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
        const Real dtst = wb ? (utest_(m,IDN,k,j,i) + ueq_(m,IDN,k,j,i))
                             : utest_(m,IDN,k,j,i);
        for (int n=nhyd_; n<nvars && !trouble; ++n) {
          // a non-positive candidate density is already PAD's business; skip the ratio
          if (!(dtst > 0.0)) break;
          const Real q_new = utest_(m,n,k,j,i)/dtst;
          const int sq = s_sc + (n - nhyd_);   // precomputed bounds (mood_bnd_val)
          const Real qmn = bmn(m,sq,k,j,i), qmx = bmx(m,sq,k,j,i);
          Real eps_m = fmax(cell_rtol*fabs(qmn), atol);
          Real eps_p = fmax(cell_rtol*fabs(qmx), atol);
          eps_m = fmax(eps_m, eps0*fabs(qmn));
          eps_p = fmax(eps_p, eps0*fabs(qmx));
          bool nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
          if (nad && use_sed) {
            nad = (SEDAlpha(w0_, m, n, k, j, i, multi_d, three_d) < 1.0);
          }
          if (nad) { trouble = true; }
        }
      }

      // debug/validation: force demotion of a chosen region (hydro.cpp)
      if (dbg_demote) {
        Real x1v = CellCenterX(i-is, indcs_.nx1, size_.d_view(m).x1min,
                               size_.d_view(m).x1max);
        if (x1v < dbg_x1max) { trouble = true; }
      }

      // demote flagged cells one tier; fofc marks "newly demoted" for the revise pass.
      // sum (incl. the ghost halo) controls the loop exit and must be consistent with
      // the neighbor blocks' decisions; sum_int (interior only) is the diagnostic
      // counter, so it is decomposition-invariant.
      fofc_(m,k,j,i) = trouble;
      if (trouble) {
        fb_level_(m,k,j,i) += 1;
        Kokkos::atomic_inc(&cnt_(0));
        if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
          Kokkos::atomic_inc(&cnt_(1));
        }
      }
    });
    auto cnt_h = Kokkos::create_mirror_view(cnt_);
    Kokkos::deep_copy(cnt_h, cnt_);
    const int ndemoted = cnt_h(0), ndemoted_int = cnt_h(1);

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
        ReconFace<IVX>(frecon, m, k, j, i, nvars, wnad_, wl_, wr_);
        if (wb) {
          wellbalance::AddFaceEquilibrium(eos_, false, nhyd_, m, k, j, i, weqf1,
                                          wl_, wr_);
        }
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
        if (wb) {
          // perturbation flux; scalars upwinded on the FULL mass flux
          for (int n=0; n<nhyd_; ++n) {flx1(m,n,k,j,i) -= feq1(m,n,k,j,i);}
          const Real fd = flx1(m,IDN,k,j,i) + feq1(m,IDN,k,j,i);
          for (int n=nhyd_; n<nvars; ++n) {
            flx1(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
          }
        } else {
          for (int n=nhyd_; n<nvars; ++n) {
            if (flx1(m,IDN,k,j,i) >= 0.0) {
              flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wl_(m,n,k,j,i);
            } else {
              flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wr_(m,n,k,j,i);
            }
          }
        }
        // face-local re-add of the explicit diffusion fluxes this revision overwrote
        if (readd_visc) {
          const ViscFaceFlux vf = ViscousFluxX1(vfp, w0_, size_.d_view(m), m, k, j, i);
          flx1(m,IM1,k,j,i) -= vf.nud*vf.fvx;
          flx1(m,IM2,k,j,i) -= vf.nud*vf.fvy;
          flx1(m,IM3,k,j,i) -= vf.nud*vf.fvz;
          if (vfp.is_ideal) {flx1(m,IEN,k,j,i) -= 0.5*vf.nud*vf.ve;}
        }
        if (readd_cond) {
          flx1(m,IEN,k,j,i) -= HeatFluxIsoX1(cond_alpha, cond_gm1, w0_, size_.d_view(m),
                                             m, k, j, i);
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
          ReconFace<IVY>(frecon, m, k, j, i, nvars, wnad_, wl_, wr_);
          if (wb) {
            wellbalance::AddFaceEquilibrium(eos_, false, nhyd_, m, k, j, i, weqf2,
                                            wl_, wr_);
          }
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
          if (wb) {
            // perturbation flux; scalars upwinded on the FULL mass flux
            for (int n=0; n<nhyd_; ++n) {flx2(m,n,k,j,i) -= feq2(m,n,k,j,i);}
            const Real fd = flx2(m,IDN,k,j,i) + feq2(m,IDN,k,j,i);
            for (int n=nhyd_; n<nvars; ++n) {
              flx2(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
            }
          } else {
            for (int n=nhyd_; n<nvars; ++n) {
              if (flx2(m,IDN,k,j,i) >= 0.0) {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wl_(m,n,k,j,i);
              } else {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wr_(m,n,k,j,i);
              }
            }
          }
          // face-local re-add of the explicit diffusion fluxes this revision overwrote
          if (readd_visc) {
            const ViscFaceFlux vf = ViscousFluxX2(vfp, w0_, size_.d_view(m), m, k, j, i);
            flx2(m,IM1,k,j,i) -= vf.nud*vf.fvx;
            flx2(m,IM2,k,j,i) -= vf.nud*vf.fvy;
            flx2(m,IM3,k,j,i) -= vf.nud*vf.fvz;
            if (vfp.is_ideal) {flx2(m,IEN,k,j,i) -= 0.5*vf.nud*vf.ve;}
          }
          if (readd_cond) {
            flx2(m,IEN,k,j,i) -= HeatFluxIsoX2(cond_alpha, cond_gm1, w0_, size_.d_view(m),
                                               m, k, j, i);
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
          ReconFace<IVZ>(frecon, m, k, j, i, nvars, wnad_, wl_, wr_);
          if (wb) {
            wellbalance::AddFaceEquilibrium(eos_, false, nhyd_, m, k, j, i, weqf3,
                                            wl_, wr_);
          }
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
          if (wb) {
            // perturbation flux; scalars upwinded on the FULL mass flux
            for (int n=0; n<nhyd_; ++n) {flx3(m,n,k,j,i) -= feq3(m,n,k,j,i);}
            const Real fd = flx3(m,IDN,k,j,i) + feq3(m,IDN,k,j,i);
            for (int n=nhyd_; n<nvars; ++n) {
              flx3(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
            }
          } else {
            for (int n=nhyd_; n<nvars; ++n) {
              if (flx3(m,IDN,k,j,i) >= 0.0) {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wl_(m,n,k,j,i);
              } else {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wr_(m,n,k,j,i);
              }
            }
          }
          // face-local re-add of the explicit diffusion fluxes this revision overwrote
          if (readd_visc) {
            const ViscFaceFlux vf = ViscousFluxX3(vfp, w0_, size_.d_view(m), m, k, j, i);
            flx3(m,IM1,k,j,i) -= vf.nud*vf.fvx;
            flx3(m,IM2,k,j,i) -= vf.nud*vf.fvy;
            flx3(m,IM3,k,j,i) -= vf.nud*vf.fvz;
            if (vfp.is_ideal) {flx3(m,IEN,k,j,i) -= 0.5*vf.nud*vf.ve;}
          }
          if (readd_cond) {
            flx3(m,IEN,k,j,i) -= HeatFluxIsoX3(cond_alpha, cond_gm1, w0_, size_.d_view(m),
                                               m, k, j, i);
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
template void Hydro::MOODLoop<Hydro_RSolver::lhllc>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::roe>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::llf_sr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hllc_sr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::llf_gr>(Driver *pdriver, int stage);
template void Hydro::MOODLoop<Hydro_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace hydro
