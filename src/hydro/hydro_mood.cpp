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
//! The loop runs before SendFlux/RecvFlux (after the immutable candidate exchange
//! when mood_frozen_flux is enabled), so revised fluxes
//! are restricted at fine/coarse AMR boundaries like any other flux (conservative).
//! Detection is evaluated on the interior plus one ghost layer so neighboring
//! MeshBlocks make identical decisions about their shared faces (as FOFC does).
//! Like FOFC, revised faces lose any viscous/conduction flux contribution.

#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "hydro/rsolvers/solve_face_hyd.hpp"
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
  auto &touched_ = mood_touched;
  auto wl_ = wl3d;
  auto wr_ = wr3d;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  const bool use_sed = mood_sed;
  const Real eps0  = mood_eps0;
  const Real rtol = mood_rtol;
  const bool local_rcfl = mood_nad_rcfl;
  const Real step_dt = pmy_pack->pmesh->dt;
  const Real atol = mood_atol;
  const bool nad_energy = mood_nad_energy;
  const bool nad_von = (mood_nad_v > 0);      // velocity in NAD (Newtonian only)
  const bool nad_vmag = (mood_nad_v == 1);    // |v| (else components)
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
  const int ndi = 1;
  const int ndj = (multi_d) ? 1 : 0;
  const int ndk = (three_d) ? 1 : 0;

  // Frozen fluxes: ApplyFallback calls this once per detection and once per revision.
  // fb_level is reset when the candidate is sent (hydro_mood_comm.cpp), detection runs
  // on owned cells only, and the neighbours' cascade levels are exchanged in between.
  const bool frozen = mood_frozen_flux;
  if (!frozen) Kokkos::deep_copy(fb_level_, 0);
  // the check after the last allowed revision must not change the levels: they describe
  // the fluxes that will actually be committed
  const bool may_demote = !frozen || mood_revision < nrevs;

  // detect / demote / revise loop
  int ndemoted_total = 0;
  const int first_rev = frozen ? mood_revision + 1 : 1;
  const int last_rev = frozen ? first_rev : nrevs;
  for (int rev=first_rev; rev<=last_rev; ++rev) {
    // detection halo for this iteration (shrinks toward 1 at the final iteration);
    // none with frozen fluxes, whose ghost levels come from their owners
    const int dh = frozen ? 0 : 1 + (nrevs - rev);
    il = is-dh; iu = ie+dh;
    if (multi_d) { jl = js-dh; ju = je+dh; }
    if (three_d) { kl = ks-dh; ku = ke+dh; }

    // NAD on the first check only unless mood_nad_checks = all; later checks of a frozen
    // stage re-test only the cells next to a face revised since the previous check
    const bool nad_now = mood_nad_all || (rev == 1);
    const bool masked = frozen && mood_touched_only && (rev > 1) && newtonian;
    const bool mark = frozen && mood_touched_only;

    if (!(frozen && mood_revise)) {

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
        if (masked && !touched_(m,k,j,i)) return;
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
      if (masked) { peos->c2p_mask = touched_; peos->use_c2p_mask = true; }
      peos->ConsToPrim(utest_, w0_, true, il, iu, jl, ju, kl, ku);
      peos->use_c2p_mask = false;

      //------------------------------------------------------------------------------------
      // (3) NaN + NAD detection; combine with PAD; filter out cells that cannot descend
      // further; demote survivors and count them
      int ndemoted = 0, ndemoted_int = 0, nviolations = 0, npad = 0, nstuck = 0;
      int nlpad = 0, nlnad = 0;
      Kokkos::parallel_reduce("mood_detect",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, int &sum, int &sum_int, int &sum_bad, int &sum_pad,
                    int &sum_stuck, int &sum_lpad, int &sum_lnad) {
        int m = (idx)/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/ni;
        int i = (idx - m*nkji - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // untouched cells keep their previous verdict (their candidate is unchanged)
        if (masked && !touched_(m,k,j,i)) return;
        // cells already at the bottom tier cannot be revised further
        const int lv = fb_level_(m,k,j,i);
        const bool owned = (i >= is && i <= ie && j >= js && j <= je &&
                            k >= ks && k <= ke);
        if (lv >= n_fb && !frozen) {
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
        // DC cannot descend further, but a failed PAD/finite test still prevents
        // convergence (frozen fluxes): keep it in the consensus without a new tier
        if (lv >= n_fb) {
          if (danger && owned) { sum_bad++; sum_stuck++; sum_lpad++; }
          fofc_(m,k,j,i) = false;
          return;
        }

        // rcfl uses only this cell's immutable stage-input state and local spacing.
        // Retain the max-component/min-spacing convention and full-step dt of rcfl;
        // no pack/domain extrema or additional stencil are needed.
        Real cell_rtol = rtol;
        if (local_rcfl && !danger && lv < 1) {
          Real speed = fmax(fabs(w0_(m,IVX,k,j,i)),
                       fmax(fabs(w0_(m,IVY,k,j,i)), fabs(w0_(m,IVZ,k,j,i))));
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
        if (!trouble && lv < 1 && nad_now) {
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
            // Local relative relaxation of the neighboring lower/upper bounds.
            Real eps_m = fmax(cell_rtol*fabs(qmn), atol);
            Real eps_p = fmax(cell_rtol*fabs(qmx), atol);
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

        // NAD on the velocity (opt-in mood_nad_v), as in the MHD detector: the candidate
        // velocity mom/rho of the trial state against the stage-input velocities over the
        // neighbourhood, with the same tolerance as density.  Components exempt smooth
        // extrema (SED); |v| has no precomputed array, so it is never exempted.
        if (!trouble && lv < 1 && nad_now && nad_von) {
          const Real dtst = utest_(m,IDN,k,j,i);
          const int nvpass = nad_vmag ? 1 : 3;
          for (int vidx=0; vidx<nvpass && !trouble; ++vidx) {
            Real q_new, qmn, qmx;
            if (nad_vmag) {
              q_new = sqrt(SQR(utest_(m,IM1,k,j,i)) + SQR(utest_(m,IM2,k,j,i))
                         + SQR(utest_(m,IM3,k,j,i))) / dtst;
              qmn = sqrt(SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i))
                       + SQR(w0_(m,IVZ,k,j,i)));
              qmx = qmn;
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
              q_new = utest_(m,IM1+vidx,k,j,i) / dtst;
              qmn = w0_(m,n,k,j,i); qmx = qmn;
              for (int dk=-ndk; dk<=ndk; ++dk) {
              for (int dj=-ndj; dj<=ndj; ++dj) {
              for (int di=-ndi; di<=ndi; ++di) {
                Real q = w0_(m,n,k+dk,j+dj,i+di);
                qmn = fmin(qmn, q);
                qmx = fmax(qmx, q);
              }}}
            }
            Real eps_m = fmax(cell_rtol*fabs(qmn), atol);
            Real eps_p = fmax(cell_rtol*fabs(qmx), atol);
            eps_m = fmax(eps_m, eps0*fabs(qmn));
            eps_p = fmax(eps_p, eps0*fabs(qmx));
            bool nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
            if (nad && use_sed && !nad_vmag) {
              nad = (SEDAlpha(w0_, m, IVX+vidx, k, j, i, multi_d, three_d) < 1.0);
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
        if (!trouble && lv < 1 && nad_now && nad_scalars) {
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

        // demote flagged cells one tier; fofc marks "newly demoted" for the revise pass.
        // sum (incl. the ghost halo) controls the loop exit and must be consistent with
        // the neighbor blocks' decisions; sum_int (interior only) is the diagnostic
        // counter, so it is decomposition-invariant.
        if (trouble && owned) sum_bad++;
        if (danger && may_demote && owned) sum_pad++;
        if (trouble && !may_demote && owned) { if (danger) sum_lpad++; else sum_lnad++; }
        fofc_(m,k,j,i) = trouble && may_demote;
        if (trouble && may_demote) {
          fb_level_(m,k,j,i) += 1;
          sum++;
          if (owned) {
            sum_int++;
          }
        }
      }, Kokkos::Sum<int>(ndemoted), Kokkos::Sum<int>(ndemoted_int),
         Kokkos::Sum<int>(nviolations), Kokkos::Sum<int>(npad), Kokkos::Sum<int>(nstuck),
         Kokkos::Sum<int>(nlpad), Kokkos::Sum<int>(nlnad));

      if (frozen) {
        mood_nviolations = nviolations;
        mood_diag_counts[0] = ndemoted_int - npad;   // new NAD demotions (to PLM)
        mood_diag_counts[1] = npad;                  // new PAD/NaN demotions
        mood_diag_counts[2] = nstuck;                // PAD/NaN failures already at DC
        mood_diag_counts[3] = nviolations;           // all inadmissible owned cells
        mood_nchanged = ndemoted_int;
        mood_left_pad = nlpad;
        mood_left_nad = nlnad;
        ndemoted_total += ndemoted_int;
        // the next revision marks the cells whose candidate it changes
        if (mark) Kokkos::deep_copy(touched_, false);
        break;  // exchange the owners' levels before revising shared faces
      }
      if (ndemoted == 0) break;
      ndemoted_total += ndemoted_int;
    }  // detection

    //------------------------------------------------------------------------------------
    // (4) revise fluxes on every face touching a newly-demoted cell, at the tier given
    // by the max cascade level of the two adjacent cells (PLM, or DC at the bottom).
    // Each face kernel writes only its own wl/wr slots (ReconFace), so there are no
    // write races, and unrevised faces keep their base-scheme states.

    // x1 faces
    par_for("mood_rev_x1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, is-dh, ie+1+dh,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (fofc_(m,k,j,i-1) || fofc_(m,k,j,i)) {
        if (mark) { touched_(m,k,j,i-1) = true; touched_(m,k,j,i) = true; }
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
          if (mark) { touched_(m,k,j-1,i) = true; touched_(m,k,j,i) = true; }
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
          if (mark) { touched_(m,k-1,j,i) = true; touched_(m,k,j,i) = true; }
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
