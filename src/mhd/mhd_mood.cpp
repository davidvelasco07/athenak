//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_mood.cpp
//! \brief MOOD a-posteriori fallback for MHD.  Mirrors the hydro implementation
//! (src/hydro/hydro_mood.cpp — see its header for the full algorithm description and
//! the light-cone halo rationale), extended for the staggered magnetic field:
//!
//! - The candidate update consists of the conserved hydro variables (flux divergence)
//!   AND a candidate cell-averaged magnetic field bcctest, built from the stage
//!   combination of the cell-averaged fields plus finite differences of the
//!   face-centered electric fields (the FOFC estimate, mhd_fofc.cpp);
//!   mood_predictor=euler uses a full-step forward-Euler update instead.
//! - Detection adds the candidate B to the NAD check: either the rotation-invariant
//!   magnitude |B| (mood_nad_b=mag) or the three components (mood_nad_b=comps),
//!   with DMP bounds from the stage-input cell-centered field.
//! - Face revision recomputes, at the demoted tier with the same Riemann solver, the
//!   conserved fluxes AND the face-centered E-fields AND (when UCT is selected) the
//!   UCT face coefficients, so the demotion enters the corner-EMF composition through
//!   exactly the same data path as the base scheme.
//! - Edge correction is then either implicit (mood_edge=blend: the corner composition
//!   blends revised and base face data, the FOFC pattern) or explicit (mood_edge=flag:
//!   mhd_corner_e.cpp additionally drops the edge reconstruction to plm/dc at any edge
//!   whose adjacent cells are demoted).
//!
//! mood_frozen_emf exchanges the initial candidate and caches PLM/DC EMFs, then replaces
//! only edges adjacent to demoted cells. Untouched high-order edges are retained.
//! This is the default for supported uniform periodic NR/SR/GR cases. Edge selection
//! has one-cell reach; Newtonian PAD additionally needs the production B interpolant.
//! See
//! validation/frozen_emf_halo/README.md for the measured bounds and limitations.
//!
//! Cross-block consistency (legacy recomposition path): shared-edge EMFs are
//! synchronized between MeshBlocks by
//! SendE/RecvE, so div(B) and conservation hold regardless; the light-cone halo is
//! sized (mhd.cpp) so the COMPOSED edge values are also decomposition-invariant —
//! the final-iteration halo equals the transverse reach of the edge composition.

#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "eos/mhd_bcc.hpp"
#include "reconstruct/recon.hpp"
#include "mhd/rsolvers/solve_face_mhd.hpp"
#include "mhd/mhd_wb.hpp"
#include "utils/mood_bounds.hpp"
#include "mhd.hpp"

namespace mhd {

namespace {

//----------------------------------------------------------------------------------------
//! \fn SEDAlpha1D() / SEDAlpha()
//! \brief Vilar-type smooth-extrema detector; identical to the hydro version
//! (hydro_mood.cpp).  The 4D overloads operate on a scalar field (|B|).
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

KOKKOS_INLINE_FUNCTION
Real SEDAlpha1D4(const DvceArray4D<Real> &q, const int m,
                 const int k, const int j, const int i,
                 const int dk, const int dj, const int di, const int o) {
  const int kc = k + o*dk, jc = j + o*dj, ic = i + o*di;
  Real du_m = 0.5*(q(m,kc,     jc,     ic     ) - q(m,kc-2*dk,jc-2*dj,ic-2*di));
  Real du_c = 0.5*(q(m,kc+dk,  jc+dj,  ic+di  ) - q(m,kc-dk,  jc-dj,  ic-di  ));
  Real du_p = 0.5*(q(m,kc+2*dk,jc+2*dj,ic+2*di) - q(m,kc,     jc,     ic     ));
  Real dv = 0.25*(du_p - du_m);
  if (dv == 0.0) return 1.0;
  Real vl = du_m - du_c;
  Real vr = du_p - du_c;
  Real alpha_l = -((dv < 0.0) ? fmax(vl, 0.0) : fmin(vl, 0.0))/dv;
  Real alpha_r =  ((dv > 0.0) ? fmax(vr, 0.0) : fmin(vr, 0.0))/dv;
  return fmin(1.0, fmin(alpha_l, alpha_r));
}

KOKKOS_INLINE_FUNCTION
Real SEDAlpha4(const DvceArray4D<Real> &q, const int m,
               const int k, const int j, const int i,
               const bool multi_d, const bool three_d) {
  Real alpha = 1.0;
  for (int o=-1; o<=1; ++o) {
    alpha = fmin(alpha, SEDAlpha1D4(q, m, k, j, i, 0, 0, 1, o));
  }
  if (multi_d) {
    for (int o=-1; o<=1; ++o) {
      alpha = fmin(alpha, SEDAlpha1D4(q, m, k, j, i, 0, 1, 0, o));
    }
  }
  if (three_d) {
    for (int o=-1; o<=1; ++o) {
      alpha = fmin(alpha, SEDAlpha1D4(q, m, k, j, i, 1, 0, 0, o));
    }
  }
  return alpha;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void MHD::MOODLoop
//! \brief detect/demote/revise loop described in the file header.

//----------------------------------------------------------------------------------------
//! \struct MoodNadPassCtx
//! \brief One relaxed-DMP (NAD) test of the MHD MOOD detector: pass `pass` of cell
//! (m,k,j,i).  Shared by the two launch forms of mmood_detect_pass: one thread per
//! (cell, pass) on GPU backends, a per-cell loop over the passes (with early exit) on CPU
//! backends, so the test logic exists once.  Passes: 0 = density, 1 = internal energy
//! (ideal gas), then B (|B| in mag mode, three components in comps mode), then v (|v| or
//! three components), then the passive-scalar concentrations.  Bounds from the
//! stage-input state, precomputed per stage (mmood_bnd_val); with WB the bounds are on
//! the perturbation and the tolerance scale smn/smx on the full state (hydro_mood.cpp).

// per-cell detection status (MoodNadPassCtx::Prep)
enum MoodCellStatus : int { ST_SKIP = 0, ST_DANGER = 1, ST_NONAD = 2, ST_TEST = 3 };

struct MoodNadPassCtx {
  DvceArray5D<Real> utest_, bcctest_, ueq_, weq_, bcceq_, w0_, bcc0_, wnad_, u0_, bnad_;
  DvceArray5D<Real> bmn, bmx;
  DvceArray4D<Real> btpad1, btpad2, btpad3, beq1, beq2, beq3, bmag_;
  DvceArray4D<bool> fofc_, touched_;
  DvceArray4D<int> fb_level_;
  DvceArray1D<int> cnt_;
  DvceArray1D<RegionSize> mbsize_;
  bool wb, ho_pad, fv_bavg, multi_d, three_d, newtonian, nad_bmag, nad_vmag, is_ideal_;
  bool use_sed, masked, frozen, may_demote, local_rcfl;
  Real gm_, cs_iso_, atol, eps0, eps0_eq, rtol, step_dt;
  int npass_he, npass_hb, npass, npass_tot, nmhd_, nhe, s_b, s_v, s_sc, ntest, n_fb;
  int is, ie, js, je, ks, ke;

  KOKKOS_INLINE_FUNCTION
  bool Owned(const int k, const int j, const int i) const {
    return (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke);
  }

  //! (a) per cell: masked/tier filter, PAD + NaN danger, rcfl band scale.  Returns the
  //! cell status; cells that leave the cascade here get fofc=false (and, on the frozen
  //! path, a DC-failure count).  NAD is only evaluated for cells still above the PLM
  //! floor (level 0): ST_TEST.
  KOKKOS_INLINE_FUNCTION
  int Prep(const int m, const int k, const int j, const int i, Real &cell_rtol) const {
    cell_rtol = rtol;
    if (masked && !touched_(m,k,j,i)) return ST_SKIP;
    const int lv = fb_level_(m,k,j,i);
    if (lv >= n_fb && !frozen) {
      fofc_(m,k,j,i) = false;
      return ST_SKIP;
    }
    bool danger = fofc_(m,k,j,i);  // PAD
    if (!danger) {
      for (int n=0; n<ntest; ++n) {
        if (!isfinite(utest_(m,n,k,j,i))) { danger = true; }
      }
      if (!isfinite(bcctest_(m,IBX,k,j,i)) || !isfinite(bcctest_(m,IBY,k,j,i)) ||
          !isfinite(bcctest_(m,IBZ,k,j,i))) { danger = true; }
    }
    // DC cannot descend further. Record physical failure separately so it
    // cannot request another correction with an unchanged cascade map.
    if (lv >= n_fb) {
      if (frozen && danger && Owned(k,j,i)) Kokkos::atomic_inc(&cnt_(4));
      fofc_(m,k,j,i) = false;
      return ST_SKIP;
    }
    // rcfl uses only this cell's immutable stage-input state and local spacing.
    // Retain the max-component/min-spacing convention and full-step dt of rcfl;
    // no pack/domain extrema or additional stencil are needed.  WB retains its
    // signal-speed contribution so a static equilibrium has a nonzero band.
    if (local_rcfl && !danger && lv < 1) {
      Real speed = fmax(fabs(w0_(m,IVX,k,j,i)),
                   fmax(fabs(w0_(m,IVY,k,j,i)), fabs(w0_(m,IVZ,k,j,i))));
      if (wb) {
        const Real d = w0_(m,IDN,k,j,i);
        const Real cs2 = is_ideal_ ? gm_*(gm_-1.0)*w0_(m,IEN,k,j,i)/d
                                   : SQR(cs_iso_);
        const Real va2 = (SQR(bcc0_(m,IBX,k,j,i)) + SQR(bcc0_(m,IBY,k,j,i))
                        + SQR(bcc0_(m,IBZ,k,j,i)))/d;
        speed += sqrt(fmax(cs2 + va2, 0.0));
      }
      Real dx = mbsize_(m).dx1;
      if (multi_d) dx = fmin(dx, mbsize_(m).dx2);
      if (three_d) dx = fmin(dx, mbsize_(m).dx3);
      cell_rtol *= fmin(1.0, step_dt*speed/dx);
    }
    return danger ? ST_DANGER : ((lv < 1) ? ST_TEST : ST_NONAD);
  }

  //! (c) per cell: combine, demote, count.  The check after the last allowed
  //! correction must not change the levels: those levels describe the fluxes/EMFs that
  //! will actually be committed.  All surviving trouble below DC requests a genuine
  //! level increase.  Requests are counted independently of may_demote (frozen path):
  //! the terminal check must distinguish pending corrections from a stable map
  //! without mutating it.  cnt(0) = demoted incl. the halo (loop exit), cnt(1) =
  //! demoted interior cells (event counter); cnt(2..4) NAD/PAD requests and DC
  //! failures, consumed only by the frozen-EMF consensus and diagnostics.
  KOKKOS_INLINE_FUNCTION
  void Combine(const int m, const int k, const int j, const int i, const int st,
               const bool nadflag) const {
    const bool danger = (st == ST_DANGER);
    const bool trouble = danger || nadflag;
    const bool owned = Owned(k,j,i);
    if (frozen && trouble && owned) {
      Kokkos::atomic_inc(&cnt_(danger ? 3 : 2));
    }
    fofc_(m,k,j,i) = trouble && may_demote;
    if (trouble && may_demote) {
      fb_level_(m,k,j,i) += 1;
      Kokkos::atomic_inc(&cnt_(0));
      if (owned) Kokkos::atomic_inc(&cnt_(1));
    }
  }

  //! full candidate density for a cell (WB: U' + U_eq), the divisor of the velocity,
  //! energy and scalar tests; computed once per cell by the callers
  KOKKOS_INLINE_FUNCTION
  Real CandidateDensity(const int m, const int k, const int j, const int i) const {
    return utest_(m,IDN,k,j,i) + (wb ? ueq_(m,IDN,k,j,i) : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  bool Test(const int m, const int pass, const int k, const int j, const int i,
            const Real cell_rtol, const Real dtst) const {
    bool nad;
    if (pass >= npass) {
      // NAD for passive scalars.  The advected CONCENTRATION r = (d*r)/d is the bounded
      // quantity -- a tracer cannot leave the range it is advected from -- and it is
      // also what the revise pass reconstructs, so the DMP is applied to r rather than
      // to the conserved scalar density (whose bounds would move with the density even
      // where the tracer itself is perfectly smooth).
      // a non-positive candidate density is already PAD's business; skip the ratio
      if (!(dtst > 0.0)) return false;
      const int n = nmhd_ + (pass - npass);
      const Real q_new = utest_(m,n,k,j,i)/dtst;
      const int sq = s_sc + (n - nmhd_);   // precomputed bounds (mmood_bnd_val)
      const Real qmn = bmn(m,sq,k,j,i), qmx = bmx(m,sq,k,j,i);
      Real eps_m = fmax(cell_rtol*fabs(qmn), atol);
      Real eps_p = fmax(cell_rtol*fabs(qmx), atol);
      eps_m = fmax(eps_m, eps0*fabs(qmn));
      eps_p = fmax(eps_p, eps0*fabs(qmx));
      nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
      if (nad && use_sed) {
        nad = (SEDAlpha(w0_, m, n, k, j, i, multi_d, three_d) < 1.0);
      }
    } else {
      const bool vpass = (pass >= npass_hb);              // velocity pass
      const bool bpass = (!vpass) && (pass >= npass_he);  // magnetic pass
      const int bidx = pass - npass_he;  // 0 in mag mode; 0..2 in comps mode
      const int vidx = pass - npass_hb;  // 0 in |v| mode; 0..2 in comps mode
      // fvN uses this one magnetic volume-average candidate for every test.
      // Preserve the historical separate NAD average for the non-FV methods.
      Real bn1 = bcctest_(m,IBX,k,j,i), bn2 = bcctest_(m,IBY,k,j,i);
      Real bn3 = bcctest_(m,IBZ,k,j,i);
      if (bpass && ho_pad && !fv_bavg) {
        bn1 = 0.5*(btpad1(m,k,j,i) + btpad1(m,k,j,i+1));
        bn2 = 0.5*(btpad2(m,k,j,i) + btpad2(m,k,j+1,i));
        bn3 = three_d ? 0.5*(btpad3(m,k,j,i) + btpad3(m,k+1,j,i))
                      : btpad3(m,k,j,i);
        if (wb) {
          bn1 += bcceq_(m,IBX,k,j,i) - 0.5*(beq1(m,k,j,i)+beq1(m,k,j,i+1));
          bn2 += bcceq_(m,IBY,k,j,i) - 0.5*(beq2(m,k,j,i)+beq2(m,k,j+1,i));
          bn3 += bcceq_(m,IBZ,k,j,i) - (three_d ?
              0.5*(beq3(m,k,j,i)+beq3(m,k+1,j,i)) : beq3(m,k,j,i));
        }
      }
      // candidate value of the checked variable
      Real q_new;
      if (vpass) {
        // candidate velocity from the trial conserved state (newtonian: mom/rho)
        if (nad_vmag) {
          q_new = sqrt(SQR(utest_(m,IM1,k,j,i)) + SQR(utest_(m,IM2,k,j,i))
                     + SQR(utest_(m,IM3,k,j,i))) / dtst;
        } else {
          q_new = utest_(m,IM1+vidx,k,j,i) / dtst;
        }
      } else if (!bpass) {
        const int n = (pass == 0) ? IDN : IEN;
        if (!newtonian) {
          q_new = utest_(m,n,k,j,i);
        } else if (n == IDN) {
          q_new = wb ? utest_(m,IDN,k,j,i) : dtst;   // WB: rho' = U'_rho
        } else {  // internal energy density (subtract kinetic AND magnetic);
                  // WB: e' = e(U'+U_eq, B*) - e_eq
          q_new = (utest_(m,IEN,k,j,i) + (wb ? ueq_(m,IEN,k,j,i) : 0.0))
                - 0.5*(SQR(utest_(m,IM1,k,j,i))
                + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dtst
                - 0.5*(SQR(bcctest_(m,IBX,k,j,i)) + SQR(bcctest_(m,IBY,k,j,i))
                     + SQR(bcctest_(m,IBZ,k,j,i)))
                - (wb ? weq_(m,IEN,k,j,i) : 0.0);
        }
      } else if (nad_bmag) {
        q_new = sqrt(SQR(bn1) + SQR(bn2) + SQR(bn3));
        if (wb) {   // signed delta|B|
          q_new -= sqrt(SQR(bcceq_(m,IBX,k,j,i)) + SQR(bcceq_(m,IBY,k,j,i))
                      + SQR(bcceq_(m,IBZ,k,j,i)));
        }
      } else {
        q_new = ((bidx == 0) ? bn1 : (bidx == 1) ? bn2 : bn3)
              - (wb ? bcceq_(m,IBX+bidx,k,j,i) : 0.0);
      }
      Real qmn, qmx, smn, smx;
      if (vpass) {
        qmn = bmn(m,s_v+vidx,k,j,i); qmx = bmx(m,s_v+vidx,k,j,i);
      } else if (!bpass) {
        qmn = bmn(m,pass,k,j,i); qmx = bmx(m,pass,k,j,i);
        if (wb) { smn = bmn(m,nhe+pass,k,j,i); smx = bmx(m,nhe+pass,k,j,i); }
      } else if (nad_bmag) {
        qmn = bmn(m,s_b,k,j,i); qmx = bmx(m,s_b,k,j,i);
        if (wb) { smn = bmn(m,s_b+1,k,j,i); smx = bmx(m,s_b+1,k,j,i); }
      } else {
        qmn = bmn(m,s_b+bidx,k,j,i); qmx = bmx(m,s_b+bidx,k,j,i);
        if (wb) {
          smn = bmn(m,s_b+3+bidx,k,j,i); smx = bmx(m,s_b+3+bidx,k,j,i);
          // a component whose equilibrium value vanishes (e.g. B2, B3 of a
          // horizontal field) has no full-state scale of its own: add the local
          // equilibrium field strength, the scale every component shares
          const Real beq = sqrt(SQR(bcceq_(m,IBX,k,j,i)) + SQR(bcceq_(m,IBY,k,j,i))
                              + SQR(bcceq_(m,IBZ,k,j,i)));
          smn = fabs(smn) + beq;
          smx = fabs(smx) + beq;
        }
      }
      if (!wb) {
        smn = qmn; smx = qmx;
      } else if (vpass) {
        // WB velocity pass: v_eq = 0, so a band relative to |v| alone is a band
        // relative to the perturbation (round-off fence at small amplitude); the
        // velocity's physical scale in a static equilibrium is the signal speed,
        // hence |v| + c_f (upper bound sqrt(c_s^2 + B^2/rho) of the fast speed)
        const Real dd = w0_(m,IDN,k,j,i);
        const Real cs2 = is_ideal_ ? gm_*(gm_-1.0)*w0_(m,IEN,k,j,i)/dd : SQR(cs_iso_);
        const Real b2 = SQR(bcc0_(m,IBX,k,j,i)) + SQR(bcc0_(m,IBY,k,j,i))
                      + SQR(bcc0_(m,IBZ,k,j,i));
        const Real cf = sqrt(fmax(cs2 + b2/dd, 0.0));
        smn = fabs(qmn) + cf; smx = fabs(qmx) + cf;
      }
      // Local relative relaxation of the neighboring lower/upper bounds.
      Real eps_m = fmax(cell_rtol*fabs(smn), atol);
      Real eps_p = fmax(cell_rtol*fabs(smx), atol);
      eps_m = fmax(eps_m, eps0*fabs(smn));
      eps_p = fmax(eps_p, eps0*fabs(smx));
      if (wb && !vpass) {
        // Optional extra round-off floor tied to the EQUILIBRIUM (mood_eps0_eq,
        // hydro_mood.cpp); normally redundant with the full-state scales above.
        // v_eq = 0, so no floor for v.
        Real qeq;
        if (!bpass) {
          qeq = weq_(m,(pass == 0) ? IDN : IEN,k,j,i);
        } else if (nad_bmag) {
          qeq = sqrt(SQR(bcceq_(m,IBX,k,j,i)) + SQR(bcceq_(m,IBY,k,j,i))
                   + SQR(bcceq_(m,IBZ,k,j,i)));
        } else {
          qeq = bcceq_(m,IBX+bidx,k,j,i);
        }
        const Real eq_floor = eps0_eq*fabs(qeq);
        eps_m = fmax(eps_m, eq_floor);
        eps_p = fmax(eps_p, eq_floor);
      }
      nad = (q_new < qmn - eps_m) || (q_new > qmx + eps_p);
      // exempt smooth extrema of this variable from NAD (never from PAD/NaN)
      if (nad && use_sed) {
        Real alpha;
        if (vpass) {
          // no precomputed |v| array; exempt smooth extrema only in comps mode
          alpha = nad_vmag ? 0.0
                : SEDAlpha(w0_, m, IVX+vidx, k, j, i, multi_d, three_d);
        } else if (!bpass) {
          const int n = (pass == 0) ? IDN : IEN;
          const DvceArray5D<Real> &qref = (newtonian) ? wnad_ : u0_;
          alpha = SEDAlpha(qref, m, n, k, j, i, multi_d, three_d);
        } else if (nad_bmag) {
          alpha = SEDAlpha4(bmag_, m, k, j, i, multi_d, three_d);
        } else {
          alpha = SEDAlpha(bnad_, m, IBX+bidx, k, j, i, multi_d, three_d);
        }
        nad = (alpha < 1.0);
      }
    }
    return nad;
  }
};

template <MHD_RSolver rsolver_method_>
void MHD::MOODLoop(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;
  int ncells1 = indcs_.nx1 + 2*indcs_.ng;
  int ncells2 = (indcs_.nx2 > 1) ? (indcs_.nx2 + 2*indcs_.ng) : 1;
  int ncells3 = (indcs_.nx3 > 1) ? (indcs_.nx3 + 2*indcs_.ng) : 1;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  int nmb1 = nmb - 1;
  int &nmhd_ = nmhd;
  int nvars = nmhd + nscalars;

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
  const bool ho_pad = newtonian && peos->eos_data.is_ideal && multi_d &&
      emf_method != MHD_EMF::ct_contact;
  const int pad_extra = ho_pad ? mhd_bcc::Reach(recon_method)-1 : 0;
  const auto pad_recon = recon_method;
  const bool fv_bavg = ho_pad && mhd_bcc::IsFV(recon_method);

  auto &u0_ = u0;
  auto &u1_ = u1;
  auto &w0_ = w0;
  auto &bcc0_ = bcc0;
  auto &b1_ = b1;
  auto &utest_ = utest;
  auto &bcctest_ = bcctest;
  auto &fofc_ = fofc;
  auto &fb_level_ = fb_level;
  auto &bmag_ = bmag_ref;
  auto wl_ = wl3d;
  auto wr_ = wr3d;
  auto bl_ = bl3d;
  auto br_ = br3d;
  auto sdet_ = sdet;   // L-HLLD carbuncle sensor (filled by CalculateFluxes)
  // Well-balanced scheme (mhd_wb.cpp): u0/u1/utest and the fluxes/face EMFs are
  // PERTURBATION quantities about a static magnetohydrostatic equilibrium, b0/bcc0 the
  // full field.  As in hydro (hydro_mood.cpp): NAD/SED and their global scales act on
  // the perturbation (rho', e', B - B_eq or |B| - |B_eq|), PAD on the full state, the
  // candidate B is formed in perturbation form (exact at rest), and revised faces go
  // through the face equilibrium exactly like the base fluxes.
  const bool wb = well_balanced;
  auto &ueq_ = u_eq;
  auto &weq_ = w_eq;
  auto &bcceq_ = bcc_eq;
  const DvceArray5D<Real> wnad_ = wb ? w_pert : w0;   // NAD reference primitives
  const bool nad_avg = mood_nad_bcc_avg && !wb;   // (WB keeps its perturbation field)
  const DvceArray5D<Real> bnad_ = nad_avg ? bcc_nad : (wb ? bcc_pert : bcc0);
  auto beq1 = b_eq.x1f;  auto beq2 = b_eq.x2f;  auto beq3 = b_eq.x3f;
  auto weqf1 = w_eq_face.x1f;  auto weqf2 = w_eq_face.x2f;  auto weqf3 = w_eq_face.x3f;
  auto beqf1 = bcc_eq_face.x1f;  auto beqf2 = bcc_eq_face.x2f;
  auto beqf3 = bcc_eq_face.x3f;
  auto feq1 = f_eq.x1f;  auto feq2 = f_eq.x2f;  auto feq3 = f_eq.x3f;
  auto e31q = e3x1_eq;  auto e21q = e2x1_eq;  auto e12q = e1x2_eq;
  auto e32q = e3x2_eq;  auto e23q = e2x3_eq;  auto e13q = e1x3_eq;
  const Real eps0_eq = mood_eps0_eq;
  auto b0x1 = b0.x1f;
  auto b0x2 = b0.x2f;
  auto b0x3 = b0.x3f;
  auto btpad1 = b0_test.x1f, btpad2 = b0_test.x2f, btpad3 = b0_test.x3f;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto e3x1_ = e3x1;
  auto e2x1_ = e2x1;
  auto e1x2_ = e1x2;
  auto e3x2_ = e3x2;
  auto e2x3_ = e2x3;
  auto e1x3_ = e1x3;

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
  const int ntest = (nad_scalars) ? nvars : nmhd_;
  const bool nad_bmag = (mood_nad_b == 0);
  const bool nad_von = (mood_nad_v > 0);      // velocity in NAD
  const bool nad_vmag = (mood_nad_v == 1);    // |v| (else components)
  const int n_fb = n_fb_tiers;
  const int nrevs = mood_max_revs;
  // face-local re-add of the explicit diffusion fluxes on revised faces (see
  // mood_candidate_diffusion in mhd.cpp); the bulk pass used the same functions
  const bool readd_visc = mood_candidate_diffusion && has_explicit_viscosity;
  const bool readd_cond = mood_candidate_diffusion && has_explicit_conduction;
  const ViscFaceParams vfp =
      (pvisc != nullptr) ? pvisc->FaceParams(eos_) : ViscFaceParams{};
  const Real cond_alpha = (pcond != nullptr) ? pcond->alpha_iso : 0.0;
  const Real cond_gm1 = eos_.gamma - 1.0;

  const bool initialize = !mood_frozen_emf || (mood_revision == 0 && !mood_revise);
  const bool is_ideal_ = eos_.is_ideal;
  const Real gm_ = eos_.gamma, cs_iso_ = eos_.iso_cs;   // WB velocity band scale

  // UCT face-coefficient arrays for the revise kernels (empty views when ct_contact —
  // the solver then skips the UCT outputs).  Flag values match mhd_fluxes.cpp.
  const bool use_uct = (emf_method != MHD_EMF::ct_contact);
  const int uct_fl = (emf_method == MHD_EMF::uct_hlld) ? 2 : (use_uct ? 1 : 0);
  auto aL1 = aL_x1f;  auto dl1 = dL_x1f;  auto dr1 = dR_x1f;
  auto vt1_1 = vy_x1f;  auto vt2_1 = vz_x1f;
  auto aL2 = aL_x2f;  auto dl2 = dL_x2f;  auto dr2 = dR_x2f;
  auto vt1_2 = vz_x2f;  auto vt2_2 = vx_x2f;
  auto aL3 = aL_x3f;  auto dl3 = dL_x3f;  auto dr3 = dR_x3f;
  auto vt1_3 = vx_x3f;  auto vt2_3 = vy_x3f;

  // For fvN, the stage-input bcc0 already contains the production volume average.
  // In the frozen path its ghosts have been exchanged from their owners: copying
  // them also avoids order reductions at the outer edge of a local face array.
  // The candidate below uses the same averaging for both PAD and magnetic NAD.
  if (initialize && nad_avg && fv_bavg) {
    Kokkos::deep_copy(bcc_nad, bcc0);
  } else if (initialize && nad_avg) {
    auto bn_ = bcc_nad;
    const int n1m = ncells1-1, n2m = ncells2-1, n3m = ncells3-1;
    const bool md = multi_d, td = three_d;
    const bool full_average = mood_frozen_emf;
    par_for("mood_bnad", DevExeSpace(), 0, nmb1, 0, n3m, 0, n2m, 0, n1m,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int ip = (full_average || i < n1m) ? i+1 : i;
      const int jp = (md && (full_average || j < n2m)) ? j+1 : j;
      const int kp = (td && (full_average || k < n3m)) ? k+1 : k;
      bn_(m,IBX,k,j,i) = 0.5*(b0x1(m,k,j,i) + b0x1(m,k,j,ip));
      bn_(m,IBY,k,j,i) = 0.5*(b0x2(m,k,j,i) + b0x2(m,k,jp,i));
      bn_(m,IBZ,k,j,i) = 0.5*(b0x3(m,k,j,i) + b0x3(m,kp,j,i));
    });
  }

  // |B| of the stage-input state, used for NAD bounds and SED in "mag" mode.  Filled
  // over the full array so the SED stencil is covered anywhere detection runs.
  if (initialize && nad_bmag) {
    const auto bsrc_ = nad_avg ? bcc_nad : bcc0;
    par_for("mood_bmag", DevExeSpace(), 0, nmb1, 0, ncells3-1, 0, ncells2-1,
            0, ncells1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      bmag_(m,k,j,i) = sqrt(SQR(bsrc_(m,IBX,k,j,i)) + SQR(bsrc_(m,IBY,k,j,i))
                          + SQR(bsrc_(m,IBZ,k,j,i)));
      if (wb) {   // signed delta|B| = |B| - |B_eq|: smooth, rotation-invariant
        bmag_(m,k,j,i) -= sqrt(SQR(bcceq_(m,IBX,k,j,i)) + SQR(bcceq_(m,IBY,k,j,i))
                             + SQR(bcceq_(m,IBZ,k,j,i)));
      }
    });
  }

  // The shrinking halo covers edge composition and the production B interpolant.
  const int h_f = mood_halo0/nrevs;

  const int ndj = (multi_d) ? 1 : 0;
  const int ndk = (three_d) ? 1 : 0;

  //--------------------------------------------------------------------------------------
  // Relaxed-DMP bounds of the NAD reference quantities over the 3^d-cell neighbourhood,
  // once per stage (the reference state is immutable through the revisions, also
  // across the per-revision calls of the frozen-EMF path): per-cell values gathered
  // into mood_val, then separable 3-point sweeps over all slots at once
  // (utils/mood_bounds.hpp; bmn/bmx name the arrays holding the result) -- bit-identical
  // to the direct scan the detection kernel used to repeat per variable, per cell, per
  // revision.  Slot layout:
  //   [0,nhe)            density, energy            WB: [nhe,2nhe)  the same, FULL state
  //   [s_b,s_b+nb)       |B| (mag) / B comps        WB: next nb     the same, FULL state
  //   [s_v,s_v+nv)       |v| / v comps              (WB scale from |v| + c_f, in-kernel)
  //   [s_sc,...)         scalar concentrations
  const int nhe = (is_ideal_ && nad_energy) ? 2 : 1;
  const int nb = nad_bmag ? 1 : 3;
  const int nv = nad_von ? (nad_vmag ? 1 : 3) : 0;
  const int s_b = nhe*(wb ? 2 : 1);
  const int s_v = s_b + nb*(wb ? 2 : 1);
  const int s_sc = s_v + nv;
  // (2D leaves the result in (mood_val, mood_tmx), 1D/3D in (mood_qmn, mood_qmx); the
  // frozen-EMF revision calls skip the computation and must name the same arrays)
  const bool bnd_in_val = multi_d && !three_d;
  auto bmn = bnd_in_val ? mood_val : mood_qmn;
  auto bmx = bnd_in_val ? mood_tmx : mood_qmx;
  if (initialize) {
    auto val = mood_val;
    const int nsc = nad_scalars ? (nvars - nmhd_) : 0;
    // one cell beyond the widest detection halo (first revision; none when frozen)
    const int dhm = mood_frozen_emf ? 0 : mood_halo0;
    const int bil = is-dhm, biu = ie+dhm;
    const int bjl = multi_d ? js-dhm : js, bju = multi_d ? je+dhm : je;
    const int bkl = three_d ? ks-dhm : ks, bku = three_d ? ke+dhm : ke;
    par_for("mmood_bnd_val", DevExeSpace(), 0, nmb1, bkl-ndk, bku+ndk, bjl-ndj, bju+ndj,
            bil-1, biu+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int p=0; p<nhe; ++p) {
        const int n = (p == 0) ? IDN : IEN;
        const Real q = (newtonian) ? wnad_(m,n,k,j,i) : u0_(m,n,k,j,i);
        val(m,p,k,j,i) = q;
        if (wb) { val(m,nhe+p,k,j,i) = q + weq_(m,n,k,j,i); }
      }
      if (nad_bmag) {
        const Real q = bmag_(m,k,j,i);
        val(m,s_b,k,j,i) = q;
        if (wb) {
          val(m,s_b+1,k,j,i) = q + sqrt(SQR(bcceq_(m,IBX,k,j,i))
                                      + SQR(bcceq_(m,IBY,k,j,i))
                                      + SQR(bcceq_(m,IBZ,k,j,i)));
        }
      } else {
        for (int b=0; b<3; ++b) {
          const Real q = bnad_(m,IBX+b,k,j,i);
          val(m,s_b+b,k,j,i) = q;
          if (wb) { val(m,s_b+3+b,k,j,i) = q + bcceq_(m,IBX+b,k,j,i); }
        }
      }
      if (nad_von) {
        if (nad_vmag) {
          val(m,s_v,k,j,i) = sqrt(SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i))
                                + SQR(w0_(m,IVZ,k,j,i)));
        } else {
          for (int v=0; v<3; ++v) { val(m,s_v+v,k,j,i) = w0_(m,IVX+v,k,j,i); }
        }
      }
      for (int sc=0; sc<nsc; ++sc) { val(m,s_sc+sc,k,j,i) = w0_(m,nmhd_+sc,k,j,i); }
    });
    MoodNeighborhoodBounds(nmb1, s_sc+nsc, bil, biu, bjl, bju, bkl, bku, multi_d,
                           three_d, mood_val, mood_qmn, mood_qmx, mood_tmx, bmn, bmx);
  }

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
  if (initialize) {
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

  // Touched-cell restriction (mood_touched_only).  A revision changes only the faces
  // adjacent to the cells it demoted, so from the second revision on the candidate,
  // the floor test and the detection need to run only where those faces can reach:
  // one cell for the flux divergence and the face-E proxy of B; with UCT the composed
  // edge EMF reads faces i-reach..i+reach-1 (cells within reach+1), and the CT update
  // plus the cell average add one more cell.  Every other cell keeps its accepted
  // candidate and flags from the previous pass, so the result is bitwise identical.
  // The wider candidate face field must be initialized throughout PAD's stencil.
  const bool touched_only = mood_touched_only && !ho_pad;
  auto touched_ = mood_touched;
  auto tmp_ = mood_tmp;
  const int r_touch = (uct_fl > 0 && multi_d) ? ReconEdgeReach(recon_method) + 2 : 1;
  peos->c2p_mask = mood_touched;

  auto revise_faces = [&](const int dh, const int force_tier) {
    const int extra = (force_tier > 0) ? ((force_tier >= n_fb) ? 1 : 2) : 0;
    const int de = dh + extra;
    const int de2 = multi_d ? de : 0, de3 = three_d ? de : 0;
    // x1 faces
    par_for("mmood_rev_x1", DevExeSpace(), 0, nmb1, ks-de3, ke+de3, js-de2, je+de2, is-dh, ie+1+dh,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (force_tier > 0 || fofc_(m,k,j,i-1) || fofc_(m,k,j,i)) {
        int lv_l = fb_level_(m,k,j,i-1), lv_r = fb_level_(m,k,j,i);
        int tier = (force_tier > 0) ? force_tier : ((lv_l > lv_r) ? lv_l : lv_r);
        ReconstructionMethod frecon = (tier >= n_fb) ?
            ReconstructionMethod::dc : ReconstructionMethod::plm;
        ReconFace<IVX>(frecon, m, k, j, i, nvars, wnad_, wl_, wr_);
        ReconFace<IVX>(frecon, m, k, j, i, 3, bnad_, bl_, br_);
        if (wb) {
          AddFaceEquilibriumMHD(eos_, true, nmhd_, m, k, j, i, weqf1, beqf1,
                                wl_, wr_, bl_, br_);
        }
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = b0x1;  auto flx = flx1;  auto eyl = e3x1_;  auto ezl = e2x1_;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFaceMHD<rsolver_method_, IVX>(eos, indcs, size, coord,
                                           m, k, j, i, is_, js_, ks_,
                                           wl, wr, bl, br, bx, flx, eyl, ezl,
                                           uct_fl, aL1, dl1, dr1, vt1_1, vt2_1, sdet_);
        if (wb) {
          // perturbation flux and face EMFs; scalars upwinded on the FULL mass flux
          for (int n=0; n<nmhd_; ++n) {
            flx1(m,n,k,j,i) -= feq1(m,n,k,j,i);
          }
          eyl(m,k,j,i) -= e31q(m,k,j,i);
          ezl(m,k,j,i) -= e21q(m,k,j,i);
          const Real fd = flx1(m,IDN,k,j,i) + feq1(m,IDN,k,j,i);
          for (int n=nmhd_; n<nvars; ++n) {
            flx1(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
          }
        } else {
          for (int n=nmhd_; n<nvars; ++n) {
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
      par_for("mmood_rev_x2", DevExeSpace(), 0, nmb1, ks-de3, ke+de3, js-dh, je+1+dh, is-de, ie+de,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (force_tier > 0 || fofc_(m,k,j-1,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k,j-1,i), lv_r = fb_level_(m,k,j,i);
          int tier = (force_tier > 0) ? force_tier : ((lv_l > lv_r) ? lv_l : lv_r);
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          ReconFace<IVY>(frecon, m, k, j, i, nvars, wnad_, wl_, wr_);
          ReconFace<IVY>(frecon, m, k, j, i, 3, bnad_, bl_, br_);
          if (wb) {
            AddFaceEquilibriumMHD(eos_, true, nmhd_, m, k, j, i, weqf2, beqf2,
                                  wl_, wr_, bl_, br_);
          }
          auto eos = eos_;
          auto indcs = indcs_;
          auto size = size_;
          auto coord = coord_;
          auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
          auto bx = b0x2;  auto flx = flx2;  auto eyl = e1x2_;  auto ezl = e3x2_;
          const int is_ = is, js_ = js, ks_ = ks;
          SolveFaceMHD<rsolver_method_, IVY>(eos, indcs, size, coord,
                                             m, k, j, i, is_, js_, ks_,
                                             wl, wr, bl, br, bx, flx, eyl, ezl,
                                             uct_fl, aL2, dl2, dr2, vt1_2, vt2_2, sdet_);
          if (wb) {
            // perturbation flux and face EMFs; scalars upwinded on the FULL mass flux
            for (int n=0; n<nmhd_; ++n) {
              flx2(m,n,k,j,i) -= feq2(m,n,k,j,i);
            }
            eyl(m,k,j,i) -= e12q(m,k,j,i);
            ezl(m,k,j,i) -= e32q(m,k,j,i);
            const Real fd = flx2(m,IDN,k,j,i) + feq2(m,IDN,k,j,i);
            for (int n=nmhd_; n<nvars; ++n) {
              flx2(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
            }
          } else {
            for (int n=nmhd_; n<nvars; ++n) {
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
      par_for("mmood_rev_x3", DevExeSpace(), 0, nmb1, ks-dh, ke+1+dh, js-de2, je+de2, is-de, ie+de,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (force_tier > 0 || fofc_(m,k-1,j,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k-1,j,i), lv_r = fb_level_(m,k,j,i);
          int tier = (force_tier > 0) ? force_tier : ((lv_l > lv_r) ? lv_l : lv_r);
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          ReconFace<IVZ>(frecon, m, k, j, i, nvars, wnad_, wl_, wr_);
          ReconFace<IVZ>(frecon, m, k, j, i, 3, bnad_, bl_, br_);
          if (wb) {
            AddFaceEquilibriumMHD(eos_, true, nmhd_, m, k, j, i, weqf3, beqf3,
                                  wl_, wr_, bl_, br_);
          }
          auto eos = eos_;
          auto indcs = indcs_;
          auto size = size_;
          auto coord = coord_;
          auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
          auto bx = b0x3;  auto flx = flx3;  auto eyl = e2x3_;  auto ezl = e1x3_;
          const int is_ = is, js_ = js, ks_ = ks;
          SolveFaceMHD<rsolver_method_, IVZ>(eos, indcs, size, coord,
                                             m, k, j, i, is_, js_, ks_,
                                             wl, wr, bl, br, bx, flx, eyl, ezl,
                                             uct_fl, aL3, dl3, dr3, vt1_3, vt2_3, sdet_);
          if (wb) {
            // perturbation flux and face EMFs; scalars upwinded on the FULL mass flux
            for (int n=0; n<nmhd_; ++n) {
              flx3(m,n,k,j,i) -= feq3(m,n,k,j,i);
            }
            eyl(m,k,j,i) -= e23q(m,k,j,i);
            ezl(m,k,j,i) -= e13q(m,k,j,i);
            const Real fd = flx3(m,IDN,k,j,i) + feq3(m,IDN,k,j,i);
            for (int n=nmhd_; n<nvars; ++n) {
              flx3(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
            }
          } else {
            for (int n=nmhd_; n<nvars; ++n) {
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
  };

  // Cache fallback EMFs from immutable stage inputs. The initial high-order
  // fluxes/EMFs have already been exchanged by the task graph. All revisions
  // reuse these caches; no high-order corner composition is repeated.
  auto &eplm = mood_eplm;
  auto &edc = mood_edc;
  auto copy_edges = [&](DvceEdgeFld4D<Real> &dst, const DvceEdgeFld4D<Real> &src) {
    Kokkos::deep_copy(dst.x1e, src.x1e);
    Kokkos::deep_copy(dst.x2e, src.x2e);
    Kokkos::deep_copy(dst.x3e, src.x3e);
  };
  if (mood_frozen_emf && initialize) {
    const int h = mood_emf_halo, h2 = multi_d ? h : 0, h3 = three_d ? h : 0;
    auto compose = [&]() {
      ComposeCornerEMF(is-h,ie+h+1,js-h2,je+h2+1,ks-h3,
                       three_d ? ke+h3+1 : ks, false);
    };
    auto base_recon = recon_method;
    if (n_fb > 1) {
      revise_faces(h, 1);
      recon_method = ReconstructionMethod::plm;
      compose();
      copy_edges(eplm, efld);
    }
    revise_faces(h, n_fb);
    recon_method = ReconstructionMethod::dc;
    compose();
    copy_edges(edc, efld);
    recon_method = base_recon;
    RestoreMOODCandidate();
  }
  auto select_frozen_edges = [&](const int h) {
    auto E1=efld.x1e, E2=efld.x2e, E3=efld.x3e;
    auto P1=eplm.x1e, P2=eplm.x2e, P3=eplm.x3e;
    auto D1=edc.x1e, D2=edc.x2e, D3=edc.x3e;
    const bool td=three_d;
    par_for("audit_select_edges", DevExeSpace(), 0,nmb1,
            td ? ks-h : ks, td ? ke+h+1 : ks, js-h,je+h+1,is-h,ie+h+1,
    KOKKOS_LAMBDA(const int m,const int k,const int j,const int i) {
      int t3=0;
      for (int dj=-1;dj<=0;++dj) for (int di=-1;di<=0;++di) {
        const int t=fb_level_(m,k,j+dj,i+di); t3=(t>t3)?t:t3;
      }
      if (t3>0) E3(m,k,j,i)=(t3>=n_fb)?D3(m,k,j,i):P3(m,k,j,i);
      if (td) {
        int t1=0,t2=0;
        for (int dk=-1;dk<=0;++dk) for (int dj=-1;dj<=0;++dj) {
          const int t=fb_level_(m,k+dk,j+dj,i); t1=(t>t1)?t:t1;
        }
        for (int dk=-1;dk<=0;++dk) for (int di=-1;di<=0;++di) {
          const int t=fb_level_(m,k+dk,j,i+di); t2=(t>t2)?t:t2;
        }
        if (t1>0) E1(m,k,j,i)=(t1>=n_fb)?D1(m,k,j,i):P1(m,k,j,i);
        if (t2>0) E2(m,k,j,i)=(t2>=n_fb)?D2(m,k,j,i):P2(m,k,j,i);
      } else {
        int t1=fb_level_(m,ks,j-1,i), t=fb_level_(m,ks,j,i);
        t1=(t>t1)?t:t1;
        int t2=fb_level_(m,ks,j,i-1); t2=(t>t2)?t:t2;
        if (t1>0) {
          E1(m,ks,j,i)=(t1>=n_fb)?D1(m,ks,j,i):P1(m,ks,j,i);
          E1(m,ke+1,j,i)=E1(m,ks,j,i);
        }
        if (t2>0) {
          E2(m,ks,j,i)=(t2>=n_fb)?D2(m,ks,j,i):P2(m,ks,j,i);
          E2(m,ke+1,j,i)=E2(m,ks,j,i);
        }
      }
    });
  };
  if (mood_frozen_emf && mood_revise) {
    revise_faces(0, 0);
    select_frozen_edges(mood_emf_halo);
    return;
  }
  int ndemoted_total = 0;
  const int first_rev = mood_frozen_emf ? mood_revision+1 : 1;
  const int last_rev = mood_frozen_emf ? first_rev : nrevs;
  const bool frozen = mood_frozen_emf;
  const bool may_demote = !frozen || mood_revision < nrevs;
  for (int rev=first_rev; rev<=last_rev; ++rev) {
    const bool masked = touched_only && (rev > 1);
    const int dh = mood_frozen_emf ? 0 : h_f*(nrevs-rev+1);
    int il = is-dh, iu = ie+dh, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-dh; ju = je+dh; }
    if (three_d) { kl = ks-dh; ku = ke+dh; }


    //------------------------------------------------------------------------------------
    // (1) candidate update: conserved hydro variables from the flux divergence, and the
    // candidate cell-averaged B from face-E finite differences (the FOFC estimate).
    if (masked) {
      // dilate the previous revision's demotion flags (still in fofc_) by r_touch with
      // a separable max filter over the full array (clamped at the array edges):
      // 1D: fofc->touched; 2D: fofc->tmp->touched; 3D: fofc->touched->tmp->touched
      const int r = r_touch;
      const int n1 = ncells1, n2 = ncells2, n3 = ncells3;
      auto dst1 = (multi_d && !three_d) ? tmp_ : touched_;
      par_for("mmood_touch1", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const int lo = (i-r < 0) ? 0 : i-r;
        const int hi = (i+r > n1-1) ? n1-1 : i+r;
        bool t = false;
        for (int ii=lo; ii<=hi; ++ii) { t = t || fofc_(m,k,j,ii); }
        dst1(m,k,j,i) = t;
      });
      if (multi_d) {
        auto src2 = dst1;
        auto dst2 = (three_d) ? tmp_ : touched_;
        par_for("mmood_touch2", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          const int lo = (j-r < 0) ? 0 : j-r;
          const int hi = (j+r > n2-1) ? n2-1 : j+r;
          bool t = false;
          for (int jj=lo; jj<=hi; ++jj) { t = t || src2(m,k,jj,i); }
          dst2(m,k,j,i) = t;
        });
      }
      if (three_d) {
        par_for("mmood_touch3", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          const int lo = (k-r < 0) ? 0 : k-r;
          const int hi = (k+r > n3-1) ? n3-1 : k+r;
          bool t = false;
          for (int kk=lo; kk<=hi; ++kk) { t = t || tmp_(m,kk,j,i); }
          touched_(m,k,j,i) = t;
        });
      }
    }
    Kokkos::deep_copy(fofc_, false);
    par_for("mmood_newu", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
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

      Real b1old = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      Real b2old = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      Real b3old = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));

      if (wb) {
        // perturbation form about the equilibrium (mhd_ct.cpp): b_eq + gam0*(b - b_eq)
        // + gam1*(b1 - b_eq) is b_eq bit for bit at rest, (2/3)X + (1/3)X is not X
        b1old -= 0.5*(beq1(m,k,j,i) + beq1(m,k,j,i+1));
        b2old -= 0.5*(beq2(m,k,j,i) + beq2(m,k,j+1,i));
        b3old -= 0.5*(beq3(m,k,j,i) + beq3(m,k+1,j,i));
        bcctest_(m,IBX,k,j,i) = bcceq_(m,IBX,k,j,i)
            + gam0*(bcc0_(m,IBX,k,j,i) - bcceq_(m,IBX,k,j,i)) + gam1*b1old;
        bcctest_(m,IBY,k,j,i) = bcceq_(m,IBY,k,j,i)
            + gam0*(bcc0_(m,IBY,k,j,i) - bcceq_(m,IBY,k,j,i)) + gam1*b2old;
        bcctest_(m,IBZ,k,j,i) = bcceq_(m,IBZ,k,j,i)
            + gam0*(bcc0_(m,IBZ,k,j,i) - bcceq_(m,IBZ,k,j,i)) + gam1*b3old;
      } else {
        bcctest_(m,IBX,k,j,i) = gam0*bcc0_(m,IBX,k,j,i) + gam1*b1old;
        bcctest_(m,IBY,k,j,i) = gam0*bcc0_(m,IBY,k,j,i) + gam1*b2old;
        bcctest_(m,IBZ,k,j,i) = gam0*bcc0_(m,IBZ,k,j,i) + gam1*b3old;
      }

      bcctest_(m,IBY,k,j,i) += dtodx1*(e3x1_(m,k,j,i+1) - e3x1_(m,k,j,i));
      bcctest_(m,IBZ,k,j,i) -= dtodx1*(e2x1_(m,k,j,i+1) - e2x1_(m,k,j,i));
      if (multi_d) {
        bcctest_(m,IBX,k,j,i) -= dtodx2*(e3x2_(m,k,j+1,i) - e3x2_(m,k,j,i));
        bcctest_(m,IBZ,k,j,i) += dtodx2*(e1x2_(m,k,j+1,i) - e1x2_(m,k,j,i));
      }
      if (three_d) {
        bcctest_(m,IBX,k,j,i) += dtodx3*(e2x3_(m,k+1,j,i) - e2x3_(m,k,j,i));
        bcctest_(m,IBY,k,j,i) -= dtodx3*(e1x3_(m,k+1,j,i) - e1x3_(m,k,j,i));
      }
    });

    //------------------------------------------------------------------------------------
    // (1a-UCT) Overwrite the INTERIOR candidate cell-averaged B with the GENUINE staggered
    // CT update: compose the candidate corner EMFs (identical code to CornerE, honouring
    // the current fb_level demotion), CT-update the candidate face-B (mhd_ct.cpp), and
    // interpolate to cell centers with the production C2P operator. The detector sees
    // the real evolved field -- including
    // the corner-EMF composition where the UCT dissipation coefficient can run away -- not
    // the FOFC face-E proxy above.  Composed over the full light-cone halo (the flux
    // kernels extend the UCT face data by mood_halo0 for this), so it is identical on
    // every block that sees a cell: decomposition-invariant demotions.
    if (uct_fl > 0 && multi_d) {
      // Compose over the WHOLE detection halo [il,iu] x [jl,ju] x [kl,ku] (edges one past
      // the upper cell), not only the interior: the flux kernels fill the UCT face data
      // out to this reach (mhd_fluxes.cpp), so every halo cell sees the same composed
      // candidate as the neighbouring block's interior and the demotion pattern is
      // block-decomposition invariant.
      const int px = pad_extra, py = multi_d ? pad_extra : 0;
      const int pz = three_d ? pad_extra : 0;
      if (!mood_frozen_emf) {
        ComposeCornerEMF(il-px, iu+1+px, jl-py, ju+1+py, kl-pz,
                         (three_d ? ku+1+pz : ks), masked);
      }
      auto bt1 = b0_test.x1f;  auto bt2 = b0_test.x2f;  auto bt3 = b0_test.x3f;
      auto e1c = efld.x1e;  auto e2c = efld.x2e;  auto e3c = efld.x3e;
      const bool td = three_d;
      // CT update of candidate face-B (mirrors mhd_ct.cpp)
      par_for("mmood_ctB1", DevExeSpace(), 0, nmb1, kl-pz, ku+pz,
              jl-py, ju+py, il-px, iu+1+px,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (masked && !touched_(m,k,j,i)) return;
        Real b = gam0*b0x1(m,k,j,i) + gam1*b1_.x1f(m,k,j,i);
        if (wb) {   // perturbation form (mhd_ct.cpp)
          b = beq1(m,k,j,i) + gam0*(b0x1(m,k,j,i) - beq1(m,k,j,i))
            + gam1*(b1_.x1f(m,k,j,i) - beq1(m,k,j,i));
        }
        b -= beta_dt*(e3c(m,k,j+1,i) - e3c(m,k,j,i))/size_.d_view(m).dx2;
        if (td) b += beta_dt*(e2c(m,k+1,j,i) - e2c(m,k,j,i))/size_.d_view(m).dx3;
        bt1(m,k,j,i) = b;
      });
      par_for("mmood_ctB2", DevExeSpace(), 0, nmb1, kl-pz, ku+pz,
              jl-py, ju+1+py, il-px, iu+px,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (masked && !touched_(m,k,j,i)) return;
        Real b = gam0*b0x2(m,k,j,i) + gam1*b1_.x2f(m,k,j,i);
        if (wb) {   // perturbation form (mhd_ct.cpp)
          b = beq2(m,k,j,i) + gam0*(b0x2(m,k,j,i) - beq2(m,k,j,i))
            + gam1*(b1_.x2f(m,k,j,i) - beq2(m,k,j,i));
        }
        b += beta_dt*(e3c(m,k,j,i+1) - e3c(m,k,j,i))/size_.d_view(m).dx1;
        if (td) b -= beta_dt*(e1c(m,k+1,j,i) - e1c(m,k,j,i))/size_.d_view(m).dx3;
        bt2(m,k,j,i) = b;
      });
      par_for("mmood_ctB3", DevExeSpace(), 0, nmb1, kl-pz, ku+1+pz,
              jl-py, ju+py, il-px, iu+px,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (masked && !touched_(m,k,j,i)) return;
        Real b = gam0*b0x3(m,k,j,i) + gam1*b1_.x3f(m,k,j,i);
        if (wb) {   // perturbation form (mhd_ct.cpp)
          b = beq3(m,k,j,i) + gam0*(b0x3(m,k,j,i) - beq3(m,k,j,i))
            + gam1*(b1_.x3f(m,k,j,i) - beq3(m,k,j,i));
        }
        b -= beta_dt*(e2c(m,k,j,i+1) - e2c(m,k,j,i))/size_.d_view(m).dx1;
        b += beta_dt*(e1c(m,k,j+1,i) - e1c(m,k,j,i))/size_.d_view(m).dx2;
        bt3(m,k,j,i) = b;
      });
      // average candidate face-B to cell centers -> bcctest over the halo
      par_for("mmood_avgB", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (masked && !touched_(m,k,j,i)) return;
        if (wb) {
          // average the PERTURBATION faces about bcc_eq (built with the ConsToPrim
          // face-to-centre arithmetic, which is not this average): exact at rest
          bcctest_(m,IBX,k,j,i) = bcceq_(m,IBX,k,j,i)
              + 0.5*((bt1(m,k,j,i) - beq1(m,k,j,i))
                   + (bt1(m,k,j,i+1) - beq1(m,k,j,i+1)));
          bcctest_(m,IBY,k,j,i) = bcceq_(m,IBY,k,j,i)
              + 0.5*((bt2(m,k,j,i) - beq2(m,k,j,i))
                   + (bt2(m,k,j+1,i) - beq2(m,k,j+1,i)));
          bcctest_(m,IBZ,k,j,i) = bcceq_(m,IBZ,k,j,i) + ((td) ?
              0.5*((bt3(m,k,j,i) - beq3(m,k,j,i))
                 + (bt3(m,k+1,j,i) - beq3(m,k+1,j,i)))
              : (bt3(m,k,j,i) - beq3(m,k,j,i)));
        } else {
          bcctest_(m,IBX,k,j,i) = 0.5*(bt1(m,k,j,i) + bt1(m,k,j,i+1));
          bcctest_(m,IBY,k,j,i) = 0.5*(bt2(m,k,j,i) + bt2(m,k,j+1,i));
          bcctest_(m,IBZ,k,j,i) = (td) ? 0.5*(bt3(m,k,j,i) + bt3(m,k+1,j,i))
                                       : bt3(m,k,j,i);
        }
        if (ho_pad) {
          bcctest_(m,IBX,k,j,i) = mhd_bcc::CellCenteredComponent(
              pad_recon, bt1, m,k,j,i, 0,0,1, 0,ncells1-1);
          bcctest_(m,IBY,k,j,i) = mhd_bcc::CellCenteredComponent(
              pad_recon, bt2, m,k,j,i, 0,1,0, 0,ncells2-1);
          bcctest_(m,IBZ,k,j,i) = mhd_bcc::CellCenteredComponent(
              pad_recon, bt3, m,k,j,i, 1,0,0, 0,ncells3-1);
        }
      });
    }

    //------------------------------------------------------------------------------------
    // (2) PAD: flag cells where conversion to primitives would require floors
    // (sets fofc; b0/w0 passed but not used/changed)
    peos->use_c2p_mask = masked;
    if (wb) {
      peos->ConsToPrimWB(utest_, u_eq, b0, w0_, bcctest_, true, il, iu, jl, ju, kl, ku);
    } else {
      peos->ConsToPrim(utest_, b0, w0_, bcctest_, true, il, iu, jl, ju, kl, ku);
    }
    peos->use_c2p_mask = false;

    //------------------------------------------------------------------------------------
    // (3) NaN + NAD detection; combine with PAD; demote survivors and count them.
    // Same tier-floor split as hydro: PAD/NaN may cascade to DC, NAD (density, energy,
    // magnetic field) demotes to the PLM tier only.
    // Counters are atomics (incremented only by flagged cells) instead of a five-value
    // parallel_reduce: exact integers either way, no reduction epilogue, negligible
    // contention (MoodNadPassCtx::Combine lists the slots).
    // The pass-type layout: 0 = density, 1 = internal energy (ideal gas), then B (|B|
    // in mag mode, three components in comps mode), then v (|v| or three components),
    // then the passive-scalar concentrations.
    const int npass_he = (is_ideal_ && nad_energy) ? 2 : 1;
    const int npass_hb = npass_he + (nad_bmag ? 1 : 3);
    const int npass = npass_hb + (nad_von ? (nad_vmag ? 1 : 3) : 0);
    const int npass_tot = npass + (nad_scalars ? (nvars - nmhd_) : 0);
    auto cnt_ = mood_cnt;
    auto stat_ = mood_stat;
    auto crtol_ = mood_crtol;
    auto nadf_ = mood_nadf;
    Kokkos::deep_copy(cnt_, 0);
    MoodNadPassCtx ctx;
    ctx.utest_ = utest_;  ctx.bcctest_ = bcctest_;  ctx.ueq_ = ueq_;  ctx.weq_ = weq_;
    ctx.bcceq_ = bcceq_;  ctx.w0_ = w0_;  ctx.bcc0_ = bcc0_;  ctx.wnad_ = wnad_;
    ctx.u0_ = u0_;  ctx.bnad_ = bnad_;  ctx.bmn = bmn;  ctx.bmx = bmx;
    ctx.btpad1 = btpad1;  ctx.btpad2 = btpad2;  ctx.btpad3 = btpad3;
    ctx.beq1 = beq1;  ctx.beq2 = beq2;  ctx.beq3 = beq3;  ctx.bmag_ = bmag_;
    ctx.fofc_ = fofc_;  ctx.touched_ = touched_;  ctx.fb_level_ = fb_level_;
    ctx.cnt_ = cnt_;  ctx.mbsize_ = size_.d_view;
    ctx.wb = wb;  ctx.ho_pad = ho_pad;  ctx.fv_bavg = fv_bavg;  ctx.multi_d = multi_d;
    ctx.three_d = three_d;  ctx.newtonian = newtonian;  ctx.nad_bmag = nad_bmag;
    ctx.nad_vmag = nad_vmag;  ctx.is_ideal_ = is_ideal_;  ctx.use_sed = use_sed;
    ctx.masked = masked;  ctx.frozen = frozen;  ctx.may_demote = may_demote;
    ctx.local_rcfl = local_rcfl;
    ctx.gm_ = gm_;  ctx.cs_iso_ = cs_iso_;  ctx.atol = atol;  ctx.eps0 = eps0;
    ctx.eps0_eq = eps0_eq;  ctx.rtol = rtol;  ctx.step_dt = step_dt;
    ctx.npass_he = npass_he;  ctx.npass_hb = npass_hb;  ctx.npass = npass;
    ctx.npass_tot = npass_tot;  ctx.nmhd_ = nmhd_;  ctx.nhe = nhe;  ctx.s_b = s_b;
    ctx.s_v = s_v;  ctx.s_sc = s_sc;  ctx.ntest = ntest;  ctx.n_fb = n_fb;
    ctx.is = is;  ctx.ie = ie;  ctx.js = js;  ctx.je = je;  ctx.ks = ks;  ctx.ke = ke;
    const MoodNadPassCtx ctx_ = ctx;

    // The detection logic lives in MoodNadPassCtx (Prep / Test / Combine); only the
    // launch shape depends on the backend.  GPU: three kernels, so that the NAD passes
    // of one cell run as independent threads -- (a) per cell: status + band scale;
    // (b) per (cell, pass) with the pass as the outer parallel index, the threads of a
    // warp sharing one pass (no divergence) and reading consecutive i; (c) per cell:
    // combine, demote,
    // count.  CPU: one kernel per cell doing (a), the pass loop (stopping at the first
    // failure) and (c), which keeps the cell's state in registers.  The decision is the
    // OR of the same tests either way, so the trouble set and the counters are
    // bit-identical across backends and against the former single sequential kernel.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
    constexpr bool pass_parallel = true;
#else
    constexpr bool pass_parallel = false;
#endif
    if (pass_parallel) {
      par_for("mmood_detect_prep", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real cr;
        stat_(m,k,j,i) = ctx_.Prep(m, k, j, i, cr);
        crtol_(m,k,j,i) = cr;
        nadf_(m,k,j,i) = 0;
      });
      // pass as the outer parallel index: every warp holds one pass over consecutive i
      // (on an idle GH200 this beat nesting the pass inside the row loop by ~1.5%)
      par_for("mmood_detect_pass", DevExeSpace(), 0, nmb1, 0, npass_tot-1, kl, ku, jl, ju,
              il, iu,
      KOKKOS_LAMBDA(const int m, const int pass, const int k, const int j, const int i) {
        if (stat_(m,k,j,i) != ST_TEST) return;
        const Real dtst = ctx_.CandidateDensity(m, k, j, i);
        // several passes of one cell may raise the flag; every writer stores the same
        // value
        if (ctx_.Test(m, pass, k, j, i, crtol_(m,k,j,i), dtst)) {
          Kokkos::atomic_store(&nadf_(m,k,j,i), 1);
        }
      });
      par_for("mmood_detect_combine", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const int st = stat_(m,k,j,i);
        if (st == ST_SKIP) return;
        ctx_.Combine(m, k, j, i, st, nadf_(m,k,j,i) != 0);
      });
    } else {
      par_for("mmood_detect", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real cr;
        const int st = ctx_.Prep(m, k, j, i, cr);
        if (st == ST_SKIP) return;
        bool nad = false;
        if (st == ST_TEST) {
          const Real dtst = ctx_.CandidateDensity(m, k, j, i);
          for (int pass=0; pass<npass_tot && !nad; ++pass) {
            nad = ctx_.Test(m, pass, k, j, i, cr, dtst);
          }
        }
        ctx_.Combine(m, k, j, i, st, nad);
      });
    }
    auto cnt_h = Kokkos::create_mirror_view(cnt_);
    Kokkos::deep_copy(cnt_h, cnt_);
    const int ndemoted = cnt_h(0), ndemoted_int = cnt_h(1);
    const int nnad = cnt_h(2), npad = cnt_h(3), ndc = cnt_h(4);

    if (mood_frozen_emf) {
      mood_nad_requests = nnad;
      mood_pad_requests = npad;
      mood_dc_bad = ndc;
      ndemoted_total += ndemoted_int;
      break;  // exchange owner levels before revising shared faces and halo edges
    }
    if (ndemoted == 0) break;
    ndemoted_total += ndemoted_int;

    //------------------------------------------------------------------------------------
    // (4) revise every face touching a newly-demoted cell at the demoted tier: conserved
    // fluxes, face-centered E-fields, and (UCT) the face coefficients, all through the
    // same per-face Riemann re-solve as the base scheme.

    revise_faces(dh, 0);
    if (mood_frozen_emf) select_frozen_edges(dh);

  }

  pmy_pack->pmesh->ecounter.nmood += ndemoted_total;

  return;
}

// function definitions for each template parameter
template void MHD::MOODLoop<MHD_RSolver::advect>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::llf>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlle>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlld>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::lhlld>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::llf_sr>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::llf_gr>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace mhd
