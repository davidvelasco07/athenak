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
//!   face-centered electric fields (the FOFC estimate, mhd_fofc.cpp).
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
//! Cross-block consistency: shared-edge EMFs are synchronized between MeshBlocks by
//! SendE/RecvE, so div(B) and conservation hold regardless; the light-cone halo is
//! sized (mhd.cpp) so the COMPOSED edge values are also decomposition-invariant —
//! the final-iteration halo equals the transverse reach of the edge composition.

#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "mhd/rsolvers/solve_face_mhd.hpp"
#include "mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

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
  auto &bcc0_ = bcc0;
  auto &b1_ = b1;
  auto &utest_ = utest;
  auto &bcctest_ = bcctest;
  auto &fofc_ = fofc;
  auto &fb_level_ = fb_level;
  auto &bmag_ = bmag_ref;
  auto wl_ = wl_split;
  auto wr_ = wr_split;
  auto bl_ = bl_split;
  auto br_ = br_split;
  auto b0x1 = b0.x1f;
  auto b0x2 = b0.x2f;
  auto b0x3 = b0.x3f;
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
  const Real atol = mood_atol;
  const Real theta = mood_nad_theta;
  const bool nad_energy = mood_nad_energy;
  const bool nad_bmag = (mood_nad_b == 0);
  const bool nad_von = (mood_nad_v > 0);      // velocity in NAD
  const bool nad_vmag = (mood_nad_v == 1);    // |v| (else components)
  const int n_fb = n_fb_tiers;
  const int nrevs = mood_max_revs;
  const bool is_ideal_ = eos_.is_ideal;
  const int scale_mode = mood_nad_scale;

  // UCT face-coefficient arrays for the revise kernels (empty views when ct_contact —
  // the solver then skips the UCT outputs).
  const int uct_fl = (emf_method == MHD_EMF::uct_hlld) ? 2 :
                     ((emf_method == MHD_EMF::uct_hll) ? 1 : 0);
  auto aL1 = aL_x1f;  auto dl1 = dL_x1f;  auto dr1 = dR_x1f;
  auto vt1_1 = vy_x1f;  auto vt2_1 = vz_x1f;
  auto aL2 = aL_x2f;  auto dl2 = dL_x2f;  auto dr2 = dR_x2f;
  auto vt1_2 = vz_x2f;  auto vt2_2 = vx_x2f;
  auto aL3 = aL_x3f;  auto dl3 = dL_x3f;  auto dr3 = dR_x3f;
  auto vt1_3 = vx_x3f;  auto vt2_3 = vy_x3f;

  // |B| of the stage-input state, used for NAD bounds and SED in "mag" mode.  Filled
  // over the full array so the SED stencil is covered anywhere detection runs.
  if (nad_bmag) {
    par_for("mood_bmag", DevExeSpace(), 0, nmb1, 0, ncells3-1, 0, ncells2-1,
            0, ncells1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      bmag_(m,k,j,i) = sqrt(SQR(bcc0_(m,IBX,k,j,i)) + SQR(bcc0_(m,IBY,k,j,i))
                          + SQR(bcc0_(m,IBZ,k,j,i)));
    });
  }

  // GLOBAL tolerance scales (see hydro_mood.cpp): index 0 = density, 1 = internal
  // energy, 2..4 = B (component ranges in comps mode; slot 2 = |B| range in mag mode).
  // All reduce over ACTIVE cells only.
  Real gscale0 = 0.0, gscale1 = 0.0;
  Real gscaleb[3] = {0.0, 0.0, 0.0};
  Real gscalev[3] = {0.0, 0.0, 0.0};   // velocity NAD scale (|v| in slot 0, or comps)
  if (scale_mode == 1 || scale_mode == 3) {
    Real gdmn = std::numeric_limits<Real>::max(), gdmx = -gdmn;
    Real gemn = gdmn, gemx = gdmx;
    Real gvmx = 0.0;
    const int ni = ie-is+1, nji = (je-js+1)*ni, nkji = (ke-ks+1)*nji;
    const int nmkji = nmb*nkji;
    const bool ideal = is_ideal_;
    const bool want_v = (scale_mode == 3);
    Kokkos::parallel_reduce("mmood_grange",
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

    // B ranges (separate reduction to keep the reducer count manageable)
    Real gb0mn = std::numeric_limits<Real>::max(), gb0mx = -gb0mn;
    Real gb1mn = gb0mn, gb1mx = gb0mx;
    Real gb2mn = gb0mn, gb2mx = gb0mx;
    const bool bmag_mode = nad_bmag;
    Kokkos::parallel_reduce("mmood_grangeb",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &b0mn, Real &b0mx, Real &b1mn, Real &b1mx,
                  Real &b2mn, Real &b2mx) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/ni;
      int i = (idx - m*nkji - k*nji - j*ni) + is;
      j += js; k += ks;
      if (bmag_mode) {
        Real b = bmag_(m,k,j,i);
        b0mn = fmin(b0mn, b); b0mx = fmax(b0mx, b);
      } else {
        Real bx = bcc0_(m,IBX,k,j,i);
        Real by = bcc0_(m,IBY,k,j,i);
        Real bz = bcc0_(m,IBZ,k,j,i);
        b0mn = fmin(b0mn, bx); b0mx = fmax(b0mx, bx);
        b1mn = fmin(b1mn, by); b1mx = fmax(b1mx, by);
        b2mn = fmin(b2mn, bz); b2mx = fmax(b2mx, bz);
      }
    }, Kokkos::Min<Real>(gb0mn), Kokkos::Max<Real>(gb0mx),
       Kokkos::Min<Real>(gb1mn), Kokkos::Max<Real>(gb1mx),
       Kokkos::Min<Real>(gb2mn), Kokkos::Max<Real>(gb2mx));
#if MPI_PARALLEL_ENABLED
    Real rmin[5] = {gdmn, gemn, gb0mn, gb1mn, gb2mn};
    Real rmax[6] = {gdmx, gemx, gb0mx, gb1mx, gb2mx, gvmx};
    MPI_Allreduce(MPI_IN_PLACE, rmin, 5, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rmax, 6, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    gdmn = rmin[0]; gemn = rmin[1]; gb0mn = rmin[2]; gb1mn = rmin[3]; gb2mn = rmin[4];
    gdmx = rmax[0]; gemx = rmax[1]; gb0mx = rmax[2]; gb1mx = rmax[3]; gb2mx = rmax[4];
    gvmx = rmax[5];
#endif
    gscale0 = fmax(gdmx - gdmn, 0.0);
    gscale1 = fmax(gemx - gemn, 0.0);
    gscaleb[0] = fmax(gb0mx - gb0mn, 0.0);
    gscaleb[1] = fmax(gb1mx - gb1mn, 0.0);
    gscaleb[2] = fmax(gb2mx - gb2mn, 0.0);

    // velocity ranges (stage-input primitive) for the NAD scale, same treatment as B
    if (nad_von) {
      Real gv0mn = std::numeric_limits<Real>::max(), gv0mx = -gv0mn;
      Real gv1mn = gv0mn, gv1mx = gv0mx;
      Real gv2mn = gv0mn, gv2mx = gv0mx;
      const bool vmag_mode = nad_vmag;
      Kokkos::parallel_reduce("mmood_grangev",
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
      Real fac = fmin(1.0, cfl_adv);
      gscale0 *= fac;
      gscale1 *= fac;
      gscaleb[0] *= fac;
      gscaleb[1] *= fac;
      gscaleb[2] *= fac;
      gscalev[0] *= fac;
      gscalev[1] *= fac;
      gscalev[2] *= fac;
    }
  }

  // Light-cone halo: the final iteration retains h_f = mood_halo0 - nrevs + 1 cells,
  // the transverse reach of the corner-EMF composition (see mhd.cpp).
  const int h_f = mood_halo0 - nrevs + 1;

  const int ndi = 1;
  const int ndj = (multi_d) ? 1 : 0;
  const int ndk = (three_d) ? 1 : 0;

  Kokkos::deep_copy(fb_level_, 0);

  int ndemoted_total = 0;
  for (int rev=1; rev<=nrevs; ++rev) {
    const int dh = h_f + (nrevs - rev);
    int il = is-dh, iu = ie+dh, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-dh; ju = je+dh; }
    if (three_d) { kl = ks-dh; ku = ke+dh; }

    const int ni   = (iu - il + 1);
    const int nji  = (ju - jl + 1)*ni;
    const int nkji = (ku - kl + 1)*nji;
    const int nmkji = nmb*nkji;

    //------------------------------------------------------------------------------------
    // (1) candidate update: conserved hydro variables from the flux divergence, and the
    // candidate cell-averaged B from face-E finite differences (the FOFC estimate).
    Kokkos::deep_copy(fofc_, false);
    par_for("mmood_newu", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size_.d_view(m).dx1;
      Real dtodx2 = beta_dt/size_.d_view(m).dx2;
      Real dtodx3 = beta_dt/size_.d_view(m).dx3;
      for (int n=0; n<nmhd_; ++n) {
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

      bcctest_(m,IBX,k,j,i) = gam0*bcc0_(m,IBX,k,j,i) + gam1*b1old;
      bcctest_(m,IBY,k,j,i) = gam0*bcc0_(m,IBY,k,j,i) + gam1*b2old;
      bcctest_(m,IBZ,k,j,i) = gam0*bcc0_(m,IBZ,k,j,i) + gam1*b3old;

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
    // average to cell centers.  The detector then sees the real evolved field -- including
    // the corner-EMF composition where the UCT dissipation coefficient can run away -- not
    // the FOFC face-E proxy above.  First cut: interior only (single-block-exact); the halo
    // keeps the proxy, so multi-block demotion in the outer halo is approximate until the
    // composition is extended over the light-cone halo (needs +3 flux/ghost coverage).
    if (uct_fl > 0 && multi_d) {
      ComposeCornerEMF(is, ie+1, js, je+1, ks, (three_d ? ke+1 : ks));
      auto bt1 = b0_test.x1f;  auto bt2 = b0_test.x2f;  auto bt3 = b0_test.x3f;
      auto e1c = efld.x1e;  auto e2c = efld.x2e;  auto e3c = efld.x3e;
      const bool td = three_d;
      // CT update of candidate face-B (mirrors mhd_ct.cpp)
      par_for("mmood_ctB1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real b = gam0*b0x1(m,k,j,i) + gam1*b1_.x1f(m,k,j,i);
        b -= beta_dt*(e3c(m,k,j+1,i) - e3c(m,k,j,i))/size_.d_view(m).dx2;
        if (td) b += beta_dt*(e2c(m,k+1,j,i) - e2c(m,k,j,i))/size_.d_view(m).dx3;
        bt1(m,k,j,i) = b;
      });
      par_for("mmood_ctB2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real b = gam0*b0x2(m,k,j,i) + gam1*b1_.x2f(m,k,j,i);
        b += beta_dt*(e3c(m,k,j,i+1) - e3c(m,k,j,i))/size_.d_view(m).dx1;
        if (td) b -= beta_dt*(e1c(m,k+1,j,i) - e1c(m,k,j,i))/size_.d_view(m).dx3;
        bt2(m,k,j,i) = b;
      });
      par_for("mmood_ctB3", DevExeSpace(), 0, nmb1, ks, ke+td, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real b = gam0*b0x3(m,k,j,i) + gam1*b1_.x3f(m,k,j,i);
        b -= beta_dt*(e2c(m,k,j,i+1) - e2c(m,k,j,i))/size_.d_view(m).dx1;
        b += beta_dt*(e1c(m,k,j+1,i) - e1c(m,k,j,i))/size_.d_view(m).dx2;
        bt3(m,k,j,i) = b;
      });
      // average candidate face-B to cell centers -> bcctest (interior)
      par_for("mmood_avgB", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        bcctest_(m,IBX,k,j,i) = 0.5*(bt1(m,k,j,i) + bt1(m,k,j,i+1));
        bcctest_(m,IBY,k,j,i) = 0.5*(bt2(m,k,j,i) + bt2(m,k,j+1,i));
        bcctest_(m,IBZ,k,j,i) = (td) ? 0.5*(bt3(m,k,j,i) + bt3(m,k+1,j,i))
                                     : bt3(m,k,j,i);
      });
    }

    //------------------------------------------------------------------------------------
    // (1b) gdu scale from the first (pre-revision) candidate
    if (scale_mode == 2 && rev == 1) {
      Real gdd = 0.0, gde = 0.0, gb0 = 0.0, gb1 = 0.0, gb2 = 0.0;
      const int nia = ie-is+1, njia = (je-js+1)*nia, nkjia = (ke-ks+1)*njia;
      const int nmkjia = nmb*nkjia;
      const bool ideal = is_ideal_;
      const bool newt = newtonian;
      const bool bmag_mode = nad_bmag;
      Kokkos::parallel_reduce("mmood_gdu",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkjia),
      KOKKOS_LAMBDA(const int &idx, Real &dd, Real &de, Real &db0, Real &db1,
                    Real &db2) {
        int m = idx/nkjia;
        int k = (idx - m*nkjia)/njia;
        int j = (idx - m*nkjia - k*njia)/nia;
        int i = (idx - m*nkjia - k*njia - j*nia) + is;
        j += js; k += ks;
        Real d = fabs(utest_(m,IDN,k,j,i) - u0_(m,IDN,k,j,i));
        if (isfinite(d)) dd = fmax(dd, d);
        if (ideal) {
          Real enew, eold;
          if (newt) {
            Real dtst = utest_(m,IDN,k,j,i);
            enew = (dtst > 0.0) ?
                   utest_(m,IEN,k,j,i) - 0.5*(SQR(utest_(m,IM1,k,j,i))
                   + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dtst
                   - 0.5*(SQR(bcctest_(m,IBX,k,j,i)) + SQR(bcctest_(m,IBY,k,j,i))
                        + SQR(bcctest_(m,IBZ,k,j,i))) : 0.0;
            eold = (dtst > 0.0) ? w0_(m,IEN,k,j,i) : 0.0;
          } else {
            enew = utest_(m,IEN,k,j,i);
            eold = u0_(m,IEN,k,j,i);
          }
          Real e = fabs(enew - eold);
          if (isfinite(e)) de = fmax(de, e);
        }
        if (bmag_mode) {
          Real bnew = sqrt(SQR(bcctest_(m,IBX,k,j,i)) + SQR(bcctest_(m,IBY,k,j,i))
                         + SQR(bcctest_(m,IBZ,k,j,i)));
          Real b = fabs(bnew - bmag_(m,k,j,i));
          if (isfinite(b)) db0 = fmax(db0, b);
        } else {
          Real bx = fabs(bcctest_(m,IBX,k,j,i) - bcc0_(m,IBX,k,j,i));
          Real by = fabs(bcctest_(m,IBY,k,j,i) - bcc0_(m,IBY,k,j,i));
          Real bz = fabs(bcctest_(m,IBZ,k,j,i) - bcc0_(m,IBZ,k,j,i));
          if (isfinite(bx)) db0 = fmax(db0, bx);
          if (isfinite(by)) db1 = fmax(db1, by);
          if (isfinite(bz)) db2 = fmax(db2, bz);
        }
      }, Kokkos::Max<Real>(gdd), Kokkos::Max<Real>(gde), Kokkos::Max<Real>(gb0),
         Kokkos::Max<Real>(gb1), Kokkos::Max<Real>(gb2));
#if MPI_PARALLEL_ENABLED
      Real rmx[5] = {gdd, gde, gb0, gb1, gb2};
      MPI_Allreduce(MPI_IN_PLACE, rmx, 5, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      gdd = rmx[0]; gde = rmx[1]; gb0 = rmx[2]; gb1 = rmx[3]; gb2 = rmx[4];
#endif
      gscale0 = gdd;
      gscale1 = gde;
      gscaleb[0] = gb0;
      gscaleb[1] = gb1;
      gscaleb[2] = gb2;

      if (nad_von) {
        Real gvd0 = 0.0, gvd1 = 0.0, gvd2 = 0.0;
        const bool vmag_mode2 = nad_vmag;
        Kokkos::parallel_reduce("mmood_gduv",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkjia),
        KOKKOS_LAMBDA(const int &idx, Real &vd0, Real &vd1, Real &vd2) {
          int m = idx/nkjia;
          int k = (idx - m*nkjia)/njia;
          int j = (idx - m*nkjia - k*njia)/nia;
          int i = (idx - m*nkjia - k*njia - j*nia) + is;
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

    //------------------------------------------------------------------------------------
    // (2) PAD: flag cells where conversion to primitives would require floors
    // (sets fofc; b0/w0 passed but not used/changed)
    peos->ConsToPrim(utest_, b0, w0_, bcctest_, true, il, iu, jl, ju, kl, ku);

    //------------------------------------------------------------------------------------
    // (3) NaN + NAD detection; combine with PAD; demote survivors and count them.
    // Same tier-floor split as hydro: PAD/NaN may cascade to DC, NAD (density, energy,
    // magnetic field) demotes to the PLM tier only.
    int ndemoted = 0, ndemoted_int = 0;
    const Real gsb0 = gscaleb[0], gsb1 = gscaleb[1], gsb2 = gscaleb[2];
    const Real gsv0 = gscalev[0], gsv1 = gscalev[1], gsv2 = gscalev[2];
    Kokkos::parallel_reduce("mmood_detect",
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

      bool danger = fofc_(m,k,j,i);  // PAD

      if (!danger) {
        for (int n=0; n<nmhd_; ++n) {
          if (!isfinite(utest_(m,n,k,j,i))) { danger = true; }
        }
        if (!isfinite(bcctest_(m,IBX,k,j,i)) || !isfinite(bcctest_(m,IBY,k,j,i)) ||
            !isfinite(bcctest_(m,IBZ,k,j,i))) { danger = true; }
      }

      bool trouble = danger;

      // NAD passes: 0 = density, 1 = internal energy (ideal gas), then B (|B| in mag
      // mode, three components in comps mode).  Bounds from the stage-input state.
      if (!trouble && lv < 1) {
        const Real dtst = utest_(m,IDN,k,j,i);
        const int npass_he = (is_ideal_ && nad_energy) ? 2 : 1;
        const int npass_hb = npass_he + (nad_bmag ? 1 : 3);
        const int npass = npass_hb + (nad_von ? (nad_vmag ? 1 : 3) : 0);
        for (int pass=0; pass<npass && !trouble; ++pass) {
          const bool vpass = (pass >= npass_hb);              // velocity pass
          const bool bpass = (!vpass) && (pass >= npass_he);  // magnetic pass
          const int bidx = pass - npass_he;  // 0 in mag mode; 0..2 in comps mode
          const int vidx = pass - npass_hb;  // 0 in |v| mode; 0..2 in comps mode
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
              q_new = dtst;
            } else {  // internal energy density (subtract kinetic AND magnetic)
              q_new = utest_(m,IEN,k,j,i) - 0.5*(SQR(utest_(m,IM1,k,j,i))
                    + SQR(utest_(m,IM2,k,j,i)) + SQR(utest_(m,IM3,k,j,i)))/dtst
                    - 0.5*(SQR(bcctest_(m,IBX,k,j,i)) + SQR(bcctest_(m,IBY,k,j,i))
                         + SQR(bcctest_(m,IBZ,k,j,i)));
            }
          } else if (nad_bmag) {
            q_new = sqrt(SQR(bcctest_(m,IBX,k,j,i)) + SQR(bcctest_(m,IBY,k,j,i))
                       + SQR(bcctest_(m,IBZ,k,j,i)));
          } else {
            q_new = bcctest_(m,IBX+bidx,k,j,i);
          }
          // DMP bounds from the stage-input state over the neighborhood
          Real qmn, qmx;
          if (vpass) {
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
          } else if (!bpass) {
            const int n = (pass == 0) ? IDN : IEN;
            const DvceArray5D<Real> &qref = (newtonian) ? w0_ : u0_;
            qmn = qref(m,n,k,j,i); qmx = qmn;
            for (int dk=-ndk; dk<=ndk; ++dk) {
            for (int dj=-ndj; dj<=ndj; ++dj) {
            for (int di=-ndi; di<=ndi; ++di) {
              Real q = qref(m,n,k+dk,j+dj,i+di);
              qmn = fmin(qmn, q);
              qmx = fmax(qmx, q);
            }}}
          } else if (nad_bmag) {
            qmn = bmag_(m,k,j,i); qmx = qmn;
            for (int dk=-ndk; dk<=ndk; ++dk) {
            for (int dj=-ndj; dj<=ndj; ++dj) {
            for (int di=-ndi; di<=ndi; ++di) {
              Real q = bmag_(m,k+dk,j+dj,i+di);
              qmn = fmin(qmn, q);
              qmx = fmax(qmx, q);
            }}}
          } else {
            const int n = IBX + bidx;
            qmn = bcc0_(m,n,k,j,i); qmx = qmn;
            for (int dk=-ndk; dk<=ndk; ++dk) {
            for (int dj=-ndj; dj<=ndj; ++dj) {
            for (int di=-ndi; di<=ndi; ++di) {
              Real q = bcc0_(m,n,k+dk,j+dj,i+di);
              qmn = fmin(qmn, q);
              qmx = fmax(qmx, q);
            }}}
          }
          // tolerance (see hydro_mood.cpp for the scale-mode rationale)
          Real gsc;
          if (vpass) {
            gsc = (vidx == 0) ? gsv0 : ((vidx == 1) ? gsv1 : gsv2);
          } else if (!bpass) {
            gsc = (pass == 0) ? gscale0 : gscale1;
          } else {
            gsc = (bidx == 0) ? gsb0 : ((bidx == 1) ? gsb1 : gsb2);
          }
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
          // exempt smooth extrema of this variable from NAD (never from PAD/NaN)
          if (nad && use_sed) {
            Real alpha;
            if (vpass) {
              // no precomputed |v| array; exempt smooth extrema only in comps mode
              alpha = nad_vmag ? 0.0
                    : SEDAlpha(w0_, m, IVX+vidx, k, j, i, multi_d, three_d);
            } else if (!bpass) {
              const int n = (pass == 0) ? IDN : IEN;
              const DvceArray5D<Real> &qref = (newtonian) ? w0_ : u0_;
              alpha = SEDAlpha(qref, m, n, k, j, i, multi_d, three_d);
            } else if (nad_bmag) {
              alpha = SEDAlpha4(bmag_, m, k, j, i, multi_d, three_d);
            } else {
              alpha = SEDAlpha(bcc0_, m, IBX+bidx, k, j, i, multi_d, three_d);
            }
            nad = (alpha < 1.0);
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
    // (4) revise every face touching a newly-demoted cell at the demoted tier: conserved
    // fluxes, face-centered E-fields, and (UCT) the face coefficients, all through the
    // same per-face Riemann re-solve as the base scheme.

    // x1 faces
    par_for("mmood_rev_x1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, is-dh, ie+1+dh,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (fofc_(m,k,j,i-1) || fofc_(m,k,j,i)) {
        int lv_l = fb_level_(m,k,j,i-1), lv_r = fb_level_(m,k,j,i);
        int tier = (lv_l > lv_r) ? lv_l : lv_r;
        ReconstructionMethod frecon = (tier >= n_fb) ?
            ReconstructionMethod::dc : ReconstructionMethod::plm;
        ReconFace<IVX>(frecon, m, k, j, i, nvars, w0_, wl_, wr_);
        ReconFace<IVX>(frecon, m, k, j, i, 3, bcc0_, bl_, br_);
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
                                           uct_fl, aL1, dl1, dr1, vt1_1, vt2_1);
        for (int n=nmhd_; n<nvars; ++n) {
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
      par_for("mmood_rev_x2", DevExeSpace(), 0, nmb1, kl, ku, js-dh, je+1+dh, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (fofc_(m,k,j-1,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k,j-1,i), lv_r = fb_level_(m,k,j,i);
          int tier = (lv_l > lv_r) ? lv_l : lv_r;
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          ReconFace<IVY>(frecon, m, k, j, i, nvars, w0_, wl_, wr_);
          ReconFace<IVY>(frecon, m, k, j, i, 3, bcc0_, bl_, br_);
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
                                             uct_fl, aL2, dl2, dr2, vt1_2, vt2_2);
          for (int n=nmhd_; n<nvars; ++n) {
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
      par_for("mmood_rev_x3", DevExeSpace(), 0, nmb1, ks-dh, ke+1+dh, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (fofc_(m,k-1,j,i) || fofc_(m,k,j,i)) {
          int lv_l = fb_level_(m,k-1,j,i), lv_r = fb_level_(m,k,j,i);
          int tier = (lv_l > lv_r) ? lv_l : lv_r;
          ReconstructionMethod frecon = (tier >= n_fb) ?
              ReconstructionMethod::dc : ReconstructionMethod::plm;
          ReconFace<IVZ>(frecon, m, k, j, i, nvars, w0_, wl_, wr_);
          ReconFace<IVZ>(frecon, m, k, j, i, 3, bcc0_, bl_, br_);
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
                                             uct_fl, aL3, dl3, dr3, vt1_3, vt2_3);
          for (int n=nmhd_; n<nvars; ++n) {
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
template void MHD::MOODLoop<MHD_RSolver::advect>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::llf>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlle>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlld>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::llf_sr>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::llf_gr>(Driver *pdriver, int stage);
template void MHD::MOODLoop<MHD_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace mhd
