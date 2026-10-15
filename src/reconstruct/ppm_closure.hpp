#ifndef RECONSTRUCT_PPM_CLOSURE_HPP_
#define RECONSTRUCT_PPM_CLOSURE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ppm_closure.hpp
//! \brief One-sided closure of the unlimited PPM (reconstruct = fv4) at physical
//! boundaries whose ghost cells hold an IMPOSED state (`inflow`, `user`).
//!
//! Unlimited PPM interpolates one 4th-order value per face, so w_L = w_R and the
//! Riemann solver returns the pure physical flux.  Across a boundary whose ghost zones
//! are imposed from outside (e.g. an equilibrium held fixed while a wave leaves the
//! domain) the solution is not smooth: the centred stencil straddles that jump, puts a
//! mid-value on the boundary face with no upwinding, and seeds undamped odd-even noise
//! that the dissipation-free interior never removes.  The closure treats the boundary
//! face like an element face of a DG/SD scheme -- two one-sided extrapolations, so the
//! Riemann solver sees the jump:
//!   interior side of the boundary face   (25 q0 - 23 q1 + 13 q2 - 3 q3)/12
//!   ghost side of the boundary face      the same from the ghost cells
//!   first face on either side            (3 q0 + 13 q1 - 5 q2 + q3)/12  (biased)
//! with q0 the cell adjacent to the boundary on that side.  All three are 4th-order
//! accurate for smooth data.  Every other boundary keeps the centred stencil: there the
//! ghost data continue the interior solution (periodic/block copies, the mirror image of
//! `reflect`, the zero-gradient extension of `outflow`/`diode`), so there is no jump to
//! see.  Needs nghost >= 4.  Faces are written only where the stencil is one-sided, so
//! the call is a no-op for blocks without such a boundary.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn PPMBoundaryClosure<ivx>()
//! \brief Overwrite the face states near imposed-state physical boundaries along `ivx`,
//! after ReconDispatch<ivx>(ReconstructionMethod::fv4, ...).  (t1l,t1u) and (t2l,t2u) are
//! the ranges of the two transverse indices used by that reconstruction, in (k,j) order
//! for x1, (k,i) for x2 and (j,i) for x3.

template <int ivx>
inline void PPMBoundaryClosure(const char *name, int nmb1, int t1l, int t1u, int t2l,
                               int t2u, const RegionIndcs &indcs,
                               const DualArray2D<BoundaryFlag> &mb_bcs,
                               const EOS_Data &eos, bool apply_floors, int nvars,
                               const DvceArray5D<Real> &q, const DvceArray5D<Real> &ql,
                               const DvceArray5D<Real> &qr) {
  const int lo = (ivx == IVX) ? indcs.is : ((ivx == IVY) ? indcs.js : indcs.ks);
  const int hi = (ivx == IVX) ? indcs.ie : ((ivx == IVY) ? indcs.je : indcs.ke);
  const int fin = (ivx == IVX) ? BoundaryFace::inner_x1
                : ((ivx == IVY) ? BoundaryFace::inner_x2 : BoundaryFace::inner_x3);
  const Real dfloor = eos.dfloor;
  const Real efloor = eos.is_ideal ? (eos.pfloor/(eos.gamma - 1.0)) : 0.0;
  const bool ideal = eos.is_ideal;
  auto bcs = mb_bcs.d_view;
  par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, t1l, t1u, t2l, t2u,
  KOKKOS_LAMBDA(int m, int n, int a, int b) {
    for (int side = 0; side < 2; ++side) {
      const BoundaryFlag f = bcs(m, fin + side);
      if (f != BoundaryFlag::inflow && f != BoundaryFlag::user) {
        continue;
      }
      // (a,b,c) -> (k,j,i) with c along ivx.  Plain arithmetic, no `if constexpr`: a CUDA
      // extended lambda may not first-capture q/ql/qr inside a constexpr-if branch.
      auto K = [&](const int c) {return (ivx == IVZ) ? c : a;};
      auto J = [&](const int c) {return (ivx == IVY) ? c : ((ivx == IVX) ? b : a);};
      auto I = [&](const int c) {return (ivx == IVX) ? c : b;};
      auto Q = [&](const int c) -> Real {return q(m, n, K(c), J(c), I(c));};
      auto set = [&](const DvceArray5D<Real> &w, const int c, Real v) {
        if (apply_floors) {
          if (n == IDN) v = fmax(v, dfloor);
          if (ideal && n == IEN) v = fmax(v, efloor);
        }
        w(m, n, K(c), J(c), I(c)) = v;
      };
      // boundary face bf; interior cells I(d), ghost cells G(d), d = 0 next to the face
      const int bf = (side == 0) ? lo : hi + 1;
      const int si = (side == 0) ? 1 : -1;               // step into the interior
      const int i0 = (side == 0) ? bf : bf - 1;
      const int g0 = (side == 0) ? bf - 1 : bf;
      const Real qi0 = Q(i0), qi1 = Q(i0 + si), qi2 = Q(i0 + 2*si), qi3 = Q(i0 + 3*si);
      const Real qg0 = Q(g0), qg1 = Q(g0 - si), qg2 = Q(g0 - 2*si), qg3 = Q(g0 - 3*si);
      const Real ei = (25.0*qi0 - 23.0*qi1 + 13.0*qi2 - 3.0*qi3)/12.0;
      const Real eg = (25.0*qg0 - 23.0*qg1 + 13.0*qg2 - 3.0*qg3)/12.0;
      const Real vi = (3.0*qi0 + 13.0*qi1 - 5.0*qi2 + qi3)/12.0;
      const Real vg = (3.0*qg0 + 13.0*qg1 - 5.0*qg2 + qg3)/12.0;
      // face slots: ql(f) = state on the low side of face f, qr(f) = high side
      if (side == 0) {
        set(ql, bf, eg);  set(qr, bf, ei);
      } else {
        set(ql, bf, ei);  set(qr, bf, eg);
      }
      set(ql, bf + si, vi);  set(qr, bf + si, vi);       // first interior face
      set(ql, bf - si, vg);  set(qr, bf - si, vg);       // first ghost face
    }
  });
}

#endif // RECONSTRUCT_PPM_CLOSURE_HPP_
