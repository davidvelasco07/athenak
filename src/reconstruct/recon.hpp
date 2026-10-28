#ifndef RECONSTRUCT_RECON_HPP_
#define RECONSTRUCT_RECON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file recon.hpp
//! \brief Per-cell reconstruction supporting every reconstruction method (DC, PLM, PPM4,
//! PPMX, WENOZ, TENO, and the unlimited FV4..FV9).  Each cell produces ql at its right
//! face and qr at its left face, written to the global per-face L/R buffers.  Templated
//! on direction (ivx = IVX|IVY|IVZ); the method is selected at runtime via a
//! grid-uniform branch (coherent across threads, so effectively branch-free on GPU).

#include "athena.hpp"
#include "eos/eos.hpp"
#include "plm.hpp"    // PLM()
#include "ppm.hpp"    // PPM4(), PPMX()
#include "wenoz.hpp"  // WENOZ()
#include "teno.hpp"   // TENO()
#include "fv.hpp"     // FV4() .. FV9()

//----------------------------------------------------------------------------------------
//! \fn ReconCellT<recon,ivx>()
//! \brief Single-variable, single-cell reconstruction with the method selected at
//! COMPILE TIME.  Reads the stencil of variable `n` centered on cell (m,k,j,i) in
//! direction `ivx`, and writes ql to face (i+1) and qr to face (i) in the global
//! per-face buffers.  Buffers are indexed by the GLOBAL cell/face index (m,n,k,j,i);
//! they are sized to the full cell range (including ghosts) so any face touched by
//! the reconstruction loop is in bounds (requires nghost >= 2).  The variable index
//! `n` is supplied by the caller (it is a parallel index of the launching kernel), so
//! this function handles exactly one variable.  Caller loops over the cells whose
//! faces are needed (one extra cell on the normal axis):
//!   ivx=IVX:  i in [is-1, ie+1], (transverse j,k over their active+ghost range)
//!   ivx=IVY:  j in [js-1, je+1], (transverse i,k over their active+ghost range)
//!   ivx=IVZ:  k in [ks-1, ke+1], (transverse i,j over their active+ghost range)
//!
//! Stencil half-width (see ReconStencilRadius): DC 0, PLM 1, PPM/WENOZ/TENO/FV4/FV5 2,
//! FV6/FV7 3, FV8/FV9 4.  The Hydro/MHD constructors require nghost at least one larger
//! than that half-width.  The method-dispatch is a constexpr-if on the template
//! parameter, so no per-cell branch survives on the device.
template <ReconstructionMethod recon, int ivx>
KOKKOS_INLINE_FUNCTION
void ReconCellT(const EOS_Data &eos, const bool apply_floors,
                const int m, const int n, const int k, const int j, const int i,
                const DvceArray5D<Real> &q,
                const DvceArray5D<Real> &wl,
                const DvceArray5D<Real> &wr) {
  // Compile-time stencil offsets along the reconstruction direction
  constexpr int di = (ivx == IVX) ? 1 : 0;
  constexpr int dj = (ivx == IVY) ? 1 : 0;
  constexpr int dk = (ivx == IVZ) ? 1 : 0;

  const Real dfloor = eos.dfloor;
  const Real efloor = eos.is_ideal ? (eos.pfloor/(eos.gamma - 1.0)) : 0.0;

  Real ql_val, qr_val;
  if constexpr (recon == ReconstructionMethod::dc) {
    ql_val = q(m, n, k, j, i);
    qr_val = q(m, n, k, j, i);
  } else if constexpr (recon == ReconstructionMethod::plm) {
    PLM(q(m, n, k - dk, j - dj, i - di),
        q(m, n, k,      j,      i),
        q(m, n, k + dk, j + dj, i + di),
        ql_val, qr_val);
  } else if constexpr (recon == ReconstructionMethod::ppm4) {
    PPM4(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
         q(m, n, k -   dk, j -   dj, i -   di),
         q(m, n, k,        j,        i),
         q(m, n, k +   dk, j +   dj, i +   di),
         q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
         ql_val, qr_val);
  } else if constexpr (recon == ReconstructionMethod::ppmx) {
    PPMX(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
         q(m, n, k -   dk, j -   dj, i -   di),
         q(m, n, k,        j,        i),
         q(m, n, k +   dk, j +   dj, i +   di),
         q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
         ql_val, qr_val);
    if (apply_floors) {
      if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
      if (eos.is_ideal && n == IEN) {
        ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
      }
    }
  } else if constexpr (recon == ReconstructionMethod::wenoz) {
    WENOZ(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          ql_val, qr_val);
    if (apply_floors) {
      if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
      if (eos.is_ideal && n == IEN) {
        ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
      }
    }
  } else if constexpr (recon == ReconstructionMethod::teno) {
    TENO(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          ql_val, qr_val);
    if (apply_floors) {
      if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
      if (eos.is_ideal && n == IEN) {
        ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
      }
    }
  } else if constexpr (IsUnlimitedFV(recon)) {
    // Unlimited cell-polynomial reconstruction (fv.hpp): cell i evaluates its own
    // polynomial at its two faces, so the two cells sharing a face give different
    // states and the Riemann solver adds O(h^N) upwind dissipation.
    // Stability relies entirely on the MOOD a-posteriori fallback (enforced in the
    // Hydro/MHD constructor).
    if constexpr (recon == ReconstructionMethod::fv4) {
      FV4(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          ql_val, qr_val);
    } else if constexpr (recon == ReconstructionMethod::fv5) {
      FV5(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          ql_val, qr_val);
    } else if constexpr (recon == ReconstructionMethod::fv6) {
      FV6(q(m, n, k - 3*dk, j - 3*dj, i - 3*di),
          q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          q(m, n, k + 3*dk, j + 3*dj, i + 3*di),
          ql_val, qr_val);
    } else if constexpr (recon == ReconstructionMethod::fv7) {
      FV7(q(m, n, k - 3*dk, j - 3*dj, i - 3*di),
          q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          q(m, n, k + 3*dk, j + 3*dj, i + 3*di),
          ql_val, qr_val);
    } else if constexpr (recon == ReconstructionMethod::fv9) {
      FV9(q(m, n, k - 4*dk, j - 4*dj, i - 4*di),
          q(m, n, k - 3*dk, j - 3*dj, i - 3*di),
          q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          q(m, n, k + 3*dk, j + 3*dj, i + 3*di),
          q(m, n, k + 4*dk, j + 4*dj, i + 4*di),
          ql_val, qr_val);
    } else {
      FV8(q(m, n, k - 4*dk, j - 4*dj, i - 4*di),
          q(m, n, k - 3*dk, j - 3*dj, i - 3*di),
          q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          q(m, n, k + 3*dk, j + 3*dj, i + 3*di),
          q(m, n, k + 4*dk, j + 4*dj, i + 4*di),
          ql_val, qr_val);
    }
    if (apply_floors) {
      if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
      if (eos.is_ideal && n == IEN) {
        ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
      }
    }
  } else {
    ql_val = q(m, n, k, j, i);
    qr_val = q(m, n, k, j, i);
  }
  // ql is the right-face value of cell (writes the LEFT state of face i+1);
  // qr is the left-face value of cell (writes the RIGHT state of face i).
  wl(m, n, k + dk, j + dj, i + di) = ql_val;
  wr(m, n, k, j, i) = qr_val;
}

//----------------------------------------------------------------------------------------
//! \fn ReconDispatch<ivx>()
//! \brief Select the (runtime, grid-uniform) reconstruction method on the host and launch
//! the per-cell, per-variable reconstruction kernel with the chosen method fixed at
//! COMPILE time, so the dispatch happens here (host side) rather than per-cell on device.
//! Each case keeps only the selected method's code in its kernel (instead of inlining all
//! methods via a runtime switch), avoiding the register/footprint bloat that would defeat
//! the purpose of templating ReconCellT.  Shared by hydro, MHD, and dyn-GRMHD.
//!
//! The par_for kernels are written directly in this function template, which has external
//! (vague) linkage: nvcc forbids extended __device__ lambdas inside functions with
//! internal linkage (e.g. an anonymous namespace).  The variable index `n` is the inner
//! parallel index, run over [0, nvars-1]; `apply_floors` toggles the WENOZ, etc. floors
//! (true for fluid primitives, false for the cell-centered B field).
template <int ivx>
inline void ReconDispatch(ReconstructionMethod recon, const char *name, int nmb1,
    int kl, int ku, int jl, int ju, int il, int iu,
    const EOS_Data &eos, bool apply_floors, int nvars,
    const DvceArray5D<Real> &q,
    const DvceArray5D<Real> &ql, const DvceArray5D<Real> &qr) {
  switch (recon) {
    case ReconstructionMethod::plm:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::plm, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::ppm4:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::ppm4, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::ppmx:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::ppmx, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::wenoz:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::wenoz, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::teno:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::teno, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::fv4:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::fv4, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::fv5:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::fv5, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::fv6:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::fv6, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::fv7:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::fv7, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::fv8:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::fv8, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::fv9:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::fv9, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
    case ReconstructionMethod::dc:
    default:
      par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          ReconCellT<ReconstructionMethod::dc, ivx>(
              eos, apply_floors, m, n, k, j, i, q, ql, qr);
        });
      break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn ReconFace<ivx>()
//! \brief Single-FACE reconstruction: computes the L state (from the cell on the left of
//! the face) and R state (from the cell on the right) of face (m,k,j,i) in direction
//! `ivx`, writing ONLY that face's slots wl(...,i) and wr(...,i).  Used by the MOOD
//! fallback to revise individual faces at a lower-order method without the write races
//! that per-cell ReconCellT would cause when driven from a face-indexed kernel (two
//! adjacent faces would both write the shared cell's other-face slot).
//! Only the fallback-tier methods (dc, plm) are supported; anything else falls back
//! to dc.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void ReconFace(const ReconstructionMethod recon,
               const int m, const int k, const int j, const int i,
               const int nvars,
               const DvceArray5D<Real> &q,
               const DvceArray5D<Real> &wl,
               const DvceArray5D<Real> &wr) {
  constexpr int di = (ivx == IVX) ? 1 : 0;
  constexpr int dj = (ivx == IVY) ? 1 : 0;
  constexpr int dk = (ivx == IVZ) ? 1 : 0;

  for (int n = 0; n < nvars; ++n) {
    Real wl_val, wr_val;
    if (recon == ReconstructionMethod::plm) {
      Real ql, qr;
      // L state of face i = right-face value (ql) of cell i-1
      PLM(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          ql, qr);
      wl_val = ql;
      // R state of face i = left-face value (qr) of cell i
      PLM(q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          ql, qr);
      wr_val = qr;
    } else {  // donor cell
      wl_val = q(m, n, k - dk, j - dj, i - di);
      wr_val = q(m, n, k, j, i);
    }
    wl(m, n, k, j, i) = wl_val;
    wr(m, n, k, j, i) = wr_val;
  }
}

#endif  // RECONSTRUCT_RECON_HPP_
