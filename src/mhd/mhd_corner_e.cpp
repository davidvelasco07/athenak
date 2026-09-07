//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_corner_e.cpp
//  \brief
//  Also includes contributions to electric field from "source terms" such as the
//  shearing box.

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "reconstruct/teno.hpp"

//----------------------------------------------------------------------------------------
//! \fn ReconstructEdge()
//! \brief Reconstruct a scalar field to L/R states at the interface between positions
//! i-1 and i (i.e., at i-1/2), using the same spatial reconstruction as the main solver.
//!
//! Takes 6 values at positions i-3, i-2, i-1, i, i+1, i+2 (PLM only uses i-2..i+1).
//! Returns: qL = left state at i-1/2 (reconstructed from cell i-1 toward right)
//!          qR = right state at i-1/2 (reconstructed from cell i toward left)

KOKKOS_INLINE_FUNCTION
void ReconstructEdge(const ReconstructionMethod recon,
                     const Real &q_im3, const Real &q_im2,
                     const Real &q_im1, const Real &q_i,
                     const Real &q_ip1, const Real &q_ip2,
                     Real &qL, Real &qR) {
  Real dum;  // unused output
  switch (recon) {
    case ReconstructionMethod::dc:
      qL = q_im1;
      qR = q_i;
      break;
    case ReconstructionMethod::plm:
      PLM(q_im2, q_im1, q_i, qL, dum);    // center i-1 → ql at i-1/2
      PLM(q_im1, q_i, q_ip1, dum, qR);    // center i   → qr at i-1/2
      break;
    case ReconstructionMethod::ppm4:
      PPM4(q_im3, q_im2, q_im1, q_i, q_ip1, qL, dum);    // center i-1 → ql at i-1/2
      PPM4(q_im2, q_im1, q_i, q_ip1, q_ip2, dum, qR);    // center i   → qr at i-1/2
      break;
    case ReconstructionMethod::ppmx:
      PPMX(q_im3, q_im2, q_im1, q_i, q_ip1, qL, dum);
      PPMX(q_im2, q_im1, q_i, q_ip1, q_ip2, dum, qR);
      break;
    case ReconstructionMethod::ppm:
      // unlimited 4th-order interface interpolation; single-valued at the edge
      qL = (7.0*(q_im1 + q_i) - (q_im2 + q_ip1))/12.0;
      qR = qL;
      break;
    case ReconstructionMethod::wenoz:
      WENOZ(q_im3, q_im2, q_im1, q_i, q_ip1, qL, dum);
      WENOZ(q_im2, q_im1, q_i, q_ip1, q_ip2, dum, qR);
      break;
    case ReconstructionMethod::teno:
      TENO(q_im3, q_im2, q_im1, q_i, q_ip1, qL, dum);
      TENO(q_im2, q_im1, q_i, q_ip1, q_ip2, dum, qR);
      break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn InterpolateEdge()
//! \brief Interpolate a face-centered scalar to the edge at i-1/2 using
//! high-order reconstruction.  Returns the average of L/R states, which gives
//! a high-order pointwise estimate at the interface for smooth fields.

KOKKOS_INLINE_FUNCTION
Real InterpolateEdge(const ReconstructionMethod recon,
                     const Real &q_im3, const Real &q_im2,
                     const Real &q_im1, const Real &q_i,
                     const Real &q_ip1, const Real &q_ip2) {
  Real qL, qR;
  ReconstructEdge(recon, q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, qL, qR);
  return 0.5*(qL + qR);
}

//----------------------------------------------------------------------------------------
//! \fn InterpolateEdgeA() / InterpolateEdgeD()
//! \brief Physically-clamped variants for the UCT composition coefficients: 'a' is a
//! convex flux weight in [0,1], 'd' a non-negative dissipation coefficient.  Unlimited
//! edge interpolation can overshoot across discontinuous coefficient data (e.g. at
//! faces revised by the MOOD fallback), which would make the corner EMF composition
//! anti-dissipative and destabilize the CT update.

KOKKOS_INLINE_FUNCTION
Real InterpolateEdgeA(const ReconstructionMethod recon,
                      const Real &q_im3, const Real &q_im2,
                      const Real &q_im1, const Real &q_i,
                      const Real &q_ip1, const Real &q_ip2) {
  return fmin(1.0, fmax(0.0,
      InterpolateEdge(recon, q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2)));
}

KOKKOS_INLINE_FUNCTION
Real InterpolateEdgeD(const ReconstructionMethod recon,
                      const Real &q_im3, const Real &q_im2,
                      const Real &q_im1, const Real &q_i,
                      const Real &q_ip1, const Real &q_ip2) {
  return fmax(0.0,
      InterpolateEdge(recon, q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2));
}

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::ComposeCornerEMF
//! \brief Compose the edge-centered corner EMFs (efld.x{1,2,3}e) with the UCT formula
//! (Berta+2024 Eq.39 / Mignone & Del Zanna 2021 Eq.33) over an arbitrary edge-point range,
//! from the current face-centered E-fields and UCT face coefficients, honouring the MOOD
//! per-edge demotion (fb_level).  Factored out of CornerE so that the MOOD detector can
//! build the genuine staggered candidate B over its light-cone halo (mhd_mood.cpp) using
//! IDENTICAL composition code to the production path.  Ideal-MHD UCT (Newtonian, SR and
//! GR alike -- the solvers hand over coordinate-frame v and lambda); the caller
//! is responsible for the 1D copy and for ct_contact.  Ranges are edge-point ranges:
//! 2D uses k = ks (pass kl=ku=ks); 3D loops k in [kl,ku].

void MHD::ComposeCornerEMF(int il, int iu, int jl, int ju, int kl, int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto e2x1_ = e2x1;
  auto e1x2_ = e1x2;
  auto bx1f_ = b0.x1f;
  auto bx2f_ = b0.x2f;
  auto bx3f_ = b0.x3f;
  auto aL1 = aL_x1f;  auto dL1 = dL_x1f;  auto dR1 = dR_x1f;
  auto vy1 = vy_x1f;  auto vz1 = vz_x1f;
  auto aL2 = aL_x2f;  auto dL2 = dL_x2f;  auto dR2 = dR_x2f;
  auto vx2 = vx_x2f;  auto vz2 = vz_x2f;
  auto aL3 = aL_x3f;  auto dL3 = dL_x3f;  auto dR3 = dR_x3f;
  auto vx3 = vx_x3f;  auto vy3 = vy_x3f;
  auto recon = recon_method;
  auto fb_ = fb_level;
  const bool eflag_ = use_mood && mood_edge_flag;
  const int nfb_ = n_fb_tiers;

  if (pmy_pack->pmesh->two_d) {
    par_for("compose_emf2_uct", DevExeSpace(), 0, nmb1, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      // Copy face-centered E1 and E2 to edges (same as GS07)
      e2(m,ks  ,j,i) = e2x1_(m,ks,j,i);
      e2(m,ke+1,j,i) = e2x1_(m,ks,j,i);
      e1(m,ks  ,j,i) = e1x2_(m,ks,j,i);
      e1(m,ke+1,j,i) = e1x2_(m,ks,j,i);

      // E3 at z-edge (i-1/2, j-1/2, k=ks)
      ReconstructionMethod er = recon;
      if (eflag_) {
        int t =        fb_(m,ks,j-1,i-1);
        t = (fb_(m,ks,j-1,i  ) > t) ? fb_(m,ks,j-1,i  ) : t;
        t = (fb_(m,ks,j  ,i-1) > t) ? fb_(m,ks,j  ,i-1) : t;
        t = (fb_(m,ks,j  ,i  ) > t) ? fb_(m,ks,j  ,i  ) : t;
        if (t >= nfb_) { er = ReconstructionMethod::dc; }
        else if (t >= 1) { er = ReconstructionMethod::plm; }
      }
      Real ax_W = InterpolateEdgeA(er,
        aL1(m,ks,j-3,i), aL1(m,ks,j-2,i), aL1(m,ks,j-1,i),
        aL1(m,ks,j,i),   aL1(m,ks,j+1,i), aL1(m,ks,j+2,i));
      Real ax_E = 1.0 - ax_W;
      Real dx_W = InterpolateEdgeD(er,
        dL1(m,ks,j-3,i), dL1(m,ks,j-2,i), dL1(m,ks,j-1,i),
        dL1(m,ks,j,i),   dL1(m,ks,j+1,i), dL1(m,ks,j+2,i));
      Real dx_E = InterpolateEdgeD(er,
        dR1(m,ks,j-3,i), dR1(m,ks,j-2,i), dR1(m,ks,j-1,i),
        dR1(m,ks,j,i),   dR1(m,ks,j+1,i), dR1(m,ks,j+2,i));
      Real ay_S = InterpolateEdgeA(er,
        aL2(m,ks,j,i-3), aL2(m,ks,j,i-2), aL2(m,ks,j,i-1),
        aL2(m,ks,j,i),   aL2(m,ks,j,i+1), aL2(m,ks,j,i+2));
      Real ay_N = 1.0 - ay_S;
      Real dy_S = InterpolateEdgeD(er,
        dL2(m,ks,j,i-3), dL2(m,ks,j,i-2), dL2(m,ks,j,i-1),
        dL2(m,ks,j,i),   dL2(m,ks,j,i+1), dL2(m,ks,j,i+2));
      Real dy_N = InterpolateEdgeD(er,
        dR2(m,ks,j,i-3), dR2(m,ks,j,i-2), dR2(m,ks,j,i-1),
        dR2(m,ks,j,i),   dR2(m,ks,j,i+1), dR2(m,ks,j,i+2));
      Real vy_S, vy_N;
      ReconstructEdge(er,
        vy1(m,ks,j-3,i), vy1(m,ks,j-2,i), vy1(m,ks,j-1,i),
        vy1(m,ks,j,i),   vy1(m,ks,j+1,i), vy1(m,ks,j+2,i),
        vy_S, vy_N);
      Real vx_W, vx_E;
      ReconstructEdge(er,
        vx2(m,ks,j,i-3), vx2(m,ks,j,i-2), vx2(m,ks,j,i-1),
        vx2(m,ks,j,i),   vx2(m,ks,j,i+1), vx2(m,ks,j,i+2),
        vx_W, vx_E);
      Real By_W, By_E;
      ReconstructEdge(er,
        bx2f_(m,ks,j,i-3), bx2f_(m,ks,j,i-2), bx2f_(m,ks,j,i-1),
        bx2f_(m,ks,j,i),   bx2f_(m,ks,j,i+1), bx2f_(m,ks,j,i+2),
        By_W, By_E);
      Real Bx_S, Bx_N;
      ReconstructEdge(er,
        bx1f_(m,ks,j-3,i), bx1f_(m,ks,j-2,i), bx1f_(m,ks,j-1,i),
        bx1f_(m,ks,j,i),   bx1f_(m,ks,j+1,i), bx1f_(m,ks,j+2,i),
        Bx_S, Bx_N);
      e3(m,ks,j,i) = -(ax_W*vx_W*By_W + ax_E*vx_E*By_E)
                     +(ay_S*vy_S*Bx_S + ay_N*vy_N*Bx_N)
                     +(dx_E*By_E - dx_W*By_W)
                     -(dy_N*Bx_N - dy_S*Bx_S);
    });
  }

  if (pmy_pack->pmesh->three_d) {
    par_for("compose_emf3_uct", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // ---- E1 at x-edge (i, j-1/2, k-1/2) ----
      {
        ReconstructionMethod er1 = recon;
        if (eflag_) {
          int t = fb_(m,k-1,j-1,i);
          t = (fb_(m,k-1,j,i) > t) ? fb_(m,k-1,j,i) : t;
          t = (fb_(m,k,j-1,i) > t) ? fb_(m,k,j-1,i) : t;
          t = (fb_(m,k,j,i) > t) ? fb_(m,k,j,i) : t;
          if (t >= nfb_) { er1 = ReconstructionMethod::dc; }
          else if (t >= 1) { er1 = ReconstructionMethod::plm; }
        }
        Real ay_S = InterpolateEdgeA(er1,
          aL2(m,k-3,j,i), aL2(m,k-2,j,i), aL2(m,k-1,j,i),
          aL2(m,k,j,i),   aL2(m,k+1,j,i), aL2(m,k+2,j,i));
        Real ay_N = 1.0 - ay_S;
        Real dy_S = InterpolateEdgeD(er1,
          dL2(m,k-3,j,i), dL2(m,k-2,j,i), dL2(m,k-1,j,i),
          dL2(m,k,j,i),   dL2(m,k+1,j,i), dL2(m,k+2,j,i));
        Real dy_N = InterpolateEdgeD(er1,
          dR2(m,k-3,j,i), dR2(m,k-2,j,i), dR2(m,k-1,j,i),
          dR2(m,k,j,i),   dR2(m,k+1,j,i), dR2(m,k+2,j,i));
        Real az_B = InterpolateEdgeA(er1,
          aL3(m,k,j-3,i), aL3(m,k,j-2,i), aL3(m,k,j-1,i),
          aL3(m,k,j,i),   aL3(m,k,j+1,i), aL3(m,k,j+2,i));
        Real az_T = 1.0 - az_B;
        Real dz_B = InterpolateEdgeD(er1,
          dL3(m,k,j-3,i), dL3(m,k,j-2,i), dL3(m,k,j-1,i),
          dL3(m,k,j,i),   dL3(m,k,j+1,i), dL3(m,k,j+2,i));
        Real dz_T = InterpolateEdgeD(er1,
          dR3(m,k,j-3,i), dR3(m,k,j-2,i), dR3(m,k,j-1,i),
          dR3(m,k,j,i),   dR3(m,k,j+1,i), dR3(m,k,j+2,i));
        Real vz_B, vz_T;
        ReconstructEdge(er1,
          vz2(m,k-3,j,i), vz2(m,k-2,j,i), vz2(m,k-1,j,i),
          vz2(m,k,j,i),   vz2(m,k+1,j,i), vz2(m,k+2,j,i),
          vz_B, vz_T);
        Real vy_S, vy_N;
        ReconstructEdge(er1,
          vy3(m,k,j-3,i), vy3(m,k,j-2,i), vy3(m,k,j-1,i),
          vy3(m,k,j,i),   vy3(m,k,j+1,i), vy3(m,k,j+2,i),
          vy_S, vy_N);
        Real By_B, By_T;
        ReconstructEdge(er1,
          bx2f_(m,k-3,j,i), bx2f_(m,k-2,j,i), bx2f_(m,k-1,j,i),
          bx2f_(m,k,j,i),   bx2f_(m,k+1,j,i), bx2f_(m,k+2,j,i),
          By_B, By_T);
        Real Bz_S, Bz_N;
        ReconstructEdge(er1,
          bx3f_(m,k,j-3,i), bx3f_(m,k,j-2,i), bx3f_(m,k,j-1,i),
          bx3f_(m,k,j,i),   bx3f_(m,k,j+1,i), bx3f_(m,k,j+2,i),
          Bz_S, Bz_N);
        e1(m,k,j,i) = -(ay_S*vy_S*Bz_S + ay_N*vy_N*Bz_N)
                      +(az_B*vz_B*By_B + az_T*vz_T*By_T)
                      +(dy_N*Bz_N - dy_S*Bz_S)
                      -(dz_T*By_T - dz_B*By_B);
      }
      // ---- E2 at y-edge (i-1/2, j, k-1/2) ----
      {
        ReconstructionMethod er2 = recon;
        if (eflag_) {
          int t = fb_(m,k-1,j,i-1);
          t = (fb_(m,k-1,j,i) > t) ? fb_(m,k-1,j,i) : t;
          t = (fb_(m,k,j,i-1) > t) ? fb_(m,k,j,i-1) : t;
          t = (fb_(m,k,j,i) > t) ? fb_(m,k,j,i) : t;
          if (t >= nfb_) { er2 = ReconstructionMethod::dc; }
          else if (t >= 1) { er2 = ReconstructionMethod::plm; }
        }
        Real az_B = InterpolateEdgeA(er2,
          aL3(m,k,j,i-3), aL3(m,k,j,i-2), aL3(m,k,j,i-1),
          aL3(m,k,j,i),   aL3(m,k,j,i+1), aL3(m,k,j,i+2));
        Real az_T = 1.0 - az_B;
        Real dz_B = InterpolateEdgeD(er2,
          dL3(m,k,j,i-3), dL3(m,k,j,i-2), dL3(m,k,j,i-1),
          dL3(m,k,j,i),   dL3(m,k,j,i+1), dL3(m,k,j,i+2));
        Real dz_T = InterpolateEdgeD(er2,
          dR3(m,k,j,i-3), dR3(m,k,j,i-2), dR3(m,k,j,i-1),
          dR3(m,k,j,i),   dR3(m,k,j,i+1), dR3(m,k,j,i+2));
        Real ax_W = InterpolateEdgeA(er2,
          aL1(m,k-3,j,i), aL1(m,k-2,j,i), aL1(m,k-1,j,i),
          aL1(m,k,j,i),   aL1(m,k+1,j,i), aL1(m,k+2,j,i));
        Real ax_E = 1.0 - ax_W;
        Real dx_W = InterpolateEdgeD(er2,
          dL1(m,k-3,j,i), dL1(m,k-2,j,i), dL1(m,k-1,j,i),
          dL1(m,k,j,i),   dL1(m,k+1,j,i), dL1(m,k+2,j,i));
        Real dx_E = InterpolateEdgeD(er2,
          dR1(m,k-3,j,i), dR1(m,k-2,j,i), dR1(m,k-1,j,i),
          dR1(m,k,j,i),   dR1(m,k+1,j,i), dR1(m,k+2,j,i));
        Real vx_W, vx_E;
        ReconstructEdge(er2,
          vx3(m,k,j,i-3), vx3(m,k,j,i-2), vx3(m,k,j,i-1),
          vx3(m,k,j,i),   vx3(m,k,j,i+1), vx3(m,k,j,i+2),
          vx_W, vx_E);
        Real vz_B, vz_T;
        ReconstructEdge(er2,
          vz1(m,k-3,j,i), vz1(m,k-2,j,i), vz1(m,k-1,j,i),
          vz1(m,k,j,i),   vz1(m,k+1,j,i), vz1(m,k+2,j,i),
          vz_B, vz_T);
        Real Bz_W, Bz_E;
        ReconstructEdge(er2,
          bx3f_(m,k,j,i-3), bx3f_(m,k,j,i-2), bx3f_(m,k,j,i-1),
          bx3f_(m,k,j,i),   bx3f_(m,k,j,i+1), bx3f_(m,k,j,i+2),
          Bz_W, Bz_E);
        Real Bx_B, Bx_T;
        ReconstructEdge(er2,
          bx1f_(m,k-3,j,i), bx1f_(m,k-2,j,i), bx1f_(m,k-1,j,i),
          bx1f_(m,k,j,i),   bx1f_(m,k+1,j,i), bx1f_(m,k+2,j,i),
          Bx_B, Bx_T);
        e2(m,k,j,i) = -(az_B*vz_B*Bx_B + az_T*vz_T*Bx_T)
                      +(ax_W*vx_W*Bz_W + ax_E*vx_E*Bz_E)
                      +(dz_T*Bx_T - dz_B*Bx_B)
                      -(dx_E*Bz_E - dx_W*Bz_W);
      }
      // ---- E3 at z-edge (i-1/2, j-1/2, k) ----
      {
        ReconstructionMethod er3 = recon;
        if (eflag_) {
          int t = fb_(m,k,j-1,i-1);
          t = (fb_(m,k,j-1,i) > t) ? fb_(m,k,j-1,i) : t;
          t = (fb_(m,k,j,i-1) > t) ? fb_(m,k,j,i-1) : t;
          t = (fb_(m,k,j,i) > t) ? fb_(m,k,j,i) : t;
          if (t >= nfb_) { er3 = ReconstructionMethod::dc; }
          else if (t >= 1) { er3 = ReconstructionMethod::plm; }
        }
        Real ax_W = InterpolateEdgeA(er3,
          aL1(m,k,j-3,i), aL1(m,k,j-2,i), aL1(m,k,j-1,i),
          aL1(m,k,j,i),   aL1(m,k,j+1,i), aL1(m,k,j+2,i));
        Real ax_E = 1.0 - ax_W;
        Real dx_W = InterpolateEdgeD(er3,
          dL1(m,k,j-3,i), dL1(m,k,j-2,i), dL1(m,k,j-1,i),
          dL1(m,k,j,i),   dL1(m,k,j+1,i), dL1(m,k,j+2,i));
        Real dx_E = InterpolateEdgeD(er3,
          dR1(m,k,j-3,i), dR1(m,k,j-2,i), dR1(m,k,j-1,i),
          dR1(m,k,j,i),   dR1(m,k,j+1,i), dR1(m,k,j+2,i));
        Real ay_S = InterpolateEdgeA(er3,
          aL2(m,k,j,i-3), aL2(m,k,j,i-2), aL2(m,k,j,i-1),
          aL2(m,k,j,i),   aL2(m,k,j,i+1), aL2(m,k,j,i+2));
        Real ay_N = 1.0 - ay_S;
        Real dy_S = InterpolateEdgeD(er3,
          dL2(m,k,j,i-3), dL2(m,k,j,i-2), dL2(m,k,j,i-1),
          dL2(m,k,j,i),   dL2(m,k,j,i+1), dL2(m,k,j,i+2));
        Real dy_N = InterpolateEdgeD(er3,
          dR2(m,k,j,i-3), dR2(m,k,j,i-2), dR2(m,k,j,i-1),
          dR2(m,k,j,i),   dR2(m,k,j,i+1), dR2(m,k,j,i+2));
        Real vx_W, vx_E;
        ReconstructEdge(er3,
          vx2(m,k,j,i-3), vx2(m,k,j,i-2), vx2(m,k,j,i-1),
          vx2(m,k,j,i),   vx2(m,k,j,i+1), vx2(m,k,j,i+2),
          vx_W, vx_E);
        Real vy_S, vy_N;
        ReconstructEdge(er3,
          vy1(m,k,j-3,i), vy1(m,k,j-2,i), vy1(m,k,j-1,i),
          vy1(m,k,j,i),   vy1(m,k,j+1,i), vy1(m,k,j+2,i),
          vy_S, vy_N);
        Real By_W, By_E;
        ReconstructEdge(er3,
          bx2f_(m,k,j,i-3), bx2f_(m,k,j,i-2), bx2f_(m,k,j,i-1),
          bx2f_(m,k,j,i),   bx2f_(m,k,j,i+1), bx2f_(m,k,j,i+2),
          By_W, By_E);
        Real Bx_S, Bx_N;
        ReconstructEdge(er3,
          bx1f_(m,k,j-3,i), bx1f_(m,k,j-2,i), bx1f_(m,k,j-1,i),
          bx1f_(m,k,j,i),   bx1f_(m,k,j+1,i), bx1f_(m,k,j+2,i),
          Bx_S, Bx_N);
        e3(m,k,j,i) = -(ax_W*vx_W*By_W + ax_E*vx_E*By_E)
                      +(ay_S*vy_S*Bx_S + ay_N*vy_N*Bx_N)
                      +(dx_E*By_E - dx_W*By_W)
                      -(dy_N*Bx_N - dy_S*Bx_S);
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::CornerE
//  \brief calculate the corner electric fields.

TaskStatus MHD::CornerE(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;

  //---- 1-D problem:
  //  copy face-centered E-fields to edges and return.
  //  Note e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]

  if (pmy_pack->pmesh->one_d) {
    // capture class variables for the kernels
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto e2x1_ = e2x1;
    auto e3x1_ = e3x1;
    par_for("emf1", DevExeSpace(), 0, nmb1, is, ie+1,
    KOKKOS_LAMBDA(int m, int i) {
      e2(m,ks  ,js  ,i) = e2x1_(m,ks,js,i);
      e2(m,ke+1,js  ,i) = e2x1_(m,ks,js,i);
      e3(m,ks  ,js  ,i) = e3x1_(m,ks,js,i);
      e3(m,ks  ,je+1,i) = e3x1_(m,ks,js,i);
    });
  }

  //---- 2-D problem:
  // Copy face-centered E1 and E2 to edges, compute E3 at corners
  // Either GS07 (CT-Contact) or UCT composition formula

  if (pmy_pack->pmesh->two_d) {
    if (emf_method == MHD_EMF::ct_contact) {
      // Compute cell-centered E3 = -(v X B) = VyBx-VxBy (needed for GS07)
      auto w0_ = w0;
      auto bcc_ = bcc0;
      auto e3cc_ = e3_cc;

      // compute cell-centered EMF in dynamical GRMHD
      if (pmy_pack->padm != nullptr) {
        auto &adm = pmy_pack->padm->adm;
        par_for("e_cc_2d", DevExeSpace(), 0, nmb1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int j, int i) {
          const Real &ux = w0_(m,IVX,ks,j,i);
          const Real &uy = w0_(m,IVY,ks,j,i);
          const Real &uz = w0_(m,IVZ,ks,j,i);
          Real iW = 1.0/sqrt(1.0
                      + adm.g_dd(m,0,0,ks,j,i)*ux*ux + 2.0*adm.g_dd(m,0,1,ks,j,i)*ux*uy
                      + 2.0*adm.g_dd(m,0,2,ks,j,i)*ux*uz + adm.g_dd(m,1,1,ks,j,i)*uy*uy
                      + 2.0*adm.g_dd(m,1,2,ks,j,i)*uy*uz + adm.g_dd(m,2,2,ks,j,i)*uz*uz);
          Real v1 = ux*iW;
          Real v2 = uy*iW;

          const Real &alpha = adm.alpha(m,ks,j,i);
          e3cc_(m,ks,j,i) = bcc_(m,IBX,ks,j,i)*(alpha*v2 - adm.beta_u(m, 1, ks, j, i))
                          - bcc_(m,IBY,ks,j,i)*(alpha*v1 - adm.beta_u(m, 0, ks, j, i));
        });
      } else if (pmy_pack->pcoord->is_general_relativistic) {
        par_for("e_cc_2d", DevExeSpace(), 0, nmb1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int j, int i) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(0, indcs.nx3, x3min, x3max);

          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

          const Real &ux = w0_(m,IVX,ks,j,i);
          const Real &uy = w0_(m,IVY,ks,j,i);
          const Real &uz = w0_(m,IVZ,ks,j,i);
          const Real &bx = bcc_(m,IBX,ks,j,i);
          const Real &by = bcc_(m,IBY,ks,j,i);
          Real tmp = glower[1][1]*ux*ux + 2.0*glower[1][2]*ux*uy + 2.0*glower[1][3]*ux*uz
                   + glower[2][2]*uy*uy + 2.0*glower[2][3]*uy*uz
                   + glower[3][3]*uz*uz;
          Real alpha = sqrt(-1.0/gupper[0][0]);
          Real gamma = sqrt(1.0 + tmp);
          Real u0 = gamma / alpha;
          Real u1 = ux - alpha * gamma * gupper[0][1];
          Real u2 = uy - alpha * gamma * gupper[0][2];
          Real u3 = uz - alpha * gamma * gupper[0][3];
          Real u_1 = glower[1][0]*u0+glower[1][1]*u1+glower[1][2]*u2+glower[1][3]*u3;
          Real u_2 = glower[2][0]*u0+glower[2][1]*u1+glower[2][2]*u2+glower[2][3]*u3;
          Real u_3 = glower[3][0]*u0+glower[3][1]*u1+glower[3][2]*u2+glower[3][3]*u3;
          Real b0 = u_1*bx + u_2*by + u_3*bcc_(m,IBZ,ks,j,i);
          Real b1 = (bx + b0 * u1) / u0;
          Real b2 = (by + b0 * u2) / u0;
          e3cc_(m,ks,j,i) = b1 * u2 - b2 * u1;
        });
      } else if (pmy_pack->pcoord->is_special_relativistic) {
        par_for("e_cc_2d", DevExeSpace(), 0, nmb1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int j, int i) {
          const Real &u1 = w0_(m,IVX,ks,j,i);
          const Real &u2 = w0_(m,IVY,ks,j,i);
          const Real &u3 = w0_(m,IVZ,ks,j,i);
          Real u0 = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
          e3cc_(m,ks,j,i) = (u2*bcc_(m,IBX,ks,j,i) - u1*bcc_(m,IBY,ks,j,i)) / u0;
        });
      } else {
        par_for("e_cc_2d", DevExeSpace(), 0, nmb1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int j, int i) {
          e3cc_(m,ks,j,i) = w0_(m,IVY,ks,j,i)*bcc_(m,IBX,ks,j,i) -
                            w0_(m,IVX,ks,j,i)*bcc_(m,IBY,ks,j,i);
        });
      }
    } // end ct_contact cell-centered EMFs

    // capture class variables for the kernels
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto e2x1_ = e2x1;
    auto e3x1_ = e3x1;
    auto e1x2_ = e1x2;
    auto e3x2_ = e3x2;
    auto flx1 = uflx.x1f;
    auto flx2 = uflx.x2f;

    // integrate E3 to corner
    //  Note e1[is:ie,  js:je+1,ks:ke+1]
    //       e2[is:ie+1,js:je,  ks:ke+1]
    //       e3[is:ie+1,js:je+1,ks:ke  ]
    if (emf_method == MHD_EMF::ct_contact) {
      // GS07 (CT-Contact) algorithm
      auto e3cc_ = e3_cc;
      par_for("emf2", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int j, const int i) {
        e2(m,ks  ,j,i) = e2x1_(m,ks,j,i);
        e2(m,ke+1,j,i) = e2x1_(m,ks,j,i);
        e1(m,ks  ,j,i) = e1x2_(m,ks,j,i);
        e1(m,ke+1,j,i) = e1x2_(m,ks,j,i);

        Real e3_l2, e3_r2, e3_l1, e3_r1;
        if (flx1(m,IDN,ks,j-1,i) >= 0.0) {
          e3_l2 = e3x2_(m,ks,j,i-1) - e3cc_(m,ks,j-1,i-1);
        } else {
          e3_l2 = e3x2_(m,ks,j,i  ) - e3cc_(m,ks,j-1,i  );
        }
        if (flx1(m,IDN,ks,j,i) >= 0.0) {
          e3_r2 = e3x2_(m,ks,j,i-1) - e3cc_(m,ks,j  ,i-1);
        } else {
          e3_r2 = e3x2_(m,ks,j,i  ) - e3cc_(m,ks,j  ,i  );
        }
        if (flx2(m,IDN,ks,j,i-1) >= 0.0) {
          e3_l1 = e3x1_(m,ks,j-1,i) - e3cc_(m,ks,j-1,i-1);
        } else {
          e3_l1 = e3x1_(m,ks,j  ,i) - e3cc_(m,ks,j  ,i-1);
        }
        if (flx2(m,IDN,ks,j,i) >= 0.0) {
          e3_r1 = e3x1_(m,ks,j-1,i) - e3cc_(m,ks,j-1,i  );
        } else {
          e3_r1 = e3x1_(m,ks,j  ,i) - e3cc_(m,ks,j  ,i  );
        }
        e3(m,ks,j,i) = 0.25*(e3_l1 + e3_r1 + e3_l2 + e3_r2 +
          e3x2_(m,ks,j,i-1) + e3x2_(m,ks,j,i) + e3x1_(m,ks,j-1,i) + e3x1_(m,ks,j,i));
      });
    } else {
      // UCT composition formula (Eq. 39, Berta et al. 2024 / MDZ21 Eq. 33)
      auto bx1f_ = b0.x1f;
      auto bx2f_ = b0.x2f;
      auto aL1 = aL_x1f;
      auto dL1 = dL_x1f;
      auto dR1 = dR_x1f;
      auto vy1 = vy_x1f;
      auto aL2 = aL_x2f;
      auto dL2 = dL_x2f;
      auto dR2 = dR_x2f;
      auto vx2 = vx_x2f;
      auto recon = recon_method;
      // explicit MOOD edge demotion: drop the edge reconstruction at any edge whose
      // adjacent cells are demoted (tier = max fb_level of the sharing cells)
      auto fb_ = fb_level;
      const bool eflag_ = use_mood && mood_edge_flag;
      const int nfb_ = n_fb_tiers;

      par_for("emf2_uct", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int j, const int i) {
        // Copy face-centered E1 and E2 to edges (same as GS07)
        e2(m,ks  ,j,i) = e2x1_(m,ks,j,i);
        e2(m,ke+1,j,i) = e2x1_(m,ks,j,i);
        e1(m,ks  ,j,i) = e1x2_(m,ks,j,i);
        e1(m,ke+1,j,i) = e1x2_(m,ks,j,i);

        // E3 at z-edge (i-1/2, j-1/2, k=ks)
        ReconstructionMethod er = recon;
        if (eflag_) {
          int t =        fb_(m,ks,j-1,i-1);
          t = (fb_(m,ks,j-1,i  ) > t) ? fb_(m,ks,j-1,i  ) : t;
          t = (fb_(m,ks,j  ,i-1) > t) ? fb_(m,ks,j  ,i-1) : t;
          t = (fb_(m,ks,j  ,i  ) > t) ? fb_(m,ks,j  ,i  ) : t;
          if (t >= nfb_) { er = ReconstructionMethod::dc; }
          else if (t >= 1) { er = ReconstructionMethod::plm; }
        }
        // --- UCT coefficients: reconstruct to edge via high-order interp ---
        // x1-face coefficients at (i-1/2, j): reconstruct along j to j-1/2
        Real ax_W = InterpolateEdgeA(er,
          aL1(m,ks,j-3,i), aL1(m,ks,j-2,i), aL1(m,ks,j-1,i),
          aL1(m,ks,j,i),   aL1(m,ks,j+1,i), aL1(m,ks,j+2,i));
        Real ax_E = 1.0 - ax_W;
        Real dx_W = InterpolateEdgeD(er,
          dL1(m,ks,j-3,i), dL1(m,ks,j-2,i), dL1(m,ks,j-1,i),
          dL1(m,ks,j,i),   dL1(m,ks,j+1,i), dL1(m,ks,j+2,i));
        Real dx_E = InterpolateEdgeD(er,
          dR1(m,ks,j-3,i), dR1(m,ks,j-2,i), dR1(m,ks,j-1,i),
          dR1(m,ks,j,i),   dR1(m,ks,j+1,i), dR1(m,ks,j+2,i));
        // x2-face coefficients at (i, j-1/2): reconstruct along i to i-1/2
        Real ay_S = InterpolateEdgeA(er,
          aL2(m,ks,j,i-3), aL2(m,ks,j,i-2), aL2(m,ks,j,i-1),
          aL2(m,ks,j,i),   aL2(m,ks,j,i+1), aL2(m,ks,j,i+2));
        Real ay_N = 1.0 - ay_S;
        Real dy_S = InterpolateEdgeD(er,
          dL2(m,ks,j,i-3), dL2(m,ks,j,i-2), dL2(m,ks,j,i-1),
          dL2(m,ks,j,i),   dL2(m,ks,j,i+1), dL2(m,ks,j,i+2));
        Real dy_N = InterpolateEdgeD(er,
          dR2(m,ks,j,i-3), dR2(m,ks,j,i-2), dR2(m,ks,j,i-1),
          dR2(m,ks,j,i),   dR2(m,ks,j,i+1), dR2(m,ks,j,i+2));

        // --- Transverse velocities: reconstruct to L/R at edge ---
        Real vy_S, vy_N;
        ReconstructEdge(er,
          vy1(m,ks,j-3,i), vy1(m,ks,j-2,i), vy1(m,ks,j-1,i),
          vy1(m,ks,j,i),   vy1(m,ks,j+1,i), vy1(m,ks,j+2,i),
          vy_S, vy_N);
        Real vx_W, vx_E;
        ReconstructEdge(er,
          vx2(m,ks,j,i-3), vx2(m,ks,j,i-2), vx2(m,ks,j,i-1),
          vx2(m,ks,j,i),   vx2(m,ks,j,i+1), vx2(m,ks,j,i+2),
          vx_W, vx_E);

        // --- Staggered B: reconstruct to L/R at edge ---
        Real By_W, By_E;
        ReconstructEdge(er,
          bx2f_(m,ks,j,i-3), bx2f_(m,ks,j,i-2), bx2f_(m,ks,j,i-1),
          bx2f_(m,ks,j,i),   bx2f_(m,ks,j,i+1), bx2f_(m,ks,j,i+2),
          By_W, By_E);
        Real Bx_S, Bx_N;
        ReconstructEdge(er,
          bx1f_(m,ks,j-3,i), bx1f_(m,ks,j-2,i), bx1f_(m,ks,j-1,i),
          bx1f_(m,ks,j,i),   bx1f_(m,ks,j+1,i), bx1f_(m,ks,j+2,i),
          Bx_S, Bx_N);

        // UCT composition (Eq. 39, Berta et al. 2024):
        // E_z = vy*Bx - vx*By (with upwind dissipation)
        e3(m,ks,j,i) = -(ax_W*vx_W*By_W + ax_E*vx_E*By_E)
                       +(ay_S*vy_S*Bx_S + ay_N*vy_N*Bx_N)
                       +(dx_E*By_E - dx_W*By_W)
                       -(dy_N*Bx_N - dy_S*Bx_S);
      });
    }
  }

  //---- 3-D problem:
  // Use GS07 algorithm to compute all three of E1, E2, and E3

  if (pmy_pack->pmesh->three_d) {
    if (emf_method == MHD_EMF::ct_contact) {
    // Compute cell-centered electric fields (only needed for GS07/CT-Contact)
    // E1=-(v X B)=VzBy-VyBz
    // E2=-(v X B)=VxBz-VzBx
    // E3=-(v X B)=VyBx-VxBy
    auto w0_ = w0;
    auto bcc_ = bcc0;
    auto e1cc_ = e1_cc;
    auto e2cc_ = e2_cc;
    auto e3cc_ = e3_cc;

    // compute cell-centered EMFs in dynamical GRMHD
    if (pmy_pack->padm != nullptr) {
      auto &adm = pmy_pack->padm->adm;
      par_for("e_cc_3d", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Calculate something that resembles the spatial components of the four-velocity
        // normalized by W.
        const Real &ux = w0_(m,IVX,k,j,i);
        const Real &uy = w0_(m,IVY,k,j,i);
        const Real &uz = w0_(m,IVZ,k,j,i);
        const Real &bx = bcc_(m,IBX,k,j,i);
        const Real &by = bcc_(m,IBY,k,j,i);
        const Real &bz = bcc_(m,IBZ,k,j,i);
        Real iW = 1.0/sqrt(1.0
                         + adm.g_dd(m,0,0,k,j,i)*ux*ux + 2.0*adm.g_dd(m,0,1,k,j,i)*ux*uy
                         + 2.0*adm.g_dd(m,0,2,k,j,i)*ux*uz + adm.g_dd(m,1,1,k,j,i)*uy*uy
                         + 2.0*adm.g_dd(m,1,2,k,j,i)*uy*uz + adm.g_dd(m,2,2,k,j,i)*uz*uz);
        const Real &alpha = adm.alpha(m, k, j, i);
        Real v1c = alpha*ux*iW - adm.beta_u(m, 0, k, j, i);
        Real v2c = alpha*uy*iW - adm.beta_u(m, 1, k, j, i);
        Real v3c = alpha*uz*iW - adm.beta_u(m, 2, k, j, i);

        e1cc_(m,k,j,i) = by * v3c - bz * v2c;
        e2cc_(m,k,j,i) = bz * v1c - bx * v3c;
        e3cc_(m,k,j,i) = bx * v2c - by * v1c;
      });
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      // compute cell-centered EMFs in GR MHD
      par_for("e_cc_3d", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Extract components of metric
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

        const Real &ux = w0_(m,IVX,k,j,i);
        const Real &uy = w0_(m,IVY,k,j,i);
        const Real &uz = w0_(m,IVZ,k,j,i);
        const Real &bx = bcc_(m,IBX,k,j,i);
        const Real &by = bcc_(m,IBY,k,j,i);
        const Real &bz = bcc_(m,IBZ,k,j,i);
        // Calculate 4-velocity
        Real tmp = glower[1][1]*ux*ux + 2.0*glower[1][2]*ux*uy + 2.0*glower[1][3]*ux*uz
                 + glower[2][2]*uy*uy + 2.0*glower[2][3]*uy*uz
                 + glower[3][3]*uz*uz;
        Real alpha = sqrt(-1.0/gupper[0][0]);
        Real gamma = sqrt(1.0 + tmp);
        Real u0 = gamma / alpha;
        Real u1 = ux - alpha * gamma * gupper[0][1];
        Real u2 = uy - alpha * gamma * gupper[0][2];
        Real u3 = uz - alpha * gamma * gupper[0][3];
        // lower vector indices
        Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
        Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
        Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;
        // calculate 4-magnetic field
        Real b0 = u_1*bx + u_2*by + u_3*bz;
        Real b1 = (bx + b0 * u1) / u0;
        Real b2 = (by + b0 * u2) / u0;
        Real b3 = (bz + b0 * u3) / u0;

        e1cc_(m,k,j,i) = b2 * u3 - b3 * u2;
        e2cc_(m,k,j,i) = b3 * u1 - b1 * u3;
        e3cc_(m,k,j,i) = b1 * u2 - b2 * u1;
      });

    // compute cell-centered EMFs in SR MHD
    } else if (pmy_pack->pcoord->is_special_relativistic) {
      par_for("e_cc_3d", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real &u1 = w0_(m,IVX,k,j,i);
        const Real &u2 = w0_(m,IVY,k,j,i);
        const Real &u3 = w0_(m,IVZ,k,j,i);
        Real u0 = sqrt(1.0 + SQR(u1) + SQR(u2) + SQR(u3));
        e1cc_(m,k,j,i) = (u3 * bcc_(m,IBY,k,j,i) - u2 * bcc_(m,IBZ,k,j,i)) / u0;
        e2cc_(m,k,j,i) = (u1 * bcc_(m,IBZ,k,j,i) - u3 * bcc_(m,IBX,k,j,i)) / u0;
        e3cc_(m,k,j,i) = (u2 * bcc_(m,IBX,k,j,i) - u1 * bcc_(m,IBY,k,j,i)) / u0;
      });

    // compute cell-centered EMFs in Newtonian MHD
    } else {
      par_for("e_cc_3d", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        e1cc_(m,k,j,i) = w0_(m,IVZ,k,j,i)*bcc_(m,IBY,k,j,i) -
                         w0_(m,IVY,k,j,i)*bcc_(m,IBZ,k,j,i);
        e2cc_(m,k,j,i) = w0_(m,IVX,k,j,i)*bcc_(m,IBZ,k,j,i) -
                         w0_(m,IVZ,k,j,i)*bcc_(m,IBX,k,j,i);
        e3cc_(m,k,j,i) = w0_(m,IVY,k,j,i)*bcc_(m,IBX,k,j,i) -
                         w0_(m,IVX,k,j,i)*bcc_(m,IBY,k,j,i);
      });
    }
    } // end ct_contact cell-centered EMFs

    // capture class variables for the kernels
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto e2x1_ = e2x1;
    auto e3x1_ = e3x1;
    auto e1x2_ = e1x2;
    auto e3x2_ = e3x2;
    auto e1x3_ = e1x3;
    auto e2x3_ = e2x3;

    // Integrate E1, E2, E3 to corners
    //  Note e1[is:ie,  js:je+1,ks:ke+1]
    //       e2[is:ie+1,js:je,  ks:ke+1]
    //       e3[is:ie+1,js:je+1,ks:ke  ]
    if (emf_method == MHD_EMF::ct_contact) {
      // GS07 (CT-Contact) algorithm
      auto e1cc_ = e1_cc;
      auto e2cc_ = e2_cc;
      auto e3cc_ = e3_cc;
      auto flx1 = uflx.x1f;
      auto flx2 = uflx.x2f;
      auto flx3 = uflx.x3f;
      par_for("emf3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // integrate E1 to corner using SG07
        Real e1_l3, e1_r3, e1_l2, e1_r2;
        if (flx2(m,IDN,k-1,j,i) >= 0.0) {
          e1_l3 = e1x3_(m,k,j-1,i) - e1cc_(m,k-1,j-1,i);
        } else {
          e1_l3 = e1x3_(m,k,j  ,i) - e1cc_(m,k-1,j  ,i);
        }
        if (flx2(m,IDN,k,j,i) >= 0.0) {
          e1_r3 = e1x3_(m,k,j-1,i) - e1cc_(m,k  ,j-1,i);
        } else {
          e1_r3 = e1x3_(m,k,j  ,i) - e1cc_(m,k  ,j  ,i);
        }
        if (flx3(m,IDN,k,j-1,i) >= 0.0) {
          e1_l2 = e1x2_(m,k-1,j,i) - e1cc_(m,k-1,j-1,i);
        } else {
          e1_l2 = e1x2_(m,k  ,j,i) - e1cc_(m,k  ,j-1,i);
        }
        if (flx3(m,IDN,k,j,i) >= 0.0) {
          e1_r2 = e1x2_(m,k-1,j,i) - e1cc_(m,k-1,j  ,i);
        } else {
          e1_r2 = e1x2_(m,k  ,j,i) - e1cc_(m,k  ,j  ,i);
        }
        e1(m,k,j,i) = 0.25*(e1_l3 + e1_r3 + e1_l2 + e1_r2 +
              e1x2_(m,k-1,j,i) + e1x2_(m,k,j,i) + e1x3_(m,k,j-1,i) + e1x3_(m,k,j,i));

        // integrate E2 to corner using SG07
        Real e2_l3, e2_r3, e2_l1, e2_r1;
        if (flx1(m,IDN,k-1,j,i) >= 0.0) {
          e2_l3 = e2x3_(m,k,j,i-1) - e2cc_(m,k-1,j,i-1);
        } else {
          e2_l3 = e2x3_(m,k,j,i  ) - e2cc_(m,k-1,j,i  );
        }
        if (flx1(m,IDN,k,j,i) >= 0.0) {
          e2_r3 = e2x3_(m,k,j,i-1) - e2cc_(m,k  ,j,i-1);
        } else {
          e2_r3 = e2x3_(m,k,j,i  ) - e2cc_(m,k  ,j,i  );
        }
        if (flx3(m,IDN,k,j,i-1) >= 0.0) {
          e2_l1 = e2x1_(m,k-1,j,i) - e2cc_(m,k-1,j,i-1);
        } else {
          e2_l1 = e2x1_(m,k  ,j,i) - e2cc_(m,k  ,j,i-1);
        }
        if (flx3(m,IDN,k,j,i) >= 0.0) {
          e2_r1 = e2x1_(m,k-1,j,i) - e2cc_(m,k-1,j,i  );
        } else {
          e2_r1 = e2x1_(m,k  ,j,i) - e2cc_(m,k  ,j,i  );
        }
        e2(m,k,j,i) = 0.25*(e2_l3 + e2_r3 + e2_l1 + e2_r1 +
              e2x3_(m,k,j,i-1) + e2x3_(m,k,j,i) + e2x1_(m,k-1,j,i) + e2x1_(m,k,j,i));

        // integrate E3 to corner using SG07
        Real e3_l2, e3_r2, e3_l1, e3_r1;
        if (flx1(m,IDN,k,j-1,i) >= 0.0) {
          e3_l2 = e3x2_(m,k,j,i-1) - e3cc_(m,k,j-1,i-1);
        } else {
          e3_l2 = e3x2_(m,k,j,i  ) - e3cc_(m,k,j-1,i  );
        }
        if (flx1(m,IDN,k,j,i) >= 0.0) {
          e3_r2 = e3x2_(m,k,j,i-1) - e3cc_(m,k,j  ,i-1);
        } else {
          e3_r2 = e3x2_(m,k,j,i  ) - e3cc_(m,k,j  ,i  );
        }
        if (flx2(m,IDN,k,j,i-1) >= 0.0) {
          e3_l1 = e3x1_(m,k,j-1,i) - e3cc_(m,k,j-1,i-1);
        } else {
          e3_l1 = e3x1_(m,k,j  ,i) - e3cc_(m,k,j  ,i-1);
        }
        if (flx2(m,IDN,k,j,i) >= 0.0) {
          e3_r1 = e3x1_(m,k,j-1,i) - e3cc_(m,k,j-1,i  );
        } else {
          e3_r1 = e3x1_(m,k,j  ,i) - e3cc_(m,k,j  ,i  );
        }
        e3(m,k,j,i) = 0.25*(e3_l1 + e3_r1 + e3_l2 + e3_r2 +
              e3x2_(m,k,j,i-1) + e3x2_(m,k,j,i) + e3x1_(m,k,j-1,i) + e3x1_(m,k,j,i));
      });
    } else {
      // UCT composition formula (Eq. 33, Mignone & Del Zanna 2021)
      // Staggered B fields
      auto bx1f_ = b0.x1f;
      auto bx2f_ = b0.x2f;
      auto bx3f_ = b0.x3f;
      // UCT coefficients from x1-faces
      auto aL1 = aL_x1f;
      auto dL1 = dL_x1f;
      auto dR1 = dR_x1f;
      auto vy1 = vy_x1f;  // vy component from x1-faces
      auto vz1 = vz_x1f;  // vz component from x1-faces
      // UCT coefficients from x2-faces
      auto aL2 = aL_x2f;
      auto dL2 = dL_x2f;
      auto dR2 = dR_x2f;
      auto vx2 = vx_x2f;  // vx component from x2-faces
      auto vz2 = vz_x2f;  // vz component from x2-faces
      // UCT coefficients from x3-faces
      auto aL3 = aL_x3f;
      auto dL3 = dL_x3f;
      auto dR3 = dR_x3f;
      auto vx3 = vx_x3f;  // vx component from x3-faces
      auto vy3 = vy_x3f;  // vy component from x3-faces
      auto recon = recon_method;
      // explicit MOOD edge demotion: drop the edge reconstruction at any edge whose
      // adjacent cells are demoted (tier = max fb_level of the sharing cells)
      auto fb_ = fb_level;
      const bool eflag_ = use_mood && mood_edge_flag;
      const int nfb_ = n_fb_tiers;

      par_for("emf3_uct", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // ---- E1 at x-edge (i, j-1/2, k-1/2) ----
        // E_x = vz*By - vy*Bz  (with upwind dissipation)
        // x2-face quantities at (i, j-1/2, k): reconstruct along k to k-1/2
        // x3-face quantities at (i, j, k-1/2): reconstruct along j to j-1/2
        {
          ReconstructionMethod er1 = recon;
          if (eflag_) {
            int t = fb_(m,k-1,j-1,i);
            t = (fb_(m,k-1,j,i) > t) ? fb_(m,k-1,j,i) : t;
            t = (fb_(m,k,j-1,i) > t) ? fb_(m,k,j-1,i) : t;
            t = (fb_(m,k,j,i) > t) ? fb_(m,k,j,i) : t;
            if (t >= nfb_) { er1 = ReconstructionMethod::dc; }
            else if (t >= 1) { er1 = ReconstructionMethod::plm; }
          }

          // UCT coefficients: reconstruct to edge using high-order interpolation
          // x2-face coefficients at (i, j-1/2, k): reconstruct along k to k-1/2
          Real ay_S = InterpolateEdgeA(er1,
            aL2(m,k-3,j,i), aL2(m,k-2,j,i), aL2(m,k-1,j,i),
            aL2(m,k,j,i),   aL2(m,k+1,j,i), aL2(m,k+2,j,i));
          Real ay_N = 1.0 - ay_S;
          Real dy_S = InterpolateEdgeD(er1,
            dL2(m,k-3,j,i), dL2(m,k-2,j,i), dL2(m,k-1,j,i),
            dL2(m,k,j,i),   dL2(m,k+1,j,i), dL2(m,k+2,j,i));
          Real dy_N = InterpolateEdgeD(er1,
            dR2(m,k-3,j,i), dR2(m,k-2,j,i), dR2(m,k-1,j,i),
            dR2(m,k,j,i),   dR2(m,k+1,j,i), dR2(m,k+2,j,i));
          // x3-face coefficients at (i, j, k-1/2): reconstruct along j to j-1/2
          Real az_B = InterpolateEdgeA(er1,
            aL3(m,k,j-3,i), aL3(m,k,j-2,i), aL3(m,k,j-1,i),
            aL3(m,k,j,i),   aL3(m,k,j+1,i), aL3(m,k,j+2,i));
          Real az_T = 1.0 - az_B;
          Real dz_B = InterpolateEdgeD(er1,
            dL3(m,k,j-3,i), dL3(m,k,j-2,i), dL3(m,k,j-1,i),
            dL3(m,k,j,i),   dL3(m,k,j+1,i), dL3(m,k,j+2,i));
          Real dz_T = InterpolateEdgeD(er1,
            dR3(m,k,j-3,i), dR3(m,k,j-2,i), dR3(m,k,j-1,i),
            dR3(m,k,j,i),   dR3(m,k,j+1,i), dR3(m,k,j+2,i));

          // Transverse velocities: reconstruct to L/R at edge
          // vz from x2-faces: reconstruct along k to B/T
          Real vz_B, vz_T;
          ReconstructEdge(er1,
            vz2(m,k-3,j,i), vz2(m,k-2,j,i), vz2(m,k-1,j,i),
            vz2(m,k,j,i),   vz2(m,k+1,j,i), vz2(m,k+2,j,i),
            vz_B, vz_T);
          // vy from x3-faces: reconstruct along j to S/N
          Real vy_S, vy_N;
          ReconstructEdge(er1,
            vy3(m,k,j-3,i), vy3(m,k,j-2,i), vy3(m,k,j-1,i),
            vy3(m,k,j,i),   vy3(m,k,j+1,i), vy3(m,k,j+2,i),
            vy_S, vy_N);

          // Staggered B: reconstruct to L/R at edge
          // By from x2-faces: reconstruct along k to B/T
          Real By_B, By_T;
          ReconstructEdge(er1,
            bx2f_(m,k-3,j,i), bx2f_(m,k-2,j,i), bx2f_(m,k-1,j,i),
            bx2f_(m,k,j,i),   bx2f_(m,k+1,j,i), bx2f_(m,k+2,j,i),
            By_B, By_T);
          // Bz from x3-faces: reconstruct along j to S/N
          Real Bz_S, Bz_N;
          ReconstructEdge(er1,
            bx3f_(m,k,j-3,i), bx3f_(m,k,j-2,i), bx3f_(m,k,j-1,i),
            bx3f_(m,k,j,i),   bx3f_(m,k,j+1,i), bx3f_(m,k,j+2,i),
            Bz_S, Bz_N);

          // UCT composition: E_x = +vz*By - vy*Bz (with dissipation)
          // All quantities (B, v, a, d) reconstructed to edge via high-order interp
          e1(m,k,j,i) = -(ay_S*vy_S*Bz_S + ay_N*vy_N*Bz_N)
                        +(az_B*vz_B*By_B + az_T*vz_T*By_T)
                        +(dy_N*Bz_N - dy_S*Bz_S)
                        -(dz_T*By_T - dz_B*By_B);
        }

        // ---- E2 at y-edge (i-1/2, j, k-1/2) ----
        // E_y = vx*Bz - vz*Bx  (with upwind dissipation)
        // x3-face quantities at (i, j, k-1/2): reconstruct along i to i-1/2
        // x1-face quantities at (i-1/2, j, k): reconstruct along k to k-1/2
        {
          ReconstructionMethod er2 = recon;
          if (eflag_) {
            int t = fb_(m,k-1,j,i-1);
            t = (fb_(m,k-1,j,i) > t) ? fb_(m,k-1,j,i) : t;
            t = (fb_(m,k,j,i-1) > t) ? fb_(m,k,j,i-1) : t;
            t = (fb_(m,k,j,i) > t) ? fb_(m,k,j,i) : t;
            if (t >= nfb_) { er2 = ReconstructionMethod::dc; }
            else if (t >= 1) { er2 = ReconstructionMethod::plm; }
          }

          // UCT coefficients: reconstruct to edge using high-order interpolation
          // x3-face coefficients at (i, j, k-1/2): reconstruct along i to i-1/2
          Real az_B = InterpolateEdgeA(er2,
            aL3(m,k,j,i-3), aL3(m,k,j,i-2), aL3(m,k,j,i-1),
            aL3(m,k,j,i),   aL3(m,k,j,i+1), aL3(m,k,j,i+2));
          Real az_T = 1.0 - az_B;
          Real dz_B = InterpolateEdgeD(er2,
            dL3(m,k,j,i-3), dL3(m,k,j,i-2), dL3(m,k,j,i-1),
            dL3(m,k,j,i),   dL3(m,k,j,i+1), dL3(m,k,j,i+2));
          Real dz_T = InterpolateEdgeD(er2,
            dR3(m,k,j,i-3), dR3(m,k,j,i-2), dR3(m,k,j,i-1),
            dR3(m,k,j,i),   dR3(m,k,j,i+1), dR3(m,k,j,i+2));
          // x1-face coefficients at (i-1/2, j, k): reconstruct along k to k-1/2
          Real ax_W = InterpolateEdgeA(er2,
            aL1(m,k-3,j,i), aL1(m,k-2,j,i), aL1(m,k-1,j,i),
            aL1(m,k,j,i),   aL1(m,k+1,j,i), aL1(m,k+2,j,i));
          Real ax_E = 1.0 - ax_W;
          Real dx_W = InterpolateEdgeD(er2,
            dL1(m,k-3,j,i), dL1(m,k-2,j,i), dL1(m,k-1,j,i),
            dL1(m,k,j,i),   dL1(m,k+1,j,i), dL1(m,k+2,j,i));
          Real dx_E = InterpolateEdgeD(er2,
            dR1(m,k-3,j,i), dR1(m,k-2,j,i), dR1(m,k-1,j,i),
            dR1(m,k,j,i),   dR1(m,k+1,j,i), dR1(m,k+2,j,i));

          // Transverse velocities: reconstruct to L/R at edge
          // vx from x3-faces: reconstruct along i to W/E
          Real vx_W, vx_E;
          ReconstructEdge(er2,
            vx3(m,k,j,i-3), vx3(m,k,j,i-2), vx3(m,k,j,i-1),
            vx3(m,k,j,i),   vx3(m,k,j,i+1), vx3(m,k,j,i+2),
            vx_W, vx_E);
          // vz from x1-faces: reconstruct along k to B/T
          Real vz_B, vz_T;
          ReconstructEdge(er2,
            vz1(m,k-3,j,i), vz1(m,k-2,j,i), vz1(m,k-1,j,i),
            vz1(m,k,j,i),   vz1(m,k+1,j,i), vz1(m,k+2,j,i),
            vz_B, vz_T);

          // Staggered B: reconstruct to L/R at edge
          // Bz from x3-faces: reconstruct along i to W/E
          Real Bz_W, Bz_E;
          ReconstructEdge(er2,
            bx3f_(m,k,j,i-3), bx3f_(m,k,j,i-2), bx3f_(m,k,j,i-1),
            bx3f_(m,k,j,i),   bx3f_(m,k,j,i+1), bx3f_(m,k,j,i+2),
            Bz_W, Bz_E);
          // Bx from x1-faces: reconstruct along k to B/T
          Real Bx_B, Bx_T;
          ReconstructEdge(er2,
            bx1f_(m,k-3,j,i), bx1f_(m,k-2,j,i), bx1f_(m,k-1,j,i),
            bx1f_(m,k,j,i),   bx1f_(m,k+1,j,i), bx1f_(m,k+2,j,i),
            Bx_B, Bx_T);

          // UCT composition: E_y = +vx*Bz - vz*Bx (with dissipation)
          // All quantities (B, v, a, d) reconstructed to edge via high-order interp
          e2(m,k,j,i) = -(az_B*vz_B*Bx_B + az_T*vz_T*Bx_T)
                        +(ax_W*vx_W*Bz_W + ax_E*vx_E*Bz_E)
                        +(dz_T*Bx_T - dz_B*Bx_B)
                        -(dx_E*Bz_E - dx_W*Bz_W);
        }

        // ---- E3 at z-edge (i-1/2, j-1/2, k) ----
        // E_z = vy*Bx - vx*By  (with upwind dissipation)
        // x1-face quantities at (i-1/2, j, k): reconstruct along j to j-1/2
        // x2-face quantities at (i, j-1/2, k): reconstruct along i to i-1/2
        {
          ReconstructionMethod er3 = recon;
          if (eflag_) {
            int t = fb_(m,k,j-1,i-1);
            t = (fb_(m,k,j-1,i) > t) ? fb_(m,k,j-1,i) : t;
            t = (fb_(m,k,j,i-1) > t) ? fb_(m,k,j,i-1) : t;
            t = (fb_(m,k,j,i) > t) ? fb_(m,k,j,i) : t;
            if (t >= nfb_) { er3 = ReconstructionMethod::dc; }
            else if (t >= 1) { er3 = ReconstructionMethod::plm; }
          }

          // UCT coefficients: reconstruct to edge using high-order interpolation
          // x1-face coefficients at (i-1/2, j, k): reconstruct along j to j-1/2
          Real ax_W = InterpolateEdgeA(er3,
            aL1(m,k,j-3,i), aL1(m,k,j-2,i), aL1(m,k,j-1,i),
            aL1(m,k,j,i),   aL1(m,k,j+1,i), aL1(m,k,j+2,i));
          Real ax_E = 1.0 - ax_W;
          Real dx_W = InterpolateEdgeD(er3,
            dL1(m,k,j-3,i), dL1(m,k,j-2,i), dL1(m,k,j-1,i),
            dL1(m,k,j,i),   dL1(m,k,j+1,i), dL1(m,k,j+2,i));
          Real dx_E = InterpolateEdgeD(er3,
            dR1(m,k,j-3,i), dR1(m,k,j-2,i), dR1(m,k,j-1,i),
            dR1(m,k,j,i),   dR1(m,k,j+1,i), dR1(m,k,j+2,i));
          // x2-face coefficients at (i, j-1/2, k): reconstruct along i to i-1/2
          Real ay_S = InterpolateEdgeA(er3,
            aL2(m,k,j,i-3), aL2(m,k,j,i-2), aL2(m,k,j,i-1),
            aL2(m,k,j,i),   aL2(m,k,j,i+1), aL2(m,k,j,i+2));
          Real ay_N = 1.0 - ay_S;
          Real dy_S = InterpolateEdgeD(er3,
            dL2(m,k,j,i-3), dL2(m,k,j,i-2), dL2(m,k,j,i-1),
            dL2(m,k,j,i),   dL2(m,k,j,i+1), dL2(m,k,j,i+2));
          Real dy_N = InterpolateEdgeD(er3,
            dR2(m,k,j,i-3), dR2(m,k,j,i-2), dR2(m,k,j,i-1),
            dR2(m,k,j,i),   dR2(m,k,j,i+1), dR2(m,k,j,i+2));

          // Transverse velocities: reconstruct to L/R at edge
          // vx from x2-faces: reconstruct along i to W/E
          Real vx_W, vx_E;
          ReconstructEdge(er3,
            vx2(m,k,j,i-3), vx2(m,k,j,i-2), vx2(m,k,j,i-1),
            vx2(m,k,j,i),   vx2(m,k,j,i+1), vx2(m,k,j,i+2),
            vx_W, vx_E);
          // vy from x1-faces: reconstruct along j to S/N
          Real vy_S, vy_N;
          ReconstructEdge(er3,
            vy1(m,k,j-3,i), vy1(m,k,j-2,i), vy1(m,k,j-1,i),
            vy1(m,k,j,i),   vy1(m,k,j+1,i), vy1(m,k,j+2,i),
            vy_S, vy_N);

          // Staggered B: reconstruct to L/R at edge
          // By from x2-faces: reconstruct along i to W/E
          Real By_W, By_E;
          ReconstructEdge(er3,
            bx2f_(m,k,j,i-3), bx2f_(m,k,j,i-2), bx2f_(m,k,j,i-1),
            bx2f_(m,k,j,i),   bx2f_(m,k,j,i+1), bx2f_(m,k,j,i+2),
            By_W, By_E);
          // Bx from x1-faces: reconstruct along j to S/N
          Real Bx_S, Bx_N;
          ReconstructEdge(er3,
            bx1f_(m,k,j-3,i), bx1f_(m,k,j-2,i), bx1f_(m,k,j-1,i),
            bx1f_(m,k,j,i),   bx1f_(m,k,j+1,i), bx1f_(m,k,j+2,i),
            Bx_S, Bx_N);

          // UCT composition: E_z = vy*Bx - vx*By (with dissipation)
          // All quantities (B, v, a, d) reconstructed to edge via high-order interp
          e3(m,k,j,i) = -(ax_W*vx_W*By_W + ax_E*vx_E*By_E)
                        +(ay_S*vy_S*Bx_S + ay_N*vy_N*Bx_N)
                        +(dx_E*By_E - dx_W*By_W)
                        -(dy_N*Bx_N - dy_S*Bx_S);
        }
      });
    }
  }

  // Resistivity/diffusion EMF is applied later by MHD::EField via
  // AddSelectedDiffusionEMF — do not call OhmicEField here.

  // UCT diagnostic: report the global max UCT dissipation coefficient |d| and any
  // non-finite corner EMFs.  d ~ chi_tilde, so a near-degenerate-fan chi_tilde blowup
  // shows up as a huge |d| (the edge clamp only bounds d>=0, not from above), which
  // then drives a non-finite corner EMF.  Prints only when anomalous.
  if (uct_diag && (emf_method == MHD_EMF::uct_hll || emf_method == MHD_EMF::uct_hlld)) {
    const bool md = pmy_pack->pmesh->multi_d;
    const bool td = pmy_pack->pmesh->three_d;
    // collect the d-coefficient arrays and EMF arrays to scan (per dimension)
    DvceArray4D<Real> darr[6]; int nd = 2;
    darr[0] = dL_x1f; darr[1] = dR_x1f;
    if (md) { darr[nd++] = dL_x2f; darr[nd++] = dR_x2f; }
    if (td) { darr[nd++] = dL_x3f; darr[nd++] = dR_x3f; }
    DvceArray4D<Real> earr[3]; int ne = 1;
    earr[0] = efld.x3e;
    if (md) earr[ne++] = efld.x1e;
    if (td) earr[ne++] = efld.x2e;
    Real dmax = 0.0;
    for (int a=0; a<nd; ++a) {
      auto da = darr[a]; Real mx = 0.0;
      Kokkos::parallel_reduce("uctdmax",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},
          {da.extent_int(0),da.extent_int(1),da.extent_int(2),da.extent_int(3)}),
        KOKKOS_LAMBDA(int m,int k,int j,int i,Real &r){r=fmax(r,fabs(da(m,k,j,i)));},
        Kokkos::Max<Real>(mx));
      dmax = fmax(dmax, mx);
    }
    int nnan = 0;
    for (int a=0; a<ne; ++a) {
      auto ea = earr[a]; int s = 0;
      Kokkos::parallel_reduce("uctnan",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},
          {ea.extent_int(0),ea.extent_int(1),ea.extent_int(2),ea.extent_int(3)}),
        KOKKOS_LAMBDA(int m,int k,int j,int i,int &r){if (!isfinite(ea(m,k,j,i))) r+=1;},
        Kokkos::Sum<int>(s));
      nnan += s;
    }
#if MPI_PARALLEL_ENABLED
    Real gd = dmax; MPI_Allreduce(&gd,&dmax,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    int gn = nnan;  MPI_Allreduce(&gn,&nnan,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0 && (dmax > 20.0 || nnan > 0)) {
      std::cout << "UCT_DIAG cycle=" << pmy_pack->pmesh->ncycle
                << " stage=" << stage << " maxd=" << dmax
                << " emf_nan=" << nnan << std::endl;
    }
  }

  return TaskStatus::complete;
}
} // namespace mhd
