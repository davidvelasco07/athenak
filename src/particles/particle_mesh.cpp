//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_mesh.cpp
//  \brief constructor / destructor / Zero for the ParticleMesh class.
//
//  Phase 1a — skeleton only. The TSC scatter (DepositMass) and the
//  boundary-sum communication are intentionally absent and land in Phase 1b/1c.

#include "particle_mesh.hpp"

#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn ParticleMesh::ParticleMesh
//! \brief allocates dmesh with the same ghost-cell layout as hydro u0.

ParticleMesh::ParticleMesh(MeshBlockPack *ppack, ParameterInput *pin, int nmeshaux_in)
    : nmeshaux(nmeshaux_in), pmy_pack(ppack) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb = pmy_pack->nmb_thispack;
  Kokkos::realloc(dmesh, nmb, nmeshaux, ncells3, ncells2, ncells1);
  Zero();
}

ParticleMesh::~ParticleMesh() {}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::Zero()
//! \brief zero every slot in every cell of dmesh.

void ParticleMesh::Zero() {
  Kokkos::deep_copy(dmesh, 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::DepositMass
//! \brief TSC scatter of particle mass onto dmesh slot 0.
//!
//! Algorithm (per particle p):
//!   1) Resolve the MeshBlock-within-pack slot m from prtcl_idata(PGID, p).
//!   2) Compute the fractional cell index xi = (xp - x1min)/dx1 + is. Then
//!      i0 = floor(xi - 1) is the leftmost of the three cells that receive
//!      contributions, and fx = (i0 + 0.5) - xi is the TSC argument offset
//!      such that the per-cell weight is TSCWeight(fx + ii) for ii in 0..2.
//!   3) Atomic-add mp * w1 * w2 * w3 / dV to each of 27 cells (3D) or 9 (2D).
//!
//! i0+ii can index into the inner ghost cells if the particle is near the MB
//! boundary; that's intentional and gets reconciled by the Phase 1c boundary
//! sum-flush comms.
void ParticleMesh::DepositMass(const DvceArray2D<Real>& prtcl_rdata_in,
                               const DvceArray2D<int>&  prtcl_idata_in,
                               int npar) {
  Zero();
  if (npar <= 0) return;

  auto &pr = prtcl_rdata_in;
  auto &pi = prtcl_idata_in;
  auto &dm = dmesh;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int gids = pmy_pack->gids;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  par_for("PMDepositMass", DevExeSpace(), 0, npar - 1,
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID, p) - gids;
      Real mp = pr(IPM, p);

      // Cell volume (uniform Cartesian for now).
      Real dV = mbsize.d_view(m).dx1;
      if (multi_d) dV *= mbsize.d_view(m).dx2;
      if (three_d) dV *= mbsize.d_view(m).dx3;
      Real inv_dV = 1.0 / dV;

      // x-axis: always active (particles only in 2D/3D).
      Real xi1 = (pr(IPX, p) - mbsize.d_view(m).x1min) / mbsize.d_view(m).dx1 + is;
      int  i0  = static_cast<int>(xi1 - 1.0);
      Real fx  = (static_cast<Real>(i0) + 0.5) - xi1;

      // y-axis: active in 2D/3D.
      int  j0 = 0;
      Real fy = 0.0;
      if (multi_d) {
        Real xi2 = (pr(IPY, p) - mbsize.d_view(m).x2min) / mbsize.d_view(m).dx2 + js;
        j0 = static_cast<int>(xi2 - 1.0);
        fy = (static_cast<Real>(j0) + 0.5) - xi2;
      }

      // z-axis: active in 3D only.
      int  k0 = 0;
      Real fz = 0.0;
      if (three_d) {
        Real xi3 = (pr(IPZ, p) - mbsize.d_view(m).x3min) / mbsize.d_view(m).dx3 + ks;
        k0 = static_cast<int>(xi3 - 1.0);
        fz = (static_cast<Real>(k0) + 0.5) - xi3;
      }

      const int nkk = three_d ? 3 : 1;
      const int njj = multi_d ? 3 : 1;
      for (int kk = 0; kk < nkk; ++kk) {
        Real w3 = three_d ? TSCWeight(fz + static_cast<Real>(kk)) : 1.0;
        int  k_cell = three_d ? (k0 + kk) : 0;
        for (int jj = 0; jj < njj; ++jj) {
          Real w2 = multi_d ? TSCWeight(fy + static_cast<Real>(jj)) : 1.0;
          int  j_cell = multi_d ? (j0 + jj) : 0;
          for (int ii = 0; ii < 3; ++ii) {
            Real w1 = TSCWeight(fx + static_cast<Real>(ii));
            int  i_cell = i0 + ii;
            Kokkos::atomic_add(&dm(m, 0, k_cell, j_cell, i_cell),
                               mp * w1 * w2 * w3 * inv_dV);
          }
        }
      }
    });
}

}  // namespace particles
