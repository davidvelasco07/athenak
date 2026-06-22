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

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "bvals/bvals.hpp"

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

  // Boundary-values object for dmesh: gives us the per-neighbour buffer index tables
  // used by FlushDepositBoundaries (and the MPI buffers for the future cross-rank path).
  pmbval = new MeshBoundaryValuesCC(ppack, pin, false);
  pmbval->InitializeBuffers(nmeshaux);
}

ParticleMesh::~ParticleMesh() {
  delete pmbval;
}

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

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::GatherGravity
//! \brief TSC gather of the gravitational acceleration (-grad phi) onto particles.
//!
//! For each particle p the same fractional-index logic as DepositMass selects the
//! 3x3x3 (3D) / 3x3 (2D) cloud of cells. At each cloud cell the cell-centered
//! acceleration is evaluated with the central-difference stencil a*(phi[m-1]-phi[m+1])
//! (a = 0.5/dx), identical to srcterms::SelfGravity, and TSC-weighted into the
//! particle's IPGX/IPGY/IPGZ slots. Mirrors Athena++ ParticleGravity.
void ParticleMesh::GatherGravity(const DvceArray5D<Real>& phi_in,
                                 DvceArray2D<Real>&        prtcl_rdata_io,
                                 const DvceArray2D<int>&   prtcl_idata_in,
                                 int npar) {
  if (npar <= 0) return;

  auto &pr = prtcl_rdata_io;
  auto &pi = prtcl_idata_in;
  auto ph  = phi_in;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int gids = pmy_pack->gids;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  par_for("PMGatherGravity", DevExeSpace(), 0, npar - 1,
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID, p) - gids;

      // Central-difference prefactors a = 0.5/dx (matches srcterms SelfGravity).
      Real a1 = 0.5 / mbsize.d_view(m).dx1;
      Real a2 = multi_d ? 0.5 / mbsize.d_view(m).dx2 : 0.0;
      Real a3 = three_d ? 0.5 / mbsize.d_view(m).dx3 : 0.0;

      // x-axis: always active.
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

      Real gx = 0.0, gy = 0.0, gz = 0.0;
      const int nkk = three_d ? 3 : 1;
      const int njj = multi_d ? 3 : 1;
      for (int kk = 0; kk < nkk; ++kk) {
        Real w3 = three_d ? TSCWeight(fz + static_cast<Real>(kk)) : 1.0;
        int  kc = three_d ? (k0 + kk) : 0;
        for (int jj = 0; jj < njj; ++jj) {
          Real w2 = multi_d ? TSCWeight(fy + static_cast<Real>(jj)) : 1.0;
          int  jc = multi_d ? (j0 + jj) : 0;
          for (int ii = 0; ii < 3; ++ii) {
            Real w1 = TSCWeight(fx + static_cast<Real>(ii));
            int  ic = i0 + ii;
            Real w  = w1 * w2 * w3;
            gx += w * a1 * (ph(m, 0, kc, jc, ic-1) - ph(m, 0, kc, jc, ic+1));
            if (multi_d) gy += w * a2 * (ph(m, 0, kc, jc-1, ic) - ph(m, 0, kc, jc+1, ic));
            if (three_d) gz += w * a3 * (ph(m, 0, kc-1, jc, ic) - ph(m, 0, kc+1, jc, ic));
          }
        }
      }
      pr(IPGX, p) = gx;
      pr(IPGY, p) = gy;
      pr(IPGZ, p) = gz;
    });
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::FlushDepositBoundaries()
//! \brief Fold ghost-zone deposit contributions into neighbour interiors, then zero the
//! ghost spill (Phase 1c). Same-level, on-rank neighbours only for now.
//!
//! For a same-level neighbour, this MeshBlock's ghost cells for buffer n (recvbuf[n].isame
//! -- the region a normal exchange would *fill* from the neighbour) coincide cell-for-cell,
//! in buffer order, with the neighbour's interior region it would *send* from
//! (sendbuf[dest].isame). So we atomic-add dmesh(m, ghost) into dmesh(dn_mb, interior)
//! using that correspondence -- no offset/geometry table needed. After all reads complete,
//! a second kernel zeroes every ghost cell so the spill is not double-counted when the
//! source is assembled.
void ParticleMesh::FlushDepositBoundaries() {
  int nmb    = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar   = dmesh.extent_int(1);

  // One-time warning for the not-yet-implemented cases (cross-rank / refinement).
  static bool warned = false;
  if (!warned && pmy_pack->pmesh->multilevel) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in ParticleMesh::FlushDepositBoundaries: particle-mesh "
                << "deposit flush across refinement (AMR/SMR) boundaries is not yet "
                << "implemented; deposits near such boundaries are dropped." << std::endl;
    }
    warned = true;
  }
#if MPI_PARALLEL_ENABLED
  if (!warned && global_variable::nranks > 1) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in ParticleMesh::FlushDepositBoundaries: cross-rank (MPI) "
                << "particle-mesh deposit flush is not yet implemented; deposits that spill "
                << "across a rank boundary are dropped." << std::endl;
    }
    warned = true;
  }
#endif

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf  = pmbval->sendbuf;
  auto &rbuf  = pmbval->recvbuf;
  auto dm = dmesh;

  // Add each MeshBlock's ghost-zone deposit into the matching neighbour interior cells.
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("PMFlush", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // same-level, on-rank neighbours only (see header note)
    if (nghbr.d_view(m,n).gid >= 0 &&
        nghbr.d_view(m,n).lev == mblev.d_view(m) &&
        nghbr.d_view(m,n).rank == my_rank) {
      // my ghost region for neighbour n (what a normal exchange would fill)
      int il = rbuf[n].isame[0].bis;
      int iu = rbuf[n].isame[0].bie;
      int jl = rbuf[n].isame[0].bjs;
      int ju = rbuf[n].isame[0].bje;
      int kl = rbuf[n].isame[0].bks;
      int ku = rbuf[n].isame[0].bke;

      // neighbour MeshBlock index and its reciprocal interior region (what it sends to me)
      int dnb = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn  = nghbr.d_view(m,n).dest;
      int il2 = sbuf[dn].isame[0].bis;
      int jl2 = sbuf[dn].isame[0].bjs;
      int kl2 = sbuf[dn].isame[0].bks;

      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj = nk*nj;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k*nj) + jl;
        k += kl;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu+1),
        [&](const int i) {
          int i2 = il2 + (i - il);
          int j2 = jl2 + (j - jl);
          int k2 = kl2 + (k - kl);
          Kokkos::atomic_add(&dm(dnb, v, k2, j2, i2), dm(m, v, k, j, i));
        });
      });
    }
    tmember.team_barrier();
  });

  // Zero every ghost cell: its deposit has been flushed to the owning neighbour interior
  // (or, at a true domain boundary, lies outside the domain). Prevents double-counting.
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n1 = dmesh.extent_int(4);
  const int n2 = dmesh.extent_int(3);
  const int n3 = dmesh.extent_int(2);
  par_for("PMFlushZeroGhost", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int v, int k, int j, int i) {
    if (i < is || i > ie || j < js || j > je || k < ks || k > ke) {
      dm(m, v, k, j, i) = 0.0;
    }
  });
}

}  // namespace particles
