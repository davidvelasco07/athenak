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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/cell_locations.hpp"
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
  // Size for the maximum MeshBlock count this rank can hold (the AthenaK convention,
  // cf. hydro u0): with AMR the pack can grow up to nmb_maxperrank, and pre-sizing
  // avoids per-regrid reallocation (DepositMass keeps a defensive realloc).
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  Kokkos::realloc(dmesh, nmb, nmeshaux, ncells3, ncells2, ncells1);
  Zero();

  // Device counter for particles skipped by the deposit/gather index guards.
  Kokkos::realloc(nbad_, 1);
  Kokkos::deep_copy(nbad_, 0);

  // Boundary-values object for dmesh: provides the buffers/MPI machinery for the future
  // cross-rank flush path, and is reused by Particles::ExchangePhi for phi's halo swap.
  pmbval = new MeshBoundaryValuesCC(ppack, pin, false);
  pmbval->InitializeBuffers(nmeshaux);

  // Cross-rank deposit-flush staging buffer + communicator. The ghost spill destined
  // off-rank is bounded by the ghost-cell count of the blocks a sink straddles; size
  // generously and grow on demand (overflow guarded + warned).
  dfemit_max_ = 4096;
  Kokkos::realloc(dfemit_, dfemit_max_, 6);
  Kokkos::realloc(dfemit_cnt_, 1);
#if MPI_PARALLEL_ENABLED
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_dfscat_);
#endif
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
  // Grow dmesh if AMR has raised the number of MeshBlocks in this pack beyond the
  // (max-sized) allocation -- defensive only; the constructor sizes for nmb_maxperrank,
  // so this should never trigger. Never shrink: oversized is harmless and downsizing
  // would churn allocations every regrid.
  {
    int nmb = pmy_pack->nmb_thispack;
    if (dmesh.extent_int(0) < nmb) {
      auto &ix = pmy_pack->pmesh->mb_indcs;
      int nc1 = ix.nx1 + 2*ix.ng;
      int nc2 = (ix.nx2 > 1) ? (ix.nx2 + 2*ix.ng) : 1;
      int nc3 = (ix.nx3 > 1) ? (ix.nx3 + 2*ix.ng) : 1;
      Kokkos::realloc(dmesh, nmb, nmeshaux, nc3, nc2, nc1);
    }
  }
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
  const int nmb_ = pmy_pack->nmb_thispack;
  const int n1 = dmesh.extent_int(4);
  const int n2 = dmesh.extent_int(3);
  const int n3 = dmesh.extent_int(2);
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto bad = nbad_;
  // level-matched coarse-lattice path: requires ng >= 4 (the coarse CIC gather stencil
  // reaches 2 coarse = 4 fine cells past a block face). With fewer ghosts fall back to
  // the fine-lattice path everywhere (one-time warning): an asymmetric near-interface
  // kernel beats a zeroed one.
  const bool multilevel = pmy_pack->pmesh->multilevel && (indcs.ng >= 4);
  if (pmy_pack->pmesh->multilevel && indcs.ng < 4) {
    static bool warned = false;
    if (!warned && global_variable::my_rank == 0) {
      std::cout << "### WARNING in ParticleMesh: level-matched coarse-lattice deposit/"
                << "gather needs nghost >= 4 (have " << indcs.ng << "); particles near "
                << "coarse-fine interfaces use the fine-lattice kernel (asymmetric)."
                << std::endl;
      warned = true;
    }
  }
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &msz = pmy_pack->pmesh->mesh_size;

  par_for("PMDepositMass", DevExeSpace(), 0, npar - 1,
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID, p) - gids;
      // Guard: a stale/foreign PGID (e.g. mid-regrid pathology) must not index out of
      // the pack. Count and skip rather than corrupt memory / SIGBUS.
      if (m < 0 || m >= nmb_) { Kokkos::atomic_add(&bad(0), 1); return; }
      Real mp = pr(IPM, p);

      // Cell volume (uniform Cartesian for now).
      Real dV = mbsize.d_view(m).dx1;
      if (multi_d) dV *= mbsize.d_view(m).dx2;
      if (three_d) dV *= mbsize.d_view(m).dx3;
      Real inv_dV = 1.0 / dV;

      // ---- level-matched coarse-lattice deposit -------------------------------------
      // If a COARSER region lies within the fine TSC footprint (2 dx), a fine-lattice
      // deposit would put part of the cloud at dx and part effectively at 2 dx: the
      // kernel the particle sees through its own deposit is asymmetric and the
      // deposit/gather self-force cancellation breaks (measured as a spurious
      // interface force on sinks). Instead deposit the WHOLE cloud with CIC on the
      // COARSE lattice in the global frame: each coarse-cell weight becomes a uniform
      // density over the 2^d fine cells covering it (conservative; the boundary flush
      // reconstitutes the exact coarse value on coarse neighbours). GatherGravity uses
      // the matching CIC coarse-lattice gather, so the pair is symmetric on one
      // uniform (coarse) lattice and the self-force cancels there. CIC (not TSC) keeps
      // the worst-case stencil reach at 2 coarse = 4 fine cells = ng. Proper nesting
      // bounds the jump to one level.
      if (multilevel && three_d) {
        const Real dxf = mbsize.d_view(m).dx1;
        if (NearCoarserLevel(nghbr, mbsize, mblev, m, multi_d, three_d,
                             pr(IPX, p), pr(IPY, p), pr(IPZ, p), 2.0*dxf)) {
          const Real dxc = 2.0*dxf, dyc = 2.0*mbsize.d_view(m).dx2;
          const Real dzc = 2.0*mbsize.d_view(m).dx3;
          const Real inv_dVc = 1.0/(dxc*dyc*dzc);
          // global coarse-lattice fractional index and bracketing cells
          const Real xic = (pr(IPX, p) - msz.x1min)/dxc;
          const Real yic = (pr(IPY, p) - msz.x2min)/dyc;
          const Real zic = (pr(IPZ, p) - msz.x3min)/dzc;
          if (!(xic > -1.0e9 && xic < 1.0e9) || !(yic > -1.0e9 && yic < 1.0e9) ||
              !(zic > -1.0e9 && zic < 1.0e9)) {
            Kokkos::atomic_add(&bad(0), 1); return;
          }
          const int ic0 = static_cast<int>(Kokkos::floor(xic - 0.5));
          const int jc0 = static_cast<int>(Kokkos::floor(yic - 0.5));
          const int kc0 = static_cast<int>(Kokkos::floor(zic - 0.5));
          const Real fxc = xic - (static_cast<Real>(ic0) + 0.5);
          const Real fyc = yic - (static_cast<Real>(jc0) + 0.5);
          const Real fzc = zic - (static_cast<Real>(kc0) + 0.5);
          // owner block's fine-cell offset from the mesh origin (exact integer)
          const int of1 = static_cast<int>((mbsize.d_view(m).x1min - msz.x1min)/dxf + 0.5);
          const int of2 = static_cast<int>((mbsize.d_view(m).x2min - msz.x2min)/
                                           mbsize.d_view(m).dx2 + 0.5);
          const int of3 = static_cast<int>((mbsize.d_view(m).x3min - msz.x3min)/
                                           mbsize.d_view(m).dx3 + 0.5);
          // whole-footprint bounds check (fine indices of the 4^3 covered fine cells);
          // a correctly-binned particle always passes (reach <= 2 coarse = ng fine)
          const int if0 = 2*ic0 - of1 + is, jf0 = 2*jc0 - of2 + js;
          const int kf0 = 2*kc0 - of3 + ks;
          if (if0 < 0 || if0 + 3 >= n1 || jf0 < 0 || jf0 + 3 >= n2 ||
              kf0 < 0 || kf0 + 3 >= n3) {
            Kokkos::atomic_add(&bad(0), 1); return;
          }
          for (int kk = 0; kk < 2; ++kk) {
            const Real w3 = (kk == 0) ? (1.0 - fzc) : fzc;
            for (int jj = 0; jj < 2; ++jj) {
              const Real w2 = (jj == 0) ? (1.0 - fyc) : fyc;
              for (int ii = 0; ii < 2; ++ii) {
                const Real w1 = (ii == 0) ? (1.0 - fxc) : fxc;
                const Real rho_c = mp*w1*w2*w3*inv_dVc;
                // uniform density into the 8 fine cells covering this coarse cell
                for (int fk = 0; fk < 2; ++fk) {
                  const int kf = kf0 + 2*kk + fk;
                  for (int fj = 0; fj < 2; ++fj) {
                    const int jf = jf0 + 2*jj + fj;
                    for (int fi = 0; fi < 2; ++fi) {
                      Kokkos::atomic_add(&dm(m, 0, kf, jf, if0 + 2*ii + fi), rho_c);
                    }
                  }
                }
              }
            }
          }
          return;   // coarse path complete for this particle
        }
      }

      // x-axis: always active (particles only in 2D/3D).
      Real xi1 = (pr(IPX, p) - mbsize.d_view(m).x1min) / mbsize.d_view(m).dx1 + is;
      // Guard the double->int casts: a non-finite or wildly out-of-block position (from
      // any upstream pathology, e.g. a diverging force) is undefined behaviour in the
      // cast and would index far outside dmesh. Skip + count instead.
      if (!(xi1 > -1.0e9 && xi1 < 1.0e9)) { Kokkos::atomic_add(&bad(0), 1); return; }
      int  i0  = static_cast<int>(xi1 - 1.0);
      Real fx  = (static_cast<Real>(i0) + 0.5) - xi1;

      // y-axis: active in 2D/3D.
      int  j0 = 0;
      Real fy = 0.0;
      if (multi_d) {
        Real xi2 = (pr(IPY, p) - mbsize.d_view(m).x2min) / mbsize.d_view(m).dx2 + js;
        if (!(xi2 > -1.0e9 && xi2 < 1.0e9)) { Kokkos::atomic_add(&bad(0), 1); return; }
        j0 = static_cast<int>(xi2 - 1.0);
        fy = (static_cast<Real>(j0) + 0.5) - xi2;
      }

      // z-axis: active in 3D only.
      int  k0 = 0;
      Real fz = 0.0;
      if (three_d) {
        Real xi3 = (pr(IPZ, p) - mbsize.d_view(m).x3min) / mbsize.d_view(m).dx3 + ks;
        if (!(xi3 > -1.0e9 && xi3 < 1.0e9)) { Kokkos::atomic_add(&bad(0), 1); return; }
        k0 = static_cast<int>(xi3 - 1.0);
        fz = (static_cast<Real>(k0) + 0.5) - xi3;
      }

      // Guard: the full 3-cell TSC footprint must lie within the array (interior +
      // ghosts). A correctly-binned particle always satisfies this (with ng >= 2);
      // a mis-binned one is counted and skipped.
      if (i0 < 0 || i0 + 2 >= n1 ||
          (multi_d && (j0 < 0 || j0 + 2 >= n2)) ||
          (three_d && (k0 < 0 || k0 + 2 >= n3))) {
        Kokkos::atomic_add(&bad(0), 1);
        return;
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

  // Surface any guarded skips (rate-limited warning). A nonzero count means particles
  // with an invalid PGID or a non-finite/out-of-range position were excluded from the
  // deposit this stage -- the run continues, but mass is missing from the source.
  auto bad_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), nbad_);
  if (bad_h(0) > 0) {
    static int nwarn = 0;
    if (nwarn < 8 && global_variable::my_rank == 0) {
      std::cout << "### WARNING in ParticleMesh::DepositMass: " << bad_h(0)
                << " particle(s) skipped by index guards (invalid PGID or position); "
                << "particle state is likely corrupt (warning " << nwarn+1 << "/8)"
                << std::endl;
      ++nwarn;
    }
    Kokkos::deep_copy(nbad_, 0);
  }
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
  const int nmb_ = pmy_pack->nmb_thispack;
  const int n1 = phi_in.extent_int(4);
  const int n2 = phi_in.extent_int(3);
  const int n3 = phi_in.extent_int(2);
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto bad = nbad_;
  // level-matched coarse-lattice path (mirrors DepositMass, incl. the ng >= 4 gate)
  const bool multilevel = pmy_pack->pmesh->multilevel && (indcs.ng >= 4);
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &msz = pmy_pack->pmesh->mesh_size;

  par_for("PMGatherGravity", DevExeSpace(), 0, npar - 1,
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID, p) - gids;
      // Guards mirroring DepositMass: never index outside phi. A skipped particle
      // keeps zero acceleration this stage (counted; DepositMass warns).
      if (m < 0 || m >= nmb_) {
        Kokkos::atomic_add(&bad(0), 1);
        pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
        return;
      }

      // ---- level-matched coarse-lattice gather ----------------------------------------
      // MUST take this branch exactly when DepositMass took its coarse branch (same
      // criterion, same reach): the self-force cancels only when deposit and gather use
      // the same kernel on the same lattice. phi at a coarse cell is the mean of its
      // 2^d covering fine values from the owner's array; for ghost regions beyond a
      // coarse-fine boundary those fine values are ProlongateCC output, whose 2^d mean
      // IS the neighbour's coarse phi exactly (the prolongation is mean-preserving), so
      // no coarse-array plumbing is needed. Force = CIC-weighted central differences on
      // the coarse lattice (a = 0.5/dx_c), matching srcterms at that level.
      if (multilevel && three_d) {
        const Real dxf = mbsize.d_view(m).dx1;
        if (NearCoarserLevel(nghbr, mbsize, mblev, m, multi_d, three_d,
                             pr(IPX, p), pr(IPY, p), pr(IPZ, p), 2.0*dxf)) {
          const Real dxc = 2.0*dxf, dyc = 2.0*mbsize.d_view(m).dx2;
          const Real dzc = 2.0*mbsize.d_view(m).dx3;
          const Real xic = (pr(IPX, p) - msz.x1min)/dxc;
          const Real yic = (pr(IPY, p) - msz.x2min)/dyc;
          const Real zic = (pr(IPZ, p) - msz.x3min)/dzc;
          if (!(xic > -1.0e9 && xic < 1.0e9) || !(yic > -1.0e9 && yic < 1.0e9) ||
              !(zic > -1.0e9 && zic < 1.0e9)) {
            Kokkos::atomic_add(&bad(0), 1);
            pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
            return;
          }
          const int ic0 = static_cast<int>(Kokkos::floor(xic - 0.5));
          const int jc0 = static_cast<int>(Kokkos::floor(yic - 0.5));
          const int kc0 = static_cast<int>(Kokkos::floor(zic - 0.5));
          const Real fxc = xic - (static_cast<Real>(ic0) + 0.5);
          const Real fyc = yic - (static_cast<Real>(jc0) + 0.5);
          const Real fzc = zic - (static_cast<Real>(kc0) + 0.5);
          const int of1 = static_cast<int>((mbsize.d_view(m).x1min - msz.x1min)/dxf + 0.5);
          const int of2 = static_cast<int>((mbsize.d_view(m).x2min - msz.x2min)/
                                           mbsize.d_view(m).dx2 + 0.5);
          const int of3 = static_cast<int>((mbsize.d_view(m).x3min - msz.x3min)/
                                           mbsize.d_view(m).dx3 + 0.5);
          // fine index of the low corner of coarse cell (ic0,jc0,kc0) in the owner array
          const int if0 = 2*ic0 - of1 + is;
          const int jf0 = 2*jc0 - of2 + js;
          const int kf0 = 2*kc0 - of3 + ks;
          // whole-stencil bounds: coarse cells [ic0-1, ic0+2] -> fine [if0-2, if0+5]
          if (if0 - 2 < 0 || if0 + 5 >= n1 || jf0 - 2 < 0 || jf0 + 5 >= n2 ||
              kf0 - 2 < 0 || kf0 + 5 >= n3) {
            Kokkos::atomic_add(&bad(0), 1);
            pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
            return;
          }
          // coarse-cell phi = mean of the 2^3 covering fine values, evaluated lazily
          // via fine low-corner indices relative to (if0,jf0,kf0)
          auto phic = [&](const int dic, const int djc, const int dkc) -> Real {
            const int i = if0 + 2*dic, j = jf0 + 2*djc, k = kf0 + 2*dkc;
            return 0.125*(ph(m,0,k  ,j  ,i) + ph(m,0,k  ,j  ,i+1) +
                          ph(m,0,k  ,j+1,i) + ph(m,0,k  ,j+1,i+1) +
                          ph(m,0,k+1,j  ,i) + ph(m,0,k+1,j  ,i+1) +
                          ph(m,0,k+1,j+1,i) + ph(m,0,k+1,j+1,i+1));
          };
          const Real a1c = 0.5/dxc, a2c = 0.5/dyc, a3c = 0.5/dzc;
          Real gx = 0.0, gy = 0.0, gz = 0.0;
          for (int kk = 0; kk < 2; ++kk) {
            const Real w3 = (kk == 0) ? (1.0 - fzc) : fzc;
            for (int jj = 0; jj < 2; ++jj) {
              const Real w2 = (jj == 0) ? (1.0 - fyc) : fyc;
              for (int ii = 0; ii < 2; ++ii) {
                const Real w1 = (ii == 0) ? (1.0 - fxc) : fxc;
                const Real w = w1*w2*w3;
                gx += w*a1c*(phic(ii-1, jj, kk) - phic(ii+1, jj, kk));
                gy += w*a2c*(phic(ii, jj-1, kk) - phic(ii, jj+1, kk));
                gz += w*a3c*(phic(ii, jj, kk-1) - phic(ii, jj, kk+1));
              }
            }
          }
          pr(IPGX, p) = gx;
          pr(IPGY, p) = gy;
          pr(IPGZ, p) = gz;
          return;   // coarse path complete for this particle
        }
      }

      // Central-difference prefactors a = 0.5/dx (matches srcterms SelfGravity).
      Real a1 = 0.5 / mbsize.d_view(m).dx1;
      Real a2 = multi_d ? 0.5 / mbsize.d_view(m).dx2 : 0.0;
      Real a3 = three_d ? 0.5 / mbsize.d_view(m).dx3 : 0.0;

      // x-axis: always active.
      Real xi1 = (pr(IPX, p) - mbsize.d_view(m).x1min) / mbsize.d_view(m).dx1 + is;
      if (!(xi1 > -1.0e9 && xi1 < 1.0e9)) {
        Kokkos::atomic_add(&bad(0), 1);
        pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
        return;
      }
      int  i0  = static_cast<int>(xi1 - 1.0);
      Real fx  = (static_cast<Real>(i0) + 0.5) - xi1;

      // y-axis: active in 2D/3D.
      int  j0 = 0;
      Real fy = 0.0;
      if (multi_d) {
        Real xi2 = (pr(IPY, p) - mbsize.d_view(m).x2min) / mbsize.d_view(m).dx2 + js;
        if (!(xi2 > -1.0e9 && xi2 < 1.0e9)) {
          Kokkos::atomic_add(&bad(0), 1);
          pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
          return;
        }
        j0 = static_cast<int>(xi2 - 1.0);
        fy = (static_cast<Real>(j0) + 0.5) - xi2;
      }

      // z-axis: active in 3D only.
      int  k0 = 0;
      Real fz = 0.0;
      if (three_d) {
        Real xi3 = (pr(IPZ, p) - mbsize.d_view(m).x3min) / mbsize.d_view(m).dx3 + ks;
        if (!(xi3 > -1.0e9 && xi3 < 1.0e9)) {
          Kokkos::atomic_add(&bad(0), 1);
          pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
          return;
        }
        k0 = static_cast<int>(xi3 - 1.0);
        fz = (static_cast<Real>(k0) + 0.5) - xi3;
      }

      // Guard: the TSC cloud [i0, i0+2] must lie within the array (matches the deposit
      // guard); beyond that the particle is genuinely mis-binned. Cloud cells whose
      // CENTRAL DIFFERENCE would poke one cell past the array edge are handled
      // per-cell below: that only happens for the extreme cell of the cloud when the
      // particle sits exactly on a MeshBlock face, where that cell's TSC weight
      // tends to zero -- so skipping just that cell is exact (weight-continuous),
      // whereas zeroing the whole particle's force would inject momentum.
      if (i0 < 0 || i0 + 2 >= n1 ||
          (multi_d && (j0 < 0 || j0 + 2 >= n2)) ||
          (three_d && (k0 < 0 || k0 + 2 >= n3))) {
        Kokkos::atomic_add(&bad(0), 1);
        pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
        return;
      }

      Real gx = 0.0, gy = 0.0, gz = 0.0;
      const int nkk = three_d ? 3 : 1;
      const int njj = multi_d ? 3 : 1;
      for (int kk = 0; kk < nkk; ++kk) {
        Real w3 = three_d ? TSCWeight(fz + static_cast<Real>(kk)) : 1.0;
        int  kc = three_d ? (k0 + kk) : 0;
        if (three_d && (kc-1 < 0 || kc+1 >= n3)) continue;   // edge cell, weight ~0
        for (int jj = 0; jj < njj; ++jj) {
          Real w2 = multi_d ? TSCWeight(fy + static_cast<Real>(jj)) : 1.0;
          int  jc = multi_d ? (j0 + jj) : 0;
          if (multi_d && (jc-1 < 0 || jc+1 >= n2)) continue;  // edge cell, weight ~0
          for (int ii = 0; ii < 3; ++ii) {
            Real w1 = TSCWeight(fx + static_cast<Real>(ii));
            int  ic = i0 + ii;
            if (ic-1 < 0 || ic+1 >= n1) continue;             // edge cell, weight ~0
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
//! ghost spill (Phase 1c + AMR). On-rank neighbours at any refinement level.
//!
//! Each nonzero ghost cell resolves its owner geometrically: it searches this block's
//! neighbour list for the (unique) on-rank MeshBlock whose bounds contain the cell's
//! centre (wrapped by the domain length if periodic), then adds its density there
//! mass-conservatively:
//!   same level    : one coincident destination cell,      add rho            (dV ratio 1)
//!   fine -> coarse: one containing coarse cell,           add rho * dV_f/dV_c (= rho/2^d)
//!   coarse -> fine: the 2^d fine cells this cell covers,  add rho to each     (uniform
//!                   split of the mass rho*dV_c over 2^d cells of volume dV_f)
//! Geometric ownership handles faces, edges and corners in one code path -- including
//! coarser "interior edge" regions for which SetNeighbors leaves no dedicated slot
//! because the coarse face neighbour covers them. Cells owned by off-rank neighbours
//! (MPI, warned once) or lying outside the domain (physical boundaries) are dropped;
//! the trailing ghost-zero pass discards them either way. After all adds complete, a
//! second kernel zeroes every ghost cell so the spill is not double-counted when the
//! gravity source is assembled.
void ParticleMesh::FlushDepositBoundaries() {
  int nmb    = pmy_pack->nmb_thispack;
  int nvar   = dmesh.extent_int(1);

  // Cross-rank spill: on >1 rank, every nonzero ghost cell is staged into dfemit_ and
  // ExchangeDepositFlush() (below) atomic-adds the ones whose containing block is an
  // off-rank same-level neighbour. Zero the counter each call.
  const bool mpi_on = (global_variable::nranks > 1);
  if (mpi_on) Kokkos::deep_copy(dfemit_cnt_, 0);
  auto dfemit = dfemit_;
  auto dfcnt  = dfemit_cnt_;
  const int dfmax = dfemit_max_;

  const int my_rank = global_variable::my_rank;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &msize = pmy_pack->pmb->mb_size;
  auto dm = dmesh;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n1 = dmesh.extent_int(4);
  const int n2 = dmesh.extent_int(3);
  const int n3 = dmesh.extent_int(2);
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  // domain lengths, for wrapping ghost-cell positions across periodic boundaries
  auto &msz = pmy_pack->pmesh->mesh_size;
  const Real Lx1 = msz.x1max - msz.x1min;
  const Real Lx2 = msz.x2max - msz.x2min;
  const Real Lx3 = msz.x3max - msz.x3min;

  par_for("PMFlush", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int v, int k, int j, int i) {
    // ghost cells only
    if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) return;
    const Real rho = dm(m, v, k, j, i);
    if (rho == 0.0) return;                       // almost always: nothing deposited here

    // physical centre of this ghost cell
    const Real dxs = msize.d_view(m).dx1;
    const Real dys = msize.d_view(m).dx2;
    const Real dzs = msize.d_view(m).dx3;
    const Real xs = msize.d_view(m).x1min + (static_cast<Real>(i-is) + 0.5)*dxs;
    const Real ys = multi_d ?
                    msize.d_view(m).x2min + (static_cast<Real>(j-js) + 0.5)*dys : 0.0;
    const Real zs = three_d ?
                    msize.d_view(m).x3min + (static_cast<Real>(k-ks) + 0.5)*dzs : 0.0;
    const int mylev = mblev.d_view(m);

    // stage this nonzero ghost cell for the cross-rank exchange (multi-rank only). Its
    // containing block is resolved host-side in ExchangeDepositFlush(); if that block is
    // an off-rank same-level neighbour the deposit is atomic-added there, else skipped
    // (on-rank containers are handled by the device loop below; off-rank level-jumps are
    // dropped, as on-rank AMR was validated separately).
    if (mpi_on) {
      const int e = Kokkos::atomic_fetch_add(&dfcnt(0), 1);
      if (e < dfmax) {
        dfemit(e, 0) = static_cast<Real>(m);
        dfemit(e, 1) = static_cast<Real>(v);
        dfemit(e, 2) = xs; dfemit(e, 3) = ys; dfemit(e, 4) = zs;
        dfemit(e, 5) = rho;
      }
    }

    // find the neighbour MeshBlock whose bounds contain this cell centre
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.d_view(m,n).gid < 0) continue;
      if (nghbr.d_view(m,n).rank != my_rank) continue;   // off-rank: via ExchangeDepositFlush
      const int dnb = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      if (dnb < 0 || dnb >= nmb) continue;               // safety

      const Real x1d = msize.d_view(dnb).x1min, x1u = msize.d_view(dnb).x1max;
      const Real x2d = msize.d_view(dnb).x2min, x2u = msize.d_view(dnb).x2max;
      const Real x3d = msize.d_view(dnb).x3min, x3u = msize.d_view(dnb).x3max;

      // wrap the cell position into the neighbour's frame (periodic domains)
      Real xc = xs, yc = ys, zc = zs;
      Real ctr = 0.5*(x1d + x1u);
      if (xc - ctr >  0.5*Lx1) xc -= Lx1;
      if (xc - ctr < -0.5*Lx1) xc += Lx1;
      if (multi_d) {
        ctr = 0.5*(x2d + x2u);
        if (yc - ctr >  0.5*Lx2) yc -= Lx2;
        if (yc - ctr < -0.5*Lx2) yc += Lx2;
      }
      if (three_d) {
        ctr = 0.5*(x3d + x3u);
        if (zc - ctr >  0.5*Lx3) zc -= Lx3;
        if (zc - ctr < -0.5*Lx3) zc += Lx3;
      }

      // containment (lower-inclusive, consistent with particle binning)
      if (xc < x1d || xc >= x1u) continue;
      if (multi_d && (yc < x2d || yc >= x2u)) continue;
      if (three_d && (zc < x3d || zc >= x3u)) continue;

      const Real dxd = msize.d_view(dnb).dx1;
      const Real dyd = msize.d_view(dnb).dx2;
      const Real dzd = msize.d_view(dnb).dx3;
      const bool dst_finer = (nghbr.d_view(m,n).lev > mylev);

      if (!dst_finer) {
        // same level (ratio 1) or coarser (ratio 1/2^d): single containing cell
        Real wvol = dxs/dxd;
        if (multi_d) wvol *= dys/dyd;
        if (three_d) wvol *= dzs/dzd;
        int it = is + static_cast<int>((xc - x1d)/dxd);
        int jt = multi_d ? js + static_cast<int>((yc - x2d)/dyd) : 0;
        int kt = three_d ? ks + static_cast<int>((zc - x3d)/dzd) : 0;
        // clamp against floating-point edge cases
        it = (it < is) ? is : (it > ie) ? ie : it;
        if (multi_d) jt = (jt < js) ? js : (jt > je) ? je : jt;
        if (three_d) kt = (kt < ks) ? ks : (kt > ke) ? ke : kt;
        Kokkos::atomic_add(&dm(dnb, v, kt, jt, it), rho*wvol);
      } else {
        // finer: split the mass uniformly over the 2 (per active dim) fine cells this
        // coarse cell covers; each receives Delta(rho_fine) = rho (dV ratio cancels)
        const int r1 = static_cast<int>(dxs/dxd + 0.5);
        const int r2 = multi_d ? static_cast<int>(dys/dyd + 0.5) : 1;
        const int r3 = three_d ? static_cast<int>(dzs/dzd + 0.5) : 1;
        Real wvol = dxs/dxd;
        if (multi_d) wvol *= dys/dyd;
        if (three_d) wvol *= dzs/dzd;
        const Real wsplit = wvol / (static_cast<Real>(r1*r2*r3));
        const int it0 = is + static_cast<int>((xc - 0.5*dxs - x1d)/dxd + 0.5);
        const int jt0 = multi_d ? js + static_cast<int>((yc - 0.5*dys - x2d)/dyd + 0.5) : 0;
        const int kt0 = three_d ? ks + static_cast<int>((zc - 0.5*dzs - x3d)/dzd + 0.5) : 0;
        for (int kk = 0; kk < r3; ++kk) {
          const int kt = three_d ? kt0 + kk : 0;
          if (three_d && (kt < ks || kt > ke)) continue;
          for (int jj = 0; jj < r2; ++jj) {
            const int jt = multi_d ? jt0 + jj : 0;
            if (multi_d && (jt < js || jt > je)) continue;
            for (int ii = 0; ii < r1; ++ii) {
              const int it = it0 + ii;
              if (it < is || it > ie) continue;
              Kokkos::atomic_add(&dm(dnb, v, kt, jt, it), rho*wsplit);
            }
          }
        }
      }
      break;   // owner found; each ghost cell is folded exactly once
    }
  });

  // add spill whose containing block is an off-rank same-level neighbour (multi-rank)
  if (mpi_on) ExchangeDepositFlush();

  // Zero every ghost cell: its deposit has been flushed to the owning neighbour interior
  // (or, at a physical/off-rank boundary, dropped). Prevents double-counting.
  par_for("PMFlushZeroGhost", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int v, int k, int j, int i) {
    if (i < is || i > ie || j < js || j > je || k < ks || k > ke) {
      dm(m, v, k, j, i) = 0.0;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::ExchangeDepositFlush()
//! \brief MPI transport for the deposit-flush ghost spill. FlushDepositBoundaries stages
//! every nonzero ghost cell (owner_m, v, x, y, z, rho); here, host-side, each cell is
//! routed to the neighbour block whose interior contains it -- if that block is
//! off-rank the deposit is shipped and atomic-added there (on-rank containers are
//! folded by the device loop). Level jumps are converted on the SENDER so the wire
//! format stays (dest cell, value): fine->coarse adds rho*dV_s/dV_d to the containing
//! coarse cell; coarse->fine splits the mass uniformly over the covered fine cells --
//! the same rules as the on-rank flush. Mirrors Particles::ExchangeCVReset (add, not
//! overwrite; density only).

void ParticleMesh::ExchangeDepositFlush() {
#if MPI_PARALLEL_ENABLED
  const int my_rank = global_variable::my_rank;
  const int nranks  = global_variable::nranks;
  if (nranks == 1) return;

  auto cnt_h = Kokkos::create_mirror_view(dfemit_cnt_);
  Kokkos::deep_copy(cnt_h, dfemit_cnt_);
  const int nemit = std::min(cnt_h(0), dfemit_max_);
  if (cnt_h(0) > dfemit_max_ && my_rank == 0) {
    std::cout << "### WARNING in ParticleMesh::ExchangeDepositFlush: dfemit overflow ("
              << cnt_h(0) << " > " << dfemit_max_ << "); some cross-rank spill dropped."
              << std::endl;
  }
  auto emit_h = Kokkos::create_mirror_view(dfemit_);
  Kokkos::deep_copy(emit_h, dfemit_);

  Mesh *pm = pmy_pack->pmesh;
  auto &ms = pm->mesh_size;
  const Real Lx1 = ms.x1max - ms.x1min;
  const Real Lx2 = ms.x2max - ms.x2min;
  const Real Lx3 = ms.x3max - ms.x3min;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const int nnghbr = pmy_pack->pmb->nnghbr;

  std::vector<int>  s_rank;
  std::vector<int>  s_int;    // 5 ints per record: dest_gid, v, i, j, k
  std::vector<Real> s_real;   // 1 real per record: rho
  for (int e = 0; e < nemit; ++e) {
    const int om = static_cast<int>(emit_h(e, 0));
    if (om < 0 || om >= pmy_pack->nmb_thispack) continue;
    const int olev = mblev.h_view(om);
    const int vv = static_cast<int>(emit_h(e, 1));
    const Real xw = emit_h(e, 2), yw = emit_h(e, 3), zw = emit_h(e, 4);
    const Real rho = emit_h(e, 5);
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.h_view(om, n).gid < 0) continue;
      const int dg = nghbr.h_view(om, n).gid;
      const auto &ll = pm->lloc_eachmb[dg];
      const int nmbx1 = pm->nmb_rootx1 << (ll.level - pm->root_level);
      const int nmbx2 = pm->nmb_rootx2 << (ll.level - pm->root_level);
      const int nmbx3 = pm->nmb_rootx3 << (ll.level - pm->root_level);
      const Real x1d = (ll.lx1 == 0) ? ms.x1min : LeftEdgeX(ll.lx1, nmbx1, ms.x1min, ms.x1max);
      const Real x1u = (ll.lx1 == nmbx1-1) ? ms.x1max
                                           : LeftEdgeX(ll.lx1+1, nmbx1, ms.x1min, ms.x1max);
      const Real x2d = (ll.lx2 == 0) ? ms.x2min : LeftEdgeX(ll.lx2, nmbx2, ms.x2min, ms.x2max);
      const Real x2u = (ll.lx2 == nmbx2-1) ? ms.x2max
                                           : LeftEdgeX(ll.lx2+1, nmbx2, ms.x2min, ms.x2max);
      const Real x3d = (ll.lx3 == 0) ? ms.x3min : LeftEdgeX(ll.lx3, nmbx3, ms.x3min, ms.x3max);
      const Real x3u = (ll.lx3 == nmbx3-1) ? ms.x3max
                                           : LeftEdgeX(ll.lx3+1, nmbx3, ms.x3min, ms.x3max);
      Real xc = xw, yc = yw, zc = zw, ctr;
      ctr = 0.5*(x1d + x1u);  if (xc-ctr >  0.5*Lx1) xc -= Lx1;  if (xc-ctr < -0.5*Lx1) xc += Lx1;
      ctr = 0.5*(x2d + x2u);  if (yc-ctr >  0.5*Lx2) yc -= Lx2;  if (yc-ctr < -0.5*Lx2) yc += Lx2;
      ctr = 0.5*(x3d + x3u);  if (zc-ctr >  0.5*Lx3) zc -= Lx3;  if (zc-ctr < -0.5*Lx3) zc += Lx3;
      if (xc < x1d || xc >= x1u || yc < x2d || yc >= x2u || zc < x3d || zc >= x3u) continue;
      // unique containing block found; route if off-rank (any level). The sender does
      // the level conversion so the wire format stays (dest cell, add-value):
      //   same level   : one cell, add rho;
      //   fine->coarse : one containing coarse cell, add rho*dV_s/dV_d (= rho/8^jump);
      //   coarse->fine : the (2^jump)^3 fine cells the source cell covers, add rho to
      //                  each (uniform split of the mass; matches the on-rank flush).
      // Block edges lie on every coarser lattice, so a cell's coverage never straddles
      // blocks and centre-containment routing stays exactly-once.
      if (nghbr.h_view(om, n).rank != my_rank) {
        const int dlev = nghbr.h_view(om, n).lev;
        const int drank = pm->rank_eachmb[dg];
        const Real dx1 = (x1u-x1d)/indcs.nx1;
        const Real dx2 = (x2u-x2d)/indcs.nx2;
        const Real dx3 = (x3u-x3d)/indcs.nx3;
        if (dlev <= olev) {
          // same level or coarser destination: one containing cell
          Real val = rho;
          for (int q = 0; q < 3*(olev - dlev); ++q) val *= 0.5;   // rho/8^jump
          s_rank.push_back(drank);
          s_int.push_back(dg); s_int.push_back(vv);
          s_int.push_back(is + static_cast<int>((xc-x1d)/dx1));
          s_int.push_back(js + static_cast<int>((yc-x2d)/dx2));
          s_int.push_back(ks + static_cast<int>((zc-x3d)/dx3));
          s_real.push_back(val);
        } else {
          // finer destination: uniform split over the covered fine cells
          const int nsub = 1 << (dlev - olev);
          const Real dxs1 = dx1*nsub, dxs2 = dx2*nsub, dxs3 = dx3*nsub; // source cell size
          const Real x0f = xc - 0.5*dxs1 + 0.5*dx1;   // first covered fine-cell centre
          const Real y0f = yc - 0.5*dxs2 + 0.5*dx2;
          const Real z0f = zc - 0.5*dxs3 + 0.5*dx3;
          for (int sk = 0; sk < nsub; ++sk) {
            for (int sj = 0; sj < nsub; ++sj) {
              for (int si = 0; si < nsub; ++si) {
                s_rank.push_back(drank);
                s_int.push_back(dg); s_int.push_back(vv);
                s_int.push_back(is + static_cast<int>((x0f + si*dx1 - x1d)/dx1));
                s_int.push_back(js + static_cast<int>((y0f + sj*dx2 - x2d)/dx2));
                s_int.push_back(ks + static_cast<int>((z0f + sk*dx3 - x3d)/dx3));
                s_real.push_back(rho);
              }
            }
          }
        }
      }
      break;   // exactly one block contains the cell centre
    }
  }

  std::vector<int> nsend(nranks, 0);
  for (int r : s_rank) nsend[r]++;
  std::vector<int> nrecv(nranks, 0);
  MPI_Alltoall(nsend.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT, mpi_comm_dfscat_);

  std::vector<int> soff(nranks, 0), sacc(nranks, 0);
  for (int r = 1; r < nranks; ++r) soff[r] = soff[r-1] + nsend[r-1];
  const int ntot_send = soff[nranks-1] + nsend[nranks-1];
  std::vector<int>  send_i(5*ntot_send);
  std::vector<Real> send_r(ntot_send);
  for (size_t rec = 0; rec < s_rank.size(); ++rec) {
    const int r = s_rank[rec];
    const int pos = soff[r] + sacc[r]; sacc[r]++;
    for (int q = 0; q < 5; ++q) send_i[5*pos+q] = s_int[5*rec+q];
    send_r[pos] = s_real[rec];
  }

  std::vector<int> roff(nranks, 0);
  for (int r = 1; r < nranks; ++r) roff[r] = roff[r-1] + nrecv[r-1];
  const int ntot_recv = roff[nranks-1] + nrecv[nranks-1];
  std::vector<int>  recv_i(5*ntot_recv);
  std::vector<Real> recv_r(ntot_recv);

  std::vector<MPI_Request> reqs;
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nrecv[r] == 0) continue;
    reqs.emplace_back();
    MPI_Irecv(&recv_i[5*roff[r]], 5*nrecv[r], MPI_INT, r, 1, mpi_comm_dfscat_, &reqs.back());
    reqs.emplace_back();
    MPI_Irecv(&recv_r[roff[r]], nrecv[r], MPI_ATHENA_REAL, r, 0, mpi_comm_dfscat_, &reqs.back());
  }
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nsend[r] == 0) continue;
    reqs.emplace_back();
    MPI_Isend(&send_i[5*soff[r]], 5*nsend[r], MPI_INT, r, 1, mpi_comm_dfscat_, &reqs.back());
    reqs.emplace_back();
    MPI_Isend(&send_r[soff[r]], nsend[r], MPI_ATHENA_REAL, r, 0, mpi_comm_dfscat_, &reqs.back());
  }
  if (!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  if (ntot_recv == 0) return;

  const int gid0 = pm->gids_eachrank[my_rank];
  const int nmb_ = pmy_pack->nmb_thispack;
  auto dm = dmesh;
  DualArray2D<int>  rint("dfrecv_i", ntot_recv, 4);   // m, v, and the (i,j,k) -> pack j,k
  DualArray2D<Real> rreal("dfrecv_r", ntot_recv, 1);
  DualArray2D<int>  rijk("dfrecv_ijk", ntot_recv, 3);
  for (int t = 0; t < ntot_recv; ++t) {
    rint.h_view(t,0) = recv_i[5*t+0] - gid0;   // local block index
    rint.h_view(t,1) = recv_i[5*t+1];          // v
    rijk.h_view(t,0) = recv_i[5*t+2];          // i
    rijk.h_view(t,1) = recv_i[5*t+3];          // j
    rijk.h_view(t,2) = recv_i[5*t+4];          // k
    rreal.h_view(t,0) = recv_r[t];
  }
  rint.template modify<HostMemSpace>();  rint.template sync<DevExeSpace>();
  rijk.template modify<HostMemSpace>();  rijk.template sync<DevExeSpace>();
  rreal.template modify<HostMemSpace>(); rreal.template sync<DevExeSpace>();
  auto ri = rint.d_view;  auto rk = rijk.d_view;  auto rr = rreal.d_view;

  Kokkos::parallel_for("dfflush_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot_recv),
  KOKKOS_LAMBDA(const int t) {
    const int m = ri(t,0), v = ri(t,1);
    const int i = rk(t,0), j = rk(t,1), k = rk(t,2);
    if (m < 0 || m >= nmb_) return;
    Kokkos::atomic_add(&dm(m, v, k, j, i), rr(t,0));
  });
#endif
}

}  // namespace particles
