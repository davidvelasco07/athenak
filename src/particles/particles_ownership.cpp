//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_ownership.cpp
//! \brief position-based particle ownership, valid across AMR regrids

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlePopulation::ReownByPosition
//! \brief Recompute each particle's PGID from its position alone, by finding the
//! MeshBlock whose bounds contain it (lower-inclusive), and stage particles whose owner
//! is on another rank into the send list of the migration chain (SendCnt -> InitRecv ->
//! SendP -> RecvP). Unlike the neighbour-based NewGID, which needs a valid PGID from the
//! previous step, this recovers ownership after an AMR regrid has renumbered MeshBlocks
//! or load balancing has moved them to another rank.
//!
//! Positions are first wrapped back into the domain along periodic axes; a type that
//! keeps a start-of-step position (prev_pos_index >= 0) has it shifted by the same
//! offset. A particle that has left the domain through a non-periodic boundary keeps its
//! PGID when retain_escaped is set (sinks); otherwise it must be frozen or waiting for a
//! final snapshot, as in NewGID.

TaskStatus ParticlePopulation::ReownByPosition(Driver *pdriver, int stage) {
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  int nmb = pmy_pack->nmb_thispack;
  int npart = nprtcl_thispack;
  if (npart <= 0) {
    pbval_part->nprtcl_send = 0;
    Kokkos::realloc(pbval_part->sendlist, 0);
    return TaskStatus::complete;
  }

  Mesh *pm = pmy_pack->pmesh;
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;
  auto periodic = [pm](BoundaryFace inner, BoundaryFace outer) {
    return pm->mesh_bcs[inner] == BoundaryFlag::periodic &&
           pm->mesh_bcs[outer] == BoundaryFlag::periodic;
  };
  const bool wrapx = periodic(BoundaryFace::inner_x1, BoundaryFace::outer_x1);
  const bool wrapy = multi_d && periodic(BoundaryFace::inner_x2, BoundaryFace::outer_x2);
  const bool wrapz = three_d && periodic(BoundaryFace::inner_x3, BoundaryFace::outer_x3);
  const int iprev = prev_pos_index;
  auto &msz = pm->mesh_size;
  const Real xmin = msz.x1min, Lx = msz.x1max - msz.x1min;
  const Real ymin = msz.x2min, Ly = msz.x2max - msz.x2min;
  const Real zmin = msz.x3min, Lz = msz.x3max - msz.x3min;

  // Device pass: re-bin each particle from its (wrapped) position against the local
  // MeshBlocks. A particle contained in no local block is a "stray": its owner is on
  // another rank (it drifted across a rank boundary, or load balancing moved its block)
  // or it has left the domain. Strays are resolved on the host, which alone knows the
  // bounds of off-rank blocks. Device counter + host mirror: a host int incremented
  // atomically from a device kernel reads back stale on GPUs.
  DualArray1D<int>  stray_idx("reown_stray_idx", npart);
  DualArray1D<Real> stray_pos("reown_stray_pos", 3*npart);
  Kokkos::View<int> d_nstray("reown_nstray");
  Kokkos::deep_copy(d_nstray, 0);

  par_for("part_reown", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    Real s;
    if (wrapx) {
      s = Kokkos::floor((pr(IPX, p) - xmin)/Lx)*Lx;
      pr(IPX, p) -= s;
      if (iprev >= 0) pr(iprev, p) -= s;
    }
    if (wrapy) {
      s = Kokkos::floor((pr(IPY, p) - ymin)/Ly)*Ly;
      pr(IPY, p) -= s;
      if (iprev >= 0) pr(iprev+1, p) -= s;
    }
    if (wrapz) {
      s = Kokkos::floor((pr(IPZ, p) - zmin)/Lz)*Lz;
      pr(IPZ, p) -= s;
      if (iprev >= 0) pr(iprev+2, p) -= s;
    }
    const Real px = pr(IPX, p), py = pr(IPY, p), pz = pr(IPZ, p);
    int mown = -1;
    for (int mm = 0; mm < nmb; ++mm) {
      if (px >= mbsize.d_view(mm).x1min && px < mbsize.d_view(mm).x1max &&
          py >= mbsize.d_view(mm).x2min && py < mbsize.d_view(mm).x2max &&
          pz >= mbsize.d_view(mm).x3min && pz < mbsize.d_view(mm).x3max) {
        mown = mm;
      }
    }
    if (mown >= 0) {
      pi(PGID, p) = gids + mown;
    } else {
      const int s = Kokkos::atomic_fetch_add(&d_nstray(), 1);
      stray_idx.d_view(s)     = p;
      stray_pos.d_view(3*s)   = px;
      stray_pos.d_view(3*s+1) = py;
      stray_pos.d_view(3*s+2) = pz;
    }
  });

  int nstray = 0;
  {
    auto h_ns = Kokkos::create_mirror_view(d_nstray);
    Kokkos::deep_copy(h_ns, d_nstray);
    nstray = h_ns();
  }
  pbval_part->nprtcl_send = 0;
  Kokkos::realloc(pbval_part->sendlist, 0);
  if (nstray == 0) return TaskStatus::complete;

  // Host pass over the strays. Every rank holds the whole tree (lloc_eachmb,
  // rank_eachmb), so the owning block is found by geometric containment; this treats
  // drift across ranks and regrid-driven redistribution alike. The sender packs the
  // destination global gid in PGID, so PGID is written back for every resolved stray.
  stray_idx.template modify<DevExeSpace>();  stray_idx.template sync<HostMemSpace>();
  stray_pos.template modify<DevExeSpace>();  stray_pos.template sync<HostMemSpace>();
  decltype(Kokkos::create_mirror_view(HostMemSpace(), prtcl_idata)) h_pi;
  if (!retain_escaped) {
    h_pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), prtcl_idata);
  }
  auto &ms = pm->mesh_size;
  const int rlev = pm->root_level;
  const int my_rank = global_variable::my_rank;
  std::vector<int> wp_idx, wp_gid;                 // (particle index, new global gid)
  std::vector<ParticleLocationData> slist;         // off-rank migration list
  for (int s = 0; s < nstray; ++s) {
    const int  p  = stray_idx.h_view(s);
    const Real px = stray_pos.h_view(3*s);
    const Real py = stray_pos.h_view(3*s+1);
    const Real pz = stray_pos.h_view(3*s+2);
    int dgid = -1;
    for (int g = 0; g < pm->nmb_total; ++g) {
      const LogicalLocation &ll = pm->lloc_eachmb[g];
      const int nx1 = pm->nmb_rootx1 << (ll.level - rlev);
      const Real x1d = LeftEdgeX(ll.lx1,   nx1, ms.x1min, ms.x1max);
      const Real x1u = LeftEdgeX(ll.lx1+1, nx1, ms.x1min, ms.x1max);
      if (px < x1d || px >= x1u) continue;
      if (multi_d) {
        const int nx2 = pm->nmb_rootx2 << (ll.level - rlev);
        const Real x2d = LeftEdgeX(ll.lx2,   nx2, ms.x2min, ms.x2max);
        const Real x2u = LeftEdgeX(ll.lx2+1, nx2, ms.x2min, ms.x2max);
        if (py < x2d || py >= x2u) continue;
      }
      if (three_d) {
        const int nx3 = pm->nmb_rootx3 << (ll.level - rlev);
        const Real x3d = LeftEdgeX(ll.lx3,   nx3, ms.x3min, ms.x3max);
        const Real x3u = LeftEdgeX(ll.lx3+1, nx3, ms.x3min, ms.x3max);
        if (pz < x3d || pz >= x3u) continue;
      }
      dgid = g; break;
    }
    if (dgid < 0) {
      // left the domain through a non-periodic boundary
      if (!retain_escaped) {
        const int status = h_pi(PSTATUS, p);
        if (status != PFROZEN && status != PDELETE_AFTER_SNAPSHOT) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Particle with tag " << h_pi(PTAG, p)
                    << " in population '" << name << "' left the domain at ("
                    << px << ", " << py << ", " << pz << ")" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      continue;   // keep PGID
    }
    wp_idx.push_back(p);  wp_gid.push_back(dgid);
    const int drank = pm->rank_eachmb[dgid];
    if (drank != my_rank) {
      ParticleLocationData d;
      d.prtcl_indx = p;  d.dest_gid = dgid;  d.dest_rank = drank;
      slist.push_back(d);
    }
  }
  const int nwp = static_cast<int>(wp_idx.size());
  if (nwp > 0) {
    DualArray1D<int> wb("reown_writeback", 2*nwp);
    for (int i = 0; i < nwp; ++i) {
      wb.h_view(2*i) = wp_idx[i];  wb.h_view(2*i+1) = wp_gid[i];
    }
    wb.template modify<HostMemSpace>();  wb.template sync<DevExeSpace>();
    auto wbd = wb.d_view;
    par_for("reown_writeback", DevExeSpace(), 0, nwp-1, KOKKOS_LAMBDA(const int i) {
      pi(PGID, wbd(2*i)) = wbd(2*i+1);
    });
  }
  const int nsend = static_cast<int>(slist.size());
  pbval_part->nprtcl_send = nsend;
  Kokkos::realloc(pbval_part->sendlist, std::max(1, nsend));
  for (int i = 0; i < nsend; ++i) { pbval_part->sendlist.h_view(i) = slist[i]; }
  pbval_part->sendlist.template modify<HostMemSpace>();
  pbval_part->sendlist.template sync<DevExeSpace>();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlePopulation::ReconcileOwnership
//! \brief ReownByPosition followed by the whole migration chain, run synchronously.
//! For use outside the task lists, after an AMR regrid or load balance. Every rank must
//! call it: the chain contains MPI collectives.

void ParticlePopulation::ReconcileOwnership(Driver *pdriver, int stage) {
  ReownByPosition(pdriver, stage);
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    SendCnt(pdriver, stage);
    InitRecv(pdriver, stage);
    SendP(pdriver, stage);
    // RecvP returns incomplete until the non-blocking receives land; there is no task
    // list to re-invoke it, so drain it here.
    while (RecvP(pdriver, stage) != TaskStatus::complete) {}
    ClearRecv(pdriver, stage);
    ClearSend(pdriver, stage);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::ReconcileOwnership
//! \brief Re-derive ownership of every population after the mesh changed. Does nothing
//! when no regrid or load balance has happened since the previous call.

void Particles::ReconcileOwnership(Driver *pdriver, int stage) {
  const int seq = pmy_pack_->pmesh->GetAMRLoadBalanceUpdateSeq();
  if (seq == owner_mesh_seq_) return;
  owner_mesh_seq_ = seq;
  for (auto *population : populations_) {
    population->ReconcileOwnership(pdriver, stage);
  }
}

} // namespace particles
