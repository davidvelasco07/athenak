//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_tasks.cpp
//! \brief functions that control Particles tasks stored in tasklists in MeshBlockPack

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "gravity/gravity.hpp"
#include "particles.hpp"
#include "particle_mesh.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::AssembleTasks
//! \brief Adds hydro tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after Hydro constructor.

void Particles::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  if (pusher == ParticlesPusher::leapfrog) {
    // Re-bin particles at the start of every cycle by recomputing PGID *absolutely* from
    // position. AthenaK's AMR regrid (end of the previous cycle) renumbers MeshBlocks but
    // does NOT remap particles, so PGID is stale afterwards; the normal per-step
    // SetNewPrtclGID is incremental (neighbour-based) and cannot recover from a regrid's
    // gid renumbering. SetGIDFromPosition re-derives the owning block from scratch, so
    // particles survive regrids. (Serial: searches this pack's blocks; MPI redistribution
    // across ranks on regrid is a TODO.)
    id.setgid = tl["before_timeintegrator"]->AddTask(&Particles::SetGIDFromPosition,
                                                     this, none);
    // Migrate any particle whose owning MeshBlock was handed to another rank by the
    // previous cycle's AMR load balancing (regrid-driven cross-rank redistribution).
    // SetGIDFromPosition above stages such particles into the sendlist; this chain ships
    // them to their new rank BEFORE the deposit below reads particle mass. No-ops on a
    // single rank / when nothing crossed. (The per-step drift migration is the separate
    // chain in "stagen" after Push.)
    id.bt_count = tl["before_timeintegrator"]->AddTask(&Particles::SendCnt, this, id.setgid);
    id.bt_irecv = tl["before_timeintegrator"]->AddTask(&Particles::InitRecv, this, id.bt_count);
    id.bt_sendp = tl["before_timeintegrator"]->AddTask(&Particles::SendP, this, id.bt_irecv);
    id.bt_recvp = tl["before_timeintegrator"]->AddTask(&Particles::RecvP, this, id.bt_sendp);
    id.bt_crecv = tl["before_timeintegrator"]->AddTask(&Particles::ClearRecv, this, id.bt_recvp);
    id.bt_csend = tl["before_timeintegrator"]->AddTask(&Particles::ClearSend, this, id.bt_crecv);

    // Self-gravitating (sink) particles are integrated WITHIN the time integrator
    // so they are stage-synchronized with the gravity solve and the gas, giving a
    // momentum-conserving RK2-KDK leapfrog (resolves the earlier non-conservative
    // before_timeintegrator placement). Per RK stage:
    //   before_stagen : Deposit (TSC scatter ρ_particle -> dmesh, read by the
    //                   gravity solve that the Driver runs right after this list)
    //   --- Driver solves Poisson for phi (ρ_gas + ρ_particle) ---
    //   stagen        : Push (gather -grad(phi) -> IPGX/Y/Z, then KDK kick/drift)
    //                   followed by the MeshBlock-crossing communication chain.
    id.deposit = tl["before_stagen"]->AddTask(&Particles::Deposit, this, none);
    id.flush   = tl["before_stagen"]->AddTask(&Particles::FlushDeposit, this, id.deposit);

    // Fill phi ghost cells (>=2 layers) from neighbour interiors before the gather, which
    // reads phi two cells deep near MeshBlock boundaries (the multigrid only keeps
    // mg_nghost valid layers). Runs in "stagen", after the Driver's per-stage solve.
    id.xphi   = tl["stagen"]->AddTask(&Particles::ExchangePhi, this, none);
    id.push   = tl["stagen"]->AddTask(&Particles::Push, this, id.xphi);
    // After the drift, re-bin by absolute position (not the incremental neighbour-based
    // SetNewPrtclGID, which mis-assigns PGID for particles crossing coarse-fine boundaries
    // under AMR). Particles that left every local block are assigned their neighbour gid
    // and staged for the cross-rank migration chain below.
    id.newgid = tl["stagen"]->AddTask(&Particles::SetGIDFromPosition, this, id.push);
    // cross-rank particle migration (same chain as tracers, driven by the sendlist
    // populated in SetGIDFromPosition); no-ops on a single rank
    id.count  = tl["stagen"]->AddTask(&Particles::SendCnt, this, id.newgid);
    id.irecv  = tl["stagen"]->AddTask(&Particles::InitRecv, this, id.count);
    id.sendp  = tl["stagen"]->AddTask(&Particles::SendP, this, id.irecv);
    id.recvp  = tl["stagen"]->AddTask(&Particles::RecvP, this, id.sendp);
    id.crecv  = tl["stagen"]->AddTask(&Particles::ClearRecv, this, id.recvp);
    id.csend  = tl["stagen"]->AddTask(&Particles::ClearSend, this, id.crecv);
    // Refresh the particle timestep every cycle from the end-of-cycle velocities
    // (dtnew was previously seeded once at t=0 by the pgens -- a latent bug once
    // particle speeds grow, e.g. under accretion). After the migration chain, so
    // arrivals contribute on their new rank.
    id.newdt  = tl["stagen"]->AddTask(&Particles::NewTimeStep, this, id.csend);

    // operator-split sink physics at the last RK stage, ordered create -> merge ->
    // accrete. Each stage is opt-in and chained onto the previous one so it runs only
    // when enabled; orbit/gravity tests with inert or no gas skip all three.
    //   create  (<particles>/creation) : new sinks are massless; the AccreteMass reset
    //           that follows seeds them conservatively (Moon & Ostriker 2025 eq. 54).
    //   merge   (<particles>/merging)  : combine sinks whose 27-cell halos overlap, so
    //           AccreteMass never sees overlapping control volumes.
    //   accrete (<particles>/accretion): control-volume gas -> sink.
    TaskID dep = none;
    if (accretion && creation) {
      id.create = tl["after_stagen"]->AddTask(&Particles::CreateSinks, this, dep);
      dep = id.create;
    }
    if (merging) {
      id.merge = tl["after_stagen"]->AddTask(&Particles::MergeSinks, this, dep);
      dep = id.merge;
    }
    if (accretion) {
      id.accrete = tl["after_stagen"]->AddTask(&Particles::AccreteMass, this, dep);
    }
  } else {
    // Tracer / cosmic-ray (drift) particles: integrated once per cycle in the
    // "before_timeintegrator" task list (no gravity coupling).
    id.push   = tl["before_timeintegrator"]->AddTask(&Particles::Push, this, none);
    id.newgid = tl["before_timeintegrator"]->AddTask(&Particles::NewGID, this, id.push);
    id.count  = tl["before_timeintegrator"]->AddTask(&Particles::SendCnt, this, id.newgid);
    id.irecv  = tl["before_timeintegrator"]->AddTask(&Particles::InitRecv, this, id.count);
    id.sendp  = tl["before_timeintegrator"]->AddTask(&Particles::SendP, this, id.irecv);
    id.recvp  = tl["before_timeintegrator"]->AddTask(&Particles::RecvP, this, id.sendp);
    id.crecv  = tl["before_timeintegrator"]->AddTask(&Particles::ClearRecv, this, id.recvp);
    id.csend  = tl["before_timeintegrator"]->AddTask(&Particles::ClearSend, this, id.crecv);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::Deposit
//! \brief Scatter particle mass onto the particle-mesh density field (dmesh) so the
//! subsequent gravity solve includes ρ_particle. No-op if this species has no
//! particle-mesh coupling (e.g. tracers).

TaskStatus Particles::Deposit(Driver *pdrive, int stage) {
  if (ppm != nullptr) {
    ppm->DepositMass(prtcl_rdata, prtcl_idata, nprtcl_thispack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::SetGIDFromPosition
//! \brief Recompute each particle's PGID absolutely, from its position, by finding the
//! MeshBlock in this pack whose bounds contain it (lower-inclusive). Unlike the
//! incremental SetNewPrtclGID, this recovers correct PGIDs after an AMR regrid has
//! renumbered MeshBlocks. A particle whose position has left every local block falls
//! back to the neighbour-based assignment from its current owner (valid because the
//! particle CFL limits the drift to <= 1 cell/step) and, when the destination is
//! off-rank, is staged into the bvals_part sendlist for the migration chain
//! (SendCnt -> SendP -> RecvP). Regrid-driven cross-rank REDISTRIBUTION (a block
//! handed to another rank by load balancing) remains a TODO.

TaskStatus Particles::SetGIDFromPosition(Driver *pdrive, int stage) {
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

  // On strictly periodic meshes, wrap particle positions back into the domain when
  // they exit (e.g. bulk-advection tests). The start-of-step position IPX0 is shifted
  // by the SAME offset so cell-crossing detection in AccreteMass stays consistent
  // across the wrap.
  const bool wrap = pmy_pack->pmesh->strictly_periodic;
  const bool has_prev = (nrdata > IPX0);   // IPX0.. exist only in the sink layout
  auto &msz = pmy_pack->pmesh->mesh_size;
  const Real xmin = msz.x1min, Lx = msz.x1max - msz.x1min;
  const Real ymin = msz.x2min, Ly = msz.x2max - msz.x2min;
  const Real zmin = msz.x3min, Lz = msz.x3max - msz.x3min;

  // cross-rank migration support: particles whose position has left every local block
  // fall back to the neighbour-based assignment (the drift moves <= 1 cell per step by
  // the particle CFL, so the destination is always among the current owner's
  // neighbours) and, when the new owner is off-rank, are staged into the bvals_part
  // sendlist for the SendCnt/SendP/RecvP chain. Device counter + host mirror (a plain
  // host int atomically incremented from a device kernel reads back stale on GPU).
  const int my_rank = global_variable::my_rank;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  // Device pass: re-bin each particle ABSOLUTELY from its (wrapped) position by finding
  // the local MeshBlock that contains it. Particles found locally get their PGID set here
  // (the common case; no communication). A particle contained in NO local block is a
  // "stray": its owner is necessarily on ANOTHER rank -- either it drifted across a rank
  // boundary, or (the regrid case) load balancing handed its block to another rank. Strays
  // are collected (index + wrapped position) for host-side global resolution below, which
  // needs the bounds of off-rank blocks -- those live only in the host tree
  // (lloc_eachmb/rank_eachmb). SetGIDFromPosition is used only by the (few) sink particles,
  // so the host pass is cheap and runs only when strays exist.
  DualArray1D<int>  stray_idx("setgid_stray_idx", npart);
  DualArray1D<Real> stray_pos("setgid_stray_pos", 3*npart);
  Kokkos::View<int> d_nstray("setgid_nstray");
  Kokkos::deep_copy(d_nstray, 0);

  par_for("part_setgid", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    // wrap positions back into the domain on strictly-periodic meshes (IPX0 shifted by the
    // same offset so AccreteMass cell-crossing detection stays consistent across the wrap)
    if (wrap) {
      Real s;
      s = Kokkos::floor((pr(IPX, p) - xmin)/Lx)*Lx;
      pr(IPX, p) -= s;  if (has_prev) pr(IPX0, p) -= s;
      s = Kokkos::floor((pr(IPY, p) - ymin)/Ly)*Ly;
      pr(IPY, p) -= s;  if (has_prev) pr(IPY0, p) -= s;
      s = Kokkos::floor((pr(IPZ, p) - zmin)/Lz)*Lz;
      pr(IPZ, p) -= s;  if (has_prev) pr(IPZ0, p) -= s;
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
      // absolute on-rank re-bin (AMR-regrid safe: derived from position alone)
      pi(PGID, p) = gids + mown;
    } else {
      // owner is off-rank (no local block contains it): collect for host global resolution
      const int s = Kokkos::atomic_fetch_add(&d_nstray(), 1);
      stray_idx.d_view(s)     = p;
      stray_pos.d_view(3*s)   = px;
      stray_pos.d_view(3*s+1) = py;
      stray_pos.d_view(3*s+2) = pz;
    }
  });

  // Host-side resolution of strays. Every rank knows the full tree (lloc_eachmb +
  // rank_eachmb), so the owning block's gid+rank is found by geometric containment; this
  // handles cross-rank DRIFT and regrid-driven cross-rank REDISTRIBUTION uniformly. The
  // sender must carry the destination (global) gid in PGID before packing, so PGID is
  // written back for every resolved stray; off-rank strays are staged for the send chain.
  int nstray = 0;
  {
    auto h_ns = Kokkos::create_mirror_view(d_nstray);
    Kokkos::deep_copy(h_ns, d_nstray);
    nstray = h_ns();
  }
  pbval_part->nprtcl_send = 0;
  Kokkos::realloc(pbval_part->sendlist, 0);
  if (nstray > 0) {
    stray_idx.template modify<DevExeSpace>();  stray_idx.template sync<HostMemSpace>();
    stray_pos.template modify<DevExeSpace>();  stray_pos.template sync<HostMemSpace>();
    Mesh *pm = pmy_pack->pmesh;
    auto &ms = pm->mesh_size;
    const int rlev = pm->root_level;
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
      if (dgid < 0) continue;   // left the domain (e.g. outflow BC); keep PGID (guarded)
      wp_idx.push_back(p);  wp_gid.push_back(dgid);
      const int drank = pm->rank_eachmb[dgid];
      if (drank != my_rank) {
        ParticleLocationData d;
        d.prtcl_indx = p;  d.dest_gid = dgid;  d.dest_rank = drank;
        slist.push_back(d);
      }
    }
    // write resolved global PGID back to the device (packed by the sender; also correct
    // for the receiver, which decodes local index as PGID - gids on its own rank)
    const int nwp = static_cast<int>(wp_idx.size());
    if (nwp > 0) {
      DualArray1D<int> wb("setgid_writeback", 2*nwp);
      for (int i = 0; i < nwp; ++i) {
        wb.h_view(2*i) = wp_idx[i];  wb.h_view(2*i+1) = wp_gid[i];
      }
      wb.template modify<HostMemSpace>();  wb.template sync<DevExeSpace>();
      auto wbd = wb.d_view;  auto pidata = pi;
      par_for("setgid_wb", DevExeSpace(), 0, nwp-1, KOKKOS_LAMBDA(const int i) {
        pidata(PGID, wbd(2*i)) = wbd(2*i+1);
      });
    }
    // publish the off-rank migration list for the SendCnt/SendP/RecvP chain
    const int nsend = static_cast<int>(slist.size());
    // optional diagnostic (set env PART_XRANK_DBG): confirms the cross-rank redistribution
    // path is exercised; cross-rank particle migration is a normal event, not an error.
    if (nsend > 0 && std::getenv("PART_XRANK_DBG") != nullptr) {
      long gtot = 0;
      for (int r = 0; r < global_variable::nranks; ++r) {
        gtot += pmy_pack->pmesh->nprtcl_eachrank[r];
      }
      std::cout << "### [part-xrank] rank " << my_rank << " ships " << nsend
                << " particle(s) to other ranks (cycle=" << pmy_pack->pmesh->ncycle
                << ", t=" << pmy_pack->pmesh->time << ", global nprtcl=" << gtot << ")"
                << std::endl;
    }
    pbval_part->nprtcl_send = nsend;
    Kokkos::realloc(pbval_part->sendlist, std::max(1, nsend));
    for (int i = 0; i < nsend; ++i) { pbval_part->sendlist.h_view(i) = slist[i]; }
    pbval_part->sendlist.template modify<HostMemSpace>();
    pbval_part->sendlist.template sync<DevExeSpace>();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::ReconcileOwnership
//! \brief Re-bin particles from position AND complete any resulting cross-rank migration,
//! synchronously. This is the task-list chain (setgid -> SendCnt -> InitRecv -> SendP ->
//! RecvP -> ClearRecv/ClearSend) collapsed into one blocking call, for use OUTSIDE the
//! task lists -- specifically Driver::Finalize, which writes the end-of-run output after
//! the last AMR regrid has renumbered (and possibly re-ranked) MeshBlocks. Without the
//! migration half, SetGIDFromPosition would leave particles whose block moved to another
//! rank holding an off-rank PGID, and the final particle-mesh output would silently drop
//! them (the same defect the per-cycle chain exists to prevent).
//! All ranks must call this: the migration chain contains MPI collectives.

void Particles::ReconcileOwnership(Driver *pdrive, int stage) {
  SetGIDFromPosition(pdrive, stage);
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    SendCnt(pdrive, stage);
    InitRecv(pdrive, stage);
    SendP(pdrive, stage);
    // RecvP returns incomplete until the non-blocking receives land; drain it here since
    // there is no task list to re-invoke it.
    while (RecvP(pdrive, stage) != TaskStatus::complete) {}
    ClearRecv(pdrive, stage);
    ClearSend(pdrive, stage);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::FlushDeposit
//! \brief Fold ghost-zone deposit contributions into neighbour MeshBlock interiors so the
//! particle-mesh density is complete across MeshBlock boundaries (Phase 1c). No-op without
//! particle-mesh coupling.

TaskStatus Particles::FlushDeposit(Driver *pdrive, int stage) {
  if (ppm != nullptr) {
    ppm->FlushDepositBoundaries();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::ExchangePhi
//! \brief Fill the gravitational potential's ghost cells from neighbouring MeshBlock
//! interiors so GatherGravity can read phi two cells deep near MeshBlock boundaries.
//! Reuses the particle-mesh boundary-values object (1 variable) for a standard cell-
//! centered halo exchange. Synchronous: correct for serial / on-rank; the MPI path
//! completes here by draining the receives (a later pass can split this across tasks
//! to overlap communication).

TaskStatus Particles::ExchangePhi(Driver *pdrive, int stage) {
  if (ppm == nullptr || pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  auto pb = ppm->pmbval;
  auto &phi  = pmy_pack->pgrav->phi;
  auto &cphi = pmy_pack->pgrav->coarse_phi;

  // Full AMR-correct CC halo exchange for phi, mirroring the validated hydro sequence
  // (RestrictU -> SendU -> RecvU -> ApplyPhysicalBCs -> Prolongate). Skipping any of
  // these steps leaves some class of ghost cells unset on a refined mesh:
  //  - without the restriction, coarse neighbours receive garbage;
  //  - without the physical-BC fill, phi ghosts at *physical* (non-periodic) domain
  //    boundaries are never written; FillCoarseInBndryCC/ProlongateCC then consume
  //    them at blocks adjacent to both a physical boundary and a coarse-fine
  //    boundary, spraying uninitialized values (e.g. 1e252 on GPU, where fresh
  //    allocations are not benign zeros) into fine-block ghosts that GatherGravity
  //    reads as enormous spurious forces.
  pb->InitRecv(1);
  // Restrict this block's fine interior into the coarse buffer, so the pack sends the
  // correct restricted data to coarser neighbours and the prolongation has a valid
  // coarse source.
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(phi, cphi);
  }
  pb->PackAndSendCC(phi, cphi);
  while (pb->RecvAndUnpackCC(phi, cphi) != TaskStatus::complete) {}
  // Physical (non-periodic) boundary fill for phi ghosts, as hydro does before its
  // prolongation. HydroBCs is variable-agnostic (outflow = zero-gradient copy), which
  // both initializes the memory and is an adequate extrapolation for the gather.
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    MeshBoundaryValues::HydroBCs(pmy_pack, pb->u_in, phi);
  }
  // Prolongate into fine-block ghost cells at coarse/fine boundaries (before clearing
  // buffers).
  if (pmy_pack->pmesh->multilevel) {
    pb->FillCoarseInBndryCC(phi, cphi);
    pb->ProlongateCC(phi, cphi);
  }
  pb->ClearSend();
  pb->ClearRecv();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::NewGID
//! \brief Wrapper task list function to set new GID for particles that move between
//! MeshBlocks.

TaskStatus Particles::NewGID(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->SetNewPrtclGID();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::SendCnt
//! \brief Wrapper task list function to set share number of particles communicated with
//! MPI between all ranks

TaskStatus Particles::SendCnt(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->CountSendsAndRecvs();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI).

TaskStatus Particles::InitRecv(Driver *pdrive, int stage) {
  // post receives for particles
  TaskStatus tstat = pbval_part->InitPrtclRecv();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::SendP()
//! \brief Wrapper task list function to pack/send particles

TaskStatus Particles::SendP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->PackAndSendPrtcls();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::RecvP
//! \brief Wrapper task list function to receive/unpack particles

TaskStatus Particles::RecvP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->RecvAndUnpackPrtcls();
  return tstat;
}


//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.

TaskStatus Particles::ClearSend(Driver *pdrive, int stage) {
  // check sends of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclSend();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed.

TaskStatus Particles::ClearRecv(Driver *pdrive, int stage) {
  // check receives of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclRecv();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::NewTimeStep
//! \brief Compute the particle timestep from end-of-cycle velocities: dtnew =
//! cfl_par * min over particles of 1/sqrt((v1/dx1)^2+(v2/dx2)^2+(v3/dx3)^2), using each
//! particle's own MeshBlock cell sizes (handles AMR levels automatically). Note
//! Mesh::NewTimeStep applies NO cfl factor to ppart->dtnew, so the safety factor lives
//! here; cfl_par=0.5 (default) guarantees at most one cell crossed per step, as the
//! control-volume accretion's cell-crossing correction assumes.

TaskStatus Particles::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;  // once per cycle
  if (nprtcl_thispack <= 0) return TaskStatus::complete;

  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int gids = pmy_pack->gids;
  const int nmb_ = pmy_pack->nmb_thispack;

  Real max_vdx = 0.0;   // max over particles of |v|/dx (per-axis)
  Kokkos::parallel_reduce("part_newdt", Kokkos::RangePolicy<>(DevExeSpace(),
                          0, nprtcl_thispack),
  KOKKOS_LAMBDA(const int p, Real &pmax) {
    const int m = pi(PGID, p) - gids;
    if (m < 0 || m >= nmb_) return;   // corrupt/unbinned particle: no dt constraint
    const Real v1 = pr(IPVX, p)/mbsize.d_view(m).dx1;
    const Real v2 = pr(IPVY, p)/mbsize.d_view(m).dx2;
    const Real v3 = pr(IPVZ, p)/mbsize.d_view(m).dx3;
    const Real vdx = sqrt(v1*v1 + v2*v2 + v3*v3);
    pmax = fmax(pmax, vdx);
  }, Kokkos::Max<Real>(max_vdx));

  if (max_vdx > 0.0) {
    dtnew = cfl_par/max_vdx;
  }
  // else: all particles at rest -- keep the previous (or pgen-seeded) dtnew

  return TaskStatus::complete;
}

} // namespace particles
