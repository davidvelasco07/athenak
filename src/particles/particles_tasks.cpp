//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_tasks.cpp
//! \brief functions that control Particles tasks stored in tasklists in MeshBlockPack

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
    // under AMR). Serial / on-rank only -- cross-rank particle communication is a TODO.
    id.newgid = tl["stagen"]->AddTask(&Particles::SetGIDFromPosition, this, id.push);
    // Refresh the particle timestep every cycle from the end-of-cycle velocities
    // (dtnew was previously seeded once at t=0 by the pgens -- a latent bug once
    // particle speeds grow, e.g. under accretion).
    id.newdt  = tl["stagen"]->AddTask(&Particles::NewTimeStep, this, id.newgid);

    // operator-split accretion: gas in the control volume -> sink, at the last RK stage.
    // Registered only when <particles>/accretion=true -- orbit/gravity tests with inert
    // or no gas skip the kernel entirely.
    if (accretion) {
      id.accrete = tl["after_stagen"]->AddTask(&Particles::AccreteMass, this, none);
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
//! renumbered MeshBlocks. Serial / on-rank: a particle whose owning block is on another
//! rank is left unchanged (cross-rank redistribution on regrid is a TODO).

TaskStatus Particles::SetGIDFromPosition(Driver *pdrive, int stage) {
  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  int gids = pmy_pack->gids;
  int nmb = pmy_pack->nmb_thispack;
  int npart = nprtcl_thispack;
  if (npart <= 0) return TaskStatus::complete;

  par_for("part_setgid", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    Real px = pr(IPX, p), py = pr(IPY, p), pz = pr(IPZ, p);
    int mcur = pi(PGID, p) - gids;
    int mown = (mcur >= 0 && mcur < nmb) ? mcur : 0;   // keep current/valid as fallback
    for (int mm = 0; mm < nmb; ++mm) {
      if (px >= mbsize.d_view(mm).x1min && px < mbsize.d_view(mm).x1max &&
          py >= mbsize.d_view(mm).x2min && py < mbsize.d_view(mm).x2max &&
          pz >= mbsize.d_view(mm).x3min && pz < mbsize.d_view(mm).x3max) {
        mown = mm;
      }
    }
    pi(PGID, p) = gids + mown;
  });
  return TaskStatus::complete;
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
