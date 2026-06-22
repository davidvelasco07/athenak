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

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
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

    id.push   = tl["stagen"]->AddTask(&Particles::Push, this, none);
    id.newgid = tl["stagen"]->AddTask(&Particles::NewGID, this, id.push);
    id.count  = tl["stagen"]->AddTask(&Particles::SendCnt, this, id.newgid);
    id.irecv  = tl["stagen"]->AddTask(&Particles::InitRecv, this, id.count);
    id.sendp  = tl["stagen"]->AddTask(&Particles::SendP, this, id.irecv);
    id.recvp  = tl["stagen"]->AddTask(&Particles::RecvP, this, id.sendp);
    id.crecv  = tl["stagen"]->AddTask(&Particles::ClearRecv, this, id.recvp);
    id.csend  = tl["stagen"]->AddTask(&Particles::ClearSend, this, id.crecv);
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

} // namespace particles
