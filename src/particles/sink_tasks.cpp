//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_tasks.cpp
//! \brief sink-particle task schedule, pushers, potential/deposit exchanges and timestep

#include <limits>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "gravity/gravity.hpp"
#include "particles.hpp"
#include "sink.hpp"
#include "sink_particles.hpp"
#include "sink_names.hpp"
#include "particle_mesh.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void SinkParticles::AssembleTasks
//! \brief Adds the sink tasks to the task lists used by the time integrators. Called by
//! ParticlePopulation::AssembleTasks for the leap_frog and rk3 pushers.

void SinkParticles::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  ParticlePopulation *pp = ppop;

  // Re-bin particles at the start of every cycle from position and ship any whose
  // owning MeshBlock now lives on another rank, BEFORE the deposit below reads particle
  // mass. The driver already re-derives ownership right after a regrid, so this chain
  // normally finds nothing to do; it also covers pgens that seed sinks with approximate
  // owners. No-ops on a single rank / when nothing crossed.
  id.setgid = tl["before_timeintegrator"]->AddTask(&ParticlePopulation::ReownByPosition,
                                                   pp, none);
  id.bt_count = tl["before_timeintegrator"]->AddTask(&ParticlePopulation::SendCnt, pp,
                                                     id.setgid);
  id.bt_irecv = tl["before_timeintegrator"]->AddTask(&ParticlePopulation::InitRecv, pp,
                                                     id.bt_count);
  id.bt_sendp = tl["before_timeintegrator"]->AddTask(&ParticlePopulation::SendP, pp,
                                                     id.bt_irecv);
  id.bt_recvp = tl["before_timeintegrator"]->AddTask(&SinkParticles::RecvP, this,
                                                     id.bt_sendp);
  id.bt_crecv = tl["before_timeintegrator"]->AddTask(&ParticlePopulation::ClearRecv, pp,
                                                     id.bt_recvp);
  id.bt_csend = tl["before_timeintegrator"]->AddTask(&ParticlePopulation::ClearSend, pp,
                                                     id.bt_crecv);

  // Self-gravitating (sink) particles are integrated WITHIN the time integrator
  // so they are stage-synchronized with the gravity solve and the gas, giving a
  // momentum-conserving RK2-KDK leapfrog. Per RK stage:
  //   before_stagen : Deposit (TSC scatter rho_particle -> dmesh, read by the
  //                   gravity solve that the Driver runs right after this list)
  //   --- Driver solves Poisson for phi (rho_gas + rho_particle) ---
  //   stagen        : Push (gather -grad(phi) -> IPGX/Y/Z, then KDK kick/drift)
  //                   followed by the MeshBlock-crossing communication chain.
  // The deposit's boundary reconciliation is a PAIR, and both halves live here:
  //   flush     -- reverse: ADD each block's ghost spill into the neighbour interiors
  //                that own it (a deposit writes into territory it does not own), then
  //                clear the ghosts;
  //   dm_rest.. -- forward: COPY the now-complete neighbour interiors back out into the
  //                ghosts, exactly as the gas's exchange does.
  // Receives are posted first (dm_irecv), the exchange runs after the flush, and the
  // buffers are released in "after_stagen".
  id.dm_irecv = tl["before_stagen"]->AddTask(&SinkParticles::DMeshInitRecv, this, none);
  id.deposit = tl["before_stagen"]->AddTask(&SinkParticles::Deposit, this, id.dm_irecv);
  id.flush   = tl["before_stagen"]->AddTask(&SinkParticles::FlushDeposit, this,
                                            id.deposit);
  id.dm_rest = tl["before_stagen"]->AddTask(&SinkParticles::DMeshRestrict, this,
                                            id.flush);
  id.dm_send = tl["before_stagen"]->AddTask(&SinkParticles::DMeshSend, this, id.dm_rest);
  id.dm_recv = tl["before_stagen"]->AddTask(&SinkParticles::DMeshRecv, this, id.dm_send);
  id.dm_prol = tl["before_stagen"]->AddTask(&SinkParticles::DMeshProlongate, this,
                                            id.dm_recv);

  // Fill phi ghost cells (>=2 layers) from neighbour interiors before the gather, which
  // reads phi two cells deep near MeshBlock boundaries (the multigrid only keeps
  // mg_nghost valid layers). Structured exactly like every other field exchange: post
  // the receives in "before_stagen", do restrict/send/recv/BC/prolongate in "stagen"
  // after the Driver's per-stage solve and before the gather, and clear the buffers in
  // "after_stagen". Recv returns incomplete and is retried by the task list instead of
  // spinning inside one call.
  id.xphi_irecv = tl["before_stagen"]->AddTask(&SinkParticles::XPhiInitRecv, this, none);
  id.xphi_rest  = tl["stagen"]->AddTask(&SinkParticles::XPhiRestrict, this, none);
  id.xphi_send  = tl["stagen"]->AddTask(&SinkParticles::XPhiSend, this, id.xphi_rest);
  id.xphi_recv  = tl["stagen"]->AddTask(&SinkParticles::XPhiRecv, this, id.xphi_send);
  id.xphi_bcs   = tl["stagen"]->AddTask(&SinkParticles::XPhiBCs, this, id.xphi_recv);
  id.xphi_prol  = tl["stagen"]->AddTask(&SinkParticles::XPhiProlongate, this,
                                        id.xphi_bcs);
  id.push   = tl["stagen"]->AddTask(&SinkParticles::Push, this, id.xphi_prol);
  // After the drift, re-bin by absolute position, which stays correct for particles
  // crossing coarse-fine boundaries under AMR. Particles that left every local block
  // are staged for the cross-rank migration chain below.
  id.newgid = tl["stagen"]->AddTask(&ParticlePopulation::ReownByPosition, pp, id.push);
  id.count  = tl["stagen"]->AddTask(&ParticlePopulation::SendCnt, pp, id.newgid);
  id.irecv  = tl["stagen"]->AddTask(&ParticlePopulation::InitRecv, pp, id.count);
  id.sendp  = tl["stagen"]->AddTask(&ParticlePopulation::SendP, pp, id.irecv);
  id.recvp  = tl["stagen"]->AddTask(&SinkParticles::RecvP, this, id.sendp);
  id.crecv  = tl["stagen"]->AddTask(&ParticlePopulation::ClearRecv, pp, id.recvp);
  id.csend  = tl["stagen"]->AddTask(&ParticlePopulation::ClearSend, pp, id.crecv);
  // Refresh the particle timestep every cycle from the end-of-cycle velocities. After
  // the migration chain, so arrivals contribute on their new rank.
  id.newdt  = tl["stagen"]->AddTask(&SinkParticles::NewTimeStep, this, id.csend);

  // operator-split sink physics at the last RK stage, ordered create -> merge ->
  // accrete. Each stage is opt-in and chained onto the previous one so it runs only
  // when enabled; orbit/gravity tests with inert or no gas skip all three.
  //   create  (creation) : new sinks are massless; the AccreteMass reset that follows
  //           seeds them conservatively (Moon & Ostriker 2025 eq. 54).
  //   merge   (merging)  : combine sinks whose 27-cell halos overlap, so AccreteMass
  //           never sees overlapping control volumes.
  //   accrete (accretion): control-volume gas -> sink.
  // release the potential exchange's buffers once the stage is done with them
  id.xphi_csend = tl["after_stagen"]->AddTask(&SinkParticles::XPhiClearSend, this, none);
  id.xphi_crecv = tl["after_stagen"]->AddTask(&SinkParticles::XPhiClearRecv, this,
                                              id.xphi_csend);
  // release the deposit's forward-exchange buffers too
  id.dm_csend = tl["after_stagen"]->AddTask(&SinkParticles::DMeshClearSend, this, none);
  id.dm_crecv = tl["after_stagen"]->AddTask(&SinkParticles::DMeshClearRecv, this,
                                            id.dm_csend);

  TaskID dep = none;
  if (accretion && creation) {
    id.create = tl["after_stagen"]->AddTask(&SinkParticles::CreateSinks, this, dep);
    dep = id.create;
  }
  if (merging) {
    id.merge = tl["after_stagen"]->AddTask(&SinkParticles::MergeSinks, this, dep);
    dep = id.merge;
    if (merge_iterative) {
      dep = tl["after_stagen"]->AddTask(&ParticlePopulation::ReownByPosition, pp, dep);
      dep = tl["after_stagen"]->AddTask(&ParticlePopulation::SendCnt, pp, dep);
      dep = tl["after_stagen"]->AddTask(&ParticlePopulation::InitRecv, pp, dep);
      dep = tl["after_stagen"]->AddTask(&ParticlePopulation::SendP, pp, dep);
      dep = tl["after_stagen"]->AddTask(&SinkParticles::RecvP, this, dep);
      dep = tl["after_stagen"]->AddTask(&ParticlePopulation::ClearRecv, pp, dep);
      dep = tl["after_stagen"]->AddTask(&ParticlePopulation::ClearSend, pp, dep);
    }
  }
  if (accretion) {
    id.accrete = tl["after_stagen"]->AddTask(&SinkParticles::AccreteMass, this, dep);
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SinkParticles::RecvP
//! \brief The population's migration receive, then the sink sanity check.

TaskStatus SinkParticles::RecvP(Driver *pdriver, int stage) {
  TaskStatus tstat = ppop->RecvP(pdriver, stage);
  if (tstat == TaskStatus::complete) CheckSinkSanity("migration receive", stage);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SinkParticles::Deposit
//! \brief Scatter particle mass onto the particle-mesh density field (dmesh) so the
//! subsequent gravity solve includes rho_particle.

TaskStatus SinkParticles::Deposit(Driver *pdrive, int stage) {
  ppm->DepositMass(ppop->prtcl_rdata, ppop->prtcl_idata, ppop->nprtcl_thispack);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SinkParticles::FlushDeposit
//! \brief Fold ghost-zone deposit contributions into neighbour MeshBlock interiors so the
//! particle-mesh density is complete across MeshBlock boundaries.

TaskStatus SinkParticles::FlushDeposit(Driver *pdrive, int stage) {
  ppm->FlushDepositBoundaries();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SinkParticles::DMesh*
//! \brief Forward ("inbox") half of the deposit's boundary exchange: copy the
//! post-flush neighbour interiors back out into this block's ghost cells, so dmesh holds
//! the same invariant as u0 -- every cell, interior or ghost, is the total particle
//! density at that location. Thin wrappers; the work and the rationale are on
//! ParticleMesh (see particle_mesh.hpp).

TaskStatus SinkParticles::DMeshInitRecv(Driver *pdrive, int stage) {
  return ppm->RefreshGhostsInitRecv();
}

TaskStatus SinkParticles::DMeshRestrict(Driver *pdrive, int stage) {
  return ppm->RefreshGhostsRestrict();
}

TaskStatus SinkParticles::DMeshSend(Driver *pdrive, int stage) {
  return ppm->RefreshGhostsSend();
}

TaskStatus SinkParticles::DMeshRecv(Driver *pdrive, int stage) {
  return ppm->RefreshGhostsRecv();
}

TaskStatus SinkParticles::DMeshProlongate(Driver *pdrive, int stage) {
  return ppm->RefreshGhostsProlongate();
}

TaskStatus SinkParticles::DMeshClearSend(Driver *pdrive, int stage) {
  return ppm->RefreshGhostsClearSend();
}

TaskStatus SinkParticles::DMeshClearRecv(Driver *pdrive, int stage) {
  return ppm->RefreshGhostsClearRecv();
}

//----------------------------------------------------------------------------------------
//! The gravitational potential's halo exchange, as the standard task sequence
//! (InitRecv -> Restrict -> Send -> Recv -> ApplyPhysicalBCs -> Prolongate -> Clear).
//! GatherGravity reads phi two cells deep near MeshBlock boundaries while the multigrid
//! leaves only mg_nghost layers valid, so phi needs this swap after every solve.
//!
//! Every step matters on a refined mesh: without the restriction coarse neighbours
//! receive unrestricted data; without the physical-BC fill, phi ghosts at non-periodic
//! domain boundaries are never written and the coarse fill/prolongation then consumes
//! uninitialized memory at blocks adjacent to both a physical and a coarse-fine boundary
//! (benign-looking zeros on CPU, ~1e252 on GPU, read by the gather as an enormous force).
//!
//! Each step is its own task, on this object's own buffers, for two reasons: Recv can
//! report incomplete and be retried by the task list instead of spinning inside one call
//! and stalling every other task on the rank mid-stage, and the exchange does not share
//! buffers/requests with the particle-mesh deposit.

TaskStatus SinkParticles::XPhiInitRecv(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  return pbval_phi->InitRecv(1);
}

TaskStatus SinkParticles::XPhiRestrict(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  pmy_pack->pgrav->SaveFaceBoundaries();
  // restrict the fine interior into the coarse buffer, so coarser neighbours are sent
  // restricted data and the prolongation has a valid coarse source
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(pmy_pack->pgrav->phi, pmy_pack->pgrav->coarse_phi);
  }
  return TaskStatus::complete;
}

TaskStatus SinkParticles::XPhiSend(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  return pbval_phi->PackAndSendCC(pmy_pack->pgrav->phi, pmy_pack->pgrav->coarse_phi);
}

TaskStatus SinkParticles::XPhiRecv(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  return pbval_phi->RecvAndUnpackCC(pmy_pack->pgrav->phi, pmy_pack->pgrav->coarse_phi);
}

TaskStatus SinkParticles::XPhiBCs(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;
  // HydroBCs is variable-agnostic (outflow = zero-gradient copy), which both initializes
  // the memory and is an adequate extrapolation for the gather
  pbval_phi->HydroBCs(pmy_pack, pbval_phi->u_in, pmy_pack->pgrav->phi);
  return TaskStatus::complete;
}

TaskStatus SinkParticles::XPhiProlongate(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel) {
    pbval_phi->FillCoarseInBndryCC(pmy_pack->pgrav->phi, pmy_pack->pgrav->coarse_phi);
    pbval_phi->ProlongateCC(pmy_pack->pgrav->phi, pmy_pack->pgrav->coarse_phi);
  }
  pmy_pack->pgrav->RestoreFaceBoundaries();
  return TaskStatus::complete;
}

TaskStatus SinkParticles::XPhiClearSend(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  return pbval_phi->ClearSend();
}

TaskStatus SinkParticles::XPhiClearRecv(Driver *pdrive, int stage) {
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  return pbval_phi->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus SinkParticles::Push
//! \brief Gather the gravitational acceleration and advance the sinks by one RK stage.

TaskStatus SinkParticles::Push(Driver *pdriver, int stage) {
  auto &pr = ppop->prtcl_rdata;
  const int npart = ppop->nprtcl_thispack;
  auto dt_ = (pmy_pack->pmesh->dt);
  auto hdt_ = 0.5*dt_;

  switch (ppop->pusher) {
    case ParticlesPusher::rk3: {
      // Evolve particle position and velocity with the same SSPRK(3,3) stages
      // as the gas. Matched gravity weights preserve equal/opposite impulses.
      if (pmy_pack->pgrav != nullptr) {
        ppm->GatherGravity(pmy_pack->pgrav->phi, ppop->prtcl_rdata, ppop->prtcl_idata,
                           npart);
      }
      const Real g0 = pdriver->gam0[stage-1];
      const Real g1 = pdriver->gam1[stage-1];
      const Real bdt = pdriver->beta[stage-1]*dt_;
      const bool first_stage = (stage == 1);
      par_for("part_rk3",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
        if (first_stage) {
          pr(IPX0,p) = pr(IPX,p); pr(IPY0,p) = pr(IPY,p); pr(IPZ0,p) = pr(IPZ,p);
          pr(IPVX0,p) = pr(IPVX,p); pr(IPVY0,p) = pr(IPVY,p);
          pr(IPVZ0,p) = pr(IPVZ,p);
        }
        // Update positions before velocities: the derivative is the input-stage
        // velocity. Migration carries all registers; periodic wrapping shifts
        // IPX0/IPY0/IPZ0 alongside the current position before the next stage.
        pr(IPX,p) = g0*pr(IPX,p) + g1*pr(IPX0,p) + bdt*pr(IPVX,p);
        pr(IPY,p) = g0*pr(IPY,p) + g1*pr(IPY0,p) + bdt*pr(IPVY,p);
        pr(IPZ,p) = g0*pr(IPZ,p) + g1*pr(IPZ0,p) + bdt*pr(IPVZ,p);
        pr(IPVX,p) = g0*pr(IPVX,p) + g1*pr(IPVX0,p) + bdt*pr(IPGX,p);
        pr(IPVY,p) = g0*pr(IPVY,p) + g1*pr(IPVY0,p) + bdt*pr(IPGY,p);
        pr(IPVZ,p) = g0*pr(IPVZ,p) + g1*pr(IPVZ0,p) + bdt*pr(IPGZ,p);
      });
      break;
    }
    case ParticlesPusher::leap_frog: {
      // Gather the gravitational acceleration -grad(phi) from the multigrid
      // potential onto each particle (writes IPGX/IPGY/IPGZ). phi already
      // includes the particle's own deposited density (folded into the Poisson
      // RHS), so on a uniform grid the symmetric TSC self-force cancels.
      if (pmy_pack->pgrav != nullptr) {
        ppm->GatherGravity(pmy_pack->pgrav->phi, ppop->prtcl_rdata, ppop->prtcl_idata,
                           npart);
      }
      // RK2-KDK leapfrog, stage-synchronized with the per-stage gravity solve:
      //   stage 1: half-kick v^n -> v^(n+1/2) using a(x^n); drift x^n -> x^(n+1)
      //   stage 2: half-kick v^(n+1/2) -> v^(n+1) using a(x^(n+1))
      const bool do_drift = (stage == 1);
      // sanity diagnostic (sanity_vmax > 0): record the kick that produced a bad velocity
      const Real svmax2 = (sanity_vmax > 0.0) ? sanity_vmax*sanity_vmax : -1.0;
      auto sflag = sanity_flag_;
      auto srec = sanity_rec_;
      auto &pid = ppop->prtcl_idata;
      par_for("part_kdk",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
        const Real vx0 = pr(IPVX,p), vy0 = pr(IPVY,p), vz0 = pr(IPVZ,p);
        // Half-kick using the gathered acceleration.
        pr(IPVX,p) += hdt_*pr(IPGX,p);
        pr(IPVY,p) += hdt_*pr(IPGY,p);
        pr(IPVZ,p) += hdt_*pr(IPGZ,p);
        if (svmax2 > 0.0) {
          const Real v2 = pr(IPVX,p)*pr(IPVX,p) + pr(IPVY,p)*pr(IPVY,p)
                        + pr(IPVZ,p)*pr(IPVZ,p);
          if (!(v2 <= svmax2) && Kokkos::atomic_compare_exchange(&sflag(0), 0, 1) == 0) {
            srec(0) = static_cast<Real>(pid(PTAG,p));
            srec(1) = vx0;  srec(2) = vy0;  srec(3) = vz0;
            srec(4) = pr(IPGX,p);  srec(5) = pr(IPGY,p);  srec(6) = pr(IPGZ,p);
            srec(7) = hdt_;
            srec(8) = pr(IPVX,p);  srec(9) = pr(IPVY,p);  srec(10) = pr(IPVZ,p);
            srec(11) = pr(IPX,p);  srec(12) = pr(IPY,p);  srec(13) = pr(IPZ,p);
            srec(14) = pr(IPM,p);
          }
        }
        // Full drift (stage 1 only). Save the pre-drift (start-of-step) position
        // first: AccreteMass uses it to detect sink-cell crossings. Unconditional
        // (once per cycle), so IPX0 is always current when accretion runs.
        if (do_drift) {
          pr(IPX0,p) = pr(IPX,p);
          pr(IPY0,p) = pr(IPY,p);
          pr(IPZ0,p) = pr(IPZ,p);
          pr(IPX,p) += dt_*pr(IPVX,p);
          pr(IPY,p) += dt_*pr(IPVY,p);
          pr(IPZ,p) += dt_*pr(IPVZ,p);
        }
      });
      break;
    }
    default:
      break;
  }

  CheckSinkSanity("push", stage);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SinkParticles::NewTimeStep
//! \brief Compute the particle timestep from end-of-cycle velocities: dtnew =
//! cfl_par * min over particles of 1/sqrt((v1/dx1)^2+(v2/dx2)^2+(v3/dx3)^2), using each
//! particle's own MeshBlock cell sizes (handles AMR levels automatically). Note
//! Mesh::NewTimeStep applies NO cfl factor to the particle dt, so the safety factor lives
//! here; cfl_par=0.5 (default) guarantees at most one cell crossed per step, as the
//! control-volume accretion's cell-crossing correction assumes.
//! Every rank resets dtnew every cycle. A rank with no particles, or with all of them at
//! rest, imposes no constraint. It used to keep its previous value (the pgen seed, or the
//! value from when it last held a particle), and since Mesh::NewTimeStep takes the MPI
//! minimum, the time step then depended on which ranks had held particles: the same run
//! took different steps on different rank counts.

TaskStatus SinkParticles::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;  // once per cycle
  CheckSinkSanity("timestep", stage);
  const int npart = ppop->nprtcl_thispack;
  ppop->dtnew = std::numeric_limits<float>::max();
  if (npart <= 0) return TaskStatus::complete;

  auto &pr = ppop->prtcl_rdata;
  auto &pi = ppop->prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int gids = pmy_pack->gids;
  const int nmb_ = pmy_pack->nmb_thispack;

  Real max_vdx = 0.0;   // max over particles of |v|/dx (per-axis)
  Kokkos::parallel_reduce("part_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
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
    ppop->dtnew = cfl_par/max_vdx;
  }
  return TaskStatus::complete;
}

} // namespace particles
