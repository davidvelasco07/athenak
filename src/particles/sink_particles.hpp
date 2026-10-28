#ifndef PARTICLES_SINK_PARTICLES_HPP_
#define PARTICLES_SINK_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles.hpp
//! \brief self-gravitating sink particles: particle-mesh gravity, accretion, creation,
//! merging. One SinkParticles object belongs to a ParticlePopulation of type sink, which
//! owns the particle arrays and the migration chain.

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

//----------------------------------------------------------------------------------------
//! \struct SinkTaskIDs
//  \brief container to hold TaskIDs of the sink tasks

struct SinkTaskIDs {
  TaskID setgid;
  TaskID deposit;
  TaskID flush;
  // forward ghost exchange of the particle-mesh deposit (the "inbox" half; see
  // ParticleMesh::RefreshGhosts*)
  TaskID dm_irecv;
  TaskID dm_rest;
  TaskID dm_send;
  TaskID dm_recv;
  TaskID dm_prol;
  TaskID dm_csend;
  TaskID dm_crecv;
  TaskID push;
  TaskID merge;
  TaskID accrete;
  TaskID create;
  TaskID newdt;
  TaskID newgid;
  TaskID count;
  TaskID irecv;
  TaskID sendp;
  TaskID recvp;
  TaskID csend;
  TaskID crecv;
  // cross-rank particle migration chain at the start of each cycle
  TaskID bt_count;
  TaskID bt_irecv;
  TaskID bt_sendp;
  TaskID bt_recvp;
  TaskID bt_csend;
  TaskID bt_crecv;
  // gravitational-potential halo exchange, split into tasks like every other exchange
  TaskID xphi_irecv;
  TaskID xphi_rest;
  TaskID xphi_send;
  TaskID xphi_recv;
  TaskID xphi_bcs;
  TaskID xphi_prol;
  TaskID xphi_csend;
  TaskID xphi_crecv;
};

namespace particles {

class ParticlePopulation;
class ParticleMesh;

//----------------------------------------------------------------------------------------
//! \class SinkParticles

class SinkParticles {
 public:
  SinkParticles(ParticlePopulation *ppop, MeshBlockPack *ppack, ParameterInput *pin,
                const std::string &input_block);
  ~SinkParticles();

  ParticlePopulation *ppop;   // the population holding the sink arrays

  Real point_mass_gm;  // temporary treatment of source term on particle

  // Particle CFL number (<particles>/cfl_par, default 0.5): dtnew = cfl_par*min(dx/|v|)
  // over particles. Mesh::NewTimeStep applies NO cfl factor to the particle dt and caps
  // per-cycle dt growth at 2x, so 0.5 guarantees sinks cross at most one cell per step
  // (required by the control-volume accretion's cell-crossing correction).
  Real cfl_par;

  // Enable control-volume gas accretion onto sinks (<particles>/accretion, default
  // false). When false the AccreteMass task is not registered at all, so orbit tests
  // with inert/no gas never execute the accretion kernel.
  bool accretion = false;

  // Enable sink creation (<particles>/creation, default false): cells exceeding the
  // Larson-Penston density threshold at a local potential minimum spawn a massless
  // sink that the next AccreteMass reset seeds conservatively (Moon & Ostriker 2025
  // section 3.4). Requires accretion = true.
  bool creation = false;
  // Optional AMR experiment: retain the uniform finest-grid creation threshold.
  bool creation_finest_only = false;
  int created_total_ = 0;   // running count, for unique tags

  // Enable sink-sink merging (<particles>/merging, default false): two sinks whose
  // 27-cell control volumes (halos) overlap are combined into one, conserving mass and
  // linear momentum. Independent of accretion (a decaying gravitational binary can merge
  // with accretion off), but typically used with it -- merging removes the overlapping
  // control volumes that AccreteMass cannot otherwise handle, so it is ordered
  // create -> merge -> accrete. See sink_merger.cpp.
  bool merging = false;
  // Require the overlapping pair to be gravitationally bound before merging
  // (<particles>/merge_bound, default false): 0.5*mu*|dv|^2 - G*m_a*m_b/r < 0. Rejects
  // unbound fly-bys that momentarily pass within the halo. Set false to merge on halo
  // overlap alone. Needs the gravity module; falls back to overlap-only (one-time
  // warning) if <gravity> is absent.
  bool merge_bound = false;
  // Reference defaults; sink_setup=legacy restores the previous defaults.
  bool merge_iterative = true;
  bool merge_cell_centers = true;
  bool merge_face_contact = false;
  bool creation_exclusion = false;
  bool creation_zero_velocity = true;
  bool accrete_old_position = false;
  bool reject_negative_accretion = true;
  int sink_convergence = 1;  // 0: off; 1: directional mass flux; 2: velocity gradient

  // Sink sanity check (<particles>/sanity_vmax, default 0 = off). When > 0, every step
  // that changes sink state (push, migration receive, creation, merging, accretion) and
  // the particle timestep check the rank's sinks for a non-finite position, velocity or
  // mass, a negative mass, or a speed above sanity_vmax. The first hit prints the full
  // record of every local sink -- plus, when the push or the accretion update produced
  // the bad velocity, the inputs it was computed from -- and aborts the run. It only
  // reads particle state, so a run that never trips it is bit-for-bit unchanged.
  Real sanity_vmax = 0.0;
  static constexpr int kSanityRec = 16;
  DvceArray1D<int>  sanity_flag_;   // [1] 0 = clean, 1 = push, 2 = accretion
  DvceArray1D<Real> sanity_rec_;    // [kSanityRec] inputs of the first flagged update
  void CheckSinkSanity(const char *where, int stage);

  // Cross-rank control-volume reset (MPI). When a sink's control volume reaches into an
  // off-rank neighbour, the accretion kernel stages the off-rank-destined reset cells
  // (owner_m, xw, yw, zw, rho, M1..3 post-reset, rho, M1..3 pre-reset) into cvemit_
  // (cvemit_cnt_ = atomic count); ExchangeCVReset() then routes them host-side to every
  // rank whose blocks' expanded (interior+ghost) bounds contain the cell; each receiver
  // scatters the values to ALL its local coincident copies of u0/w0, at whatever
  // refinement level those copies live (see sink_accretion.cpp for the level rule --
  // the pre-reset values are what make the coarser-target case conservative).
  // Sized once accretion is enabled.
  static constexpr int NCVEMIT = 12;   // m, x, y, z, 4 post-reset, 4 pre-reset
  DvceArray2D<Real> cvemit_;      // [cvemit_max_][NCVEMIT] staging buffer (device)
  DvceArray1D<int>  cvemit_cnt_;  // [1] atomic write counter (device)
  int cvemit_max_ = 0;
  // Count of sinks whose control volume could not be processed this step (defensively
  // skipped, e.g. a >1 level jump in the reach). Reported once with a rate limit: a
  // silently skipped sink stops accreting while gas keeps flowing in, which shows up
  // much later as a pile-up and a collapsing timestep.
  DvceArray1D<int>  accskip_;     // [1] atomic count of skipped sinks (device)
  int accskip_warned_ = 0;
#if MPI_PARALLEL_ENABLED
  // dedicated communicator for the CV scatter
  MPI_Comm mpi_comm_cvscat_ = MPI_COMM_NULL;
#endif

  // Particle-mesh coupling layer: deposit, ghost flush, gravity gather.
  ParticleMesh *ppm = nullptr;

  // Dedicated boundary-values object for the potential's halo exchange. Its own object,
  // NOT the particle-mesh deposit's (ppm->pmbval): one object per exchanged field is the
  // convention here, each carrying its own buffers, MPI requests and communicators, and
  // sharing one between two different fields couples their exchange state.
  MeshBoundaryValuesCC *pbval_phi = nullptr;

  // container to hold names of TaskIDs
  SinkTaskIDs id;

  // ---- host / MPI phase timers (env SINK_TIMERS) --------------------------------------
  // Wall-clock accounting for the sink module's HOST and MPI regions -- precisely what a
  // Kokkos profiler cannot see. kokkos-tools reports device kernels by label, but the
  // per-step collectives, the host<->device particle mirrors and the host colouring loop
  // are invisible to it, and those are the suspected cost at scale.
  //
  //   SINK_TIMERS=1  accumulate and report. Low perturbation, but Kokkos kernels are
  //                  asynchronous, so a region can absorb the completion of work launched
  //                  before it, charged at whatever call first synchronizes.
  //   SINK_TIMERS=2  additionally Kokkos::fence() + MPI_Barrier before each region and
  //                  account that separately as "wait". The region's own cost is then
  //                  isolated, and "wait" is the load imbalance that a collective merely
  //                  EXPOSES rather than causes. Perturbing by construction: an
  //                  attribution run, never a timing run.
  //
  // ST_REFRESH is nested inside ST_CREATE and ST_MERGE, and ST_ALLTOALL inside
  // ST_CVRESET, so those totals are INCLUSIVE -- do not add the column up.
  enum SinkTimerId {ST_COLOUR=0, ST_CVRESET, ST_ALLTOALL, ST_CREATE, ST_MERGE, ST_REFRESH,
                    ST_NUM};
  int st_mode_ = 0;
  double st_time_[ST_NUM] = {};
  double st_wait_[ST_NUM] = {};
  std::int64_t st_calls_[ST_NUM] = {};
  Kokkos::Timer st_clock_;  // SINK_TIMERS clock
  double StBegin(int id);
  void   StEnd(int id, double t0);
  void   ReportSinkTimers(double wall_seconds);
  // scope guard, so a region with several early returns still stops its timer
  struct SinkTimerScope {
    SinkParticles *p; int id; double t0;
    SinkTimerScope(SinkParticles *pp, int i) : p(pp), id(i), t0(pp->StBegin(i)) {}
    ~SinkTimerScope() { p->StEnd(id, t0); }
  };

  void RefreshMeshParticleCounts();  // after an array resize
  // Resize the particle arrays to npart_new and, with them, every buffer that was sized
  // from the ppc-derived count in the constructor, then refresh the Mesh's bookkeeping.
  // A pgen that seeds MORE particles than ppc implied must use this rather than a bare
  // Kokkos::resize: cvemit_ is allocated as nprtcl_thispack*64 records, so a ppc = 0
  // (creation-mode) run that then seeds N sinks would leave the cross-rank control-volume
  // staging buffer 64 records deep for every sink on the rank. The overflow is guarded
  // and warned rather than silent, but the reset it drops is gas that never gets removed.
  void ResizeForSeededParticles(int npart_new);
  // Rebuild the sinks of a run restarted from a restart file written WITHOUT particle
  // data, from the track (trk) output of feature/sink-particles-only, which stores every
  // column of every sink and is written in the same cycle as each restart: take the dump
  // whose cycle equals the restart cycle, map its columns by name, give each sink to the
  // block that contains it, and resume the creation-tag counter after the largest tag.
  void RestoreSinksFromTrk(const std::string &path);
  // <block>/restart_trk names the trk file to import on a restart without a particle
  // restart file (-p); <block>/restart_without_sinks starts such a restart with none.
  std::string restart_trk;
  bool restart_without_sinks = false;
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus Deposit(Driver *pdriver, int stage);
  TaskStatus FlushDeposit(Driver *pdriver, int stage);
  // Forward ghost exchange of dmesh, completing the pair the flush starts: the flush
  // ADDS each block's ghost spill into the neighbour interiors that own it (the reverse
  // half), and these copy the now-complete interiors back OUT into the ghosts (the
  // forward half, which is the only half the gas needs). Thin wrappers so the steps can
  // be task-list entries; the work and the rationale live on ParticleMesh.
  TaskStatus DMeshInitRecv(Driver *pdriver, int stage);
  TaskStatus DMeshRestrict(Driver *pdriver, int stage);
  TaskStatus DMeshSend(Driver *pdriver, int stage);
  TaskStatus DMeshRecv(Driver *pdriver, int stage);
  TaskStatus DMeshProlongate(Driver *pdriver, int stage);
  TaskStatus DMeshClearSend(Driver *pdriver, int stage);
  TaskStatus DMeshClearRecv(Driver *pdriver, int stage);
  // Gravitational-potential halo exchange. GatherGravity reads phi two cells deep near
  // MeshBlock boundaries but the multigrid leaves only mg_nghost layers valid, so phi
  // needs a halo swap after each solve. Split into the same task sequence every other
  // field exchange uses -- receives posted in "before_stagen", restrict/send/recv/BC/
  // prolongate in "stagen" before the gather, buffers cleared in "after_stagen".
  TaskStatus XPhiInitRecv(Driver *pdriver, int stage);
  TaskStatus XPhiRestrict(Driver *pdriver, int stage);
  TaskStatus XPhiSend(Driver *pdriver, int stage);
  TaskStatus XPhiRecv(Driver *pdriver, int stage);
  TaskStatus XPhiBCs(Driver *pdriver, int stage);
  TaskStatus XPhiProlongate(Driver *pdriver, int stage);
  TaskStatus XPhiClearSend(Driver *pdriver, int stage);
  TaskStatus XPhiClearRecv(Driver *pdriver, int stage);
  TaskStatus Push(Driver *pdriver, int stage);
  // the population's migration receive, followed by the sanity check
  TaskStatus RecvP(Driver *pdriver, int stage);
  TaskStatus MergeSinks(Driver *pdriver, int stage);
  TaskStatus AccreteMass(Driver *pdriver, int stage);
  void ExchangeCVReset();   // MPI: apply off-rank control-volume reset cells
  TaskStatus CreateSinks(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing these sinks
};

// The sink population of a MeshBlockPack, for problem generators: FindSinkPopulation
// returns nullptr if there is none, SinkPopulation aborts.
ParticlePopulation* FindSinkPopulation(MeshBlockPack *pmbp);
ParticlePopulation* SinkPopulation(MeshBlockPack *pmbp);

} // namespace particles
#endif // PARTICLES_SINK_PARTICLES_HPP_
