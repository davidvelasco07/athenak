#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for Particles class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations

// constants that enumerate ParticlesPusher options
enum class ParticlesPusher {drift, leapfrog, lagrangian_tracer, lagrangian_mc};

// constants that enumerate ParticleTypes
enum class ParticleType {cosmic_ray, sink};

//----------------------------------------------------------------------------------------
//! \struct ParticlesTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticlesTaskIDs {
  TaskID setgid;
  TaskID deposit;
  TaskID flush;
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
  // regrid-driven cross-rank particle migration chain (before_timeintegrator)
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

// forward declaration
class ParticleMesh;

//----------------------------------------------------------------------------------------
//! \class Particles

class Particles {
  friend class ParticlesBoundaryValues;
 public:
  Particles(MeshBlockPack *ppack, ParameterInput *pin);
  ~Particles();

  // data
  ParticleType particle_type;
  int nprtcl_thispack;             // number of particles this MeshBlockPack
  int nrdata, nidata;
  DvceArray2D<Real> prtcl_rdata;   // real number properties each particle (x,v,etc.)
  DvceArray2D<int>  prtcl_idata;   // integer properties each particle (gid, tag, etc.)
  Real dtnew;

  ParticlesPusher pusher;
  Real point_mass_gm;  // temporary treatment of source term on particle

  // Particle CFL number (<particles>/cfl_par, default 0.5): dtnew = cfl_par*min(dx/|v|)
  // over particles. Mesh::NewTimeStep applies NO cfl factor to ppart->dtnew and caps
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
  int created_total_ = 0;   // running count, for unique tags

  // Enable sink-sink merging (<particles>/merging, default false): two sinks whose
  // 27-cell control volumes (halos) overlap are combined into one, conserving mass and
  // linear momentum. Independent of accretion (a decaying gravitational binary can merge
  // with accretion off), but typically used with it -- merging removes the overlapping
  // control volumes that AccreteMass cannot otherwise handle, so it is ordered
  // create -> merge -> accrete. See particles_merger.cpp.
  bool merging = false;
  // Require the overlapping pair to be gravitationally bound before merging
  // (<particles>/merge_bound, default true): 0.5*mu*|dv|^2 - G*m_a*m_b/r < 0. Rejects
  // unbound fly-bys that momentarily pass within the halo. Set false to merge on halo
  // overlap alone. Needs the gravity module; falls back to overlap-only (one-time
  // warning) if <gravity> is absent.
  bool merge_bound = true;

  // Cross-rank control-volume reset (MPI). When a sink's control volume reaches into an
  // off-rank neighbour, the accretion kernel stages the off-rank-destined reset cells
  // (owner_m, xw, yw, zw, rho, M1..3 post-reset, rho, M1..3 pre-reset) into cvemit_
  // (cvemit_cnt_ = atomic count); ExchangeCVReset() then routes them host-side to every
  // rank whose blocks' expanded (interior+ghost) bounds contain the cell; each receiver
  // scatters the values to ALL its local coincident copies of u0/w0, at whatever
  // refinement level those copies live (see particles_accretion.cpp for the level rule --
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
  MPI_Comm mpi_comm_cvscat_ = MPI_COMM_NULL;   // dedicated communicator for the CV scatter
#endif

  // Particle-mesh coupling layer (allocated for sink type, nullptr for tracers).
  ParticleMesh *ppm = nullptr;

  // Dedicated boundary-values object for the potential's halo exchange. Its own object,
  // NOT the particle-mesh deposit's (ppm->pmbval): one object per exchanged field is the
  // convention here, each carrying its own buffers, MPI requests and communicators, and
  // sharing one between two different fields couples their exchange state.
  MeshBoundaryValuesCC *pbval_phi = nullptr;

  // Boundary communication buffers and functions for particles
  ParticlesBoundaryValues *pbval_part;

  // container to hold names of TaskIDs
  ParticlesTaskIDs id;

  // functions...
  void CreateParticleTags(ParameterInput *pin);
  void RefreshMeshParticleCounts();  // after pgen-side array resize
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SetGIDFromPosition(Driver *pdriver, int stage);
  void ReconcileOwnership(Driver *pdriver, int stage);  // setgid + cross-rank migration
  TaskStatus Deposit(Driver *pdriver, int stage);
  TaskStatus FlushDeposit(Driver *pdriver, int stage);
  // Gravitational-potential halo exchange. GatherGravity reads phi two cells deep near
  // MeshBlock boundaries but the multigrid leaves only mg_nghost layers valid, so phi
  // needs a halo swap after each solve. Split into the same task sequence every other
  // field exchange uses -- receives posted in "before_stagen", restrict/send/recv/BC/
  // prolongate in "stagen" before the gather, buffers cleared in "after_stagen" -- rather
  // than one call that posts, sends and then SPINS until the receives drain. The spin
  // blocked the whole task list mid-stage on every rank, which is both a serialization
  // point and a structure no other exchange in the code uses.
  TaskStatus XPhiInitRecv(Driver *pdriver, int stage);
  TaskStatus XPhiRestrict(Driver *pdriver, int stage);
  TaskStatus XPhiSend(Driver *pdriver, int stage);
  TaskStatus XPhiRecv(Driver *pdriver, int stage);
  TaskStatus XPhiBCs(Driver *pdriver, int stage);
  TaskStatus XPhiProlongate(Driver *pdriver, int stage);
  TaskStatus XPhiClearSend(Driver *pdriver, int stage);
  TaskStatus XPhiClearRecv(Driver *pdriver, int stage);
  TaskStatus Push(Driver *pdriver, int stage);
  TaskStatus MergeSinks(Driver *pdriver, int stage);
  TaskStatus AccreteMass(Driver *pdriver, int stage);
  void ExchangeCVReset();   // MPI: apply off-rank control-volume reset cells
  TaskStatus CreateSinks(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);
  TaskStatus NewGID(Driver *pdriver, int stage);
  TaskStatus SendCnt(Driver *pdriver, int stage);
  TaskStatus InitRecv(Driver *pdriver, int stage);
  TaskStatus SendP(Driver *pdriver, int stage);
  TaskStatus RecvP(Driver *pdriver, int stage);
  TaskStatus ClearSend(Driver *pdriver, int stage);
  TaskStatus ClearRecv(Driver *pdriver, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Particles
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
