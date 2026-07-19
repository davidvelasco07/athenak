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
  TaskID xphi;
  TaskID push;
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

  // Cross-rank control-volume reset (MPI). When a sink's control volume reaches into an
  // off-rank neighbour, the accretion kernel stages the off-rank-destined reset cells
  // (owner_m, xw, yw, zw, rho, M1, M2, M3) into cvemit_ (cvemit_cnt_ = atomic count);
  // ExchangeCVReset() then routes them host-side to every rank whose blocks' expanded
  // (interior+ghost) bounds contain the cell; each receiver scatters the values to ALL
  // its local coincident copies of u0/w0. Sized once accretion is enabled.
  DvceArray2D<Real> cvemit_;      // [cvemit_max_][8] staging buffer (device)
  DvceArray1D<int>  cvemit_cnt_;  // [1] atomic write counter (device)
  int cvemit_max_ = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm mpi_comm_cvscat_ = MPI_COMM_NULL;   // dedicated communicator for the CV scatter
#endif

  // Particle-mesh coupling layer (allocated for sink type, nullptr for tracers).
  ParticleMesh *ppm = nullptr;

  // Boundary communication buffers and functions for particles
  ParticlesBoundaryValues *pbval_part;

  // container to hold names of TaskIDs
  ParticlesTaskIDs id;

  // functions...
  void CreateParticleTags(ParameterInput *pin);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SetGIDFromPosition(Driver *pdriver, int stage);
  TaskStatus Deposit(Driver *pdriver, int stage);
  TaskStatus FlushDeposit(Driver *pdriver, int stage);
  TaskStatus ExchangePhi(Driver *pdriver, int stage);
  TaskStatus Push(Driver *pdriver, int stage);
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
