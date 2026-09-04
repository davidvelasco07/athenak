//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"
#include "particle_mesh.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  // No particles yet => no particle timestep constraint. See the declaration in
  // particles.hpp: leaving this indeterminate makes Mesh::NewTimeStep collapse the global
  // dt to zero on any creation-driven run (ppc = 0), which hangs at t = 0.
  dtnew = std::numeric_limits<Real>::max();
  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // control-volume gas accretion onto sinks: opt-in (accretion tests set it true);
  // orbit/gravity tests with inert or no gas leave it off so the kernel never runs
  accretion = pin->GetOrAddBoolean("particles","accretion",false);
  // sink creation (LP threshold + potential minimum; see particles_creation.cpp)
  creation = pin->GetOrAddBoolean("particles","creation",false);
  // sink-sink merging on overlapping control volumes (see particles_merger.cpp)
  merging = pin->GetOrAddBoolean("particles","merging",false);
  merge_bound = pin->GetOrAddBoolean("particles","merge_bound",true);

  // particle CFL number (see particles.hpp); 0.5 guarantees <= 1 cell crossed per step
  cfl_par = pin->GetOrAddReal("particles","cfl_par",0.5);

  // read number of particles per cell, and calculate number of particles this pack
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

  // compute number of particles as real number, since ppc can be < 1
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
  // then cast to integer
  nprtcl_thispack = static_cast<int>(r_npart);

  // select particle type
  {
    std::string ptype = pin->GetString("particles","type");
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
    } else if (ptype.compare("sink") == 0) {
      particle_type = ParticleType::sink;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type = '" << ptype << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // select pusher algorithm
  {
    std::string ppush = pin->GetString("particles","pusher");
    if (ppush.compare("drift") == 0) {
      pusher = ParticlesPusher::drift;
    } else if (ppush.compare("leapfrog") == 0) {
      pusher = ParticlesPusher::leapfrog;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher must be specified in <particles> block"
                <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // TODO(SMOON) This is temporary treatment of source term on particle.
  // Later, we need to implement a more general way to include source terms
  point_mass_gm = pin->GetOrAddReal("particles","point_mass_gm",0.0);

  // set dimensions of particle arrays. Note particles only work in 2D/3D
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particles only work in 2D/3D, but 1D problem initialized" <<std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (particle_type) {
    case ParticleType::cosmic_ray:
      {
        int ndim=4;
        if (pmy_pack->pmesh->three_d) {ndim+=2;}
        nrdata = ndim;
        nidata = 2;
        break;
      }
    case ParticleType::sink:
      {
        // Layout: IPX,IPVX,IPY,IPVY,IPZ,IPVZ, IPM, IPGX,IPGY,IPGZ (see athena.hpp).
        // 3D-only layout always; 2D runs leave IPZ/IPVZ/IPGZ at zero. The leapfrog
        // pusher reads IPZ/IPVZ unconditionally, so a 2D-shortened nrdata would
        // out-of-bounds. Pay the 24 B/particle to keep the layout uniform.
        nrdata = NRDATA_SINK;  // = 10
        nidata = 2;            // PGID, PTAG
        break;
      }
    default:
      break;
  }
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  // allocate particle-mesh coupling layer for mass-bearing species
  if (particle_type == ParticleType::sink) {
    // 1 slot for ρ_particles; extend (to 4 with momentum) when needed.
    ppm = new ParticleMesh(ppack, pin, 1);
    // Own boundary-values object for the potential halo exchange (1 variable). Separate
    // from ppm->pmbval so phi's exchange state cannot interact with the deposit's.
    pbval_phi = new MeshBoundaryValuesCC(ppack, pin, false);
    pbval_phi->InitializeBuffers(1);
  }

  // staging buffer + communicator for the cross-rank control-volume reset (MPI). Sized
  // to hold every reset cell of every sink twice over (new + old CV = 54 cells), which
  // bounds the off-rank-destined subset; overflow is guarded + warned in the kernel.
  if (particle_type == ParticleType::sink && accretion) {
    cvemit_max_ = std::max(1, nprtcl_thispack)*64;
    Kokkos::realloc(cvemit_, cvemit_max_, NCVEMIT);
    Kokkos::realloc(cvemit_cnt_, 1);
    Kokkos::realloc(accskip_, 1);
#if MPI_PARALLEL_ENABLED
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_cvscat_);
#endif
  }

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::RefreshMeshParticleCounts()
//! \brief Re-derive the Mesh's global particle bookkeeping (nprtcl_thisrank,
//! nprtcl_eachrank, nprtcl_total) from the CURRENT nprtcl_thispack. The Mesh sets these
//! once, from the ppc-derived count, BEFORE the problem generator runs; any pgen that
//! resizes the particle arrays (count-robust single/two-sink seeding) must call this
//! afterwards or downstream consumers (e.g. tracked-particle output sized from
//! nprtcl_eachrank) index past the resized arrays.

void Particles::RefreshMeshParticleCounts() {
  Mesh *pm = pmy_pack->pmesh;
  pm->nprtcl_thisrank = nprtcl_thispack;
  pm->nprtcl_eachrank[global_variable::my_rank] = nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&(pm->nprtcl_thisrank), 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#endif
  pm->nprtcl_total = 0;
  for (int n = 0; n < global_variable::nranks; ++n) {
    pm->nprtcl_total += pm->nprtcl_eachrank[n];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::ResizeForSeededParticles(int npart_new)
//! \brief Resize the particle arrays to npart_new, grow the buffers that were sized from
//! the ppc-derived count, and refresh the Mesh's global particle bookkeeping.
//!
//! The constructor sizes cvemit_ (the cross-rank control-volume reset staging buffer) as
//! max(1, nprtcl_thispack)*64 records, from the count ppc implied BEFORE the pgen runs. A
//! benchmark or IC that seeds sinks explicitly typically runs with ppc = 0, so that buffer
//! would hold 64 records total no matter how many sinks are seeded. Grow it here, so a
//! pgen only has to state how many particles it wants.

void Particles::ResizeForSeededParticles(int npart_new) {
  Kokkos::resize(prtcl_rdata, nrdata, npart_new);
  Kokkos::resize(prtcl_idata, nidata, npart_new);
  nprtcl_thispack = npart_new;

  // grow-only: never shrink below what a previous sizing already allocated
  if (particle_type == ParticleType::sink && accretion) {
    const int need = std::max(1, npart_new)*64;
    if (need > cvemit_max_) {
      cvemit_max_ = need;
      Kokkos::realloc(cvemit_, cvemit_max_, NCVEMIT);
    }
  }

  RefreshMeshParticleCounts();
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
  if (ppm != nullptr) {
    delete ppm;
    ppm = nullptr;
  }
  if (pbval_phi != nullptr) {
    delete pbval_phi;
    pbval_phi = nullptr;
  }
}

//----------------------------------------------------------------------------------------
// CreateParticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  std::string assign = pin->GetOrAddString("particles","assign_tag","index_order");

  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (assign.compare("index_order") == 0) {
    int tagstart = 0;
    for (int n=1; n<=global_variable::my_rank; ++n) {
      tagstart += pmy_pack->pmesh->nprtcl_eachrank[n-1];
    }

    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = tagstart + p;
    });

  // tags are assigned sequentially across ranks
  } else if (assign.compare("rank_order") == 0) {
    int myrank = global_variable::my_rank;
    int nranks = global_variable::nranks;
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = myrank + nranks*p;
    });

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assignment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles
