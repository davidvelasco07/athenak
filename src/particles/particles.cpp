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
  }

  // staging buffer + communicator for the cross-rank control-volume reset (MPI). Sized
  // to hold every reset cell of every sink twice over (new + old CV = 54 cells), which
  // bounds the off-rank-destined subset; overflow is guarded + warned in the kernel.
  if (particle_type == ParticleType::sink && accretion) {
    cvemit_max_ = std::max(1, nprtcl_thispack)*64;
    Kokkos::realloc(cvemit_, cvemit_max_, 8);
    Kokkos::realloc(cvemit_cnt_, 1);
#if MPI_PARALLEL_ENABLED
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_cvscat_);
#endif
  }

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
  if (ppm != nullptr) {
    delete ppm;
    ppm = nullptr;
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
