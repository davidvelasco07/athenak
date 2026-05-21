//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_mesh.cpp
//  \brief constructor / destructor / Zero for the ParticleMesh class.
//
//  Phase 1a — skeleton only. The TSC scatter (DepositMass) and the
//  boundary-sum communication are intentionally absent and land in Phase 1b/1c.

#include "particle_mesh.hpp"

#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn ParticleMesh::ParticleMesh
//! \brief allocates dmesh with the same ghost-cell layout as hydro u0.

ParticleMesh::ParticleMesh(MeshBlockPack *ppack, ParameterInput *pin, int nmeshaux_in)
    : nmeshaux(nmeshaux_in), pmy_pack(ppack) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb = pmy_pack->nmb_thispack;
  Kokkos::realloc(dmesh, nmb, nmeshaux, ncells3, ncells2, ncells1);
  Zero();
}

ParticleMesh::~ParticleMesh() {}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::Zero()
//! \brief zero every slot in every cell of dmesh.

void ParticleMesh::Zero() {
  Kokkos::deep_copy(dmesh, 0.0);
}

}  // namespace particles
