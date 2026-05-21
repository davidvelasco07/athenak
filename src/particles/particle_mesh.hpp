#ifndef PARTICLES_PARTICLE_MESH_HPP_
#define PARTICLES_PARTICLE_MESH_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_mesh.hpp
//  \brief Particle-mesh (PM) coupling layer for AthenaK particles.
//
//  Owns a per-pack 5D auxiliary field `dmesh` of shape
//      (nmb_in_pack, nmeshaux, ncells3, ncells2, ncells1)
//  with the same ghost-cell layout as the hydro `u0` array. Subsequent phases
//  (1b, 1c, ...) populate it via TSC scatter from the particle list and
//  exchange ghost-cell sums across MeshBlock boundaries.
//
//  This is the AthenaK port of Athena++'s ParticleMesh class (units_revision
//  branch of changgoo/athena), restructured for pack-level (not per-MB)
//  ownership and Kokkos parallelism.

#include "athena.hpp"
#include "parameter_input.hpp"

class MeshBlockPack;

namespace particles {

class Particles;

//----------------------------------------------------------------------------------------
//! \fn KOKKOS_INLINE_FUNCTION Real TSCWeight(Real dxi)
//! \brief Triangular-Shaped-Cloud weight kernel.
//!
//! `dxi` is the signed cell-fraction distance from the cloud center to a
//! candidate cell center (e.g. dxi = 0 if the particle is exactly at the
//! candidate cell's center; dxi = +1 if one cell to its right).
//!
//! Returns:
//!   |dxi| < 0.5     : 0.75 - dxi^2
//!   0.5 <= |dxi| < 1.5 : 0.5 * (1.5 - |dxi|)^2
//!   |dxi| >= 1.5    : 0
//!
//! The three nonzero values sum to 1 when evaluated at three contiguous cell
//! centers around the particle's position.
KOKKOS_INLINE_FUNCTION
Real TSCWeight(Real dxi) {
  Real a = (dxi < 0.0) ? -dxi : dxi;
  if (a >= 1.5) return 0.0;
  return (a < 0.5) ? (0.75 - a*a)
                   : (0.5 * (1.5 - a) * (1.5 - a));
}

//----------------------------------------------------------------------------------------
//! \class ParticleMesh
//! \brief Per-pack auxiliary mesh field for particle-mesh deposition and gather.
//!
//! Allocated for sink particles (and other mass-bearing species in future).
//! Not allocated for tracer/cosmic-ray particles.

class ParticleMesh {
 public:
  ParticleMesh(MeshBlockPack *ppack, ParameterInput *pin, int nmeshaux_in);
  ~ParticleMesh();

  // Number of auxiliary slots stored at each cell (e.g. 1 for ρ_particles
  // alone, 4 for ρ + 3-momentum, etc.). Fixed at construction.
  int nmeshaux;

  // Shape (nmb, nmeshaux, ncells3, ncells2, ncells1) with hydro-style ghosts.
  DvceArray5D<Real> dmesh;

  // Zero every slot in every cell (called before each deposition pass).
  void Zero();

 private:
  MeshBlockPack *pmy_pack;
};

}  // namespace particles
#endif  // PARTICLES_PARTICLE_MESH_HPP_
