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
class MeshBoundaryValuesCC;

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

  // TSC scatter of particle mass into slot 0 of dmesh.
  //   prtcl_rdata: (nrdata, nprtcl) -- reads IPX/IPY/IPZ and IPM
  //   prtcl_idata: (nidata, nprtcl) -- reads PGID
  //   npar:        number of particles to deposit
  //
  // Implicitly calls Zero() first. Deposits into both interior and ghost
  // cells of dmesh; the boundary-sum communication added in Phase 1c will
  // fold the ghost-cell contributions back into neighbor interiors.
  //
  // Each particle contributes to 27 cells in 3D (or 9 in 2D), each via a
  // Kokkos::atomic_add. No write conflicts; small contention since the
  // 27-cell footprint is tight.
  void DepositMass(const DvceArray2D<Real>& prtcl_rdata,
                   const DvceArray2D<int>&  prtcl_idata,
                   int npar);

  // TSC gather of the gravitational acceleration (-grad phi) from the mesh
  // onto each particle, written into prtcl_rdata slots IPGX/IPGY/IPGZ.
  //   phi:         (nmb, 1, ncells3, ncells2, ncells1) potential from the
  //                multigrid Poisson solve (pgrav->phi).
  //   prtcl_rdata: (nrdata, nprtcl) -- reads IPX/IPY/IPZ, writes IPGX/IPGY/IPGZ
  //   prtcl_idata: (nidata, nprtcl) -- reads PGID
  //   npar:        number of particles to gather onto.
  //
  // The cell-centered acceleration is the same central-difference stencil used
  // by srcterms SelfGravity, a*(phi[m-1]-phi[m+1]) with a = 0.5/dx, TSC-weighted
  // over the same 27-cell (3D) / 9-cell (2D) cloud as DepositMass. This is the
  // AthenaK port of Athena++ ParticleGravity::{FindGravitationalForce,
  // InterpolateGravitationalForce}. Requires valid phi ghost cells two layers
  // deep for particles within two cells of a MeshBlock boundary (see Phase 1c).
  void GatherGravity(const DvceArray5D<Real>& phi,
                     DvceArray2D<Real>&       prtcl_rdata,
                     const DvceArray2D<int>&  prtcl_idata,
                     int npar);

  // Boundary "sum-flush" for the deposit (Phase 1c + AMR). DepositMass writes TSC
  // tails into the ghost cells of the depositing MeshBlock; those contributions
  // physically belong to the neighbour's interior. FlushDepositBoundaries() adds each
  // MeshBlock's ghost-zone deposit into the matching interior cells of its neighbours
  // and then zeroes the ghost spill, so dmesh holds the complete per-cell density.
  //
  // Handles on-rank neighbours at ANY level, mass-conservatively:
  //   same level   : ghost cell -> the coincident neighbour interior cell (add rho)
  //   fine->coarse : ghost cell -> the coarse cell containing it (add rho*dV_f/dV_c)
  //   coarse->fine : ghost cell -> every fine cell it covers   (add rho, uniform split)
  // Ownership is resolved geometrically: each nonzero ghost cell searches this block's
  // neighbour list for the (unique) MeshBlock whose bounds contain the cell centre
  // (with periodic wrapping), so faces/edges/corners and graded octree refinement all
  // use one code path -- including coarser "interior edge" regions that SetNeighbors
  // leaves unpopulated because the coarse face neighbour covers them.
  // Off-rank same-level neighbours are folded via a sparse cross-rank exchange
  // (ExchangeDepositFlush, mirroring the accretion CV scatter); off-rank level-jumps
  // are still dropped (warned once).
  void FlushDepositBoundaries();
  void ExchangeDepositFlush();   // MPI: add off-rank ghost-spill deposits to owners

  // Boundary-value helper for dmesh: provides the buffers/MPI machinery for the
  // future cross-rank flush path, and is reused by Particles::ExchangePhi for the
  // potential's halo exchange.
  MeshBoundaryValuesCC *pmbval = nullptr;

 private:
  MeshBlockPack *pmy_pack;
  // Device counter of particles skipped by the deposit/gather guards (invalid PGID,
  // non-finite position, or out-of-range cell index). Checked host-side after each
  // deposit; a nonzero count triggers a rate-limited warning instead of memory
  // corruption / SIGBUS.
  DvceArray1D<int> nbad_;
  // Cross-rank deposit-flush staging: off-rank-destined ghost spill (owner_m, v, x, y, z,
  // rho); ExchangeDepositFlush() routes + atomic-adds into off-rank interiors.
  DvceArray2D<Real> dfemit_;
  DvceArray1D<int>  dfemit_cnt_;
  int dfemit_max_ = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm mpi_comm_dfscat_ = MPI_COMM_NULL;
#endif
};

}  // namespace particles
#endif  // PARTICLES_PARTICLE_MESH_HPP_
