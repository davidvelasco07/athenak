//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_amr.cpp
//! \brief Drifting particles on an adaptive mesh whose refined region keeps moving.
//!
//! Every cycle can refine or derefine MeshBlocks and load-balance them across ranks, so
//! particle ownership has to be re-derived from position after each regrid. The user
//! history records the global particle count, the sum of tags, the number of particles
//! whose PGID does not name the local MeshBlock containing them, and the number of mesh
//! updates so far.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "parameter_input.hpp"
#include "particles/cosmic_ray.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {

Real particle_dt = 0.05;
Real drift_vx = 0.0, drift_vy = 0.0, drift_vz = 0.0;
Real region_radius = 0.0;
Real region_x0 = 0.0, region_vx = 0.0;

Real ParticleAMRTimestep(MeshBlockPack*) { return particle_dt; }

// Eight particles per initial MeshBlock, on a 2x2x2 lattice inside it. Velocities vary
// smoothly with position so particles separate, cross blocks and change ranks.
void InitialParticleInjection(MeshBlockPack *pmbp,
                              particles::ParticleCreation *creation) {
  const int nmb = pmbp->nmb_thispack;
  creation->Resize(8*nmb);
  if (nmb == 0) return;
  auto pr = creation->prtcl_rdata;
  auto pi = creation->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  const int gids = pmbp->gids;
  const Real vx = drift_vx, vy = drift_vy, vz = drift_vz;
  par_for("particle_amr_init", DevExeSpace(), 0, 8*nmb-1,
  KOKKOS_LAMBDA(const int p) {
    const int m = p/8, q = p%8;
    auto size = mbsize.d_view(m);
    const Real x = size.x1min + (0.25 + 0.5*(q & 1))*(size.x1max - size.x1min);
    const Real y = size.x2min + (0.25 + 0.5*((q >> 1) & 1))*(size.x2max - size.x2min);
    const Real z = size.x3min + (0.25 + 0.5*((q >> 2) & 1))*(size.x3max - size.x3min);
    pi(particles::cosmic_ray::PGID,p) = gids + m;
    pr(particles::cosmic_ray::IPX,p) = x;
    pr(particles::cosmic_ray::IPY,p) = y;
    pr(particles::cosmic_ray::IPZ,p) = z;
    pr(particles::cosmic_ray::IPVX,p) = vx*(1.0 + 0.5*Kokkos::sin(3.0*y));
    pr(particles::cosmic_ray::IPVY,p) = vy*(1.0 + 0.5*Kokkos::sin(3.0*z));
    pr(particles::cosmic_ray::IPVZ,p) = vz*(1.0 + 0.5*Kokkos::sin(3.0*x));
  });
}

// Refine MeshBlocks whose centre lies within region_radius of a point moving along x1
// (periodically); derefine everything else.
void MovingRegionRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &size = pmbp->pmb->mb_size;
  auto &meshsize = pmbp->pmesh->mesh_size;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  const Real lx = meshsize.x1max - meshsize.x1min;
  Real xc = region_x0 + region_vx*pmbp->pmesh->time - meshsize.x1min;
  xc = meshsize.x1min + (xc - std::floor(xc/lx)*lx);
  const Real r2 = region_radius*region_radius;
  for (int m=0; m<nmb; ++m) {
    auto s = size.h_view(m);
    Real dx = 0.5*(s.x1min + s.x1max) - xc;
    dx -= lx*std::round(dx/lx);
    const Real dy = 0.5*(s.x2min + s.x2max);
    const Real dz = 0.5*(s.x3min + s.x3max);
    refine_flag.h_view(m + mbs) = (dx*dx + dy*dy + dz*dz < r2) ? 1 : -1;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

void AMRHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 5;
  pdata->label[0] = "npart";
  pdata->label[1] = "tag_sum";
  pdata->label[2] = "owner_err";
  pdata->label[3] = "nmb";
  pdata->label[4] = "mesh_seq";

  auto *population = pm->pmb_pack->ppart->FindPopulation("particles");
  auto pr = population->prtcl_rdata;
  auto pi = population->prtcl_idata;
  auto &mbsize = pm->pmb_pack->pmb->mb_size;
  const int npart = population->nprtcl_thispack;
  const int gids = pm->pmb_pack->gids;
  const int nmb = pm->pmb_pack->nmb_thispack;

  Real tag_sum = 0.0;
  Kokkos::parallel_reduce(
      "particle_amr_tag_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        local_sum += static_cast<Real>(pi(particles::cosmic_ray::PTAG,p));
      }, tag_sum);

  Real owner_errors = 0.0;
  Kokkos::parallel_reduce(
      "particle_amr_owner_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &local_sum) {
        const Real x = pr(particles::cosmic_ray::IPX,p);
        const Real y = pr(particles::cosmic_ray::IPY,p);
        const Real z = pr(particles::cosmic_ray::IPZ,p);
        int expected_gid = -1;
        for (int m=0; m<nmb; ++m) {
          auto size = mbsize.d_view(m);
          if (x >= size.x1min && x < size.x1max &&
              y >= size.x2min && y < size.x2max &&
              z >= size.x3min && z < size.x3max) {
            expected_gid = gids + m;
          }
        }
        if (pi(particles::cosmic_ray::PGID,p) != expected_gid) local_sum += 1.0;
      }, owner_errors);

  // hst sums user columns over ranks; count MeshBlocks and updates on rank 0 only
  const bool root = (global_variable::my_rank == 0);
  pdata->hdata[0] = static_cast<Real>(npart);
  pdata->hdata[1] = tag_sum;
  pdata->hdata[2] = owner_errors;
  pdata->hdata[3] = root ? static_cast<Real>(pm->nmb_total) : 0.0;
  pdata->hdata[4] = root ? static_cast<Real>(pm->GetAMRLoadBalanceUpdateSeq()) : 0.0;
}

} // namespace

void ProblemGenerator::ParticleAMR(ParameterInput *pin, const bool restart) {
  user_hist_func = AMRHistory;
  user_particle_dt_func = ParticleAMRTimestep;
  user_initial_particle_injection_func = InitialParticleInjection;
  user_ref_func = MovingRegionRefinement;

  particle_dt = pin->GetOrAddReal("problem", "particle_dt", 0.05);
  drift_vx = pin->GetOrAddReal("problem", "drift_vx", 0.3);
  drift_vy = pin->GetOrAddReal("problem", "drift_vy", 0.2);
  drift_vz = pin->GetOrAddReal("problem", "drift_vz", 0.1);
  region_radius = pin->GetOrAddReal("problem", "region_radius", 0.6);
  region_x0 = pin->GetOrAddReal("problem", "region_x0", -0.5);
  region_vx = pin->GetOrAddReal("problem", "region_vx", 2.0);

  if (!pmy_mesh_->adaptive) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle AMR test requires adaptive mesh refinement"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto *population = (pmbp->ppart == nullptr) ? nullptr :
                     pmbp->ppart->FindPopulation("particles");
  if (population == nullptr || population->particle_type != ParticleType::cosmic_ray) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle AMR test requires cosmic-ray particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
