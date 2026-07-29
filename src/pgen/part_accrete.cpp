//========================================================================================
// AthenaXXX astrophysical plasma code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file part_accrete.cpp
//! \brief Validation test for sink-particle control-volume accretion (Phase 4a).
//!
//! One sink particle at the box centre in uniform isothermal gas. With gas self-gravity
//! on (<hydro_srcterms>/self_gravity=true) the gas falls toward the sink, enters the
//! control volume and is accreted; with it off the gas stays static and nothing should
//! accrete (null test). A finalize hook reports the sink mass, the total gas mass and
//! their sum, which should equal the initial total (periodic BC -> no gas leaves).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "gravity/gravity.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
Real rho0_, mass0_, vbox_;
void AccreteFinalize(ParameterInput *pin, Mesh *pm);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  pgen_final_func = &AccreteFinalize;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << "part_accrete requires <particles> and <hydro> blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real mpar = pin->GetOrAddReal("problem", "mass", 1.0);
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 0.1);
  rho0_ = rho0;
  mass0_ = mpar;
  vbox_ = (pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min)
        * (pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min)
        * (pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min);

  // Sink position: box centre plus an optional offset. The offset matters for refined
  // meshes: the box centre of an even block decomposition lies exactly ON a block corner,
  // where the lower-inclusive containment test that assigns ownership is decided by
  // round-off in the block bounds -- the sink is then owned by the block whose face it
  // sits on, i.e. the low side. Offsetting by half a cell of the intended level puts the
  // sink unambiguously inside one block, which is what selects whether its control volume
  // looks out onto coarser or finer neighbours.
  Real offx = pin->GetOrAddReal("problem", "offx", 0.0);
  Real offy = pin->GetOrAddReal("problem", "offy", 0.0);
  Real offz = pin->GetOrAddReal("problem", "offz", 0.0);
  Real cx = 0.5*(pmy_mesh_->mesh_size.x1min + pmy_mesh_->mesh_size.x1max) + offx;
  Real cy = 0.5*(pmy_mesh_->mesh_size.x2min + pmy_mesh_->mesh_size.x2max) + offy;
  Real cz = 0.5*(pmy_mesh_->mesh_size.x3min + pmy_mesh_->mesh_size.x3max) + offz;

  // uniform isothermal gas at rest
  auto &u0 = pmbp->phydro->u0;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  par_for("accrete_gas", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
  });

  // One sink, seeded on whichever rank owns the block that contains it. Sizing the
  // particle arrays from <particles>/ppc gives the intended single particle only when one
  // rank holds the whole mesh: on N ranks each rank rounds ppc*(its cells) down to zero
  // and the sink vanishes. So resize to the locally-owned count instead (the
  // count-robust pattern used by part_merge/part_orbit).
  auto gids = pmbp->gids;
  auto &mbsz = pmbp->pmb->mb_size;
  int mown = -1;
  for (int mm = 0; mm < nmb; ++mm) {
    if (cx >= mbsz.h_view(mm).x1min && cx < mbsz.h_view(mm).x1max &&
        cy >= mbsz.h_view(mm).x2min && cy < mbsz.h_view(mm).x2max &&
        cz >= mbsz.h_view(mm).x3min && cz < mbsz.h_view(mm).x3max) {
      mown = mm; break;
    }
  }
  const int nloc = (mown >= 0) ? 1 : 0;
  Kokkos::resize(pmbp->ppart->prtcl_rdata, pmbp->ppart->nrdata, nloc);
  Kokkos::resize(pmbp->ppart->prtcl_idata, pmbp->ppart->nidata, nloc);
  pmbp->ppart->nprtcl_thispack = nloc;
  pmbp->ppart->RefreshMeshParticleCounts();
  if (nloc > 0) {
    auto pr = pmbp->ppart->prtcl_rdata;
    auto pi = pmbp->ppart->prtcl_idata;
    const bool has_prev = (pmbp->ppart->nrdata > IPX0);
    const int gidown = gids + mown;
    par_for("accrete_part", DevExeSpace(), 0, 0,
    KOKKOS_LAMBDA(int) {
      pr(IPX, 0) = cx;
      pr(IPY, 0) = cy;
      pr(IPZ, 0) = cz;
      pr(IPVX, 0) = 0.0;
      pr(IPVY, 0) = 0.0;
      pr(IPVZ, 0) = 0.0;
      pr(IPM, 0) = mpar;
      pr(IPGX, 0) = 0.0;
      pr(IPGY, 0) = 0.0;
      pr(IPGZ, 0) = 0.0;
      if (has_prev) { pr(IPX0, 0) = cx; pr(IPY0, 0) = cy; pr(IPZ0, 0) = cz; }
      pi(PGID, 0) = gidown;
      pi(PTAG, 0) = 0;
    });
  }

  Real dxmin = std::min({mbsz.h_view(0).dx1, mbsz.h_view(0).dx2, mbsz.h_view(0).dx3});
  Real &dtnew_ = pmbp->ppart->dtnew;
  dtnew_ = dxmin;  // particle drift dt; the hydro CFL will usually be smaller

  if (global_variable::my_rank == 0) {
    std::printf("part_accrete: M_sink0=%.6e  rho0=%.6e  M_gas0=%.6e  total0=%.6e\n",
                mpar, rho0, rho0*vbox_, mpar + rho0*vbox_);
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void AccreteFinalize()
//! \brief report sink mass, total gas mass and their sum (mass-conservation check).

void AccreteFinalize(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->phydro == nullptr) return;

  auto u0 = pmbp->phydro->u0;
  auto &mbsz = pmbp->pmb->mb_size;
  auto &ix = pmbp->pmesh->mb_indcs;
  int is = ix.is, ie = ix.ie, js = ix.js, je = ix.je, ks = ix.ks, ke = ix.ke;
  int nmb = pmbp->nmb_thispack;

  Real mgas = 0.0;
  Kokkos::parallel_reduce("accrete_gasmass",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, ks, js, is}, {nmb, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, Real &sum) {
      Real dV = mbsz.d_view(m).dx1*mbsz.d_view(m).dx2*mbsz.d_view(m).dx3;
      sum += u0(m, IDN, k, j, i)*dV;
    }, mgas);

  // sink mass (host copy)
  int npart = pmbp->ppart->nprtcl_thispack;
  auto pr = pmbp->ppart->prtcl_rdata;
  auto pr_h = Kokkos::create_mirror_view(pr);
  Kokkos::deep_copy(pr_h, pr);
  Real msink = 0.0;
  for (int p = 0; p < npart; ++p) msink += pr_h(IPM, p);

  // Both sums are rank-LOCAL: mgas covers this rank's blocks and msink this rank's
  // particles (the sink lives on exactly one rank). Reduce before reporting, or a
  // multi-rank run prints a fraction of the gas and, on every rank but the sink's
  // owner, no sink at all -- which would make any 1-vs-N-rank comparison meaningless.
#if MPI_PARALLEL_ENABLED
  Real loc[2] = {mgas, msink}, glb[2] = {0.0, 0.0};
  MPI_Allreduce(loc, glb, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  mgas = glb[0];  msink = glb[1];
#endif

  if (global_variable::my_rank == 0) {
    std::printf("\n=== part_accrete finalize: t=%.6f  ncycle=%d ===\n", pm->time, pm->ncycle);
    std::printf("  M_sink = %.10e  (initial %.10e, grew by %.4e)\n",
                msink, mass0_, msink - mass0_);
    std::printf("  M_gas  = %.10e  (initial %.10e)\n", mgas, rho0_*vbox_);
    std::printf("  total  = %.10e  (initial %.10e, drift %.4e)\n",
                msink + mgas, mass0_ + rho0_*vbox_, (msink + mgas) - (mass0_ + rho0_*vbox_));
  }
}
}  // namespace
