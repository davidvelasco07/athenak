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

  Real cx = 0.5*(pmy_mesh_->mesh_size.x1min + pmy_mesh_->mesh_size.x1max);
  Real cy = 0.5*(pmy_mesh_->mesh_size.x2min + pmy_mesh_->mesh_size.x2max);
  Real cz = 0.5*(pmy_mesh_->mesh_size.x3min + pmy_mesh_->mesh_size.x3max);

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

  // one sink at the box centre
  auto &pr = pmbp->ppart->prtcl_rdata;
  auto &pi = pmbp->ppart->prtcl_idata;
  int npart = pmbp->ppart->nprtcl_thispack;
  auto gids = pmbp->gids;
  auto &mbsz = pmbp->pmb->mb_size;
  par_for("accrete_part", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(int p) {
    pr(IPX, p) = cx;
    pr(IPY, p) = cy;
    pr(IPZ, p) = cz;
    pr(IPVX, p) = 0.0;
    pr(IPVY, p) = 0.0;
    pr(IPVZ, p) = 0.0;
    pr(IPM, p) = mpar;
    pr(IPGX, p) = 0.0;
    pr(IPGY, p) = 0.0;
    pr(IPGZ, p) = 0.0;
    int mown = 0;
    for (int mm = 0; mm < nmb; ++mm) {
      if (cx >= mbsz.d_view(mm).x1min && cx < mbsz.d_view(mm).x1max &&
          cy >= mbsz.d_view(mm).x2min && cy < mbsz.d_view(mm).x2max &&
          cz >= mbsz.d_view(mm).x3min && cz < mbsz.d_view(mm).x3max) {
        mown = mm;
      }
    }
    pi(PGID, p) = gids + mown;
  });

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
