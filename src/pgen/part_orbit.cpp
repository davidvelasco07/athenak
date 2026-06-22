//========================================================================================
// AthenaXXX astrophysical plasma code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file part_orbit.cpp
//! \brief Minimal validation test for the sink-particle gravity slice.
//!
//! Two equal-mass sink particles in a box with negligible uniform gas and multigrid
//! self-gravity. Exercises the full per-stage loop: TSC deposit -> Gravity::AssembleSource
//! -> Poisson solve -> GatherGravity -> RK2-KDK kick/drift. With the default circular
//! velocity the pair should orbit their (stationary) centre of mass; set <problem>/vorbit=0
//! for a head-on free-fall instead. A pgen finalize hook prints the final particle state
//! and the conserved diagnostics (centre of mass, total momentum, z angular momentum).
//!
//! Particle count: the constructor sets nprtcl = ppc * nmb * ncells. For exactly two
//! particles on a single NxNxN block use ppc = 2/N^3 (a negative power of two, exact in
//! double precision -- e.g. 2/64^3 = 2^-17).

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
// post-run diagnostics: final particle state + conserved quantities
void OrbitFinalize(ParameterInput *pin, Mesh *pm);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  pgen_final_func = &OrbitFinalize;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->phydro == nullptr || pmbp->pgrav == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_orbit requires <particles>, <hydro> and <gravity> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // parameters
  Real mpar = pin->GetOrAddReal("problem", "mass", 1.0);
  Real dsep = pin->GetOrAddReal("problem", "separation", 0.3);
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0e-8);
  Real Gconst = pmbp->pgrav->four_pi_G / (4.0*M_PI);
  // circular speed of each particle about the COM for two equal point masses:
  //   v^2 = G m / (2 d)
  Real vcirc = std::sqrt(Gconst*mpar/(2.0*dsep));
  // velocity as a fraction of the circular value: vfrac=1 -> orbit, vfrac=0 -> free-fall
  Real vfrac = pin->GetOrAddReal("problem", "vfrac", 1.0);
  Real vorb = vfrac*vcirc;

  // box centre, plus an optional rigid offset of the particle pair (used to place the
  // pair away from MeshBlock boundaries when decomposing, for testing/isolation)
  Real cx = 0.5*(pmy_mesh_->mesh_size.x1min + pmy_mesh_->mesh_size.x1max)
          + pin->GetOrAddReal("problem", "offx", 0.0);
  Real cy = 0.5*(pmy_mesh_->mesh_size.x2min + pmy_mesh_->mesh_size.x2max)
          + pin->GetOrAddReal("problem", "offy", 0.0);
  Real cz = 0.5*(pmy_mesh_->mesh_size.x3min + pmy_mesh_->mesh_size.x3max)
          + pin->GetOrAddReal("problem", "offz", 0.0);

  if (global_variable::my_rank == 0) {
    std::printf("part_orbit: m=%.3e d=%.3e G=%.6e vcirc=%.6e vorb=%.6e\n",
                mpar, dsep, Gconst, vcirc, vorb);
  }

  // uniform, negligible gas (isothermal EOS -> no energy variable)
  auto &u0 = pmbp->phydro->u0;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  par_for("orbit_gas", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
  });

  // two sink particles, symmetric about the centre, on a circular orbit in the xy-plane
  // (counter-clockwise: particle at -x moves -y, particle at +x moves +y; COM at rest)
  auto &pr = pmbp->ppart->prtcl_rdata;
  auto &pi = pmbp->ppart->prtcl_idata;
  int npart = pmbp->ppart->nprtcl_thispack;
  auto gids = pmbp->gids;
  par_for("orbit_parts", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(int p) {
    Real sgn = (p == 0) ? -1.0 : 1.0;
    pi(PGID, p) = gids;
    pr(IPX, p) = cx + sgn*0.5*dsep;
    pr(IPY, p) = cy;
    pr(IPZ, p) = cz;
    pr(IPVX, p) = 0.0;
    pr(IPVY, p) = sgn*vorb;
    pr(IPVZ, p) = 0.0;
    pr(IPM, p) = mpar;
    pr(IPGX, p) = 0.0;
    pr(IPGY, p) = 0.0;
    pr(IPGZ, p) = 0.0;
  });

  // constant particle timestep: keep sub-cell motion per step
  auto &mbsize = pmbp->pmb->mb_size;
  Real dxmin = std::min({mbsize.h_view(0).dx1, mbsize.h_view(0).dx2, mbsize.h_view(0).dx3});
  Real &dtnew_ = pmbp->ppart->dtnew;
  dtnew_ = 0.25*dxmin/std::max(vorb, 1.0e-30);

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void OrbitFinalize()
//! \brief print final particle state and conserved diagnostics after the run.

void OrbitFinalize(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->ppart == nullptr) return;
  if (global_variable::my_rank != 0) return;

  int npart = pmbp->ppart->nprtcl_thispack;
  auto pr = pmbp->ppart->prtcl_rdata;
  auto pr_h = Kokkos::create_mirror_view(pr);
  Kokkos::deep_copy(pr_h, pr);

  Real Mtot = 0.0, comx = 0.0, comy = 0.0, comz = 0.0;
  Real px = 0.0, py = 0.0, pz = 0.0, Lz = 0.0;
  std::printf("\n=== part_orbit finalize: t=%.6f  ncycle=%d  npart=%d ===\n",
              pm->time, pm->ncycle, npart);
  for (int p = 0; p < npart; ++p) {
    Real m  = pr_h(IPM, p);
    Real x  = pr_h(IPX, p),  y  = pr_h(IPY, p),  z  = pr_h(IPZ, p);
    Real vx = pr_h(IPVX, p), vy = pr_h(IPVY, p), vz = pr_h(IPVZ, p);
    std::printf("  p%d: x=(% .6e,% .6e,% .6e) v=(% .6e,% .6e,% .6e) m=%.3e\n",
                p, x, y, z, vx, vy, vz, m);
    std::printf("       g=(% .6e,% .6e,% .6e)\n",
                pr_h(IPGX, p), pr_h(IPGY, p), pr_h(IPGZ, p));
    Mtot += m;
    comx += m*x; comy += m*y; comz += m*z;
    px += m*vx; py += m*vy; pz += m*vz;
    Lz += m*(x*vy - y*vx);
  }
  if (Mtot > 0.0) { comx /= Mtot; comy /= Mtot; comz /= Mtot; }
  std::printf("  COM=(% .6e,% .6e,% .6e)  Ptot=(% .3e,% .3e,% .3e)  Lz=% .6e\n",
              comx, comy, comz, px, py, pz, Lz);
  if (npart == 2) {
    Real dx = pr_h(IPX, 1) - pr_h(IPX, 0);
    Real dy = pr_h(IPY, 1) - pr_h(IPY, 0);
    Real dz = pr_h(IPZ, 1) - pr_h(IPZ, 0);
    std::printf("  separation=% .6e\n", std::sqrt(dx*dx + dy*dy + dz*dz));
  }
}
}  // namespace
