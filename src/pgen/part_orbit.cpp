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
#include "mesh/mesh_refinement.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "particles/particle_mesh.hpp"
#include "gravity/gravity.hpp"

namespace {
// post-run diagnostics: final particle state + conserved quantities
void OrbitFinalize(ParameterInput *pin, Mesh *pm);
// AMR: refine MeshBlocks that contain (or lie within refine_buf of) a sink particle
void SinkRefinement(MeshBlockPack *pmbp);
Real sink_refine_buf_ = 0.05;
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  pgen_final_func = &OrbitFinalize;
  user_ref_func = SinkRefinement;
  sink_refine_buf_ = pin->GetOrAddReal("problem", "refine_buf", 0.05);
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
  auto &mbsz = pmbp->pmb->mb_size;
  par_for("orbit_parts", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(int p) {
    Real sgn = (p == 0) ? -1.0 : 1.0;
    Real px = cx + sgn*0.5*dsep;
    Real py = cy;
    Real pz = cz;
    pr(IPX, p) = px;
    pr(IPY, p) = py;
    pr(IPZ, p) = pz;
    pr(IPVX, p) = 0.0;
    pr(IPVY, p) = sgn*vorb;
    pr(IPVZ, p) = 0.0;
    pr(IPM, p) = mpar;
    pr(IPGX, p) = 0.0;
    pr(IPGY, p) = 0.0;
    pr(IPGZ, p) = 0.0;
    // assign the MeshBlock that actually contains this particle (lower-inclusive),
    // not just gids -- required when the mesh is decomposed into multiple blocks
    int mown = 0;
    for (int mm = 0; mm < nmb; ++mm) {
      if (px >= mbsz.d_view(mm).x1min && px < mbsz.d_view(mm).x1max &&
          py >= mbsz.d_view(mm).x2min && py < mbsz.d_view(mm).x2max &&
          pz >= mbsz.d_view(mm).x3min && pz < mbsz.d_view(mm).x3max) {
        mown = mm;
      }
    }
    pi(PGID, p) = gids + mown;
  });

  // constant particle timestep: keep sub-cell motion per step
  auto &mbsize = pmbp->pmb->mb_size;
  Real dxmin = std::min({mbsize.h_view(0).dx1, mbsize.h_view(0).dx2, mbsize.h_view(0).dx3});
  Real &dtnew_ = pmbp->ppart->dtnew;
  dtnew_ = 0.25*dxmin/std::max(vorb, 1.0e-30);

  // optional flush diagnostic: deposit + flush at the (symmetric) IC and report mass
  // conservation and mirror symmetry of the particle-mesh density across each split plane.
  if (pin->GetOrAddBoolean("problem", "diag_flush", false)) {
    auto *ppm = pmbp->ppart->ppm;
    ppm->DepositMass(pr, pi, npart);
    ppm->FlushDepositBoundaries();
    auto dmv = ppm->dmesh;
    auto &ix = pmbp->pmesh->mb_indcs;
    int is2 = ix.is, ie2 = ix.ie, js2 = ix.js, je2 = ix.je, ks2 = ix.ks, ke2 = ix.ke;
    Real Mtot = 0.0, Mxn = 0.0, Myn = 0.0, Mzn = 0.0;
    Kokkos::parallel_reduce("flushdiag",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, ks2, js2, is2}, {nmb, ke2+1, je2+1, ie2+1}),
      KOKKOS_LAMBDA(int m, int k, int j, int i, Real &tt, Real &xx, Real &yy, Real &zz) {
        Real dV = mbsz.d_view(m).dx1*mbsz.d_view(m).dx2*mbsz.d_view(m).dx3;
        Real mass = dmv(m, 0, k, j, i)*dV;
        Real xcc = mbsz.d_view(m).x1min + (i - is2 + 0.5)*mbsz.d_view(m).dx1;
        Real ycc = mbsz.d_view(m).x2min + (j - js2 + 0.5)*mbsz.d_view(m).dx2;
        Real zcc = mbsz.d_view(m).x3min + (k - ks2 + 0.5)*mbsz.d_view(m).dx3;
        tt += mass;
        if (xcc < 0.0) xx += mass;
        if (ycc < 0.0) yy += mass;
        if (zcc < 0.0) zz += mass;
      }, Mtot, Mxn, Myn, Mzn);
    if (global_variable::my_rank == 0) {
      std::printf("=== flush diagnostic (deposit+flush at IC) ===\n");
      std::printf("  total deposited mass = %.10e   (expect %.10e)\n",
                  Mtot, static_cast<Real>(npart)*mpar);
      std::printf("  mass x<0 = %.8e   x>0 = %.8e\n", Mxn, Mtot - Mxn);
      std::printf("  mass y<0 = %.8e   y>0 = %.8e\n", Myn, Mtot - Myn);
      std::printf("  mass z<0 = %.8e   z>0 = %.8e\n", Mzn, Mtot - Mzn);
    }
  }

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

//----------------------------------------------------------------------------------------
//! \fn void SinkRefinement(MeshBlockPack *pmbp)
//! \brief AMR criterion: refine a MeshBlock to the next level if any sink particle lies
//! inside it expanded by refine_buf, derefine otherwise. Refining the sink's block plus a
//! buffer keeps the sink interior to the fine region (away from coarse-fine boundaries).
//! Particle-driven (no gas dependence), so it works with inert gas / a clean orbit.

void SinkRefinement(MeshBlockPack *pmbp) {
  if (pmbp->ppart == nullptr) return;
  auto refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto &size = pmbp->pmb->mb_size;
  auto pr = pmbp->ppart->prtcl_rdata;
  int npart = pmbp->ppart->nprtcl_thispack;
  Real rbuf = sink_refine_buf_;

  par_for("SinkAMR", DevExeSpace(), 0, nmb-1, KOKKOS_LAMBDA(int m) {
    Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
    int flag = -1;  // derefine unless a sink is near
    for (int p = 0; p < npart; ++p) {
      Real px = pr(IPX, p), py = pr(IPY, p), pz = pr(IPZ, p);
      if (px > (x1min-rbuf) && px < (x1max+rbuf) &&
          py > (x2min-rbuf) && py < (x2max+rbuf) &&
          pz > (x3min-rbuf) && pz < (x3max+rbuf)) {
        flag = 1;
      }
    }
    refine_flag.d_view(m+mbs) = flag;
  });
  // We wrote refine_flag on the device, so mark the device copy modified AND sync it back
  // to the host: CheckForRefinement() reads refine_flag.h_view after the criteria loop and
  // then calls modify<HostMemSpace>(). Leaving the device-modified flag pending here would
  // trip Kokkos' "concurrent modification" abort on GPU (silent on CPU, where host==device).
  // This matches the contract of the built-in criteria (see refinement_criteria.cpp).
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}
}  // namespace
