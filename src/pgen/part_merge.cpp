//========================================================================================
// AthenaXXX astrophysical plasma code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file part_merge.cpp
//! \brief Validation test for sink-particle merging (particles_merger.cpp).
//!
//! Two sinks of (generally unequal) mass are placed a separation apart on the x-axis,
//! given a common bulk drift plus an optional mutual approach velocity, and evolved
//! under multigrid self-gravity with inert gas. The mutual attraction (and/or the
//! approach velocity) brings their 27-cell control volumes into overlap, at which point
//! <particles>/merging combines them into a single sink at the centre of mass with the
//! momentum-conserving mean velocity.
//!
//! Diagnostics (rank 0): the initial total mass and momentum are printed at setup; the
//! finalize hook prints the final particle state, total mass/momentum, and the expected
//! post-merge mass and COM velocity. Merging conserves mass and linear momentum exactly,
//! so with the isolated (internal-force-only) two-body dynamics the final totals must
//! match the initial ones to machine precision, and npart must drop from 2 to 1.
//!
//! Unequal masses make the test non-trivial: it exercises the mass-weighted COM, the
//! momentum-conserving velocity, and survivor selection (the more massive sink survives
//! and keeps its tag).

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
void MergeFinalize(ParameterInput *pin, Mesh *pm);
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  pgen_final_func = &MergeFinalize;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->phydro == nullptr || pmbp->pgrav == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_merge requires <particles>, <hydro> and <gravity> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // parameters
  const Real m0 = pin->GetOrAddReal("problem", "mass0", 1.0);
  const Real m1 = pin->GetOrAddReal("problem", "mass1", 2.0);
  const Real dsep = pin->GetOrAddReal("problem", "separation", 0.3);
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0e-8);
  // common bulk drift added to both sinks (gives the pair a nonzero total momentum)
  const Real ux = pin->GetOrAddReal("problem", "driftx", 0.0);
  const Real uy = pin->GetOrAddReal("problem", "drifty", 0.0);
  const Real uz = pin->GetOrAddReal("problem", "driftz", 0.05);
  // mutual approach speed along the separation axis (sink0 moves +n^, sink1 moves -n^)
  const Real vapp = pin->GetOrAddReal("problem", "approach", 0.0);
  // impact parameter: offset the two sinks by +-impact/2 along a direction perpendicular
  // to the separation axis, so a fast approach is a GRAZING flyby (finite closest
  // approach) -- the case the boundedness gate is meant to reject.
  const Real bimp = pin->GetOrAddReal("problem", "impact", 0.0);
  // separation-axis direction (unit-normalised); default x-axis reproduces the head-on
  // test. Set e.g. (1,1,1) for a diagonal approach exercising all three dimensions.
  Real nx = pin->GetOrAddReal("problem", "sep_dirx", 1.0);
  Real ny = pin->GetOrAddReal("problem", "sep_diry", 0.0);
  Real nz = pin->GetOrAddReal("problem", "sep_dirz", 0.0);
  {
    Real nn = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (nn < 1.0e-12) { nx = 1.0; ny = 0.0; nz = 0.0; nn = 1.0; }
    nx /= nn; ny /= nn; nz /= nn;
  }
  // perpendicular direction for the impact offset: p^ = n^ x z^, falling back to y^ when
  // n^ is (anti)parallel to z^.
  Real px_ = ny, py_ = -nx, pz_ = 0.0;
  {
    Real pn = std::sqrt(px_*px_ + py_*py_ + pz_*pz_);
    if (pn < 1.0e-9) { px_ = 0.0; py_ = 1.0; pz_ = 0.0; pn = 1.0; }
    px_ /= pn; py_ /= pn; pz_ /= pn;
  }

  // number of sinks (default 2). nsink>2 places them equally spaced by `separation` along
  // sep_dir, with masses (k+1)*mass0 (so all distinct) -- for multi-way / chain merges.
  const int MAXSINK = 32;
  int nsink = pin->GetOrAddInteger("problem", "nsink", 2);
  if (nsink < 1) nsink = 1;
  if (nsink > MAXSINK) nsink = MAXSINK;
  // rigid offset of the whole group from the box centre (place a cluster inside one rank,
  // or straddle a rank boundary at x=0 etc.)
  const Real offx = pin->GetOrAddReal("problem", "offx", 0.0);
  const Real offy = pin->GetOrAddReal("problem", "offy", 0.0);
  const Real offz = pin->GetOrAddReal("problem", "offz", 0.0);

  const Real cx = 0.5*(pmy_mesh_->mesh_size.x1min + pmy_mesh_->mesh_size.x1max) + offx;
  const Real cy = 0.5*(pmy_mesh_->mesh_size.x2min + pmy_mesh_->mesh_size.x2max) + offy;
  const Real cz = 0.5*(pmy_mesh_->mesh_size.x3min + pmy_mesh_->mesh_size.x3max) + offz;

  // sink k at cen + (k-(nsink-1)/2)*d*n^ (+ impact perp offset, 2-sink only); moving toward
  // the group centre by vapp along n^, plus the common drift.
  Real spos[MAXSINK][3], svel[MAXSINK][3], seedm[MAXSINK];
  const Real drift[3] = {ux, uy, uz}, ndir[3] = {nx, ny, nz}, pdir[3] = {px_, py_, pz_};
  const Real cen[3] = {cx, cy, cz};
  const Real kmid = 0.5*(nsink - 1);
  for (int s = 0; s < nsink; ++s) {
    const Real off = static_cast<Real>(s) - kmid;      // signed offset index (± about centre)
    const Real sgn = (off > 0) - (off < 0);            // -1 / 0 / +1
    for (int c = 0; c < 3; ++c) {
      spos[s][c] = cen[c] + off*dsep*ndir[c];
      svel[s][c] = drift[c] - sgn*vapp*ndir[c];
    }
    seedm[s] = (nsink == 2) ? (s == 0 ? m0 : m1) : m0*static_cast<Real>(s + 1);
  }
  if (nsink == 2) {   // impact (grazing) offset only meaningful for a pair
    for (int c = 0; c < 3; ++c) {
      spos[0][c] -= 0.5*bimp*pdir[c];
      spos[1][c] += 0.5*bimp*pdir[c];
    }
  }

  if (global_variable::my_rank == 0) {
    Real Mtot = 0.0, Px = 0.0, Py = 0.0, Pz = 0.0;
    for (int s = 0; s < nsink; ++s) {
      Mtot += seedm[s];
      Px += seedm[s]*svel[s][0]; Py += seedm[s]*svel[s][1]; Pz += seedm[s]*svel[s][2];
    }
    std::printf("part_merge: nsink=%d d=%.3e sep_dir=(%.3f,%.3f,%.3f) "
                "drift=(%.3e,%.3e,%.3e) approach=%.3e off=(%.3f,%.3f,%.3f)\n",
                nsink, dsep, nx, ny, nz, ux, uy, uz, vapp, offx, offy, offz);
    std::printf("  INITIAL: Mtot=%.10e  Ptot=(% .10e,% .10e,% .10e)\n", Mtot, Px, Py, Pz);
    std::printf("  EXPECT post-merge: M=%.10e  v_com=(% .6e,% .6e,% .6e)\n",
                Mtot, Px/Mtot, Py/Mtot, Pz/Mtot);
  }

  // uniform, negligible, gravity-inert gas (isothermal EOS -> no energy variable)
  auto &u0 = pmbp->phydro->u0;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks,
      ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  par_for("merge_gas", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
  });

  // seed the sinks, each on the rank/block that contains it (count-robust, as in
  // part_orbit): resize the particle arrays to the locally-owned count.
  auto gids = pmbp->gids;
  auto &mbsz = pmbp->pmb->mb_size;
  int mown_h[MAXSINK], nloc = 0;
  for (int s = 0; s < nsink; ++s) {
    mown_h[s] = -1;
    for (int mm = 0; mm < nmb; ++mm) {
      if (spos[s][0] >= mbsz.h_view(mm).x1min && spos[s][0] < mbsz.h_view(mm).x1max &&
          spos[s][1] >= mbsz.h_view(mm).x2min && spos[s][1] < mbsz.h_view(mm).x2max &&
          spos[s][2] >= mbsz.h_view(mm).x3min && spos[s][2] < mbsz.h_view(mm).x3max) {
        mown_h[s] = mm; break;
      }
    }
    if (mown_h[s] >= 0) nloc++;
  }
  Kokkos::resize(pmbp->ppart->prtcl_rdata, pmbp->ppart->nrdata, nloc);
  Kokkos::resize(pmbp->ppart->prtcl_idata, pmbp->ppart->nidata, nloc);
  pmbp->ppart->nprtcl_thispack = nloc;
  pmbp->ppart->RefreshMeshParticleCounts();
  if (nloc > 0) {
    auto pr = pmbp->ppart->prtcl_rdata;
    auto pi = pmbp->ppart->prtcl_idata;
    const bool has_prev = (pmbp->ppart->nrdata > IPX0);
    int slot = 0;
    for (int s = 0; s < nsink; ++s) {
      if (mown_h[s] < 0) continue;
      const Real qx = spos[s][0], qy = spos[s][1], qz = spos[s][2];
      const Real qvx = svel[s][0], qvy = svel[s][1], qvz = svel[s][2], pm = seedm[s];
      const int gidown = gids + mown_h[s];
      const int tag = s, sp = slot;
      par_for("merge_parts", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(int) {
        pr(IPX, sp) = qx;  pr(IPY, sp) = qy;  pr(IPZ, sp) = qz;
        pr(IPVX, sp) = qvx; pr(IPVY, sp) = qvy; pr(IPVZ, sp) = qvz;
        pr(IPM, sp) = pm;
        pr(IPGX, sp) = 0.0; pr(IPGY, sp) = 0.0; pr(IPGZ, sp) = 0.0;
        if (has_prev) { pr(IPX0, sp) = qx; pr(IPY0, sp) = qy; pr(IPZ0, sp) = qz; }
        pi(PGID, sp) = gidown;
        pi(PTAG, sp) = tag;
      });
      slot++;
    }
  }

  // small constant particle timestep floor (NewTimeStep refines it each cycle)
  Real dxmin = std::min({mbsz.h_view(0).dx1, mbsz.h_view(0).dx2, mbsz.h_view(0).dx3});
  Real vmax = std::max({std::fabs(vapp), std::fabs(ux), std::fabs(uy), std::fabs(uz),
                        1.0e-3});
  pmbp->ppart->dtnew = 0.1*dxmin/vmax;
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void MergeFinalize()
//! \brief print final particle state + conserved diagnostics after the run.

void MergeFinalize(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->ppart == nullptr) return;

  // sum this rank's sink mass/momentum/count, then reduce across ranks: under MPI the
  // surviving sink can be on any rank, so a rank-0-only readout would miss it.
  int npart = pmbp->ppart->nprtcl_thispack;
  auto pr_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_rdata);
  Kokkos::deep_copy(pr_h, pmbp->ppart->prtcl_rdata);
  Real loc[8] = {0,0,0,0,0,0,0,0};   // Mtot, m*x,y,z, m*vx,vy,vz, npart
  for (int p = 0; p < npart; ++p) {
    Real m = pr_h(IPM, p);
    loc[0] += m;
    loc[1] += m*pr_h(IPX, p);  loc[2] += m*pr_h(IPY, p);  loc[3] += m*pr_h(IPZ, p);
    loc[4] += m*pr_h(IPVX, p); loc[5] += m*pr_h(IPVY, p); loc[6] += m*pr_h(IPVZ, p);
  }
  loc[7] = static_cast<Real>(npart);
  Real glb[8];
  for (int i = 0; i < 8; ++i) glb[i] = loc[i];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(loc, glb, 8, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank != 0) return;
  const Real Mtot = glb[0];
  Real comx = 0, comy = 0, comz = 0;
  if (Mtot > 0.0) { comx = glb[1]/Mtot; comy = glb[2]/Mtot; comz = glb[3]/Mtot; }
  std::printf("\n=== part_merge finalize: t=%.6f  ncycle=%d  npart=%d ===\n",
              pm->time, pm->ncycle, static_cast<int>(glb[7] + 0.5));
  std::printf("  FINAL:   Mtot=%.10e  Ptot=(% .10e,% .10e,% .10e)\n",
              Mtot, glb[4], glb[5], glb[6]);
  std::printf("  COM=(% .6e,% .6e,% .6e)   (%d sink%s remain)\n",
              comx, comy, comz, static_cast<int>(glb[7] + 0.5),
              static_cast<int>(glb[7] + 0.5) == 1 ? "" : "s");
}
}  // namespace
