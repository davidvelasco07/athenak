//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_merger.cpp
//! \brief sink-particle merging on overlapping control volumes.
//!
//! Sinks are merged into one when their 27-cell control volumes (the "halos" of Moon &
//! Ostriker 2025 / Gong & Ostriker 2013) overlap and -- unless disabled -- the pair is
//! gravitationally bound. The control volume is the cube of (2*rctrl+1)^3 cells centred
//! on the sink cell (rctrl=1 => 3x3x3 = 27 cells), i.e. a cube of half-width
//! (rctrl+0.5)*dx per axis. Two such cubes overlap when their centres are closer than the
//! sum of the half-widths on EVERY axis (axis-aligned box overlap): the faithful "the
//! halos overlap" criterion. Merging removes the overlapping control volumes that
//! AccreteMass cannot otherwise handle, so this runs create -> MERGE -> accrete.
//!
//! A merged group conserves mass and linear momentum exactly and sits at the centre of
//! mass:  M = sum m_a ;  v = (sum m_a v_a)/M ;  x = (sum m_a x_a)/M.  The relative orbital
//! angular momentum about the COM is absorbed by the (unresolved) merged object. The MOST
//! MASSIVE sink survives (ties -> lowest PTAG) and keeps its tag; the others are removed.
//! Chained/simultaneous overlaps are resolved with union-find, so 3+ sinks merge at once.
//!
//! MPI (cross-rank): a MeshBlockPack holds only this rank's particles, so a pair whose
//! halos overlap while owned by DIFFERENT ranks is invisible to a purely local scan. To
//! handle it, every rank Allgathers ALL sinks (position, velocity, mass, per-sink dx,
//! tag) each step, runs the SAME global union-find + conservative reduction (deterministic
//! -> identical on every rank), then mutates only its LOCAL particles, keyed by tag:
//!   * a local particle whose tag is a merge survivor  -> updated to the merged state
//!     (its PGID re-derived from the COM; if the COM left this rank the migration chain
//!      moves it next cycle);
//!   * a local particle whose tag was absorbed          -> removed (array compacted);
//!   * any other local particle                         -> untouched.
//! The survivor lives on the rank that owned the most-massive member, so the merged
//! particle ends up on exactly one of the interacting ranks. Because the group decision is
//! global and the reduction sums in ascending-PTAG order, the result is
//! decomposition-invariant, and on a single rank it is bit-identical to the local path
//! (the same in-place update + compaction, keyed by tag instead of index).
//!
//! Sink counts are few, so the O(N^2) global test and the per-step Allgather are cheap;
//! the collective RefreshMeshParticleCounts is reached on every post-gate exit path so all
//! ranks participate (a merge changes the global particle total).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "gravity/gravity.hpp"
#include "particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {

namespace {
// Union-find with path halving (host, tiny N).
int uf_find(std::vector<int> &parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}
void uf_union(std::vector<int> &parent, int a, int b) {
  a = uf_find(parent, a);
  b = uf_find(parent, b);
  if (a != b) parent[b] = a;
}
constexpr int REC = 11;   // per-sink record: x,y,z,vx,vy,vz,m,dx1,dx2,dx3,tag
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::MergeSinks
//! \brief merge sinks whose control volumes overlap (last RK stage only).

TaskStatus Particles::MergeSinks(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;
  if (particle_type != ParticleType::sink || !merging) return TaskStatus::complete;

  const int nranks = global_variable::nranks;
  const int my_rank = global_variable::my_rank;

  // boundedness constant G = four_pi_G/(4 pi); fall back to overlap-only (one-time
  // warning) if merge_bound is requested without a gravity module.
  bool do_bound = merge_bound;
  Real Ggrav = 0.0;
  if (do_bound) {
    if (pmy_pack->pgrav != nullptr) {
      Ggrav = pmy_pack->pgrav->four_pi_G/(4.0*M_PI);
    } else {
      static bool warned = false;
      if (!warned && my_rank == 0) {
        std::cout << "### WARNING in Particles::MergeSinks: merge_bound=true but no "
                  << "<gravity> module; merging on halo overlap alone." << std::endl;
        warned = true;
      }
      do_bound = false;
    }
  }

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto msz_h = mbsize.h_view;
  const int gids = pmy_pack->gids;
  const int nmb = pmy_pack->nmb_thispack;
  const int rctrl = 1;
  const Real hfac = static_cast<Real>(rctrl) + 0.5;   // control-volume half-width in dx
  const bool per = pmy_pack->pmesh->strictly_periodic;
  auto &msz = pmy_pack->pmesh->mesh_size;
  const Real L[3] = {msz.x1max-msz.x1min, msz.x2max-msz.x2min, msz.x3max-msz.x3min};
  const Real dlo[3] = {msz.x1min, msz.x2min, msz.x3min};

  auto refresh_return = [&]() { RefreshMeshParticleCounts(); return TaskStatus::complete; };

  // ---- host mirror of this rank's particles + local records ----
  const int nloc = nprtcl_thispack;
  auto hr = Kokkos::create_mirror(prtcl_rdata);
  auto hi = Kokkos::create_mirror(prtcl_idata);
  Kokkos::deep_copy(hr, prtcl_rdata);
  Kokkos::deep_copy(hi, prtcl_idata);
  std::vector<Real> loc(static_cast<size_t>(nloc)*REC);
  for (int p = 0; p < nloc; ++p) {
    int m = hi(PGID, p) - gids;
    Real dx1 = msz_h(0).dx1, dx2 = msz_h(0).dx2, dx3 = msz_h(0).dx3;
    if (m >= 0 && m < nmb) { dx1 = msz_h(m).dx1; dx2 = msz_h(m).dx2; dx3 = msz_h(m).dx3; }
    Real *rec = &loc[static_cast<size_t>(p)*REC];
    rec[0]=hr(IPX,p);  rec[1]=hr(IPY,p);  rec[2]=hr(IPZ,p);
    rec[3]=hr(IPVX,p); rec[4]=hr(IPVY,p); rec[5]=hr(IPVZ,p);
    rec[6]=hr(IPM,p);  rec[7]=dx1; rec[8]=dx2; rec[9]=dx3;
    rec[10]=static_cast<Real>(hi(PTAG,p));
  }

  // ---- gather every rank's sinks (cheap; sinks are few) ----
  std::vector<Real> G;
  int Ntot = nloc;
#if MPI_PARALLEL_ENABLED
  if (nranks > 1) {
    std::vector<int> cnt(nranks), cntR(nranks), dspR(nranks);
    MPI_Allgather(&nloc, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int off = 0; Ntot = 0;
    for (int r = 0; r < nranks; ++r) {
      cntR[r] = cnt[r]*REC; dspR[r] = off; off += cntR[r]; Ntot += cnt[r];
    }
    G.resize(static_cast<size_t>(Ntot)*REC);
    MPI_Allgatherv(loc.data(), nloc*REC, MPI_ATHENA_REAL, G.data(),
                   cntR.data(), dspR.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD);
  } else {
    G = std::move(loc);
  }
#else
  G = std::move(loc);
#endif
  if (Ntot < 2) return refresh_return();

  // ---- PTAGs must be globally unique --------------------------------------------------
  // Everything below keys survivors and absorbed sinks by tag, so two distinct sinks
  // sharing a tag are silently taken for one object: they get "merged" however far apart
  // they are, their masses are summed, and every rank writes the sum onto its own copy --
  // a mass that doubles every cycle. Creation numbers tags from a global base to prevent
  // this, but check it here too, where the assumption is actually relied on, and where a
  // violation can be reported against the offending tag instead of being inferred later
  // from a runaway mass. Cheap: a sort over the (few) sinks, no communication, and G is
  // identical on every rank so the verdict is too.
  {
    std::vector<int> tags(Ntot);
    for (int i = 0; i < Ntot; ++i) {
      tags[i] = static_cast<int>(G[static_cast<size_t>(i)*REC + 10]);
    }
    std::vector<int> sorted_tags = tags;
    std::sort(sorted_tags.begin(), sorted_tags.end());
    auto dup = std::adjacent_find(sorted_tags.begin(), sorted_tags.end());
    if (dup != sorted_tags.end()) {
      if (my_rank == 0) {
        std::cout << "### FATAL ERROR in Particles::MergeSinks: PTAG " << *dup
                  << " is used by more than one sink (of " << Ntot << " globally). Sink "
                  << "tags must be unique across ALL ranks -- merging keys on them, so "
                  << "duplicates fuse unrelated sinks and their mass diverges."
                  << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
  }

  // ---- global union-find over all sinks (identical on every rank) ----
  auto GX = [&](int i, int c) -> Real { return G[static_cast<size_t>(i)*REC + c]; };
  auto mimg = [&](Real d, int c) { return per ? d - L[c]*std::floor(d/L[c] + 0.5) : d; };
  std::vector<int> parent(Ntot);
  for (int i = 0; i < Ntot; ++i) parent[i] = i;
  for (int a = 0; a < Ntot; ++a) {
    for (int b = a+1; b < Ntot; ++b) {
      Real sx = mimg(GX(a,0)-GX(b,0), 0), sy = mimg(GX(a,1)-GX(b,1), 1),
           sz = mimg(GX(a,2)-GX(b,2), 2);
      Real tx = hfac*(GX(a,7)+GX(b,7)), ty = hfac*(GX(a,8)+GX(b,8)),
           tz = hfac*(GX(a,9)+GX(b,9));
      if (std::fabs(sx) >= tx || std::fabs(sy) >= ty || std::fabs(sz) >= tz) continue;
      if (do_bound) {
        Real dvx=GX(a,3)-GX(b,3), dvy=GX(a,4)-GX(b,4), dvz=GX(a,5)-GX(b,5);
        Real r = std::sqrt(sx*sx+sy*sy+sz*sz), M = GX(a,6)+GX(b,6);
        if (!(0.5*(dvx*dvx+dvy*dvy+dvz*dvz)*r < Ggrav*M)) continue;
      }
      uf_union(parent, a, b);
    }
  }
  std::vector<std::vector<int>> groups(Ntot);
  for (int i = 0; i < Ntot; ++i) groups[uf_find(parent, i)].push_back(i);
  bool any = false;
  for (int r = 0; r < Ntot && !any; ++r) any = (groups[r].size() > 1);
  if (!any) return refresh_return();

  // ---- reduce merged groups: survivor tag + conserved (M, COM, v) ----
  struct Surv { Real x, y, z, vx, vy, vz, M; };
  std::map<int, Surv> survmap;    // survivor tag -> merged state
  std::set<int> absorbed;         // tags removed by a merge
  for (int root = 0; root < Ntot; ++root) {
    auto &mem = groups[root];
    if (mem.size() < 2) continue;
    std::sort(mem.begin(), mem.end(), [&](int a, int b){ return GX(a,10) < GX(b,10); });
    const int ref = mem[0];
    const Real xr = GX(ref,0), yr = GX(ref,1), zr = GX(ref,2);
    Real M=0, sx=0, sy=0, sz=0, svx=0, svy=0, svz=0, mmax=-1.0;
    int survtag = static_cast<int>(GX(ref,10));
    for (int i : mem) {
      const Real m = GX(i,6);
      const Real ox = mimg(GX(i,0)-xr,0), oy = mimg(GX(i,1)-yr,1), oz = mimg(GX(i,2)-zr,2);
      M += m; sx += m*ox; sy += m*oy; sz += m*oz;
      svx += m*GX(i,3); svy += m*GX(i,4); svz += m*GX(i,5);
      if (m > mmax) { mmax = m; survtag = static_cast<int>(GX(i,10)); }
    }
    if (!(M > 0.0)) continue;
    Real xc = xr + sx/M, yc = yr + sy/M, zc = zr + sz/M;
    if (per) {
      xc -= L[0]*std::floor((xc-dlo[0])/L[0]);
      yc -= L[1]*std::floor((yc-dlo[1])/L[1]);
      zc -= L[2]*std::floor((zc-dlo[2])/L[2]);
    }
    survmap[survtag] = {xc, yc, zc, svx/M, svy/M, svz/M, M};
    for (int i : mem) {
      const int tg = static_cast<int>(GX(i,10));
      if (tg != survtag) absorbed.insert(tg);
    }
    if (my_rank == 0) {
      std::printf("MergeSinks: %d sinks -> tag %d (M=%.6e) at (%.5f, %.5f, %.5f) cycle=%d\n",
                  static_cast<int>(mem.size()), survtag, M, xc, yc, zc,
                  pmy_pack->pmesh->ncycle);
    }
  }

  // ---- pre-merge totals, measured from the ACTUAL particle arrays ----------------
  // NOT from the gathered list G. Summing G before and after the group reduction only
  // proves the reduction arithmetic is self-consistent: if G itself is wrong -- e.g. it
  // lists one physical sink twice because two ranks assigned the same PTAG -- then both
  // sums double-count it identically and the check reports dM=0 while the sink's mass
  // doubles every cycle. That is exactly what happened before tags were made globally
  // unique: 230 cycles of "dM=0.000e+00" while the mass ran to 1e72. So measure the real
  // state instead, and compare it after the arrays have been mutated and compacted.
  // Safe under MPI: every rank derives survmap from the same G, so all ranks take this
  // branch together and the reductions below cannot deadlock.
  Real m_pre_loc = 0.0, p_pre_loc[3] = {0.0, 0.0, 0.0};
  const bool check_cons = !survmap.empty();
  if (check_cons) {
    for (int p = 0; p < nloc; ++p) {
      const Real m = hr(IPM, p);
      m_pre_loc += m;
      p_pre_loc[0] += m*hr(IPVX, p);
      p_pre_loc[1] += m*hr(IPVY, p);
      p_pre_loc[2] += m*hr(IPVZ, p);
    }
  }

  // ---- mutate this rank's local particles, keyed by tag ----
  std::vector<char> dead(nloc, 0);
  for (int p = 0; p < nloc; ++p) {
    const int tag = hi(PTAG, p);
    if (absorbed.count(tag)) { dead[p] = 1; continue; }
    auto it = survmap.find(tag);
    if (it == survmap.end()) continue;          // not merged -> untouched
    const Surv &s = it->second;
    hr(IPX,p)=s.x;   hr(IPY,p)=s.y;   hr(IPZ,p)=s.z;
    hr(IPVX,p)=s.vx; hr(IPVY,p)=s.vy; hr(IPVZ,p)=s.vz;
    hr(IPM,p)=s.M;
    // treat the merge as NOT a cell crossing so the following AccreteMass does not
    // re-accrete a spurious old control volume
    hr(IPX0,p)=s.x;  hr(IPY0,p)=s.y;  hr(IPZ0,p)=s.z;
    // re-derive the owning block from the COM (lower-inclusive). If the COM left this
    // rank, keep the old PGID; the migration chain relocates the sink next cycle.
    for (int mm = 0; mm < nmb; ++mm) {
      if (s.x >= msz_h(mm).x1min && s.x < msz_h(mm).x1max &&
          s.y >= msz_h(mm).x2min && s.y < msz_h(mm).x2max &&
          s.z >= msz_h(mm).x3min && s.z < msz_h(mm).x3max) {
        hi(PGID,p) = gids + mm; break;
      }
    }
  }

  // ---- compact out absorbed slots (order-preserving) ----
  std::vector<int> keep;
  keep.reserve(nloc);
  for (int p = 0; p < nloc; ++p) if (!dead[p]) keep.push_back(p);
  const int nnew = static_cast<int>(keep.size());
  Kokkos::resize(prtcl_rdata, nrdata, nnew);
  Kokkos::resize(prtcl_idata, nidata, nnew);
  auto hr2 = Kokkos::create_mirror(prtcl_rdata);
  auto hi2 = Kokkos::create_mirror(prtcl_idata);
  for (int n = 0; n < nnew; ++n) {
    const int src = keep[n];
    for (int v = 0; v < nrdata; ++v) hr2(v, n) = hr(v, src);
    for (int v = 0; v < nidata; ++v) hi2(v, n) = hi(v, src);
  }
  Kokkos::deep_copy(prtcl_rdata, hr2);
  Kokkos::deep_copy(prtcl_idata, hi2);
  nprtcl_thispack = nnew;

  // ---- post-merge totals from the ACTUAL arrays, and the verdict -------------------
  // A merge moves mass and momentum between particles, so both totals must be unchanged.
  // Reported as a relative error and escalated to a WARNING past a tolerance far above
  // round-off, so a real violation is impossible to read as noise.
  if (check_cons) {
    Real m_post_loc = 0.0, p_post_loc[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n < nnew; ++n) {
      const Real m = hr2(IPM, n);
      m_post_loc += m;
      p_post_loc[0] += m*hr2(IPVX, n);
      p_post_loc[1] += m*hr2(IPVY, n);
      p_post_loc[2] += m*hr2(IPVZ, n);
    }
    Real loc[8] = {m_pre_loc, p_pre_loc[0], p_pre_loc[1], p_pre_loc[2],
                   m_post_loc, p_post_loc[0], p_post_loc[1], p_post_loc[2]};
    Real glb[8];
    for (int c = 0; c < 8; ++c) glb[c] = loc[c];
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      MPI_Allreduce(loc, glb, 8, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    const Real dM = glb[4] - glb[0];
    const Real dP[3] = {glb[5]-glb[1], glb[6]-glb[2], glb[7]-glb[3]};
    const Real mscale = std::max(std::abs(glb[0]), 1.0e-300);
    const Real pscale = std::max({std::abs(glb[1]), std::abs(glb[2]),
                                  std::abs(glb[3]), 1.0e-300});
    const Real relM = std::abs(dM)/mscale;
    const Real relP = std::max({std::abs(dP[0]), std::abs(dP[1]),
                                std::abs(dP[2])})/pscale;
    if (my_rank == 0) {
      std::printf("  merge conservation (measured): dM/M=% .3e  dP/P=% .3e"
                  "  (M %.6e -> %.6e)\n", relM, relP, glb[0], glb[4]);
      if (relM > 1.0e-10 || relP > 1.0e-8) {
        std::cout << "### WARNING in Particles::MergeSinks: the merge did NOT conserve "
                  << "sink mass/momentum (dM/M=" << relM << ", dP/P=" << relP
                  << "). Check for duplicate PTAGs across ranks." << std::endl;
      }
    }
  }
  return refresh_return();
}

}  // namespace particles
