//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_accretion.cpp
//! \brief control-volume sink-particle accretion (Gong & Ostriker 2013; Moon & Ostriker
//! 2025, ApJ 10.3847/1538-4357/add477).
//!
//! Port of the reference implementation in Chang-Goo Kim's Athena++ fork
//! (athena/src/particles/sink_particles.cpp). The 27 cells centred on the sink act as
//! internal ghost zones. Once per step (last RK stage), for each sink:
//!   1(a) integrate conserved gas (rho, momenta) over the control volume -> M^{n+1}
//!   1(b) reset the control volume by outflow extrapolation from the cells just outside
//!        it (6 faces copy, 12 edges avg-of-2, 8 corners avg-of-3, centre avg of the 6
//!        reset faces)
//!   1(c) re-integrate -> M^{n+1}_ctrl
//!   1(d) dM_sink = M^{n+1} - M^{n+1}_ctrl  (exactly the flux that entered the volume)
//!   2    if the sink moved to a new cell this step, repeat 1(a)-(d) on the OLD control
//!        volume (centred on the start-of-step position IPX0/IPY0/IPZ0) and add its
//!        difference -- conserves mass/momentum through cell crossings (the overlap
//!        region is deliberately re-reset; see reference comments).
//!   3    dump conservatively onto the sink: v <- (m v + dM)/(m + dm), m <- m + dm.
//!
//! Kernel structure: ONE team, serial loop over sinks, team-parallel inner loops. The
//! reference semantics are strictly sequential across sinks (a later sink's integral
//! must see an earlier sink's reset, or overlapping control volumes double-accrete), so
//! sinks are processed in order; each 27-cell phase is team-parallel. Reset values are
//! computed into team scratch from the pre-reset u0 (shell cells read only outside-CV
//! cells; the centre reads the 6 scratch faces), which makes the re-integration 1(c)
//! free (sum of scratch) and removes all write-ordering hazards.
//!
//! CROSS-BLOCK control volumes: all reads (control volume + extrapolation stencil,
//! <= rctrl+1 cells from the sink cell) come from the OWNER block's array -- valid in
//! its ghost zones, which hold post-update neighbour interiors at "after_stagen". The
//! reset values are then GEOMETRICALLY SCATTERED to every coincident cell copy in the
//! pack: the owner's cells (interior or ghost), and for each same-level on-rank
//! neighbour the cell of its (interior+ghost) array containing the same physical
//! position (periodic-wrapped) -- the containment pattern of
//! ParticleMesh::FlushDepositBoundaries. Writing ALL copies keeps every block
//! bit-consistent immediately, with no extra boundary exchange (this replaces the
//! Athena++ ghost-particle mechanism and its NGHOST >= req+rctrl+1 requirement).
//! Duplicate writes carry identical values, so no atomics are needed.
//!
//! CROSS-RANK control volumes: reset cells whose reach touches an off-rank same-level
//! neighbour are additionally staged into a device buffer and exchanged after the
//! kernel (ExchangeCVReset): counts negotiated with MPI_Alltoall, then per-peer
//! Isend/Irecv of (position, rho, M1..3) records on a dedicated communicator. Each
//! receiver applies the SAME geometric scatter locally -- every local block whose
//! interior+ghost array contains the (wrapped) position gets the value, interior and
//! ghost copies alike, so multi-rank runs stay bit-consistent with single-rank ones.
//! The sink cell itself is binned in the GLOBAL mesh frame (not the owner block's
//! frame): block-frame binning resolves a position within round-off of a block edge
//! differently in different blocks' frames (catastrophic cancellation), which breaks
//! decomposition invariance.
//!
//! Sinks are SKIPPED (no accretion, warning-free) only when the region they would
//! touch overlaps a different-level (AMR) neighbour -- the reference also punts on
//! AMR; the sink-region AMR refinement halo (see be_sink.cpp) keeps real runs clear
//! of this case.
//!
//! After scattering the reset conserved values, the matching PRIMITIVES are written to
//! w0 for the same cells (isothermal: w_d = u_d, w_vi = M_i/rho). In AthenaK,
//! Hydro/MHD::ConToPrim runs in "stagen" BEFORE this task ("after_stagen"), so without
//! this the reset would be invisible to the next cycle's flux computation. (Athena++
//! orders CONS2PRIM after the particle interaction for exactly this reason.)
//!
//! LIMITATIONS (documented scope): isothermal only (resets IDN+IM1..3; no energy/B);
//! same-level neighbourhoods only (sink near an AMR coarse-fine boundary skips
//! accretion). Cross-rank particle MIGRATION is separate machinery: a sink whose
//! position leaves this rank's blocks keeps a stale PGID (SetGIDFromPosition is
//! on-rank only); accretion stays correct while the sink remains within the ghost
//! depth of its stale owner, i.e. for slowly-moving sinks near a rank boundary.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "particles.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::AccreteMass
//! \brief accrete gas in the control volume onto each sink particle (last RK stage only).

TaskStatus Particles::AccreteMass(Driver *pdriver, int stage) {
  // operator split: accrete once per step, at the final stage, after the hydro update
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;
  if (particle_type != ParticleType::sink) return TaskStatus::complete;
  if (!(pmy_pack->pmesh->three_d)) return TaskStatus::complete;
  // gas conserved and primitive variables
  if (pmy_pack->phydro == nullptr && pmy_pack->pmhd == nullptr) {
    return TaskStatus::complete;
  }
  const bool is_mhd = (pmy_pack->pmhd != nullptr);
  auto u0 = is_mhd ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;
  auto w0 = is_mhd ? pmy_pack->pmhd->w0 : pmy_pack->phydro->w0;

  // cross-rank control-volume reset: when running on >1 rank, sinks whose CV reaches an
  // off-rank neighbour stage those reset cells into cvemit_ for ExchangeCVReset() (below).
  // Grow the staging buffer with the sink count and zero its counter each step.
  const bool mpi_on = (global_variable::nranks > 1);
  if (mpi_on) {
    const int need = std::max(1, nprtcl_thispack)*64;
    if (need > cvemit_max_) {
      cvemit_max_ = need;
      Kokkos::realloc(cvemit_, cvemit_max_, 8);
    }
    Kokkos::deep_copy(cvemit_cnt_, 0);
  }
  auto cvemit = cvemit_;
  auto cvcnt = cvemit_cnt_;
  const int cvmax = cvemit_max_;

  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;
  const int gids = pmy_pack->gids;
  const int npart = nprtcl_thispack;
  const int rctrl = 1;      // control volume = (2*rctrl+1)^3 = 27 cells (paper standard)
  const int nmb_ = pmy_pack->nmb_thispack;

  // neighbour tables for the geometric scatter (FlushDepositBoundaries pattern)
  const int my_rank = global_variable::my_rank;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &msz = pmy_pack->pmesh->mesh_size;
  const Real Lx1 = msz.x1max - msz.x1min;
  const Real Lx2 = msz.x2max - msz.x2min;
  const Real Lx3 = msz.x3max - msz.x3min;

  // team scratch: 27 reset values x 4 conserved vars, plus a 4-vector accumulator for
  // the per-sink accreted (dm, dM1, dM2, dM3)
  const int scr_level = 0;
  const size_t scr_bytes = ScrArray2D<Real>::shmem_size(4, 27)
                         + ScrArray1D<Real>::shmem_size(4);

  Kokkos::TeamPolicy<> policy(DevExeSpace(), 1, Kokkos::AUTO);
  policy.set_scratch_size(scr_level, Kokkos::PerTeam(scr_bytes));

  Kokkos::parallel_for("sink_accrete", policy,
  KOKKOS_LAMBDA(TeamMember_t tm) {
    ScrArray2D<Real> scr(tm.team_scratch(scr_level), 4, 27);
    ScrArray1D<Real> acc(tm.team_scratch(scr_level), 4);

    // serial over sinks (reference semantics; sinks are few), team-parallel inside
    for (int p = 0; p < npart; ++p) {
      const int m = pi(PGID, p) - gids;
      // Guard invalid PGID before ANY indexed access (mirrors ParticleMesh::DepositMass):
      // a corrupt/ejected particle must skip accretion, not fault.
      if (m < 0 || m >= nmb_) continue;
      bool touches_offrank = false;   // CV reaches an off-rank same-level neighbour
      const Real dx1 = mbsize.d_view(m).dx1;
      const Real dx2 = mbsize.d_view(m).dx2;
      const Real dx3 = mbsize.d_view(m).dx3;
      const Real dV = dx1*dx2*dx3;

      // cell containing the sink now and at the start of the step -- guard the
      // double->int conversions against non-finite/wild positions (UB otherwise),
      // and use floor (truncation is wrong for fractional offsets in (-1,0)).
      // BIN IN THE GLOBAL MESH FRAME, then convert to the owner-local index: block-frame
      // binning is frame-DEPENDENT for a position within round-off of a block boundary
      // (catastrophic cancellation resolves the same physical point to different cells
      // in different blocks' frames), which breaks decomposition invariance when ranks
      // hold different owner blocks. One global origin -> one answer on every rank.
      const Real xi1 = (pr(IPX, p) - msz.x1min)/dx1;
      const Real xi2 = (pr(IPY, p) - msz.x2min)/dx2;
      const Real xi3 = (pr(IPZ, p) - msz.x3min)/dx3;
      const Real yi1 = (pr(IPX0, p) - msz.x1min)/dx1;
      const Real yi2 = (pr(IPY0, p) - msz.x2min)/dx2;
      const Real yi3 = (pr(IPZ0, p) - msz.x3min)/dx3;
      if (!(xi1 > -1.0e9 && xi1 < 1.0e9) || !(xi2 > -1.0e9 && xi2 < 1.0e9) ||
          !(xi3 > -1.0e9 && xi3 < 1.0e9) || !(yi1 > -1.0e9 && yi1 < 1.0e9) ||
          !(yi2 > -1.0e9 && yi2 < 1.0e9) || !(yi3 > -1.0e9 && yi3 < 1.0e9)) {
        continue;
      }
      // owner block's integer cell offset from the mesh origin (exact by construction:
      // block edges lie on the global cell lattice)
      const int off1 = static_cast<int>((mbsize.d_view(m).x1min - msz.x1min)/dx1 + 0.5);
      const int off2 = static_cast<int>((mbsize.d_view(m).x2min - msz.x2min)/dx2 + 0.5);
      const int off3 = static_cast<int>((mbsize.d_view(m).x3min - msz.x3min)/dx3 + 0.5);
      const int ip  = static_cast<int>(Kokkos::floor(xi1)) - off1 + is;
      const int jp  = static_cast<int>(Kokkos::floor(xi2)) - off2 + js;
      const int kp  = static_cast<int>(Kokkos::floor(xi3)) - off3 + ks;
      const int ip0 = static_cast<int>(Kokkos::floor(yi1)) - off1 + is;
      const int jp0 = static_cast<int>(Kokkos::floor(yi2)) - off2 + js;
      const int kp0 = static_cast<int>(Kokkos::floor(yi3)) - off3 + ks;

      // ---- eligibility gate --------------------------------------------------------
      // The region this sink touches (new + old control volumes and their extrapolation
      // reads) spans at most rctrl+2.5 cells from the CURRENT sink position. Skip the
      // sink if that region overlaps a different-level (AMR) or off-rank neighbour:
      // reads through owner ghosts would mix restriction/prolongation there, and the
      // scatter could not reach an off-rank interior. (Computed redundantly by every
      // team thread; all take identical branches.)
      const int mylev = mblev.d_view(m);
      const Real xs = pr(IPX, p), ys = pr(IPY, p), zs = pr(IPZ, p);
      const Real h1 = (rctrl + 2.5)*dx1, h2 = (rctrl + 2.5)*dx2, h3 = (rctrl + 2.5)*dx3;
      bool eligible = true;
      for (int n = 0; n < nnghbr; ++n) {
        if (nghbr.d_view(m,n).gid < 0) continue;
        const bool same_rank = (nghbr.d_view(m,n).rank == my_rank);
        const bool same_lev  = (nghbr.d_view(m,n).lev == mylev);
        if (same_rank && same_lev) continue;
        // does the sink's reach overlap this neighbour's box (periodic-wrapped)?
        // neighbour boxes are only known on-rank; for off-rank neighbours use the
        // owner's own box shifted -- not available -- so fall back to a conservative
        // test: any not-same treatment within the 56-neighbour list whose OFFSET
        // direction the sink is close to. Distance of sink to the owner face in the
        // neighbour's direction:
        if (!same_rank) {
          // off-rank: conservative proximity test (neighbour box unknown on device)
          const bool near = (xs - mbsize.d_view(m).x1min < h1) ||
                            (mbsize.d_view(m).x1max - xs < h1) ||
                            (ys - mbsize.d_view(m).x2min < h2) ||
                            (mbsize.d_view(m).x2max - ys < h2) ||
                            (zs - mbsize.d_view(m).x3min < h3) ||
                            (mbsize.d_view(m).x3max - zs < h3);
          if (near) {
            // off-rank SAME level: reset cells are staged for the cross-rank exchange
            // (ExchangeCVReset). off-rank DIFFERENT level (AMR): still skip.
            if (same_lev) { touches_offrank = true; } else { eligible = false; break; }
          }
          continue;
        }
        // on-rank different level: precise overlap test against its box
        const int dnb = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        if (dnb < 0 || dnb >= nmb_) continue;
        const Real x1d = mbsize.d_view(dnb).x1min, x1u = mbsize.d_view(dnb).x1max;
        const Real x2d = mbsize.d_view(dnb).x2min, x2u = mbsize.d_view(dnb).x2max;
        const Real x3d = mbsize.d_view(dnb).x3min, x3u = mbsize.d_view(dnb).x3max;
        Real xc = xs, yc = ys, zc = zs, ctr;
        ctr = 0.5*(x1d + x1u);
        if (xc - ctr >  0.5*Lx1) xc -= Lx1;
        if (xc - ctr < -0.5*Lx1) xc += Lx1;
        ctr = 0.5*(x2d + x2u);
        if (yc - ctr >  0.5*Lx2) yc -= Lx2;
        if (yc - ctr < -0.5*Lx2) yc += Lx2;
        ctr = 0.5*(x3d + x3u);
        if (zc - ctr >  0.5*Lx3) zc -= Lx3;
        if (zc - ctr < -0.5*Lx3) zc += Lx3;
        if (xc + h1 > x1d && xc - h1 < x1u &&
            yc + h2 > x2d && yc - h2 < x2u &&
            zc + h3 > x3d && zc - h3 < x3u) {
          eligible = false; break;
        }
      }
      if (!eligible) continue;

      // zero the per-sink accumulator
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {
        for (int v = 0; v < 4; ++v) acc(v) = 0.0;
      });
      tm.team_barrier();

      // ---- process one control volume centred on (ic,jc,kc) ----------------------
      // (executed collectively by the whole team; all threads take identical branches)
      auto process_cv = [&](const int ic, const int jc, const int kc) {
        // hard array-bounds guard for the read stencil (reaches rctrl+1 from the CV
        // centre); with nghost >= rctrl+2 this passes for any validly binned sink,
        // including old-CV centres one cell into the ghost region after a crossing
        if (ic < rctrl+1 || ic > ncells1-rctrl-2 ||
            jc < rctrl+1 || jc > ncells2-rctrl-2 ||
            kc < rctrl+1 || kc > ncells3-rctrl-2) {
          return;
        }

        // Step (a): integrate conserved vars over the 27 cells (pre-reset), and
        // Step (b): compute the 26 shell reset values into scratch from pre-reset u0.
        // Unified extrapolation rule for a shell cell at offset (di,dj,dk), nnz = number
        // of non-zero offsets: value = (1/nnz) * sum over each non-zero axis of u0 at
        // the position with THAT axis offset doubled. This reproduces the reference:
        // faces (nnz=1) copy from one-further-out; edges (nnz=2) avg-of-2; corners
        // (nnz=3) avg-of-3. All reads are outside the control volume.
        Real s1[4];
        for (int v = 0; v < 4; ++v) {
          Real sum = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, 27),
          [&](const int c, Real &psum) {
            const int di = c%3 - 1, dj = (c/3)%3 - 1, dk = c/9 - 1;
            psum += u0(m, v, kc+dk, jc+dj, ic+di);
            const int nnz = (di != 0) + (dj != 0) + (dk != 0);
            if (nnz > 0) {
              Real val = 0.0;
              if (di != 0) val += u0(m, v, kc+dk,   jc+dj,   ic+2*di);
              if (dj != 0) val += u0(m, v, kc+dk,   jc+2*dj, ic+di);
              if (dk != 0) val += u0(m, v, kc+2*dk, jc+dj,   ic+di);
              scr(v, c) = val/nnz;
            }
          }, Kokkos::Sum<Real>(sum));
          s1[v] = sum;
        }
        tm.team_barrier();

        // centre cell: average of the 6 (scratch) face values; then Step (c)+(d):
        // M_ctrl = sum of scratch, accreted difference accumulated into acc
        Kokkos::single(Kokkos::PerTeam(tm), [&]() {
          for (int v = 0; v < 4; ++v) {
            scr(v, 13) = (scr(v, 12) + scr(v, 14) +    // (di=-1/+1, dj=dk=0)
                          scr(v, 10) + scr(v, 16) +    // (dj=-1/+1)
                          scr(v, 4)  + scr(v, 22))/6.0; // (dk=-1/+1)
            Real s2 = 0.0;
            for (int c = 0; c < 27; ++c) s2 += scr(v, c);
            acc(v) += (s1[v] - s2)*dV;
          }
        });
        tm.team_barrier();

        // Scatter the reset values to EVERY coincident cell copy in the pack: the
        // owner's cell (interior or ghost), plus, for each same-level on-rank
        // neighbour, the cell of its interior+ghost array containing the same
        // physical position (periodic-wrapped). Also write matching primitives to w0
        // (isothermal: w_d = u_d, w_vi = M_i/rho); without the w0 update the reset is
        // invisible to the next cycle's fluxes (ConToPrim already ran this stage).
        // Duplicate writes carry identical values -> no atomics needed.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, 27), [&](const int c) {
          const int di = c%3 - 1, dj = (c/3)%3 - 1, dk = c/9 - 1;
          const int i = ic + di, j = jc + dj, k = kc + dk;
          // owner copy
          for (int v = 0; v < 4; ++v) u0(m, v, k, j, i) = scr(v, c);
          const Real wd = scr(0, c);
          if (wd > 0.0) {
            w0(m, IDN, k, j, i) = wd;
            w0(m, IVX, k, j, i) = scr(1, c)/wd;
            w0(m, IVY, k, j, i) = scr(2, c)/wd;
            w0(m, IVZ, k, j, i) = scr(3, c)/wd;
          }
          // physical centre of this cell in the owner's frame (linear extension is
          // valid in the ghost region too)
          const Real xw = mbsize.d_view(m).x1min + (static_cast<Real>(i-is) + 0.5)*dx1;
          const Real yw = mbsize.d_view(m).x2min + (static_cast<Real>(j-js) + 0.5)*dx2;
          const Real zw = mbsize.d_view(m).x3min + (static_cast<Real>(k-ks) + 0.5)*dx3;
          // every coincident copy in same-level on-rank neighbours
          for (int n = 0; n < nnghbr; ++n) {
            if (nghbr.d_view(m,n).gid < 0) continue;
            if (nghbr.d_view(m,n).rank != my_rank) continue;   // gated above
            if (nghbr.d_view(m,n).lev != mylev) continue;      // gated above
            const int dnb = nghbr.d_view(m,n).gid - mbgid.d_view(0);
            if (dnb < 0 || dnb >= nmb_) continue;              // safety
            const Real x1d = mbsize.d_view(dnb).x1min, x1u = mbsize.d_view(dnb).x1max;
            const Real x2d = mbsize.d_view(dnb).x2min, x2u = mbsize.d_view(dnb).x2max;
            const Real x3d = mbsize.d_view(dnb).x3min, x3u = mbsize.d_view(dnb).x3max;
            // wrap into the neighbour's frame (periodic domains)
            Real xc = xw, yc = yw, zc = zw, ctr;
            ctr = 0.5*(x1d + x1u);
            if (xc - ctr >  0.5*Lx1) xc -= Lx1;
            if (xc - ctr < -0.5*Lx1) xc += Lx1;
            ctr = 0.5*(x2d + x2u);
            if (yc - ctr >  0.5*Lx2) yc -= Lx2;
            if (yc - ctr < -0.5*Lx2) yc += Lx2;
            ctr = 0.5*(x3d + x3u);
            if (zc - ctr >  0.5*Lx3) zc -= Lx3;
            if (zc - ctr < -0.5*Lx3) zc += Lx3;
            // index in the neighbour's interior+ghost array (same level -> same dx)
            const int in_ = static_cast<int>(Kokkos::floor((xc - x1d)/dx1)) + is;
            const int jn_ = static_cast<int>(Kokkos::floor((yc - x2d)/dx2)) + js;
            const int kn_ = static_cast<int>(Kokkos::floor((zc - x3d)/dx3)) + ks;
            if (in_ < 0 || in_ >= ncells1 || jn_ < 0 || jn_ >= ncells2 ||
                kn_ < 0 || kn_ >= ncells3) continue;
            // skip the trivial self-write (same block, unwrapped)
            if (dnb == m && in_ == i && jn_ == j && kn_ == k) continue;
            for (int v = 0; v < 4; ++v) u0(dnb, v, kn_, jn_, in_) = scr(v, c);
            if (wd > 0.0) {
              w0(dnb, IDN, kn_, jn_, in_) = wd;
              w0(dnb, IVX, kn_, jn_, in_) = scr(1, c)/wd;
              w0(dnb, IVY, kn_, jn_, in_) = scr(2, c)/wd;
              w0(dnb, IVZ, kn_, jn_, in_) = scr(3, c)/wd;
            }
          }
          // stage this reset cell for the cross-rank exchange (owner + physical centre +
          // conserved values); ExchangeCVReset() routes it to any off-rank block whose
          // interior contains the position. Only for sinks flagged near a rank boundary.
          if (touches_offrank) {
            const int e = Kokkos::atomic_fetch_add(&cvcnt(0), 1);
            if (e < cvmax) {
              cvemit(e, 0) = static_cast<Real>(m);
              cvemit(e, 1) = xw; cvemit(e, 2) = yw; cvemit(e, 3) = zw;
              cvemit(e, 4) = scr(0, c); cvemit(e, 5) = scr(1, c);
              cvemit(e, 6) = scr(2, c); cvemit(e, 7) = scr(3, c);
            }
          }
        });
        tm.team_barrier();
      };
      // -----------------------------------------------------------------------------

      // Step 1: the current control volume
      process_cv(ip, jp, kp);
      // Step 2: the old control volume, if the sink crossed a cell boundary this step
      // (sequential after Step 1; the overlap region is deliberately re-reset and its
      // additional change accounted, per the reference)
      if (ip != ip0 || jp != jp0 || kp != kp0) {
        process_cv(ip0, jp0, kp0);
      }

      // Step 3: conservative update of the sink mass and velocity
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {
        const Real mp = pr(IPM, p);
        const Real mnew = mp + acc(0);
        if (mnew > 0.0) {   // guard pathological dm < -mp (extrapolation outflow)
          const Real minv = 1.0/mnew;
          pr(IPVX, p) = (mp*pr(IPVX, p) + acc(1))*minv;
          pr(IPVY, p) = (mp*pr(IPVY, p) + acc(2))*minv;
          pr(IPVZ, p) = (mp*pr(IPVZ, p) + acc(3))*minv;
          pr(IPM, p)  = mnew;
        }
      });
      tm.team_barrier();
    }
  });

  // apply reset cells that landed in off-rank neighbour interiors (multi-rank only)
  if (mpi_on) ExchangeCVReset();

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::ExchangeCVReset()
//! \brief MPI transport for the control-volume reset. The accretion kernel stages every
//! reset cell of near-a-rank-boundary sinks into cvemit_ (owner_m, x, y, z, rho, M1..3).
//! Here, host-side: for each staged cell find the off-rank same-level neighbour block(s)
//! whose INTERIOR contains it (dest geometry from Mesh::lloc_eachmb + mesh_size, so no
//! device access to off-rank blocks is needed), negotiate counts, Isend/Irecv the payload
//! (dest_gid,i,j,k + rho,M1..3), and the receiver overwrites u0/w0. Same-level only
//! (AMR level-jumps are gated out); mirrors the ParticlesBoundaryValues MPI pattern.

void Particles::ExchangeCVReset() {
#if MPI_PARALLEL_ENABLED
  const int my_rank = global_variable::my_rank;
  const int nranks  = global_variable::nranks;
  if (nranks == 1) return;

  // how many cells were staged this step
  auto cnt_h = Kokkos::create_mirror_view(cvemit_cnt_);
  Kokkos::deep_copy(cnt_h, cvemit_cnt_);
  const int nemit = std::min(cnt_h(0), cvemit_max_);
  if (cnt_h(0) > cvemit_max_ && my_rank == 0) {
    std::cout << "### WARNING in Particles::ExchangeCVReset: cvemit overflow ("
              << cnt_h(0) << " > " << cvemit_max_ << "); some cross-rank resets dropped."
              << std::endl;
  }
  auto emit_h = Kokkos::create_mirror_view(cvemit_);
  Kokkos::deep_copy(emit_h, cvemit_);

  Mesh *pm = pmy_pack->pmesh;
  auto &ms = pm->mesh_size;
  const Real Lx1 = ms.x1max - ms.x1min;
  const Real Lx2 = ms.x2max - ms.x2min;
  const Real Lx3 = ms.x3max - ms.x3min;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const int nnghbr = pmy_pack->pmb->nnghbr;

  // build the send plan: a reset cell must reach EVERY rank owning a same-level
  // neighbour block whose interior+ghost array contains it -- the receiver then writes
  // all its local coincident copies (interior AND ghosts), mirroring the on-rank
  // scatter. Writing only interiors is NOT enough: the receiving rank's ghost copies
  // would stay pre-reset and its next flux computation near the boundary would diverge
  // from the single-rank run. Payload: 7 reals (x, y, z, rho, M1, M2, M3) per record;
  // one record per (cell, destination rank), deduplicated across that rank's blocks.
  std::vector<int>  s_rank;    // destination rank per record
  std::vector<Real> s_real;    // 7 reals per record
  for (int e = 0; e < nemit; ++e) {
    const int om = static_cast<int>(emit_h(e, 0));
    if (om < 0 || om >= pmy_pack->nmb_thispack) continue;
    const int olev = mblev.h_view(om);
    const Real xw = emit_h(e, 1), yw = emit_h(e, 2), zw = emit_h(e, 3);
    int sent_ranks[8]; int nsent = 0;   // per-cell rank dedup (few peers in practice)
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.h_view(om, n).gid < 0) continue;
      if (nghbr.h_view(om, n).rank == my_rank) continue;   // on-rank done on device
      if (nghbr.h_view(om, n).lev != olev) continue;       // AMR level-jump gated out
      const int drank = nghbr.h_view(om, n).rank;
      bool dup = false;
      for (int q = 0; q < nsent; ++q) if (sent_ranks[q] == drank) { dup = true; break; }
      if (dup) continue;
      const int dg = nghbr.h_view(om, n).gid;
      const auto &ll = pm->lloc_eachmb[dg];
      const int nmbx1 = pm->nmb_rootx1 << (ll.level - pm->root_level);
      const int nmbx2 = pm->nmb_rootx2 << (ll.level - pm->root_level);
      const int nmbx3 = pm->nmb_rootx3 << (ll.level - pm->root_level);
      const Real x1d = (ll.lx1 == 0) ? ms.x1min : LeftEdgeX(ll.lx1, nmbx1, ms.x1min, ms.x1max);
      const Real x1u = (ll.lx1 == nmbx1-1) ? ms.x1max
                                           : LeftEdgeX(ll.lx1+1, nmbx1, ms.x1min, ms.x1max);
      const Real x2d = (ll.lx2 == 0) ? ms.x2min : LeftEdgeX(ll.lx2, nmbx2, ms.x2min, ms.x2max);
      const Real x2u = (ll.lx2 == nmbx2-1) ? ms.x2max
                                           : LeftEdgeX(ll.lx2+1, nmbx2, ms.x2min, ms.x2max);
      const Real x3d = (ll.lx3 == 0) ? ms.x3min : LeftEdgeX(ll.lx3, nmbx3, ms.x3min, ms.x3max);
      const Real x3u = (ll.lx3 == nmbx3-1) ? ms.x3max
                                           : LeftEdgeX(ll.lx3+1, nmbx3, ms.x3min, ms.x3max);
      const Real dx1 = (x1u-x1d)/indcs.nx1;
      const Real dx2 = (x2u-x2d)/indcs.nx2;
      const Real dx3 = (x3u-x3d)/indcs.nx3;
      // wrap the cell centre into the neighbour's frame (periodic domains)
      Real xc = xw, yc = yw, zc = zw, ctr;
      ctr = 0.5*(x1d + x1u);  if (xc-ctr >  0.5*Lx1) xc -= Lx1;  if (xc-ctr < -0.5*Lx1) xc += Lx1;
      ctr = 0.5*(x2d + x2u);  if (yc-ctr >  0.5*Lx2) yc -= Lx2;  if (yc-ctr < -0.5*Lx2) yc += Lx2;
      ctr = 0.5*(x3d + x3u);  if (zc-ctr >  0.5*Lx3) zc -= Lx3;  if (zc-ctr < -0.5*Lx3) zc += Lx3;
      // containment in the EXPANDED (interior + ng ghosts) bounds
      const Real g1 = indcs.ng*dx1, g2 = indcs.ng*dx2, g3 = indcs.ng*dx3;
      if (xc < x1d-g1 || xc >= x1u+g1 || yc < x2d-g2 || yc >= x2u+g2 ||
          zc < x3d-g3 || zc >= x3u+g3) continue;
      if (nsent < 8) sent_ranks[nsent++] = drank;
      s_rank.push_back(drank);
      s_real.push_back(xw); s_real.push_back(yw); s_real.push_back(zw);
      s_real.push_back(emit_h(e,4)); s_real.push_back(emit_h(e,5));
      s_real.push_back(emit_h(e,6)); s_real.push_back(emit_h(e,7));
    }
  }

  // per-destination-rank counts of records to send
  std::vector<int> nsend(nranks, 0);
  for (int r : s_rank) nsend[r]++;
  std::vector<int> nrecv(nranks, 0);
  MPI_Alltoall(nsend.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT, mpi_comm_cvscat_);

  // pack sends contiguously per destination rank (stable order: preserves the emit
  // order, so the old-CV overlap re-reset correctly supersedes the new-CV values)
  std::vector<int> soff(nranks, 0), sacc(nranks, 0);
  for (int r = 1; r < nranks; ++r) soff[r] = soff[r-1] + nsend[r-1];
  const int ntot_send = soff[nranks-1] + nsend[nranks-1];
  std::vector<Real> send_r(7*ntot_send);
  for (size_t rec = 0; rec < s_rank.size(); ++rec) {
    const int r = s_rank[rec];
    const int pos = soff[r] + sacc[r]; sacc[r]++;
    for (int q = 0; q < 7; ++q) send_r[7*pos+q] = s_real[7*rec+q];
  }

  // receive layout
  std::vector<int> roff(nranks, 0);
  for (int r = 1; r < nranks; ++r) roff[r] = roff[r-1] + nrecv[r-1];
  const int ntot_recv = roff[nranks-1] + nrecv[nranks-1];
  std::vector<Real> recv_r(7*ntot_recv);

  // post non-blocking recvs then sends, per peer rank
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nrecv[r] == 0) continue;
    reqs.emplace_back();
    MPI_Irecv(&recv_r[7*roff[r]], 7*nrecv[r], MPI_ATHENA_REAL, r, 0, mpi_comm_cvscat_,
              &reqs.back());
  }
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nsend[r] == 0) continue;
    reqs.emplace_back();
    MPI_Isend(&send_r[7*soff[r]], 7*nsend[r], MPI_ATHENA_REAL, r, 0, mpi_comm_cvscat_,
              &reqs.back());
  }
  // records destined for this rank never go through MPI (neighbour rank != my_rank), so
  // no self-copy is needed.
  if (!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

  if (ntot_recv == 0) return;

  // apply on the device with the SAME geometric scatter as the on-rank path: for each
  // received cell, write every local coincident copy -- any local block whose
  // interior+ghost array contains the (wrapped) position, interior or ghost alike.
  // Parallel over blocks, serial over records within a block: preserves the record
  // order so an old-CV overlap re-reset supersedes the new-CV value (no write race --
  // different blocks write disjoint arrays).
  const bool is_mhd = (pmy_pack->pmhd != nullptr);
  auto u0 = is_mhd ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;
  auto w0 = is_mhd ? pmy_pack->pmhd->w0 : pmy_pack->phydro->w0;
  const int nmb_ = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;

  DualArray2D<Real> rreal("cvrecv_r", ntot_recv, 7);
  for (int t = 0; t < ntot_recv; ++t) {
    for (int q = 0; q < 7; ++q) rreal.h_view(t,q) = recv_r[7*t+q];
  }
  rreal.template modify<HostMemSpace>();  rreal.template sync<DevExeSpace>();
  auto rr = rreal.d_view;
  const int is_ = is, js_ = js, ks_ = ks;

  Kokkos::parallel_for("cvreset_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb_),
  KOKKOS_LAMBDA(const int m) {
    const Real x1d = mbsize.d_view(m).x1min, x1u = mbsize.d_view(m).x1max;
    const Real x2d = mbsize.d_view(m).x2min, x2u = mbsize.d_view(m).x2max;
    const Real x3d = mbsize.d_view(m).x3min, x3u = mbsize.d_view(m).x3max;
    const Real dx1 = mbsize.d_view(m).dx1;
    const Real dx2 = mbsize.d_view(m).dx2;
    const Real dx3 = mbsize.d_view(m).dx3;
    for (int t = 0; t < ntot_recv; ++t) {
      // wrap the cell centre into this block's frame (periodic domains)
      Real xc = rr(t,0), yc = rr(t,1), zc = rr(t,2), ctr;
      ctr = 0.5*(x1d + x1u);
      if (xc - ctr >  0.5*Lx1) xc -= Lx1;
      if (xc - ctr < -0.5*Lx1) xc += Lx1;
      ctr = 0.5*(x2d + x2u);
      if (yc - ctr >  0.5*Lx2) yc -= Lx2;
      if (yc - ctr < -0.5*Lx2) yc += Lx2;
      ctr = 0.5*(x3d + x3u);
      if (zc - ctr >  0.5*Lx3) zc -= Lx3;
      if (zc - ctr < -0.5*Lx3) zc += Lx3;
      const int i = static_cast<int>(Kokkos::floor((xc - x1d)/dx1)) + is_;
      const int j = static_cast<int>(Kokkos::floor((yc - x2d)/dx2)) + js_;
      const int k = static_cast<int>(Kokkos::floor((zc - x3d)/dx3)) + ks_;
      if (i < 0 || i >= ncells1 || j < 0 || j >= ncells2 ||
          k < 0 || k >= ncells3) continue;
      const Real rho = rr(t,3), m1 = rr(t,4), m2 = rr(t,5), m3 = rr(t,6);
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = m1;
      u0(m, IM2, k, j, i) = m2;
      u0(m, IM3, k, j, i) = m3;
      if (rho > 0.0) {
        w0(m, IDN, k, j, i) = rho;
        w0(m, IVX, k, j, i) = m1/rho;
        w0(m, IVY, k, j, i) = m2/rho;
        w0(m, IVZ, k, j, i) = m3/rho;
      }
    }
  });
#endif
}

}  // namespace particles
