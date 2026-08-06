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
//! pack: the owner's cells (interior or ghost), and for each on-rank neighbour -- at ANY
//! refinement level, by the three-way rule below -- the cell(s) of its (interior+ghost)
//! array covering the same physical position (periodic-wrapped) -- the containment
//! pattern of ParticleMesh::FlushDepositBoundaries. Writing ALL copies keeps every block
//! bit-consistent immediately, with no extra boundary exchange (this replaces the
//! Athena++ ghost-particle mechanism and its NGHOST >= req+rctrl+1 requirement).
//! The same-level and finer writes are assignments carrying identical values, so no
//! atomics are needed; the coarser writes accumulate and are serialized (see below).
//!
//! CROSS-RANK control volumes: reset cells whose reach touches an off-rank neighbour are
//! additionally staged into a device buffer and exchanged after the kernel
//! (ExchangeCVReset): counts negotiated with MPI_Alltoall, then per-peer Isend/Irecv of
//! (position, post-reset rho/M1..3, pre-reset rho/M1..3, owner level) records on a
//! dedicated communicator. Each receiver applies the SAME geometric scatter locally --
//! every local block whose interior+ghost array contains the (wrapped) position gets the
//! value, interior and ghost copies alike, so multi-rank runs stay bit-consistent with
//! single-rank ones. The sink cell itself is binned in the GLOBAL mesh frame (not the
//! owner block's frame): block-frame binning resolves a position within round-off of a
//! block edge differently in different blocks' frames (catastrophic cancellation), which
//! breaks decomposition invariance.
//!
//! AMR (DIFFERENT-LEVEL) control volumes: the control volume always lives at the OWNER
//! block's resolution, and every read stays inside the owner's array -- its ghost zones
//! hold prolongated (coarser neighbour) or restricted (finer neighbour) data at
//! "after_stagen", so the 27-cell integral is a faithful, conservative representation of
//! the neighbour's gas either way. What must be level-aware is the SCATTER, because the
//! cells that actually own the reset data may be coarser or finer than the control volume.
//! For a reset cell with pre-reset value U_old and post-reset U_new, each target copy is
//! updated by:
//!   same level  : U <- U_new                       (assignment, as before)
//!   finer target: U <- U_new for each of the 2^3 sub-cells covering the reset cell
//!                 (uniform split; conservative for cell-averaged densities, and the
//!                 sub-structure inside a control volume is deliberately erased anyway)
//!   coarser tgt : U += (U_new - U_old)/2^3         (conservative PARTIAL restriction)
//! The coarser rule is the only subtle one. A coarse cell holds 8 fine cells, of which the
//! control volume may cover only some, so it cannot simply be assigned; but since
//! AthenaK's prolongation is conservative, the 8 fine values average to the coarse value,
//! and adding 1/8 of each covered cell's CHANGE removes exactly the mass that the owner's
//! integral counted as accreted: dV_coarse*(dU/8) = dV_fine*dU. Partial coverage is
//! therefore handled exactly, with no assumption that the control volume aligns with the
//! coarse lattice. Duplicate coarse targets must be applied ONCE (unlike the idempotent
//! assignments), so the coarser pass deduplicates by gid and runs serially -- which also
//! keeps the matching w0 write consistent with the accumulated u0.
//!
//! A sink is skipped only defensively (level jump > 1, which proper nesting forbids, or a
//! wild position); skips are counted and warned about, since a silently non-accreting sink
//! shows up much later as a gas pile-up and a collapsing timestep.
//!
//! After scattering the reset conserved values, the matching PRIMITIVES are written to
//! w0 for the same cells (isothermal: w_d = u_d, w_vi = M_i/rho). In AthenaK,
//! Hydro/MHD::ConToPrim runs in "stagen" BEFORE this task ("after_stagen"), so without
//! this the reset would be invisible to the next cycle's flux computation. (Athena++
//! orders CONS2PRIM after the particle interaction for exactly this reason.)
//!
//! LIMITATIONS (documented scope): isothermal only (resets IDN+IM1..3; no energy/B).
//! Across a level jump the extrapolation stencil reads prolongated/restricted ghost data,
//! so the reset values there carry the coarser side's accuracy -- conservation is exact
//! but the reset profile is only as good as the ghost data. Keeping a sink's control
//! volume inside one level (the sink-proximity refinement halo in be_sink.cpp /
//! bb_collapse.cpp) is therefore still preferable; this path exists so that the
//! unavoidable transients -- a sink drifting toward a level jump, or the lag between a
//! sink moving and the next regrid -- accrete conservatively instead of not at all.

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
      Kokkos::realloc(cvemit_, cvemit_max_, NCVEMIT);
    }
    Kokkos::deep_copy(cvemit_cnt_, 0);
  }
  auto cvemit = cvemit_;
  auto cvcnt = cvemit_cnt_;
  const int cvmax = cvemit_max_;
  Kokkos::deep_copy(accskip_, 0);
  auto accskip = accskip_;

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

  // team scratch: per control-volume cell, 4 post-reset conserved values (rows 0-3) and
  // the 4 pre-reset ones (rows 4-7; needed for the conservative coarser-target update),
  // plus a 4-vector accumulator for the per-sink accreted (dm, dM1, dM2, dM3)
  const int scr_level = 0;
  const size_t scr_bytes = ScrArray2D<Real>::shmem_size(8, 27)
                         + ScrArray1D<Real>::shmem_size(4);

  Kokkos::TeamPolicy<> policy(DevExeSpace(), 1, Kokkos::AUTO);
  policy.set_scratch_size(scr_level, Kokkos::PerTeam(scr_bytes));

  Kokkos::parallel_for("sink_accrete", policy,
  KOKKOS_LAMBDA(TeamMember_t tm) {
    ScrArray2D<Real> scr(tm.team_scratch(scr_level), 8, 27);
    ScrArray1D<Real> acc(tm.team_scratch(scr_level), 4);

    // serial over sinks (reference semantics; sinks are few), team-parallel inside
    for (int p = 0; p < npart; ++p) {
      const int m = pi(PGID, p) - gids;
      // Guard invalid PGID before ANY indexed access (mirrors ParticleMesh::DepositMass):
      // a corrupt/ejected particle must skip accretion, not fault.
      if (m < 0 || m >= nmb_) continue;
      bool touches_offrank = false;   // CV reaches an off-rank neighbour (any level)
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
      // reads) spans at most rctrl+2.5 cells from the CURRENT sink position. Level jumps
      // in that region are HANDLED (see the level rule in the file comment), so the only
      // reasons to bail out are defensive: a level jump of more than one, which proper
      // nesting forbids and whose scatter factors are not derived here. Off-rank
      // neighbours within reach set touches_offrank, which stages this sink's reset cells
      // for ExchangeCVReset. (Computed redundantly by every team thread; all take
      // identical branches.)
      const int mylev = mblev.d_view(m);
      const Real xs = pr(IPX, p), ys = pr(IPY, p), zs = pr(IPZ, p);
      const Real h1 = (rctrl + 2.5)*dx1, h2 = (rctrl + 2.5)*dx2, h3 = (rctrl + 2.5)*dx3;
      // proximity of the sink to any owner face, i.e. "the reach leaves this block".
      // Used for off-rank neighbours, whose boxes are not addressable on the device.
      const bool near_face = (xs - mbsize.d_view(m).x1min < h1) ||
                             (mbsize.d_view(m).x1max - xs < h1) ||
                             (ys - mbsize.d_view(m).x2min < h2) ||
                             (mbsize.d_view(m).x2max - ys < h2) ||
                             (zs - mbsize.d_view(m).x3min < h3) ||
                             (mbsize.d_view(m).x3max - zs < h3);
      bool eligible = true;
      for (int n = 0; n < nnghbr; ++n) {
        if (nghbr.d_view(m,n).gid < 0) continue;
        const int dlev = nghbr.d_view(m,n).lev - mylev;
        if (dlev > 1 || dlev < -1) {   // proper nesting violated: refuse to guess
          eligible = false; break;
        }
        if (nghbr.d_view(m,n).rank != my_rank && near_face) touches_offrank = true;
      }
      if (!eligible) {
        Kokkos::single(Kokkos::PerTeam(tm), [&]() {
          Kokkos::atomic_fetch_add(&accskip(0), 1);
        });
        continue;
      }

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
        // including old-CV centres one cell into the ghost region after a crossing.
        // Counted like the eligibility skips: bailing out here also means this sink
        // stops accreting, which must not happen silently.
        if (ic < rctrl+1 || ic > ncells1-rctrl-2 ||
            jc < rctrl+1 || jc > ncells2-rctrl-2 ||
            kc < rctrl+1 || kc > ncells3-rctrl-2) {
          Kokkos::single(Kokkos::PerTeam(tm), [&]() {
            Kokkos::atomic_fetch_add(&accskip(0), 1);
          });
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
            const Real uold = u0(m, v, kc+dk, jc+dj, ic+di);
            psum += uold;
            scr(4+v, c) = uold;   // pre-reset value, for the coarser-target update
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
        // owner's cell (interior or ghost), plus, for each on-rank neighbour at THIS
        // level or FINER, the cell(s) of its interior+ghost array covering the same
        // physical position (periodic-wrapped). Coarser neighbours accumulate a
        // difference instead and are handled by the serial pass below. Also write
        // matching primitives to w0 (isothermal: w_d = u_d, w_vi = M_i/rho); without the
        // w0 update the reset is invisible to the next cycle's fluxes (ConToPrim already
        // ran this stage). These writes are assignments carrying identical values, so
        // duplicates are harmless and no atomics are needed.
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
          // every coincident copy in on-rank neighbours at THIS level or finer.
          // Coarser neighbours accumulate a difference instead and must be applied once
          // per block, so they are handled serially below (a coarse block appears in
          // several neighbour slots, and += is not idempotent).
          for (int n = 0; n < nnghbr; ++n) {
            if (nghbr.d_view(m,n).gid < 0) continue;
            if (nghbr.d_view(m,n).rank != my_rank) continue;   // shipped by ExchangeCVReset
            const int dlev = nghbr.d_view(m,n).lev - mylev;
            if (dlev < 0) continue;                            // coarser: serial pass
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
            if (dlev == 0) {
              // index in the neighbour's interior+ghost array (same level -> same dx;
              // use the OWNER's dx so this path is bit-for-bit the pre-AMR behaviour)
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
            } else {
              // FINER neighbour: uniform split of this reset cell over the sub-cells it
              // covers. Sample the reset cell at the sub-cell centres (dlev == 1 => 2 per
              // axis) so each sub-cell is found by the same containment rule, and no two
              // reset cells can target the same sub-cell (their boxes are disjoint).
              const Real dn1 = mbsize.d_view(dnb).dx1;
              const Real dn2 = mbsize.d_view(dnb).dx2;
              const Real dn3 = mbsize.d_view(dnb).dx3;
              const int r = 1 << dlev;
              const Real rinv = 1.0/static_cast<Real>(r);
              for (int sk = 0; sk < r; ++sk) {
              for (int sj = 0; sj < r; ++sj) {
              for (int si = 0; si < r; ++si) {
                const Real xsub = xc + ((si + 0.5)*rinv - 0.5)*dx1;
                const Real ysub = yc + ((sj + 0.5)*rinv - 0.5)*dx2;
                const Real zsub = zc + ((sk + 0.5)*rinv - 0.5)*dx3;
                const int in_ = static_cast<int>(Kokkos::floor((xsub - x1d)/dn1)) + is;
                const int jn_ = static_cast<int>(Kokkos::floor((ysub - x2d)/dn2)) + js;
                const int kn_ = static_cast<int>(Kokkos::floor((zsub - x3d)/dn3)) + ks;
                if (in_ < 0 || in_ >= ncells1 || jn_ < 0 || jn_ >= ncells2 ||
                    kn_ < 0 || kn_ >= ncells3) continue;
                for (int v = 0; v < 4; ++v) u0(dnb, v, kn_, jn_, in_) = scr(v, c);
                if (wd > 0.0) {
                  w0(dnb, IDN, kn_, jn_, in_) = wd;
                  w0(dnb, IVX, kn_, jn_, in_) = scr(1, c)/wd;
                  w0(dnb, IVY, kn_, jn_, in_) = scr(2, c)/wd;
                  w0(dnb, IVZ, kn_, jn_, in_) = scr(3, c)/wd;
                }
              }}}
            }
          }
          // stage this reset cell for the cross-rank exchange (owner + physical centre +
          // post-reset and pre-reset conserved values); ExchangeCVReset() routes it to any
          // off-rank block whose expanded bounds contain the position, at any level. Only
          // for sinks flagged near a rank boundary.
          if (touches_offrank) {
            const int e = Kokkos::atomic_fetch_add(&cvcnt(0), 1);
            if (e < cvmax) {
              cvemit(e, 0) = static_cast<Real>(m);
              cvemit(e, 1) = xw; cvemit(e, 2) = yw; cvemit(e, 3) = zw;
              cvemit(e, 4) = scr(0, c); cvemit(e, 5) = scr(1, c);
              cvemit(e, 6) = scr(2, c); cvemit(e, 7) = scr(3, c);
              cvemit(e, 8) = scr(4, c); cvemit(e, 9) = scr(5, c);
              cvemit(e,10) = scr(6, c); cvemit(e,11) = scr(7, c);
            }
          }
        });
        tm.team_barrier();

        // COARSER on-rank neighbours: conservative partial restriction of the reset,
        // coarse += (post - pre)/2^3 per covered fine cell. Serial in one thread because
        // (a) a coarse block occupies several neighbour slots and the accumulation must
        // happen exactly once per block, (b) several of the 27 cells land in the same
        // coarse cell, and (c) w0 must be derived from the FINAL accumulated u0.
        Kokkos::single(Kokkos::PerTeam(tm), [&]() {
          for (int n = 0; n < nnghbr; ++n) {
            if (nghbr.d_view(m,n).gid < 0) continue;
            if (nghbr.d_view(m,n).rank != my_rank) continue;
            const int dlev = nghbr.d_view(m,n).lev - mylev;
            if (dlev >= 0) continue;                       // handled team-parallel above
            const int dgid = nghbr.d_view(m,n).gid;
            bool seen = false;                             // apply once per coarse block
            for (int q = 0; q < n; ++q) {
              if (nghbr.d_view(m,q).gid == dgid) { seen = true; break; }
            }
            if (seen) continue;
            const int dnb = dgid - mbgid.d_view(0);
            if (dnb < 0 || dnb >= nmb_) continue;
            const Real x1d = mbsize.d_view(dnb).x1min, x1u = mbsize.d_view(dnb).x1max;
            const Real x2d = mbsize.d_view(dnb).x2min, x2u = mbsize.d_view(dnb).x2max;
            const Real x3d = mbsize.d_view(dnb).x3min, x3u = mbsize.d_view(dnb).x3max;
            const Real dn1 = mbsize.d_view(dnb).dx1;
            const Real dn2 = mbsize.d_view(dnb).dx2;
            const Real dn3 = mbsize.d_view(dnb).dx3;
            // one coarse cell holds 2^(3*|dlev|) cells of the control volume
            const int r = 1 << (-dlev);
            const Real fac = 1.0/static_cast<Real>(r*r*r);
            for (int c = 0; c < 27; ++c) {
              const int di = c%3 - 1, dj = (c/3)%3 - 1, dk = c/9 - 1;
              const Real xw2 = mbsize.d_view(m).x1min
                             + (static_cast<Real>(ic+di-is) + 0.5)*dx1;
              const Real yw2 = mbsize.d_view(m).x2min
                             + (static_cast<Real>(jc+dj-js) + 0.5)*dx2;
              const Real zw2 = mbsize.d_view(m).x3min
                             + (static_cast<Real>(kc+dk-ks) + 0.5)*dx3;
              Real xc = xw2, yc = yw2, zc = zw2, ctr;
              ctr = 0.5*(x1d + x1u);
              if (xc - ctr >  0.5*Lx1) xc -= Lx1;
              if (xc - ctr < -0.5*Lx1) xc += Lx1;
              ctr = 0.5*(x2d + x2u);
              if (yc - ctr >  0.5*Lx2) yc -= Lx2;
              if (yc - ctr < -0.5*Lx2) yc += Lx2;
              ctr = 0.5*(x3d + x3u);
              if (zc - ctr >  0.5*Lx3) zc -= Lx3;
              if (zc - ctr < -0.5*Lx3) zc += Lx3;
              const int in_ = static_cast<int>(Kokkos::floor((xc - x1d)/dn1)) + is;
              const int jn_ = static_cast<int>(Kokkos::floor((yc - x2d)/dn2)) + js;
              const int kn_ = static_cast<int>(Kokkos::floor((zc - x3d)/dn3)) + ks;
              if (in_ < 0 || in_ >= ncells1 || jn_ < 0 || jn_ >= ncells2 ||
                  kn_ < 0 || kn_ >= ncells3) continue;
              for (int v = 0; v < 4; ++v) {
                u0(dnb, v, kn_, jn_, in_) += fac*(scr(v, c) - scr(4+v, c));
              }
              const Real wdc = u0(dnb, IDN, kn_, jn_, in_);
              if (wdc > 0.0) {
                w0(dnb, IDN, kn_, jn_, in_) = wdc;
                w0(dnb, IVX, kn_, jn_, in_) = u0(dnb, IM1, kn_, jn_, in_)/wdc;
                w0(dnb, IVY, kn_, jn_, in_) = u0(dnb, IM2, kn_, jn_, in_)/wdc;
                w0(dnb, IVZ, kn_, jn_, in_) = u0(dnb, IM3, kn_, jn_, in_)/wdc;
              }
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

  // report defensively skipped sinks. A skipped sink keeps attracting gas but never
  // removes it, so this must never pass unnoticed; rate-limited to a few messages.
  if (accskip_warned_ < 5) {
    auto nskip_h = Kokkos::create_mirror_view(accskip_);
    Kokkos::deep_copy(nskip_h, accskip_);
    if (nskip_h(0) > 0) {
      accskip_warned_++;
      std::cout << "### WARNING in Particles::AccreteMass: " << nskip_h(0)
                << " sink control volume(s) skipped this step (a level jump greater than "
                << "one in reach, a wild position, or a control volume reaching outside "
                << "the owner's array); those sinks are NOT accreting."
                << (accskip_warned_ == 5 ? " Further warnings suppressed." : "")
                << std::endl;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::ExchangeCVReset()
//! \brief MPI transport for the control-volume reset. The accretion kernel stages every
//! reset cell of near-a-rank-boundary sinks into cvemit_ (owner_m, x, y, z, post-reset
//! rho/M1..3, pre-reset rho/M1..3). Here, host-side: for each staged cell find the
//! off-rank neighbour block(s) whose expanded bounds contain it (dest geometry from
//! Mesh::lloc_eachmb + mesh_size, so no device access to off-rank blocks is needed),
//! negotiate counts, Isend/Irecv the payload, and the receiver applies it to every local
//! coincident copy. Records carry the OWNER's refinement level, so the receiver can apply
//! the same three-way level rule as the on-rank scatter (assign at the same level, assign
//! to the covered sub-cells of a finer block, conservative partial restriction into a
//! coarser block). Mirrors the ParticlesBoundaryValues MPI pattern.

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
  // wire-record width: x, y, z, 4 post-reset, 4 pre-reset, owner level
  constexpr int NCVREC = 12;

  // build the send plan: a reset cell must reach EVERY rank owning a neighbour block whose
  // interior+ghost array contains it -- the receiver then writes all its local coincident
  // copies (interior AND ghosts), mirroring the on-rank scatter. Writing only interiors is
  // NOT enough: the receiving rank's ghost copies would stay pre-reset and its next flux
  // computation near the boundary would diverge from the single-rank run. Payload: 12
  // reals (x, y, z, post-reset rho/M1..3, pre-reset rho/M1..3, owner level) per record;
  // one record per (cell, destination rank), deduplicated across that rank's blocks -- the
  // receiver rescans all of its own blocks, so one record per rank suffices even when that
  // rank holds several blocks at different levels.
  std::vector<int>  s_rank;    // destination rank per record
  std::vector<Real> s_real;    // NCVREC reals per record
  for (int e = 0; e < nemit; ++e) {
    const int om = static_cast<int>(emit_h(e, 0));
    if (om < 0 || om >= pmy_pack->nmb_thispack) continue;
    const int olev = mblev.h_view(om);
    const Real xw = emit_h(e, 1), yw = emit_h(e, 2), zw = emit_h(e, 3);
    int sent_ranks[8]; int nsent = 0;   // per-cell rank dedup (few peers in practice)
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.h_view(om, n).gid < 0) continue;
      if (nghbr.h_view(om, n).rank == my_rank) continue;   // on-rank done on device
      const int dl = nghbr.h_view(om, n).lev - olev;
      if (dl > 1 || dl < -1) continue;                     // proper nesting violated
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
      s_real.push_back(emit_h(e,8)); s_real.push_back(emit_h(e,9));
      s_real.push_back(emit_h(e,10)); s_real.push_back(emit_h(e,11));
      s_real.push_back(static_cast<Real>(olev));
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
  std::vector<Real> send_r(NCVREC*ntot_send);
  for (size_t rec = 0; rec < s_rank.size(); ++rec) {
    const int r = s_rank[rec];
    const int pos = soff[r] + sacc[r]; sacc[r]++;
    for (int q = 0; q < NCVREC; ++q) send_r[NCVREC*pos+q] = s_real[NCVREC*rec+q];
  }

  // receive layout
  std::vector<int> roff(nranks, 0);
  for (int r = 1; r < nranks; ++r) roff[r] = roff[r-1] + nrecv[r-1];
  const int ntot_recv = roff[nranks-1] + nrecv[nranks-1];
  std::vector<Real> recv_r(NCVREC*ntot_recv);

  // post non-blocking recvs then sends, per peer rank
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nrecv[r] == 0) continue;
    reqs.emplace_back();
    MPI_Irecv(&recv_r[NCVREC*roff[r]], NCVREC*nrecv[r], MPI_ATHENA_REAL, r, 0,
              mpi_comm_cvscat_, &reqs.back());
  }
  for (int r = 0; r < nranks; ++r) {
    if (r == my_rank || nsend[r] == 0) continue;
    reqs.emplace_back();
    MPI_Isend(&send_r[NCVREC*soff[r]], NCVREC*nsend[r], MPI_ATHENA_REAL, r, 0,
              mpi_comm_cvscat_, &reqs.back());
  }
  // records destined for this rank never go through MPI (neighbour rank != my_rank), so
  // no self-copy is needed.
  if (!reqs.empty()) MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

  // opt-in trace (same switch as the cross-rank particle migration): without it there is
  // no way to tell a run where no control volume crossed a rank boundary from one where
  // the crossing silently did nothing.
  static bool xdbg = (std::getenv("PART_XRANK_DBG") != nullptr);
  if (xdbg && (ntot_send > 0 || ntot_recv > 0)) {
    std::cout << "  [cvscat] rank " << my_rank << " sent " << ntot_send
              << " recvd " << ntot_recv << " control-volume reset cells" << std::endl;
  }

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

  DualArray2D<Real> rreal("cvrecv_r", ntot_recv, NCVREC);
  for (int t = 0; t < ntot_recv; ++t) {
    for (int q = 0; q < NCVREC; ++q) rreal.h_view(t,q) = recv_r[NCVREC*t+q];
  }
  rreal.template modify<HostMemSpace>();  rreal.template sync<DevExeSpace>();
  auto rr = rreal.d_view;
  const int is_ = is, js_ = js, ks_ = ks;
  auto &dmblev = pmy_pack->pmb->mb_lev;

  Kokkos::parallel_for("cvreset_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb_),
  KOKKOS_LAMBDA(const int m) {
    const Real x1d = mbsize.d_view(m).x1min, x1u = mbsize.d_view(m).x1max;
    const Real x2d = mbsize.d_view(m).x2min, x2u = mbsize.d_view(m).x2max;
    const Real x3d = mbsize.d_view(m).x3min, x3u = mbsize.d_view(m).x3max;
    const Real dx1 = mbsize.d_view(m).dx1;
    const Real dx2 = mbsize.d_view(m).dx2;
    const Real dx3 = mbsize.d_view(m).dx3;
    const int mylev = dmblev.d_view(m);
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
      const Real rho = rr(t,3), m1 = rr(t,4), m2 = rr(t,5), m3 = rr(t,6);
      // dlev < 0: this block is COARSER than the control volume the record came from.
      // dlev == 0 / > 0: same level / finer. Same three-way rule as the on-rank scatter
      // (see the file comment). Records are applied serially within a block, so the
      // coarser accumulation needs no atomics and w0 stays consistent with the final u0.
      const int dlev = mylev - static_cast<int>(rr(t,11));
      if (dlev == 0) {
        const int i = static_cast<int>(Kokkos::floor((xc - x1d)/dx1)) + is_;
        const int j = static_cast<int>(Kokkos::floor((yc - x2d)/dx2)) + js_;
        const int k = static_cast<int>(Kokkos::floor((zc - x3d)/dx3)) + ks_;
        if (i < 0 || i >= ncells1 || j < 0 || j >= ncells2 ||
            k < 0 || k >= ncells3) continue;
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
      } else if (dlev > 0) {
        // finer: uniform split over the sub-cells covering the reset cell. The record's
        // cell width is this block's width scaled by the level difference.
        const int r = 1 << dlev;
        const Real rinv = 1.0/static_cast<Real>(r);
        const Real cw1 = dx1*r, cw2 = dx2*r, cw3 = dx3*r;   // reset cell width
        for (int sk = 0; sk < r; ++sk) {
        for (int sj = 0; sj < r; ++sj) {
        for (int si = 0; si < r; ++si) {
          const Real xs = xc + ((si + 0.5)*rinv - 0.5)*cw1;
          const Real ys = yc + ((sj + 0.5)*rinv - 0.5)*cw2;
          const Real zs = zc + ((sk + 0.5)*rinv - 0.5)*cw3;
          const int i = static_cast<int>(Kokkos::floor((xs - x1d)/dx1)) + is_;
          const int j = static_cast<int>(Kokkos::floor((ys - x2d)/dx2)) + js_;
          const int k = static_cast<int>(Kokkos::floor((zs - x3d)/dx3)) + ks_;
          if (i < 0 || i >= ncells1 || j < 0 || j >= ncells2 ||
              k < 0 || k >= ncells3) continue;
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
        }}}
      } else {
        // coarser: conservative partial restriction of this one reset cell
        const int r = 1 << (-dlev);
        const Real fac = 1.0/static_cast<Real>(r*r*r);
        const int i = static_cast<int>(Kokkos::floor((xc - x1d)/dx1)) + is_;
        const int j = static_cast<int>(Kokkos::floor((yc - x2d)/dx2)) + js_;
        const int k = static_cast<int>(Kokkos::floor((zc - x3d)/dx3)) + ks_;
        if (i < 0 || i >= ncells1 || j < 0 || j >= ncells2 ||
            k < 0 || k >= ncells3) continue;
        u0(m, IDN, k, j, i) += fac*(rho - rr(t,7));
        u0(m, IM1, k, j, i) += fac*(m1  - rr(t,8));
        u0(m, IM2, k, j, i) += fac*(m2  - rr(t,9));
        u0(m, IM3, k, j, i) += fac*(m3  - rr(t,10));
        const Real wdc = u0(m, IDN, k, j, i);
        if (wdc > 0.0) {
          w0(m, IDN, k, j, i) = wdc;
          w0(m, IVX, k, j, i) = u0(m, IM1, k, j, i)/wdc;
          w0(m, IVY, k, j, i) = u0(m, IM2, k, j, i)/wdc;
          w0(m, IVZ, k, j, i) = u0(m, IM3, k, j, i)/wdc;
        }
      }
    }
  });
#endif
}

}  // namespace particles
