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
//! After scattering the reset conserved values, the matching PRIMITIVES are written to
//! w0 for the same cells (isothermal: w_d = u_d, w_vi = M_i/rho). In AthenaK,
//! Hydro/MHD::ConToPrim runs in "stagen" BEFORE this task ("after_stagen"), so without
//! this the reset would be invisible to the next cycle's flux computation. (Athena++
//! orders CONS2PRIM after the particle interaction for exactly this reason.)
//!
//! LIMITATIONS (documented scope): isothermal only (resets IDN+IM1..3; no energy/B);
//! interior sinks only (control volume must sit >= rctrl cells inside the owning
//! MeshBlock's active zone -- sinks in the boundary shell skip accretion); AMR
//! coarse-fine boundaries and cross-rank control volumes not handled (the reference
//! also punts on AMR).

#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
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

  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int gids = pmy_pack->gids;
  const int npart = nprtcl_thispack;
  const int rctrl = 1;      // control volume = (2*rctrl+1)^3 = 27 cells (paper standard)
  const int nmb_ = pmy_pack->nmb_thispack;

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
      const Real dx1 = mbsize.d_view(m).dx1;
      const Real dx2 = mbsize.d_view(m).dx2;
      const Real dx3 = mbsize.d_view(m).dx3;
      const Real dV = dx1*dx2*dx3;

      // cell containing the sink now and at the start of the step -- guard the
      // double->int conversions against non-finite/wild positions (UB otherwise),
      // and use floor (truncation is wrong for fractional offsets in (-1,0))
      const Real xi1 = (pr(IPX, p) - mbsize.d_view(m).x1min)/dx1;
      const Real xi2 = (pr(IPY, p) - mbsize.d_view(m).x2min)/dx2;
      const Real xi3 = (pr(IPZ, p) - mbsize.d_view(m).x3min)/dx3;
      const Real yi1 = (pr(IPX0, p) - mbsize.d_view(m).x1min)/dx1;
      const Real yi2 = (pr(IPY0, p) - mbsize.d_view(m).x2min)/dx2;
      const Real yi3 = (pr(IPZ0, p) - mbsize.d_view(m).x3min)/dx3;
      if (!(xi1 > -1.0e9 && xi1 < 1.0e9) || !(xi2 > -1.0e9 && xi2 < 1.0e9) ||
          !(xi3 > -1.0e9 && xi3 < 1.0e9) || !(yi1 > -1.0e9 && yi1 < 1.0e9) ||
          !(yi2 > -1.0e9 && yi2 < 1.0e9) || !(yi3 > -1.0e9 && yi3 < 1.0e9)) {
        continue;
      }
      const int ip  = static_cast<int>(Kokkos::floor(xi1)) + is;
      const int jp  = static_cast<int>(Kokkos::floor(xi2)) + js;
      const int kp  = static_cast<int>(Kokkos::floor(xi3)) + ks;
      const int ip0 = static_cast<int>(Kokkos::floor(yi1)) + is;
      const int jp0 = static_cast<int>(Kokkos::floor(yi2)) + js;
      const int kp0 = static_cast<int>(Kokkos::floor(yi3)) + ks;

      // zero the per-sink accumulator
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {
        for (int v = 0; v < 4; ++v) acc(v) = 0.0;
      });
      tm.team_barrier();

      // ---- process one control volume centred on (ic,jc,kc) ----------------------
      // (executed collectively by the whole team; all threads take identical branches)
      auto process_cv = [&](const int ic, const int jc, const int kc) {
        // interior-sink guard: control volume and its distance-2 extrapolation reads
        // must stay in this MeshBlock's active zone (+ its valid ghosts)
        if (ic < is+rctrl || ic > ie-rctrl ||
            jc < js+rctrl || jc > je-rctrl ||
            kc < ks+rctrl || kc > ke-rctrl) {
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

        // scatter the reset conserved values to u0 and the matching primitives to w0
        // (isothermal: w_d = u_d, w_vi = M_i/rho); without the w0 update the reset is
        // invisible to the next cycle's fluxes (ConToPrim already ran this stage)
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, 27), [&](const int c) {
          const int di = c%3 - 1, dj = (c/3)%3 - 1, dk = c/9 - 1;
          const int i = ic + di, j = jc + dj, k = kc + dk;
          for (int v = 0; v < 4; ++v) u0(m, v, k, j, i) = scr(v, c);
          const Real wd = scr(0, c);
          if (wd > 0.0) {
            w0(m, IDN, k, j, i) = wd;
            w0(m, IVX, k, j, i) = scr(1, c)/wd;
            w0(m, IVY, k, j, i) = scr(2, c)/wd;
            w0(m, IVZ, k, j, i) = scr(3, c)/wd;
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

  return TaskStatus::complete;
}

}  // namespace particles
