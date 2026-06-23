//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_accretion.cpp
//! \brief control-volume sink-particle accretion (Phase 4a).
//!
//! Port of the Gong & Ostriker (2013) / Moon control-volume accretion from Chang-Goo
//! Kim's Athena++ fork (athena/src/particles/sink_particles.cpp). For each sink, the
//! conserved gas (rho, momenta) in the 27-cell control volume centred on the sink cell is
//! integrated, the control volume is then reset by an outflow-like extrapolation from the
//! cells just outside it, and the difference between the pre- and post-reset content is
//! the mass/momentum that flowed in during the step -- which is dumped conservatively onto
//! the sink:
//!     dM_sink = M^{n+1} - M^{n+1}_ctrl ,   v_new = (m v + dM)/(m + dm),  m += dm.
//! Operator-split: runs only at the last RK stage. Resets IDN + IM1..IM3 (isothermal); a
//! later pass should also handle energy/B and the magnetic field.
//!
//! LIMITATIONS (first cut): control volume must lie in active cells (interior sink); sinks
//! near a MeshBlock boundary, overlapping control volumes, the sink-cell-crossing (old
//! control-volume) correction, and AMR are not yet handled.

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
  // gas conserved variables
  if (pmy_pack->phydro == nullptr && pmy_pack->pmhd == nullptr) {
    return TaskStatus::complete;
  }
  auto u0 = (pmy_pack->pmhd != nullptr) ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;

  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int gids = pmy_pack->gids;
  const int npart = nprtcl_thispack;
  const int rctrl = 1;

  par_for("sink_accrete", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID, p) - gids;
    Real dx1 = mbsize.d_view(m).dx1;
    Real dx2 = mbsize.d_view(m).dx2;
    Real dx3 = mbsize.d_view(m).dx3;
    Real dV = dx1*dx2*dx3;

    // cell containing the sink
    int ip = static_cast<int>((pr(IPX, p) - mbsize.d_view(m).x1min)/dx1) + is;
    int jp = static_cast<int>((pr(IPY, p) - mbsize.d_view(m).x2min)/dx2) + js;
    int kp = static_cast<int>((pr(IPZ, p) - mbsize.d_view(m).x3min)/dx3) + ks;

    // require the control volume (and its distance-2 reads) to stay in this MeshBlock's
    // active+ghost cells; skip otherwise (interior-sink restriction for now)
    if (ip < is+rctrl || ip > ie-rctrl ||
        jp < js+rctrl || jp > je-rctrl ||
        kp < ks+rctrl || kp > ke-rctrl) {
      return;
    }

    // Step 1(a): integrate conserved rho, momenta over the control volume (pre-reset)
    Real s1[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = kp-rctrl; k <= kp+rctrl; ++k) {
      for (int j = jp-rctrl; j <= jp+rctrl; ++j) {
        for (int i = ip-rctrl; i <= ip+rctrl; ++i) {
          for (int v = 0; v < 4; ++v) s1[v] += u0(m, v, k, j, i);
        }
      }
    }

    // Step 1(b): reset the control volume by extrapolation from the cells just outside it.
    // rctrl=1 -> a single shell: 6 faces (copy), 8 corners (avg of 3), 12 edges (avg of 2),
    // then the centre cell (6-point average of the reset faces). All shell cells read only
    // cells outside the control volume; the centre reads the reset faces, so it goes last.
    for (int v = 0; v < 4; ++v) {
      // 6 faces: copy from the neighbour one cell further out
      for (int s = -1; s <= 1; s += 2) {
        u0(m, v, kp, jp, ip+s) = u0(m, v, kp, jp, ip+2*s);
        u0(m, v, kp, jp+s, ip) = u0(m, v, kp, jp+2*s, ip);
        u0(m, v, kp+s, jp, ip) = u0(m, v, kp+2*s, jp, ip);
      }
      // 8 corners: average of the three face-adjacent neighbours just outside
      for (int kc = -1; kc <= 1; kc += 2) {
        for (int jc = -1; jc <= 1; jc += 2) {
          for (int ic = -1; ic <= 1; ic += 2) {
            u0(m, v, kp+kc, jp+jc, ip+ic) = (1.0/3.0)*(
                u0(m, v, kp+kc,    jp+jc,    ip+2*ic) +
                u0(m, v, kp+kc,    jp+2*jc,  ip+ic) +
                u0(m, v, kp+2*kc,  jp+jc,    ip+ic));
          }
        }
      }
      // 12 edges: average of the two face-adjacent neighbours just outside
      for (int kc = -1; kc <= 1; kc += 2) {       // edges along x1: (ip, jp+jc, kp+kc)
        for (int jc = -1; jc <= 1; jc += 2) {
          u0(m, v, kp+kc, jp+jc, ip) = 0.5*(
              u0(m, v, kp+2*kc, jp+jc,   ip) +
              u0(m, v, kp+kc,   jp+2*jc, ip));
        }
      }
      for (int kc = -1; kc <= 1; kc += 2) {       // edges along x2: (ip+ic, jp, kp+kc)
        for (int ic = -1; ic <= 1; ic += 2) {
          u0(m, v, kp+kc, jp, ip+ic) = 0.5*(
              u0(m, v, kp+2*kc, jp, ip+ic) +
              u0(m, v, kp+kc,   jp, ip+2*ic));
        }
      }
      for (int jc = -1; jc <= 1; jc += 2) {       // edges along x3: (ip+ic, jp+jc, kp)
        for (int ic = -1; ic <= 1; ic += 2) {
          u0(m, v, kp, jp+jc, ip+ic) = 0.5*(
              u0(m, v, kp, jp+2*jc, ip+ic) +
              u0(m, v, kp, jp+jc,   ip+2*ic));
        }
      }
      // centre cell: 6-point average of the (now reset) face cells
      u0(m, v, kp, jp, ip) = (1.0/6.0)*(
          u0(m, v, kp, jp, ip-1) + u0(m, v, kp, jp, ip+1) +
          u0(m, v, kp, jp-1, ip) + u0(m, v, kp, jp+1, ip) +
          u0(m, v, kp-1, jp, ip) + u0(m, v, kp+1, jp, ip));
    }

    // Step 1(c): integrate the reset control volume
    Real s2[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = kp-rctrl; k <= kp+rctrl; ++k) {
      for (int j = jp-rctrl; j <= jp+rctrl; ++j) {
        for (int i = ip-rctrl; i <= ip+rctrl; ++i) {
          for (int v = 0; v < 4; ++v) s2[v] += u0(m, v, k, j, i);
        }
      }
    }

    // Step 1(d) + Step 3: accreted mass/momentum -> conservatively update the sink
    Real dm  = (s1[0] - s2[0])*dV;
    Real dM1 = (s1[1] - s2[1])*dV;
    Real dM2 = (s1[2] - s2[2])*dV;
    Real dM3 = (s1[3] - s2[3])*dV;
    Real mp = pr(IPM, p);
    Real minv = 1.0/(mp + dm);
    pr(IPVX, p) = (mp*pr(IPVX, p) + dM1)*minv;
    pr(IPVY, p) = (mp*pr(IPVY, p) + dM2)*minv;
    pr(IPVZ, p) = (mp*pr(IPVZ, p) + dM3)*minv;
    pr(IPM, p)  = mp + dm;
  });

  return TaskStatus::complete;
}

}  // namespace particles
