//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "gravity/gravity.hpp"
#include "particles.hpp"
#include "particle_mesh.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  //auto &indcs = pmy_pack->pmesh->mb_indcs;
  //int is = indcs.is;
  //int js = indcs.js;
  //int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  //auto &mbsize = pmy_pack->pmb->mb_size;
  //auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto dt_ = (pmy_pack->pmesh->dt);
  auto hdt_ = 0.5*dt_;
  //auto gids = pmy_pack->gids;

  switch (pusher) {
    case ParticlesPusher::drift:
      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        //int m = pi(PGID,p) - gids;
        //int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        pr(IPX,p) += 0.5*dt_*pr(IPVX,p);

        if (multi_d) {
          //int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
          pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
        }
        if (three_d) {
          //int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
          pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);
        }
      });
      break;
    case ParticlesPusher::rk3: {
      // Evolve particle position and velocity with the same SSPRK(3,3) stages
      // as the gas. Matched gravity weights preserve equal/opposite impulses.
      if (ppm != nullptr && pmy_pack->pgrav != nullptr) {
        ppm->GatherGravity(pmy_pack->pgrav->phi, prtcl_rdata, prtcl_idata,
                           nprtcl_thispack);
      }
      const Real g0 = pdriver->gam0[stage-1];
      const Real g1 = pdriver->gam1[stage-1];
      const Real bdt = pdriver->beta[stage-1]*dt_;
      const bool first_stage = (stage == 1);
      par_for("part_rk3",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        if (first_stage) {
          pr(IPX0,p) = pr(IPX,p); pr(IPY0,p) = pr(IPY,p); pr(IPZ0,p) = pr(IPZ,p);
          pr(IPVX0,p) = pr(IPVX,p); pr(IPVY0,p) = pr(IPVY,p);
          pr(IPVZ0,p) = pr(IPVZ,p);
        }
        // Update positions before velocities: the derivative is the input-stage
        // velocity. Migration carries all registers; periodic wrapping shifts
        // IPX0/IPY0/IPZ0 alongside the current position before the next stage.
        pr(IPX,p) = g0*pr(IPX,p) + g1*pr(IPX0,p) + bdt*pr(IPVX,p);
        pr(IPY,p) = g0*pr(IPY,p) + g1*pr(IPY0,p) + bdt*pr(IPVY,p);
        pr(IPZ,p) = g0*pr(IPZ,p) + g1*pr(IPZ0,p) + bdt*pr(IPVZ,p);
        pr(IPVX,p) = g0*pr(IPVX,p) + g1*pr(IPVX0,p) + bdt*pr(IPGX,p);
        pr(IPVY,p) = g0*pr(IPVY,p) + g1*pr(IPVY0,p) + bdt*pr(IPGY,p);
        pr(IPVZ,p) = g0*pr(IPVZ,p) + g1*pr(IPVZ0,p) + bdt*pr(IPGZ,p);
      });
      break;
    }
    case ParticlesPusher::leapfrog: {
      // Gather the gravitational acceleration -grad(phi) from the multigrid
      // potential onto each particle (writes IPGX/IPGY/IPGZ). phi already
      // includes the particle's own deposited density (folded into the Poisson
      // RHS), so on a uniform grid the symmetric TSC self-force cancels.
      if (ppm != nullptr && pmy_pack->pgrav != nullptr) {
        ppm->GatherGravity(pmy_pack->pgrav->phi, prtcl_rdata, prtcl_idata,
                           nprtcl_thispack);
      }
      // RK2-KDK leapfrog, stage-synchronized with the per-stage gravity solve:
      //   stage 1: half-kick v^n -> v^(n+1/2) using a(x^n); drift x^n -> x^(n+1)
      //   stage 2: half-kick v^(n+1/2) -> v^(n+1) using a(x^(n+1))
      const bool do_drift = (stage == 1);
      par_for("part_kdk",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        // Half-kick using the gathered acceleration.
        pr(IPVX,p) += hdt_*pr(IPGX,p);
        pr(IPVY,p) += hdt_*pr(IPGY,p);
        pr(IPVZ,p) += hdt_*pr(IPGZ,p);
        // Full drift (stage 1 only). Save the pre-drift (start-of-step) position
        // first: AccreteMass uses it to detect sink-cell crossings. Unconditional
        // (once per cycle), so IPX0 is always current when accretion runs.
        if (do_drift) {
          pr(IPX0,p) = pr(IPX,p);
          pr(IPY0,p) = pr(IPY,p);
          pr(IPZ0,p) = pr(IPZ,p);
          pr(IPX,p) += dt_*pr(IPVX,p);
          pr(IPY,p) += dt_*pr(IPVY,p);
          pr(IPZ,p) += dt_*pr(IPVZ,p);
        }
      });
      break;
    }
  default:
    break;
  }

  return TaskStatus::complete;
}
} // namespace particles
