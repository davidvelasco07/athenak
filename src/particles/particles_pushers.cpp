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
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto dt_ = (pmy_pack->pmesh->dt);
  auto hdt_ = 0.5*dt_;
  auto gids = pmy_pack->gids;

  switch (pusher) {
    case ParticlesPusher::drift:
      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        int m = pi(PGID,p) - gids;
        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        pr(IPX,p) += hdt_*pr(IPVX,p);
        if (multi_d) {
          int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
          pr(IPY,p) += hdt_*pr(IPVY,p);
        }
        if (three_d) {
          int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
          pr(IPZ,p) += hdt_*pr(IPVZ,p);
        }
      });
      break;
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
        // Full drift (stage 1 only).
        if (do_drift) {
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
