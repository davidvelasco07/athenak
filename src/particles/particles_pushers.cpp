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
#include "particles.hpp"

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
  auto &gm_ = point_mass_gm;

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
    case ParticlesPusher::leapfrog:
      // TODO(SMOON) currently, particle integration is done before main time integrator.
      // Therefore, Particle task list is executed just once (see driver.cpp).
      // In the future, we need to integrate particles within time integrator, such that
      // total gas + particle momentum is conserved.
      // Before that, let's simply not distinguish stage 1 and 2 here.
//      if (stage == 1) {
        par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
        KOKKOS_LAMBDA(const int p) {
          // Step 1. Opening kick from v^n to v^(n+1/2)
          Real r3 = pow(SQR(pr(IPX,p)) + SQR(pr(IPY,p)) + SQR(pr(IPZ,p)), 1.5);
          Real src = -hdt_*gm_/r3;
          pr(IPVX,p) += src*pr(IPX,p);
          pr(IPVY,p) += src*pr(IPY,p);
          pr(IPVZ,p) += src*pr(IPZ,p);

          // Step 2. Drift from x^n to x^(n+1)
          pr(IPX,p) += dt_*pr(IPVX,p);
          pr(IPY,p) += dt_*pr(IPVY,p);
          pr(IPZ,p) += dt_*pr(IPVZ,p);
        });
//      } else if (stage == 2) {
        par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
        KOKKOS_LAMBDA(const int p) {
          // Step 3. Closing kick from v^(n+1/2) to v^(n+1)
          Real r3 = pow(SQR(pr(IPX,p)) + SQR(pr(IPY,p)) + SQR(pr(IPZ,p)), 1.5);
          Real src = -hdt_*gm_/r3;
          pr(IPVX,p) += src*pr(IPX,p);
          pr(IPVY,p) += src*pr(IPY,p);
          pr(IPVZ,p) += src*pr(IPZ,p);
        });
//      }
      break;
  default:
    break;
  }

  return TaskStatus::complete;
}
} // namespace particles
