//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_mood_comm.cpp
//! \brief Exchange immutable stage primitives and high-order face fluxes for MOOD.
//!
//! A face belongs to the cell with the same indices (its positive-side owner).
//! The existing CC neighbor exchange fills its shared and ghost copies. All
//! subsequent revisions read stage primitives and replace only flagged faces.
//! This is independent of the Newtonian/SR/GR Riemann-solver specialization.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

namespace hydro {

TaskStatus Hydro::SendMOODPrimitives(Driver *pdrive, int stage) {
  const auto &q = pmy_pack->pmesh->mb_indcs;
  const int nmb1 = pmy_pack->nmb_thispack-1, nv = nhydro;
  auto data = mood_prim, w = w0;
  pbval_mood_prim->InitRecv(nv);
  par_for("hyd_mood_pack_primitives", DevExeSpace(), 0,nmb1,
          q.ks,q.ke,q.js,q.je,q.is,q.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nv; ++n) data(m,n,k,j,i) = w(m,n,k,j,i);
  });
  return pbval_mood_prim->PackAndSendCC(mood_prim, mood_coarse);
}

TaskStatus Hydro::RecvMOODPrimitives(Driver *pdrive, int stage) {
  auto status = pbval_mood_prim->RecvAndUnpackCC(mood_prim, mood_coarse);
  if (status != TaskStatus::complete) return status;
  pbval_mood_prim->ClearSend();
  pbval_mood_prim->ClearRecv();
  Kokkos::deep_copy(w0, mood_prim);
  return TaskStatus::complete;
}

TaskStatus Hydro::SendMOODCandidate(Driver *pdrive, int stage) {
  const int ndir = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  // each stage starts from the base scheme everywhere, independent of the previous
  // stage's cascade levels
  Kokkos::deep_copy(fb_level, 0);
  Kokkos::deep_copy(mood_touched, false);
  mood_revision = 0;
  mood_wait_flags = false;
  mood_wait_convergence = false;
  mood_revise = false;
  auto status = pbval_mood_candidate->InitRecv(ndir*nhydro);
  if (status != TaskStatus::complete) return status;
  return pbval_mood_candidate->PackAndSendOwnedFluxCC(uflx);
}

TaskStatus Hydro::RecvMOODCandidate(Driver *pdrive, int stage) {
  auto status = pbval_mood_candidate->RecvAndUnpackOwnedFluxCC(uflx);
  if (status != TaskStatus::complete) return status;
  pbval_mood_candidate->ClearSend();
  pbval_mood_candidate->ClearRecv();
  return TaskStatus::complete;
}

} // namespace hydro
