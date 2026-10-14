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
  const auto &q = pmy_pack->pmesh->mb_indcs;
  const bool md = pmy_pack->pmesh->multi_d, td = pmy_pack->pmesh->three_d;
  const int nmb1 = pmy_pack->nmb_thispack-1, nv = nhydro;
  pbval_mood_candidate->InitRecv((td ? 3 : (md ? 2 : 1))*nv);
  auto data = mood_candidate;
  auto f1 = uflx.x1f, f2 = uflx.x2f, f3 = uflx.x3f;
  par_for("hyd_mood_pack_candidate", DevExeSpace(), 0,nmb1,
          q.ks,q.ke,q.js,q.je,q.is,q.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nv; ++n) {
      data(m,n,k,j,i) = f1(m,n,k,j,i);
      if (md) data(m,nv+n,k,j,i) = f2(m,n,k,j,i);
      if (td) data(m,2*nv+n,k,j,i) = f3(m,n,k,j,i);
    }
  });
  return pbval_mood_candidate->PackAndSendCC(mood_candidate, mood_coarse);
}

TaskStatus Hydro::RecvMOODCandidate(Driver *pdrive, int stage) {
  auto status = pbval_mood_candidate->RecvAndUnpackCC(mood_candidate, mood_coarse);
  if (status != TaskStatus::complete) return status;
  pbval_mood_candidate->ClearSend();
  pbval_mood_candidate->ClearRecv();
  const bool md = pmy_pack->pmesh->multi_d, td = pmy_pack->pmesh->three_d;
  const int nmb1 = pmy_pack->nmb_thispack-1, nv = nhydro;
  auto data = mood_candidate;
  auto f1 = uflx.x1f, f2 = uflx.x2f, f3 = uflx.x3f;
  par_for("hyd_mood_unpack_candidate", DevExeSpace(), 0,nmb1,
          0,data.extent_int(2)-1,0,data.extent_int(3)-1,0,data.extent_int(4)-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nv; ++n) {
      f1(m,n,k,j,i) = data(m,n,k,j,i);
      if (md) f2(m,n,k,j,i) = data(m,nv+n,k,j,i);
      if (td) f3(m,n,k,j,i) = data(m,2*nv+n,k,j,i);
    }
  });
  return TaskStatus::complete;
}

} // namespace hydro
