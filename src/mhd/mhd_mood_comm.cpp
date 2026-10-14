//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_mood_comm.cpp
//! \brief Exchange immutable stage data before the frozen-EMF MOOD loop.
//!
//! On a uniform periodic mesh, assign every face/edge to the cell with the same
//! integer indices (the positive-side owner). Pack its value into a cell-centered
//! payload. Existing CC exchange then fills face, edge and corner ghost regions,
//! including shared upper faces/edges. Only owned values are sent, so there are
//! no ambiguous writes at shared edges. This also works with a periodic self
//! neighbor and with the existing rank-packed MPI boundary path.
//!
//! Separate boundary objects keep these messages independent of the stage U/B
//! exchanges. All storage is on the device. No global-sized arrays or global
//! collectives are introduced. Physical and refinement boundaries need their own
//! owner/extension rules and are rejected for this method in the constructor.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"

namespace mhd {

TaskStatus MHD::SendMOODPrimitives(Driver *pdrive, int stage) {
  const auto &q = pmy_pack->pmesh->mb_indcs;
  const int nmb1 = pmy_pack->nmb_thispack-1, nv = nmhd;
  auto data = mood_prim, w = w0, b = bcc0;
  pbval_mood_prim->InitRecv(nv+3);
  par_for("mood_pack_primitives", DevExeSpace(), 0,nmb1,
          q.ks,q.ke,q.js,q.je,q.is,q.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nv; ++n) data(m,n,k,j,i) = w(m,n,k,j,i);
    for (int n=0; n<3; ++n) data(m,nv+n,k,j,i) = b(m,n,k,j,i);
  });
  return pbval_mood_prim->PackAndSendCC(mood_prim, mood_coarse);
}

TaskStatus MHD::RecvMOODPrimitives(Driver *pdrive, int stage) {
  auto status = pbval_mood_prim->RecvAndUnpackCC(mood_prim, mood_coarse);
  if (status != TaskStatus::complete) return status;
  pbval_mood_prim->ClearSend();
  pbval_mood_prim->ClearRecv();
  const int nmb1 = pmy_pack->nmb_thispack-1, nv = nmhd;
  auto data = mood_prim, w = w0, b = bcc0;
  par_for("mood_unpack_primitives", DevExeSpace(), 0,nmb1,
          0,data.extent_int(2)-1,0,data.extent_int(3)-1,0,data.extent_int(4)-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nv; ++n) w(m,n,k,j,i) = data(m,n,k,j,i);
    for (int n=0; n<3; ++n) b(m,n,k,j,i) = data(m,nv+n,k,j,i);
  });
  return TaskStatus::complete;
}

TaskStatus MHD::SendMOODCandidate(Driver *pdrive, int stage) {
  const auto &q = pmy_pack->pmesh->mb_indcs;
  const bool td = pmy_pack->pmesh->three_d;
  const int nmb1 = pmy_pack->nmb_thispack-1, nv = nmhd;
  const int off = (td ? 3 : 2)*nv;
  // Start each stage with the base-order edge stencil, independently of the
  // previous stage's detector flags. This is the only base corner composition.
  Kokkos::deep_copy(fb_level, 0);
  ComposeCornerEMF(q.is,q.ie+1,q.js,q.je+1,q.ks,td ? q.ke+1 : q.ks,false);
  pbval_mood_candidate->InitRecv(off+3);
  auto data = mood_candidate;
  auto f1 = uflx.x1f, f2 = uflx.x2f, f3 = uflx.x3f;
  auto e1 = efld.x1e, e2 = efld.x2e, e3 = efld.x3e;
  par_for("mood_pack_candidate", DevExeSpace(), 0,nmb1,
          q.ks,q.ke,q.js,q.je,q.is,q.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nv; ++n) {
      data(m,n,k,j,i) = f1(m,n,k,j,i);
      data(m,nv+n,k,j,i) = f2(m,n,k,j,i);
      if (td) data(m,2*nv+n,k,j,i) = f3(m,n,k,j,i);
    }
    data(m,off,k,j,i) = e1(m,k,j,i);
    data(m,off+1,k,j,i) = e2(m,k,j,i);
    data(m,off+2,k,j,i) = e3(m,k,j,i);
  });
  return pbval_mood_candidate->PackAndSendCC(mood_candidate, mood_coarse);
}

TaskStatus MHD::RecvMOODCandidate(Driver *pdrive, int stage) {
  auto status = pbval_mood_candidate->RecvAndUnpackCC(mood_candidate, mood_coarse);
  if (status != TaskStatus::complete) return status;
  pbval_mood_candidate->ClearSend();
  pbval_mood_candidate->ClearRecv();
  RestoreMOODCandidate();
  return TaskStatus::complete;
}

void MHD::RestoreMOODCandidate() {
  const bool td = pmy_pack->pmesh->three_d;
  const int nmb1 = pmy_pack->nmb_thispack-1, nv = nmhd;
  const int off = (td ? 3 : 2)*nv;
  auto data = mood_candidate;
  auto f1 = uflx.x1f, f2 = uflx.x2f, f3 = uflx.x3f;
  auto e1 = efld.x1e, e2 = efld.x2e, e3 = efld.x3e;
  par_for("mood_restore_candidate", DevExeSpace(), 0,nmb1,
          0,data.extent_int(2)-1,0,data.extent_int(3)-1,0,data.extent_int(4)-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nv; ++n) {
      f1(m,n,k,j,i) = data(m,n,k,j,i);
      f2(m,n,k,j,i) = data(m,nv+n,k,j,i);
      if (td) f3(m,n,k,j,i) = data(m,2*nv+n,k,j,i);
    }
    e1(m,k,j,i) = data(m,off,k,j,i);
    e2(m,k,j,i) = data(m,off+1,k,j,i);
    e3(m,k,j,i) = data(m,off+2,k,j,i);
    if (!td) {
      e1(m,k+1,j,i) = e1(m,k,j,i);
      e2(m,k+1,j,i) = e2(m,k,j,i);
    }
  });
}

} // namespace mhd
