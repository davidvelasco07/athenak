//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_int.cpp
//! \brief Uniform-grid integer cell halos, packed by destination rank.

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

TaskStatus MeshBoundaryValuesCCInt::InitRecvIntCC() {
  if (pmy_pack->pmesh->multilevel || !pmy_pack->pmesh->strictly_periodic) {
    std::cerr << "Integer CC halos require a uniform periodic mesh\n";
    std::exit(EXIT_FAILURE);
  }
  const auto q = pmy_pack->pmesh->mb_indcs;
  if (width_ < 1 || width_ > q.ng) {
    std::cerr << "Integer CC halo width must be between 1 and nghost\n";
    std::exit(EXIT_FAILURE);
  }
#if MPI_PARALLEL_ENABLED
  if (rank_packed_mesh_seq_ != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
    initialized_ = false;
  }
#endif
  if (!initialized_) {
    InitializeBuffers(1);
    const int nmb = pmy_pack->nmb_thispack, nn = pmy_pack->pmb->nnghbr;
    auto trim = [&](int &sl, int &su, int &rl, int &ru, int lo, int hi) {
      if (ru < lo) { sl = lo; su = lo+width_-1; rl = lo-width_; ru = lo-1; }
      if (rl > hi) { sl = hi-width_+1; su = hi; rl = hi+1; ru = hi+width_; }
    };
    for (int n=0; n<nn; ++n) {
      if (recvbuf[n].isame_ndat == 0) continue;
      auto &s = sendbuf[n].isame[0];
      auto &r = recvbuf[n].isame[0];
      trim(s.bis,s.bie,r.bis,r.bie,q.is,q.ie);
      trim(s.bjs,s.bje,r.bjs,r.bje,q.js,q.je);
      trim(s.bks,s.bke,r.bks,r.bke,q.ks,q.ke);
      sendbuf[n].isame_ndat = (s.bie-s.bis+1)*(s.bje-s.bjs+1)*(s.bke-s.bks+1);
      recvbuf[n].isame_ndat = (r.bie-r.bis+1)*(r.bje-r.bjs+1)*(r.bke-r.bks+1);
      // This object uses only integer storage; drop the generic Real buffers.
      sendbuf[n].vars = DvceArray2D<Real>();
      recvbuf[n].vars = DvceArray2D<Real>();
    }
    Kokkos::realloc(local_offset_,nmb*nn);
    auto offsets = Kokkos::create_mirror_view(local_offset_);
    int total = 0;
    for (int m=0; m<nmb; ++m) for (int n=0; n<nn; ++n) {
      offsets(m*nn+n) = -1;
      const auto nb = pmy_pack->pmb->nghbr.h_view(m,n);
      if (nb.gid >= 0 && nb.rank == global_variable::my_rank) {
        offsets(m*nn+n) = total;
        total += recvbuf[n].isame_ndat;
      }
    }
    Kokkos::deep_copy(local_offset_,offsets);
    Kokkos::realloc(local_recv_,std::max(1,total));
#if MPI_PARALLEL_ENABLED
    BuildRankPackedVarMetadata(1);
    Kokkos::realloc(int_send_,rank_sendbuf_vars_.extent_int(0));
    Kokkos::realloc(int_recv_,rank_recvbuf_vars_.extent_int(0));
    rank_sendbuf_vars_ = DvceArray1D<Real>();
    rank_recvbuf_vars_ = DvceArray1D<Real>();
#endif
    initialized_ = true;
  }
#if MPI_PARALLEL_ENABLED
  std::fill(recv_var_reqs_.begin(),recv_var_reqs_.end(),MPI_REQUEST_NULL);
  for (std::size_t i=0; i<recv_var_msgs_.size(); ++i) {
    const auto &msg = recv_var_msgs_[i];
    if (MPI_Irecv(int_recv_.data()+msg.offset,msg.data_size,MPI_INT,msg.rank,1,
                  comm_vars,&recv_var_reqs_[i]) != MPI_SUCCESS) {
      std::cerr << "MPI error posting integer CC receives\n";
      std::exit(EXIT_FAILURE);
    }
  }
#endif
  return TaskStatus::complete;
}

TaskStatus MeshBoundaryValuesCCInt::PackAndSendIntCC(DvceArray4D<int> &a) {
  const int nmb = pmy_pack->nmb_thispack, nn = pmy_pack->pmb->nnghbr;
  const int rank = global_variable::my_rank;
  auto nb = pmy_pack->pmb->nghbr.d_view;
  auto gids = pmy_pack->pmb->mb_gid.d_view;
  auto &sbuf = sendbuf;
  auto local = local_recv_, offsets = local_offset_;
#if MPI_PARALLEL_ENABLED
  auto send = int_send_, sendoff = send_agg_offset_;
#endif
  Kokkos::parallel_for("SendIntCC",Kokkos::TeamPolicy<>(nmb*nn,Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t team) {
    const int m = team.league_rank()/nn, n = team.league_rank()%nn;
    if (nb(m,n).gid < 0) return;
    const auto idx = sbuf[n].isame[0];
    const int ni = idx.bie-idx.bis+1, nj = idx.bje-idx.bjs+1;
    const int nk = idx.bke-idx.bks+1;
    const int dm = nb(m,n).gid-gids(0), dn = nb(m,n).dest;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,nk*nj),[&](int kj) {
      const int k = idx.bks+kj/nj, j = idx.bjs+kj%nj;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,idx.bis,idx.bie+1),[&](int i) {
        const int bi = i-idx.bis+ni*kj;
        if (nb(m,n).rank == rank) {
          local(offsets(dm*nn+dn)+bi) = a(m,k,j,i);
        } else {
#if MPI_PARALLEL_ENABLED
          send(sendoff(m*nn+n)+bi) = a(m,k,j,i);
#endif
        }
      });
    });
  });
#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  std::fill(send_var_reqs_.begin(),send_var_reqs_.end(),MPI_REQUEST_NULL);
  for (std::size_t i=0; i<send_var_msgs_.size(); ++i) {
    const auto &msg = send_var_msgs_[i];
    if (MPI_Isend(int_send_.data()+msg.offset,msg.data_size,MPI_INT,msg.rank,1,
                  comm_vars,&send_var_reqs_[i]) != MPI_SUCCESS) {
      std::cerr << "MPI error posting integer CC sends\n";
      std::exit(EXIT_FAILURE);
    }
  }
#endif
  return TaskStatus::complete;
}

TaskStatus MeshBoundaryValuesCCInt::RecvAndUnpackIntCC(
    DvceArray4D<int> &a, DvceArray4D<bool> &changed) {
#if MPI_PARALLEL_ENABLED
  bool pending = false;
  for (auto &req : recv_var_reqs_) {
    int complete;
    if (MPI_Test(&req,&complete,MPI_STATUS_IGNORE) != MPI_SUCCESS) {
      std::cerr << "MPI error testing integer CC receives\n";
      std::exit(EXIT_FAILURE);
    }
    pending = pending || !complete;
  }
  if (pending) return TaskStatus::incomplete;
  auto recv = int_recv_, recvoff = recv_agg_offset_;
#endif
  const int nmb = pmy_pack->nmb_thispack, nn = pmy_pack->pmb->nnghbr;
  const int rank = global_variable::my_rank;
  auto nb = pmy_pack->pmb->nghbr.d_view;
  auto &rbuf = recvbuf;
  auto local = local_recv_, offsets = local_offset_;
  Kokkos::parallel_for("RecvIntCC",Kokkos::TeamPolicy<>(nmb*nn,Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t team) {
    const int m = team.league_rank()/nn, n = team.league_rank()%nn;
    if (nb(m,n).gid < 0) return;
    const auto idx = rbuf[n].isame[0];
    const int ni = idx.bie-idx.bis+1, nj = idx.bje-idx.bjs+1;
    const int nk = idx.bke-idx.bks+1;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team,nk*nj),[&](int kj) {
      const int k = idx.bks+kj/nj, j = idx.bjs+kj%nj;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,idx.bis,idx.bie+1),[&](int i) {
        const int bi = i-idx.bis+ni*kj;
        int tier = 0;
        if (nb(m,n).rank == rank) tier = local(offsets(m*nn+n)+bi);
#if MPI_PARALLEL_ENABLED
        else tier = recv(recvoff(m*nn+n)+bi);
#endif
        changed(m,k,j,i) = tier > a(m,k,j,i);
        a(m,k,j,i) = tier;
      });
    });
  });
  return TaskStatus::complete;
}
