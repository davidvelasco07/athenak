//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_fc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for face-centered (FC)
//! Mesh variables.
//! Prolongation of FC variables  occurs in ProlongateFC() function called from task list

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <utility>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BValFC constructor:

MeshBoundaryValuesFC::MeshBoundaryValuesFC(MeshBlockPack *pp, ParameterInput *pin) :
  MeshBoundaryValues(pp, pin, false) {
}

//----------------------------------------------------------------------------------------
//! \!fn void MeshBoundaryValuesFC::PackAndSendFC()
//! \brief Pack face-centered Mesh variables into boundary buffers and send to neighbors.
//!
//! As for cell-centered data, this routine packs ALL the buffers on ALL the faces, edges,
//! and corners simultaneously for all three components of face-fields on ALL the
//! MeshBlocks.
//!
//! Input array must be DvceFaceFld4D dimensioned (nmb, nx3, nx2, nx1)
//! DvceFaceFld4D of coarsened (restricted) fields also required with SMR/AMR

TaskStatus MeshBoundaryValuesFC::PackAndSendFC(DvceFaceFld4D<Real> &b,
                                               DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  int nmnv = 3*nmb;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank()/3;
    const int v = tmember.league_rank()%3;

    // scalar loop over neighbors to prevent race condition in overlapping assignments
    for (int n=0; n<nnghbr; ++n) {
      // only load buffers when neighbor exists
      if (nghbr.d_view(m,n).gid >= 0) {
        // if neighbor is at coarser level, use cindices to pack buffer
        // Note indices can be different for each component of face-centered field.
        int il, iu, jl, ju, kl, ku, ndat;
        if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
          il = sbuf[n].icoar[v].bis;
          iu = sbuf[n].icoar[v].bie;
          jl = sbuf[n].icoar[v].bjs;
          ju = sbuf[n].icoar[v].bje;
          kl = sbuf[n].icoar[v].bks;
          ku = sbuf[n].icoar[v].bke;
          ndat = sbuf[n].icoar_ndat;
        // if neighbor is at same level, use sindices to pack buffer
        } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
          il = sbuf[n].isame[v].bis;
          iu = sbuf[n].isame[v].bie;
          jl = sbuf[n].isame[v].bjs;
          ju = sbuf[n].isame[v].bje;
          kl = sbuf[n].isame[v].bks;
          ku = sbuf[n].isame[v].bke;
          ndat = sbuf[n].isame_ndat;
        // if neighbor is at finer level, use findices to pack buffer
        } else {
          il = sbuf[n].ifine[v].bis;
          iu = sbuf[n].ifine[v].bie;
          jl = sbuf[n].ifine[v].bjs;
          ju = sbuf[n].ifine[v].bje;
          kl = sbuf[n].ifine[v].bks;
          ku = sbuf[n].ifine[v].bke;
          ndat = sbuf[n].ifine_ndat;
        }
        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
        int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        int dn = nghbr.d_view(m,n).dest;

        // copy field components directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,n).rank == my_rank) {
          // if neighbor is at same or finer level, load data from b0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
            [&](const int idx) {
              int k = (idx)/nji;
              int j = (idx - k*nji)/ni;
              int i = (idx - k*nji - j*ni) + il;
              k += kl;
              j += jl;
              if (v==0) {
                rbuf[dn].vars(dm,i-il + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
              } else if (v==1) {
                rbuf[dn].vars(dm,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
              } else if (v==2) {
                rbuf[dn].vars(dm,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
              }
            });
          // if neighbor is at coarser level, load data from coarse_b0
          } else {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
            [&](const int idx) {
              int k = (idx)/nji;
              int j = (idx - k*nji)/ni;
              int i = (idx - k*nji - j*ni) + il;
              k += kl;
              j += jl;
              if (v==0) {
                rbuf[dn].vars(dm,i-il + ni*(j-jl + nj*(k-kl))) = cb.x1f(m,k,j,i);
              } else if (v==1) {
                rbuf[dn].vars(dm,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = cb.x2f(m,k,j,i);
              } else if (v==2) {
                rbuf[dn].vars(dm,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = cb.x3f(m,k,j,i);
              }
            });
          }

        // else copy field components into send buffer for MPI communication below
        } else {
          // if neighbor is at same or finer level, load data from b0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
            [&](const int idx) {
              int k = (idx)/nji;
              int j = (idx - k*nji)/ni;
              int i = (idx - k*nji - j*ni) + il;
              k += kl;
              j += jl;
              if (v==0) {
                sbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
              } else if (v==1) {
                sbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
              } else if (v==2) {
                sbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
              }
            });
          // if neighbor is at coarser level, load data from coarse_b0
          } else {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
            [&](const int idx) {
              int k = (idx)/nji;
              int j = (idx - k*nji)/ni;
              int i = (idx - k*nji - j*ni) + il;
              k += kl;
              j += jl;
              if (v==0) {
                sbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl))) = cb.x1f(m,k,j,i);
              } else if (v==1) {
                sbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = cb.x2f(m,k,j,i);
              } else if (v==2) {
                sbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl))) = cb.x3f(m,k,j,i);
              }
            });
          }
        }
      } // end if-neighbor-exists block
      tmember.team_barrier();
    }
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors = true;
  if (rank_packed_bvals_nvars_ != 3) {
    BuildRankPackedVarMetadata(3);
  }
  std::fill(send_var_reqs_.begin(), send_var_reqs_.end(), MPI_REQUEST_NULL);
  std::fill(send_var_hdr_reqs_.begin(), send_var_hdr_reqs_.end(), MPI_REQUEST_NULL);

  for (const auto &msg : send_var_msgs_) {
    for (int e = 0; e < msg.nentries; ++e) {
      const auto &entry = send_var_entries_[msg.entry_offset + e];
      int hidx = msg.hdr_offset + 3*e;
      rank_sendhdr_vars_(hidx    ) = entry.lid;
      rank_sendhdr_vars_(hidx + 1) = entry.dn;
      rank_sendhdr_vars_(hidx + 2) = entry.data_size;
    }
  }

  for (const auto &entry : send_var_entries_) {
    auto src = Kokkos::subview(sendbuf[entry.n].vars, entry.m,
                               std::make_pair(0, entry.data_size));
    auto dst = Kokkos::subview(rank_sendbuf_vars_,
                               std::make_pair(entry.offset, entry.offset + entry.data_size));
    Kokkos::deep_copy(dst, src);
  }
  Kokkos::fence();

  for (std::size_t i = 0; i < send_var_msgs_.size(); ++i) {
    const auto &msg = send_var_msgs_[i];
    int hdr_size = 3*msg.nentries;
    int ierr = MPI_Isend(rank_sendhdr_vars_.data() + msg.hdr_offset, hdr_size,
                         MPI_INT, msg.rank, 0, comm_vars, &send_var_hdr_reqs_[i]);
    if (ierr != MPI_SUCCESS) no_errors = false;
    ierr = MPI_Isend(rank_sendbuf_vars_.data() + msg.offset, msg.data_size,
                     MPI_ATHENA_REAL, msg.rank, 1, comm_vars, &send_var_reqs_[i]);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }

  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus MeshBoundaryValuesFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b,
                                                 DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (std::size_t i = 0; i < recv_var_reqs_.size(); ++i) {
    int test;
    int ierr = MPI_Test(&recv_var_hdr_reqs_[i], &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
    ierr = MPI_Test(&recv_var_reqs_[i], &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  int nmb_max = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  for (const auto &msg : recv_var_msgs_) {
    int data_offset = msg.offset;
    for (int e = 0; e < msg.nentries; ++e) {
      int hidx = msg.hdr_offset + 3*e;
      int lid = rank_recvhdr_vars_(hidx);
      int dn = rank_recvhdr_vars_(hidx + 1);
      int dsize = rank_recvhdr_vars_(hidx + 2);
      if ((lid < 0) || (lid >= nmb_max) || (dn < 0) || (dn >= nnghbr) || (dsize < 0)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Invalid rank-packed recv header in MHD FC"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      auto src = Kokkos::subview(rank_recvbuf_vars_,
                                 std::make_pair(data_offset, data_offset + dsize));
      auto dst = Kokkos::subview(recvbuf[dn].vars, lid, std::make_pair(0, dsize));
      Kokkos::deep_copy(dst, src);
      data_offset += dsize;
    }
  }
  Kokkos::fence();
#endif

  //----- STEP 2: buffers have all completed, so unpack 3-components of field

  auto &mblev = pmy_pack->pmb->mb_lev;
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (3*nmb), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank()/3;
    const int v = tmember.league_rank()%3;

    // scalar loop over neighbors to prevent race condition in overlapping assignments
    for (int n=0; n<nnghbr; ++n) {
      // only unpack buffers when neighbor exists
      if (nghbr.d_view(m,n).gid >= 0) {
        // if neighbor is at coarser level, use cindices to unpack buffer
        int il, iu, jl, ju, kl, ku, ndat;
        if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
          il = rbuf[n].icoar[v].bis;
          iu = rbuf[n].icoar[v].bie;
          jl = rbuf[n].icoar[v].bjs;
          ju = rbuf[n].icoar[v].bje;
          kl = rbuf[n].icoar[v].bks;
          ku = rbuf[n].icoar[v].bke;
          ndat = rbuf[n].icoar_ndat;
        // if neighbor is at same level, use sindices to unpack buffer
        } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
          il = rbuf[n].isame[v].bis;
          iu = rbuf[n].isame[v].bie;
          jl = rbuf[n].isame[v].bjs;
          ju = rbuf[n].isame[v].bje;
          kl = rbuf[n].isame[v].bks;
          ku = rbuf[n].isame[v].bke;
          ndat = rbuf[n].isame_ndat;
        // if neighbor is at finer level, use findices to unpack buffer
        } else {
          il = rbuf[n].ifine[v].bis;
          iu = rbuf[n].ifine[v].bie;
          jl = rbuf[n].ifine[v].bjs;
          ju = rbuf[n].ifine[v].bje;
          kl = rbuf[n].ifine[v].bks;
          ku = rbuf[n].ifine[v].bke;
          ndat = rbuf[n].ifine_ndat;
        }
        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // if neighbor is at same or finer level, load data directly into b0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
          [&](const int idx) {
            int k = (idx)/nji;
            int j = (idx - k*nji)/ni;
            int i = (idx - k*nji - j*ni) + il;
            k += kl;
            j += jl;
            if (v==0) {
              b.x1f(m,k,j,i) = rbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==1) {
              b.x2f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==2) {
              b.x3f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            }
          });
        // if neighbor is at coarser level, load data into coarse_b0
        } else {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
          [&](const int idx) {
            int k = (idx)/nji;
            int j = (idx - k*nji)/ni;
            int i = (idx - k*nji - j*ni) + il;
            k += kl;
            j += jl;
            if (v==0) {
              cb.x1f(m,k,j,i) = rbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==1) {
              cb.x2f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==2) {
              cb.x3f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            }
          });
        }
        tmember.team_barrier();
      }  // end if-neighbor-exists block
    }
  });  // end par_for_outer

  return TaskStatus::complete;
}
