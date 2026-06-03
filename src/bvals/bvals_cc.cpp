//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! Mesh variables.
//! Prolongation of CC variables  occurs in ProlongateCC() function called from task list

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BValCC constructor:

MeshBoundaryValuesCC::MeshBoundaryValuesCC(MeshBlockPack *pp, ParameterInput *pin,
                                           bool z4c) :
  MeshBoundaryValues(pp, pin, z4c) {
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors.
//!
//! This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously
//! for ALL the MeshBlocks. This reduces the number of kernel launches when there are a
//! large number of MeshBlocks per MPI rank. Buffer data are then sent (via MPI) or copied
//! directly for periodic or block boundaries.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)
//! 5D Kokkos View of coarsened (restricted) array data also required with SMR/AMR

TaskStatus MeshBoundaryValuesCC::PackAndSendCC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
  // Opt-B: one team packs one (m,n) buffer, flattening ALL (var,k,j,i) elements across
  // the team's threads.  Dropping the per-variable league factor and the old
  // nkj x i loop split means corner/edge buffers no longer each occupy a whole team for a
  // handful of elements; every team gets its buffer's full element count to spread over.
  int nmn = nmb*nnghbr;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/nnghbr;
    const int n = (tmember.league_rank() - m*nnghbr);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      const int lev  = nghbr.d_view(m,n).lev;
      const int mlev = mblev.d_view(m);
      int il, iu, jl, ju, kl, ku;
      if (lev < mlev) {            // neighbor coarser: pack coarse-grid indices
        il = sbuf[n].icoar[0].bis; iu = sbuf[n].icoar[0].bie;
        jl = sbuf[n].icoar[0].bjs; ju = sbuf[n].icoar[0].bje;
        kl = sbuf[n].icoar[0].bks; ku = sbuf[n].icoar[0].bke;
      } else if (lev == mlev) {    // neighbor same level
        il = sbuf[n].isame[0].bis; iu = sbuf[n].isame[0].bie;
        jl = sbuf[n].isame[0].bjs; ju = sbuf[n].isame[0].bje;
        kl = sbuf[n].isame[0].bks; ku = sbuf[n].isame[0].bke;
      } else {                     // neighbor finer
        il = sbuf[n].ifine[0].bis; iu = sbuf[n].ifine[0].bie;
        jl = sbuf[n].ifine[0].bjs; ju = sbuf[n].ifine[0].bje;
        kl = sbuf[n].ifine[0].bks; ku = sbuf[n].ifine[0].bke;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int ncells = ni*nj*nk;
      const int ntot = nvar*ncells;   // all (var,k,j,i) elements for this buffer

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
      // in MeshBlockPacks, so array index equals (target_id - first_id)
      const int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      const int dn = nghbr.d_view(m,n).dest;
      const bool same_rank = (nghbr.d_view(m,n).rank == my_rank);

      // The flat element index `idx` in [0,ntot) decodes to (v,k,j,i) AND equals the
      // buffer linear index (i-il + ni*(j-jl + nj*(k-kl + nk*v))).
      if (same_rank && lev == mlev) {
        // SAME-RANK, SAME-LEVEL: copy directly u0(src interior) -> u0(dest ghosts),
        // skipping the buffer (matching unpack skips these).  Dest ghost base = recv
        // "isame" indices of the destination slot dn; regions have identical extents.
        const int dil = rbuf[dn].isame[0].bis;
        const int djl = rbuf[dn].isame[0].bjs;
        const int dkl = rbuf[dn].isame[0].bks;
        Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot), [&](const int idx) {
          const int v  = idx / ncells;
          const int r  = idx - v*ncells;
          const int kk = r / (ni*nj);
          const int r2 = r - kk*(ni*nj);
          const int jj = r2 / ni;
          const int ii = r2 - jj*ni;
          a(dm, v, dkl+kk, djl+jj, dil+ii) = a(m, v, kl+kk, jl+jj, il+ii);
        });
      } else {
        // Staged in buffer: same-rank finer/coarser -> rbuf[dn](dm); cross-rank -> sbuf[n](m).
        // Source is coarse_u0 when neighbor is coarser, else u0.
        Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot), [&](const int idx) {
          const int v  = idx / ncells;
          const int r  = idx - v*ncells;
          const int kk = r / (ni*nj);
          const int r2 = r - kk*(ni*nj);
          const int jj = r2 / ni;
          const int ii = r2 - jj*ni;
          const Real val = (lev < mlev) ? ca(m, v, kl+kk, jl+jj, il+ii)
                                        : a (m, v, kl+kk, jl+jj, il+ii);
          if (same_rank) {
            rbuf[dn].vars(dm, idx) = val;
          } else {
            sbuf[n].vars(m, idx) = val;
          }
        });
      }
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer

  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // If neighbor is at same level and data is for Z4c module, append data from coarse
      // array for higher-order prolongation
      if ((nghbr.d_view(m,n).lev == mblev.d_view(m)) && (is_z4c) && (multilevel)) {
        il = sbuf[n].isame_z4c.bis;
        iu = sbuf[n].isame_z4c.bie;
        jl = sbuf[n].isame_z4c.bjs;
        ju = sbuf[n].isame_z4c.bje;
        kl = sbuf[n].isame_z4c.bks;
        ku = sbuf[n].isame_z4c.bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
        int ndat = nvar*sbuf[n].isame_ndat; // size of same level data already in buff

        // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
        // in MeshBlockPacks, so array index equals (target_id - first_id)
        int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        int dn = nghbr.d_view(m,n).dest;

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // Inner (vector) loop over i
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            // load data from coarse_u0
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              rbuf[dn].vars(dm,ndat+ (i-il + ni*(j-jl + nj*(k-kl + nk*v))))=ca(m,v,k,j,i);
            });

          // else copy into send buffer for MPI communication below
          } else {
            // load data from coarse_u0
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              sbuf[n].vars(m,ndat+ (i-il + ni*(j-jl + nj*(k-kl + nk*v))) )=ca(m,v,k,j,i);
            });
          }
        });
      }
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  auto &is_z4c = is_z4c_;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          int data_size = nvar;
          if ( nghbr.h_view(m,n).lev < pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= sendbuf[n].icoar_ndat;
          } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
            if (is_z4c) {
              data_size *= sendbuf[n].isame_z4c_ndat;
            } else {
              data_size *= sendbuf[n].isame_ndat;
            }
          } else {
            data_size *= sendbuf[n].ifine_ndat;
          }
          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
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

TaskStatus MeshBoundaryValuesCC::RecvAndUnpackCC(DvceArray5D<Real> &a,
                                                 DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
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
#endif

  //----- STEP 2: buffers have all completed, so unpack

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &mblev = pmy_pack->pmb->mb_lev;
  int my_rank = global_variable::my_rank;

  // Opt-B: one team unpacks one (m,n) buffer, flattening all (var,k,j,i) elements across
  // the team's threads (drops the per-variable league factor and the nkj x i loop split).
  int nmn = nmb*nnghbr;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/nnghbr;
    const int n = (tmember.league_rank() - m*nnghbr);

    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      const int lev  = nghbr.d_view(m,n).lev;
      const int mlev = mblev.d_view(m);
      // SAME-RANK, SAME-LEVEL neighbors were written directly u0->u0 in the pack kernel
      // (no buffer staged), so skip them here.  Condition is team-uniform.
      if (nghbr.d_view(m,n).rank == my_rank && lev == mlev) {
        tmember.team_barrier();
        return;
      }
      int il, iu, jl, ju, kl, ku;
      if (lev < mlev) {            // neighbor coarser
        il = rbuf[n].icoar[0].bis; iu = rbuf[n].icoar[0].bie;
        jl = rbuf[n].icoar[0].bjs; ju = rbuf[n].icoar[0].bje;
        kl = rbuf[n].icoar[0].bks; ku = rbuf[n].icoar[0].bke;
      } else if (lev == mlev) {    // neighbor same level (cross-rank only; same-rank skipped)
        il = rbuf[n].isame[0].bis; iu = rbuf[n].isame[0].bie;
        jl = rbuf[n].isame[0].bjs; ju = rbuf[n].isame[0].bje;
        kl = rbuf[n].isame[0].bks; ku = rbuf[n].isame[0].bke;
      } else {                     // neighbor finer
        il = rbuf[n].ifine[0].bis; iu = rbuf[n].ifine[0].bie;
        jl = rbuf[n].ifine[0].bjs; ju = rbuf[n].ifine[0].bje;
        kl = rbuf[n].ifine[0].bks; ku = rbuf[n].ifine[0].bke;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int ncells = ni*nj*nk;
      const int ntot = nvar*ncells;
      const bool coarser = (lev < mlev);

      // flat idx in [0,ntot) decodes to (v,k,j,i) and equals the buffer linear index.
      Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot), [&](const int idx) {
        const int v  = idx / ncells;
        const int r  = idx - v*ncells;
        const int kk = r / (ni*nj);
        const int r2 = r - kk*(ni*nj);
        const int jj = r2 / ni;
        const int ii = r2 - jj*ni;
        const Real val = rbuf[n].vars(m, idx);
        if (coarser) {
          ca(m, v, kl+kk, jl+jj, il+ii) = val;   // neighbor coarser -> coarse_u0
        } else {
          a (m, v, kl+kk, jl+jj, il+ii) = val;    // same/finer -> u0
        }
      });
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // If neighbor is at same level and data is for Z4c module, unpack data from coarse
      // array for higher-order prolongation
      if ((nghbr.d_view(m,n).lev == mblev.d_view(m)) && (is_z4c) && (multilevel)) {
        il = rbuf[n].isame_z4c.bis;
        iu = rbuf[n].isame_z4c.bie;
        jl = rbuf[n].isame_z4c.bjs;
        ju = rbuf[n].isame_z4c.bje;
        kl = rbuf[n].isame_z4c.bks;
        ku = rbuf[n].isame_z4c.bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
        int ndat = nvar*rbuf[n].isame_ndat; // size of same level data packed in buff

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // load data into coarse_u0
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            ca(m,v,k,j,i) = rbuf[n].vars(m,ndat + (i-il + ni*(j-jl + nj*(k-kl + nk*v))) );
          });
        });
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  return TaskStatus::complete;
}
