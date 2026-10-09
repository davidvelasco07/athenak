//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file amr_shared_faces.cpp
//! \brief Makes the two copies of every MeshBlock face touched by an AMR regrid agree.
//!
//! Each MeshBlock stores and evolves its own copy of the face-centered field on each of
//! its six faces, and boundary communication never overwrites these active faces. Before
//! a regrid the copies agree: same-level neighbors hold identical values, and a coarse
//! face equals the average of the fine faces covering it. A regrid breaks this in two
//! cases, because RefineFC fills every face of a new child by prolongation from its
//! parent:
//!   - a child next to a same-level MeshBlock that already existed keeps the
//!     prolongated face, while the neighbor keeps its evolved fine values;
//!   - a child created on the coarse side of a level jump (refined from a coarser
//!     parent while the block across the face was refined too) holds prolongated values
//!     instead of the restriction of the fine faces opposite it.
//! The two sides then reconstruct from different normal fields on the same face, compute
//! different fluxes there and break the conservation of mass, momentum and energy, and
//! the face carries a jump in B that no per-MeshBlock div(B) check can see.
//!
//! ReconcileSharedFacesFC() is called after the first boundary exchange of a regrid and
//! before RepairAMRFC(). Each newly refined child takes, on each face:
//!   - from a same-level neighbor that was not refined by this regrid, its copy;
//!   - from a same-level neighbor that was also refined (from another parent), its copy
//!     when that neighbor lies on the child's inner side (the lower block's copy wins);
//!   - from finer neighbors, the restriction (average) of their faces.
//! Every replaced group of fine faces keeps the average of the parent face it came from,
//! so RepairAMRFC's Toth & Roe rebuild of the interior faces stays divergence-free.
//! Derefined MeshBlocks need nothing: the restriction of consistent copies is
//! consistent with every neighbor.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "bvals/bvals.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// one face (kind 0) or one quadrant of a face (kind 1) moved from a sender to a receiver
struct SharedFaceItem {
  int peer;      // rank of the MeshBlock on the other side
  int lm;        // local index of the MeshBlock on this rank (sender or receiver)
  int rgid;      // gid of the receiving MeshBlock (sort key)
  int ax, side;  // face normal axis and side (0 inner, 1 outer) of the RECEIVING block
  int kind;      // 0: same-level face, 1: restriction of a finer block's face
  int o1, o2;    // quadrant of the receiving face covered by the finer block (kind 1)
  int offset;    // start of the item's data in the send or recv buffer
};

bool ItemLess(const SharedFaceItem &a, const SharedFaceItem &b) {
  if (a.peer != b.peer) return a.peer < b.peer;
  if (a.rgid != b.rgid) return a.rgid < b.rgid;
  if (a.ax != b.ax) return a.ax < b.ax;
  if (a.side != b.side) return a.side < b.side;
  if (a.o2 != b.o2) return a.o2 < b.o2;
  return a.o1 < b.o1;
}

using LocKey = std::array<std::int32_t, 4>;   // (level, lx1, lx2, lx3)
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ReconcileSharedFacesFC()
//! \brief See file header. Uses amr_origin (set in RedistAndRefineMeshBlocks) and the
//! logical locations of all MeshBlocks, both known on every rank, so senders and
//! receivers derive the same list of transfers without communication.

void MeshRefinement::ReconcileSharedFacesFC(DvceFaceFld4D<Real> &b) {
  Mesh *pm = pmy_mesh;
  auto &indcs = pm->mb_indcs;
  const int nmb_total = pm->nmb_total;
  const int gids = pm->gids_eachrank[global_variable::my_rank];
  const int nmb = pm->nmb_eachrank[global_variable::my_rank];
  const int my_rank = global_variable::my_rank;

  // nothing to do unless this regrid refined some MeshBlock
  bool any_refined = false;
  for (int g=0; g<nmb_total; ++g) {
    if (amr_origin.h_view(g) > 0) {any_refined = true; break;}
  }
  if (!any_refined) return;

  // geometry: active axes, block size and the refinement ratio along each axis
  const bool active[3] = {true, pm->multi_d, pm->three_d};
  const int nx[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  const int is[3] = {indcs.is, indcs.js, indcs.ks};
  const int ie[3] = {indcs.ie, indcs.je, indcs.ke};
  const int nroot[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  const BoundaryFlag pflag = BoundaryFlag::periodic;
  const bool periodic[3] = {pm->mesh_bcs[BoundaryFace::inner_x1] == pflag,
                            pm->mesh_bcs[BoundaryFace::inner_x2] == pflag,
                            pm->mesh_bcs[BoundaryFace::inner_x3] == pflag};

  // sorted list of the logical locations of all MeshBlocks
  std::vector<std::pair<LocKey, int>> loc(nmb_total);
  for (int g=0; g<nmb_total; ++g) {
    const LogicalLocation &ll = pm->lloc_eachmb[g];
    loc[g] = {LocKey{ll.level, ll.lx1, ll.lx2, ll.lx3}, g};
  }
  std::sort(loc.begin(), loc.end());
  auto find = [&](int lev, const int q[3]) -> int {
    LocKey k{lev, q[0], q[1], q[2]};
    auto it = std::lower_bound(loc.begin(), loc.end(), std::make_pair(k, -1));
    return (it != loc.end() && it->first == k) ? it->second : -1;
  };
  auto origin = [&](int g) { return amr_origin.h_view(g); };
  // does block x take its copy of the face across its side 'side' from same-level y?
  auto takes_same = [&](int gx, int gy, int side) {
    return origin(gx) > 0 && (origin(gy) == 0 || (origin(gy) > 0 && side == 0));
  };

  // enumerate transfers involving MeshBlocks on this rank
  std::vector<SharedFaceItem> sends, recvs;
  for (int lm=0; lm<nmb; ++lm) {
    const int gx = gids + lm;
    const LogicalLocation &ll = pm->lloc_eachmb[gx];
    const int lv = ll.level;
    const int lx[3] = {ll.lx1, ll.lx2, ll.lx3};
    for (int ax=0; ax<3; ++ax) {
      if (!active[ax]) continue;
      const int t1 = (ax + 1) % 3, t2 = (ax + 2) % 3;
      const int nl = nroot[ax] << (lv - pm->root_level);
      for (int side=0; side<2; ++side) {
        int q[3] = {lx[0], lx[1], lx[2]};
        q[ax] = lx[ax] + ((side == 1) ? 1 : -1);
        if (q[ax] < 0 || q[ax] >= nl) {
          if (!periodic[ax]) continue;
          q[ax] = (q[ax] + nl) % nl;
        }
        const int gy = find(lv, q);
        if (gy >= 0) {             // same-level neighbor
          if (takes_same(gx, gy, side)) {
            recvs.push_back({pm->rank_eachmb[gy], lm, gx, ax, side, 0, 0, 0, 0});
          }
          if (takes_same(gy, gx, 1 - side)) {
            sends.push_back({pm->rank_eachmb[gy], lm, gy, ax, 1 - side, 0, 0, 0, 0});
          }
          continue;
        }
        if (origin(gx) > 0) {      // finer neighbors: receive the restriction of each
          for (int o2=0; o2<(active[t2] ? 2 : 1); ++o2) {
            for (int o1=0; o1<(active[t1] ? 2 : 1); ++o1) {
              int qf[3];
              qf[ax] = 2*q[ax] + ((side == 1) ? 0 : 1);
              qf[t1] = active[t1] ? 2*lx[t1] + o1 : lx[t1];
              qf[t2] = active[t2] ? 2*lx[t2] + o2 : lx[t2];
              const int gf = find(lv + 1, qf);
              if (gf >= 0) {
                recvs.push_back({pm->rank_eachmb[gf], lm, gx, ax, side, 1, o1, o2, 0});
              }
            }
          }
        }
        if (lv > pm->root_level) {  // coarser neighbor: send it my restricted face
          int qc[3];
          qc[ax] = q[ax] >> 1;
          qc[t1] = active[t1] ? (lx[t1] >> 1) : lx[t1];
          qc[t2] = active[t2] ? (lx[t2] >> 1) : lx[t2];
          const int gc = find(lv - 1, qc);
          if (gc >= 0 && origin(gc) > 0) {
            const int o1 = active[t1] ? (lx[t1] & 1) : 0;
            const int o2 = active[t2] ? (lx[t2] & 1) : 0;
            sends.push_back({pm->rank_eachmb[gc], lm, gc, ax, 1 - side, 1, o1, o2, 0});
          }
        }
      }
    }
  }

  // data per item, and buffer offsets in a deterministic order shared by both sides
  auto item_size = [&](const SharedFaceItem &it) {
    const int t1 = (it.ax + 1) % 3, t2 = (it.ax + 2) % 3;
    const int r1 = (it.kind == 1 && active[t1]) ? 2 : 1;
    const int r2 = (it.kind == 1 && active[t2]) ? 2 : 1;
    return (nx[t1]/r1)*(nx[t2]/r2);
  };
  std::sort(sends.begin(), sends.end(), ItemLess);
  std::sort(recvs.begin(), recvs.end(), ItemLess);
  int nsend_data = 0, nrecv_data = 0;
  for (auto &it : sends) {it.offset = nsend_data; nsend_data += item_size(it);}
  for (auto &it : recvs) {it.offset = nrecv_data; nrecv_data += item_size(it);}
  // items with a peer on this rank are read straight from the send buffer: the sorted
  // self-sends and self-recvs pair up one to one
  std::vector<int> self_send;
  for (auto &it : sends) {if (it.peer == my_rank) self_send.push_back(it.offset);}
  int iself = 0;
  for (auto &it : recvs) {
    if (it.peer == my_rank) it.offset = -1 - self_send[iself++];   // < 0: in send buffer
  }

  const int nsends = static_cast<int>(sends.size());
  const int nrecvs = static_cast<int>(recvs.size());
  DvceArray1D<Real> sbuf("shared_fc_send", std::max(nsend_data, 1));
  DvceArray1D<Real> rbuf("shared_fc_recv", std::max(nrecv_data, 1));
  // item tables on device: lm, ax, side, kind, o1, o2, offset
  DualArray2D<int> sitems("shared_fc_sitems", std::max(nsends, 1), 7);
  DualArray2D<int> ritems("shared_fc_ritems", std::max(nrecvs, 1), 7);
  for (int i=0; i<nsends; ++i) {
    const auto &it = sends[i];
    int v[7] = {it.lm, it.ax, it.side, it.kind, it.o1, it.o2, it.offset};
    for (int c=0; c<7; ++c) sitems.h_view(i, c) = v[c];
  }
  for (int i=0; i<nrecvs; ++i) {
    const auto &it = recvs[i];
    int v[7] = {it.lm, it.ax, it.side, it.kind, it.o1, it.o2, it.offset};
    for (int c=0; c<7; ++c) ritems.h_view(i, c) = v[c];
  }
  sitems.template modify<HostMemSpace>();
  sitems.template sync<DevExeSpace>();
  ritems.template modify<HostMemSpace>();
  ritems.template sync<DevExeSpace>();

  const int nx1 = nx[0], nx2 = nx[1], nx3 = nx[2];
  const int is1 = is[0], is2 = is[1], is3 = is[2];
  const int ie1 = ie[0], ie2 = ie[1], ie3 = ie[2];
  const bool act1 = active[1], act2 = active[2];
  auto x1f = b.x1f, x2f = b.x2f, x3f = b.x3f;

  // pack: a sender's copy of the shared face (kind 0) or its restriction (kind 1). The
  // face of the sender is its outer face when the receiver lies on its outer side.
  if (nsends > 0) {
    auto si = sitems;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nsends, Kokkos::AUTO);
    Kokkos::parallel_for("PackSharedFC", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int n = tmember.league_rank();
      const int m = si.d_view(n, 0), ax = si.d_view(n, 1), side = si.d_view(n, 2);
      const int kind = si.d_view(n, 3), off = si.d_view(n, 6);
      const int t1 = (ax + 1) % 3, t2 = (ax + 2) % 3;
      const int nt[3] = {nx1, nx2, nx3}, st[3] = {is1, is2, is3}, et[3] = {ie1, ie2, ie3};
      const bool at[3] = {true, act1, act2};
      const int r1 = (kind == 1 && at[t1]) ? 2 : 1, r2 = (kind == 1 && at[t2]) ? 2 : 1;
      const int n1 = nt[t1]/r1, n2 = nt[t2]/r2;
      const int plane = (side == 1) ? st[ax] : et[ax] + 1;   // receiver side -> my face
      const Real w = 1.0/static_cast<Real>(r1*r2);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, n1*n2), [&](const int idx) {
        const int c1 = idx % n1, c2 = idx / n1;
        Real val = 0.0;
        for (int g2=0; g2<r2; ++g2) {
          for (int g1=0; g1<r1; ++g1) {
            int c[3];
            c[ax] = plane;
            c[t1] = st[t1] + r1*c1 + g1;
            c[t2] = st[t2] + r2*c2 + g2;
            Real f;
            if (ax == 0) {
              f = x1f(m, c[2], c[1], c[0]);
            } else if (ax == 1) {
              f = x2f(m, c[2], c[1], c[0]);
            } else {
              f = x3f(m, c[2], c[1], c[0]);
            }
            if (kind == 0) {
              val = f;
            } else {
              val += w*f;
            }
          }
        }
        sbuf(off + idx) = val;
      });
    });
  }

#if MPI_PARALLEL_ENABLED
  // one message per peer rank; items for a peer are contiguous in both buffers
  Kokkos::fence();
  std::vector<MPI_Request> reqs;
  for (int i=0; i<nrecvs;) {
    const int peer = recvs[i].peer;
    int j = i, cnt = 0;
    while (j < nrecvs && recvs[j].peer == peer) {cnt += item_size(recvs[j]); ++j;}
    if (peer != my_rank) {
      reqs.emplace_back();
      MPI_Irecv(rbuf.data() + recvs[i].offset, cnt, MPI_ATHENA_REAL, peer, 0,
                fc_share_comm, &reqs.back());
    }
    i = j;
  }
  for (int i=0; i<nsends;) {
    const int peer = sends[i].peer;
    int j = i, cnt = 0;
    while (j < nsends && sends[j].peer == peer) {cnt += item_size(sends[j]); ++j;}
    if (peer != my_rank) {
      reqs.emplace_back();
      MPI_Isend(sbuf.data() + sends[i].offset, cnt, MPI_ATHENA_REAL, peer, 0,
                fc_share_comm, &reqs.back());
    }
    i = j;
  }
  if (!reqs.empty()) {
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  }
#endif

  // unpack into the receiver's own face (kind 1: the quadrant covered by the finer block)
  if (nrecvs > 0) {
    auto ri = ritems;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nrecvs, Kokkos::AUTO);
    Kokkos::parallel_for("UnpackSharedFC", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int n = tmember.league_rank();
      const int m = ri.d_view(n, 0), ax = ri.d_view(n, 1), side = ri.d_view(n, 2);
      const int kind = ri.d_view(n, 3), o1 = ri.d_view(n, 4), o2 = ri.d_view(n, 5);
      const int off = ri.d_view(n, 6);
      const int t1 = (ax + 1) % 3, t2 = (ax + 2) % 3;
      const int nt[3] = {nx1, nx2, nx3}, st[3] = {is1, is2, is3}, et[3] = {ie1, ie2, ie3};
      const bool at[3] = {true, act1, act2};
      const int r1 = (kind == 1 && at[t1]) ? 2 : 1, r2 = (kind == 1 && at[t2]) ? 2 : 1;
      const int n1 = nt[t1]/r1, n2 = nt[t2]/r2;
      const int plane = (side == 1) ? et[ax] + 1 : st[ax];
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, n1*n2), [&](const int idx) {
        const int c1 = idx % n1, c2 = idx / n1;
        const Real val = (off >= 0) ? rbuf(off + idx) : sbuf(-1 - off + idx);
        int c[3];
        c[ax] = plane;
        c[t1] = st[t1] + o1*n1 + c1;
        c[t2] = st[t2] + o2*n2 + c2;
        if (ax == 0) {
          x1f(m, c[2], c[1], c[0]) = val;
        } else if (ax == 1) {
          x2f(m, c[2], c[1], c[0]) = val;
        } else {
          x3f(m, c[2], c[1], c[0]) = val;
        }
      });
    });
  }
  return;
}
