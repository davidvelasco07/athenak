#ifndef SPARSE_LOCAL_MESH_HPP_
#define SPARSE_LOCAL_MESH_HPP_
// Uniform, single-rank cell lookup for initial sparse physics integrations.
// This is explicit local addressing, not a substitute for sparse MPI/AMR exchange.
#include <stdexcept>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"

namespace sparse {
struct LocalMesh {
  DvceArray2D<const int> neighbors;  // block, flattened (oz+1,oy+1,ox+1)
  int nx1, nx2, nx3, is, js, ks;
  KOKKOS_INLINE_FUNCTION
  bool Locate(int &b, int &k, int &j, int &i) const {
    const int ox = i < is ? -1 : (i >= is+nx1 ? 1 : 0);
    const int oy = j < js ? -1 : (j >= js+nx2 ? 1 : 0);
    const int oz = k < ks ? -1 : (k >= ks+nx3 ? 1 : 0);
    b = neighbors(b,(oz+1)*9+(oy+1)*3+ox+1);
    i -= ox*nx1; j -= oy*nx2; k -= oz*nx3;
    return b >= 0;
  }
};

inline LocalMesh MakeLocalMesh(MeshBlockPack *pack) {
  auto *mesh = pack->pmesh;
  if (global_variable::nranks != 1 || mesh->multilevel ||
      mesh->nmb_packs_thisrank != 1 || pack->nmb_thispack != mesh->nmb_total) {
    throw std::runtime_error("Sparse pilot requires one rank/pack and a uniform mesh");
  }
  const int bx = mesh->nmb_rootx1, by = mesh->nmb_rootx2, bz = mesh->nmb_rootx3;
  std::vector<int> blocks(bx*by*bz,-1);
  for (int b = 0; b < pack->nmb_thispack; ++b) {
    const auto loc = mesh->lloc_eachmb[pack->pmb->mb_gid.h_view(b)];
    blocks[(loc.lx3*by+loc.lx2)*bx+loc.lx1] = b;
  }
  DvceArray2D<int> neighbors("sparse_local_neighbors",pack->nmb_thispack,27);
  auto host = Kokkos::create_mirror_view(neighbors);
  for (int b = 0; b < pack->nmb_thispack; ++b) {
    const auto loc = mesh->lloc_eachmb[pack->pmb->mb_gid.h_view(b)];
    for (int oz = -1; oz <= 1; ++oz) for (int oy = -1; oy <= 1; ++oy) {
      for (int ox = -1; ox <= 1; ++ox) {
        int x = loc.lx1+ox, y = loc.lx2+oy, z = loc.lx3+oz;
        bool valid = true;
        int coord[3] = {x,y,z}, extent[3] = {bx,by,bz};
        for (int d = 0; d < 3; ++d) {
          if (coord[d] < 0 || coord[d] >= extent[d]) {
            const int face = 2*d+(coord[d] >= extent[d]);
            const bool inactive = (d == 1 && !mesh->multi_d) ||
                                  (d == 2 && !mesh->three_d);
            if (inactive || mesh->mesh_bcs[face] == BoundaryFlag::periodic) {
              coord[d] = (coord[d]+extent[d])%extent[d];
            } else { valid = false; }
          }
        }
        host(b,(oz+1)*9+(oy+1)*3+ox+1) = valid ?
            blocks[(coord[2]*by+coord[1])*bx+coord[0]] : -1;
      }
    }
  }
  Kokkos::deep_copy(neighbors,host);
  auto &ind = mesh->mb_indcs;
  return {neighbors,ind.nx1,ind.nx2,ind.nx3,ind.is,ind.js,ind.ks};
}
}  // namespace sparse
#endif
