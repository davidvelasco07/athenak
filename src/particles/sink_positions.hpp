#ifndef PARTICLES_SINK_POSITIONS_HPP_
#define PARTICLES_SINK_POSITIONS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_positions.hpp
//! \brief GatherAllSinkPositions(): every sink's (x,y,z) on every rank, for use by AMR
//! refinement criteria in problem generators.
//!
//! WHY THIS EXISTS. A user refinement function fills `refine_flag` for the MeshBlocks of
//! the rank that runs it (CheckForRefinement then Allgathers the flags across ranks). So a
//! criterion of the form "refine blocks near a sink" must be evaluated against ALL sinks,
//! not just `ppart->nprtcl_thispack` (the ones this rank owns). Using only local particles
//! makes the refinement -- and therefore the whole mesh -- depend on the rank count: blocks
//! that neighbour an off-rank sink are never flagged, and because these criteria default to
//! "derefine" they are actively coarsened. The sink then sits against an asymmetric
//! coarse-fine boundary, the particle-mesh deposit/gather pair stops cancelling, and the
//! sink picks up a large spurious force. Measured on part_orbit_amr with 4 ranks before this
//! was fixed: 78 MeshBlocks instead of 120, and an out-of-plane force g_z ~ 0.9 where
//! symmetry requires 0.
//!
//! Returns a device-resident flat array [x0,y0,z0, x1,y1,z1, ...] of length 3*nsink_all.
//! MUST be called by every rank (it contains MPI collectives). Cheap: sinks number in the
//! tens, and the gather is O(nranks).

#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/meshblock_pack.hpp"
#include "particles/particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn DualArray1D<Real> GatherAllSinkPositions(MeshBlockPack*, int&)

inline DualArray1D<Real> GatherAllSinkPositions(MeshBlockPack *pmbp, int &nsink_all) {
  const int npart = (pmbp->ppart == nullptr) ? 0 : pmbp->ppart->nprtcl_thispack;
  nsink_all = npart;

  // pull this rank's positions to the host
  std::vector<Real> pos(3*std::max(npart, 0));
  if (npart > 0) {
    auto pr = pmbp->ppart->prtcl_rdata;
    auto pr_h = Kokkos::create_mirror_view(pr);
    Kokkos::deep_copy(pr_h, pr);
    for (int p = 0; p < npart; ++p) {
      pos[3*p    ] = pr_h(IPX, p);
      pos[3*p + 1] = pr_h(IPY, p);
      pos[3*p + 2] = pr_h(IPZ, p);
    }
  }

#if MPI_PARALLEL_ENABLED
  const int nranks = global_variable::nranks;
  if (nranks > 1) {
    std::vector<int> ncnt(nranks, 0);
    MPI_Allgather(&npart, 1, MPI_INT, ncnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> rcnt(nranks), displ(nranks, 0);
    nsink_all = 0;
    for (int r = 0; r < nranks; ++r) { rcnt[r] = 3*ncnt[r]; nsink_all += ncnt[r]; }
    for (int r = 1; r < nranks; ++r) { displ[r] = displ[r-1] + rcnt[r-1]; }
    std::vector<Real> all(3*std::max(nsink_all, 0));
    MPI_Allgatherv(pos.data(), 3*npart, MPI_ATHENA_REAL, all.data(), rcnt.data(),
                   displ.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD);
    pos.swap(all);
  }
#endif

  // upload to the device for use inside the refinement kernel
  DualArray1D<Real> spos("sink_pos_all", 3*std::max(nsink_all, 1));
  for (int i = 0; i < 3*nsink_all; ++i) { spos.h_view(i) = pos[i]; }
  spos.template modify<HostMemSpace>();
  spos.template sync<DevExeSpace>();
  return spos;
}

#endif  // PARTICLES_SINK_POSITIONS_HPP_
