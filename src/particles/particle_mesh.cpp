//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_mesh.cpp
//  \brief constructor / destructor / Zero for the ParticleMesh class.
//
//  Phase 1a — skeleton only. The TSC scatter (DepositMass) and the
//  boundary-sum communication are intentionally absent and land in Phase 1b/1c.

#include "particle_mesh.hpp"

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "bvals/bvals.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn ParticleMesh::ParticleMesh
//! \brief allocates dmesh with the same ghost-cell layout as hydro u0.

ParticleMesh::ParticleMesh(MeshBlockPack *ppack, ParameterInput *pin, int nmeshaux_in)
    : nmeshaux(nmeshaux_in), pmy_pack(ppack) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  // Size for the maximum MeshBlock count this rank can hold (the AthenaK convention,
  // cf. hydro u0): with AMR the pack can grow up to nmb_maxperrank, and pre-sizing
  // avoids per-regrid reallocation (DepositMass keeps a defensive realloc).
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  Kokkos::realloc(dmesh, nmb, nmeshaux, ncells3, ncells2, ncells1);
  Zero();

  // Device counter for particles skipped by the deposit/gather index guards.
  Kokkos::realloc(nbad_, 1);
  Kokkos::deep_copy(nbad_, 0);

  // Boundary-values object for dmesh: provides the buffers/MPI machinery for the future
  // cross-rank flush path, and is reused by Particles::ExchangePhi for phi's halo swap.
  pmbval = new MeshBoundaryValuesCC(ppack, pin, false);
  pmbval->InitializeBuffers(nmeshaux);
}

ParticleMesh::~ParticleMesh() {
  delete pmbval;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::Zero()
//! \brief zero every slot in every cell of dmesh.

void ParticleMesh::Zero() {
  Kokkos::deep_copy(dmesh, 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::DepositMass
//! \brief TSC scatter of particle mass onto dmesh slot 0.
//!
//! Algorithm (per particle p):
//!   1) Resolve the MeshBlock-within-pack slot m from prtcl_idata(PGID, p).
//!   2) Compute the fractional cell index xi = (xp - x1min)/dx1 + is. Then
//!      i0 = floor(xi - 1) is the leftmost of the three cells that receive
//!      contributions, and fx = (i0 + 0.5) - xi is the TSC argument offset
//!      such that the per-cell weight is TSCWeight(fx + ii) for ii in 0..2.
//!   3) Atomic-add mp * w1 * w2 * w3 / dV to each of 27 cells (3D) or 9 (2D).
//!
//! i0+ii can index into the inner ghost cells if the particle is near the MB
//! boundary; that's intentional and gets reconciled by the Phase 1c boundary
//! sum-flush comms.
void ParticleMesh::DepositMass(const DvceArray2D<Real>& prtcl_rdata_in,
                               const DvceArray2D<int>&  prtcl_idata_in,
                               int npar) {
  // Grow dmesh if AMR has raised the number of MeshBlocks in this pack beyond the
  // (max-sized) allocation -- defensive only; the constructor sizes for nmb_maxperrank,
  // so this should never trigger. Never shrink: oversized is harmless and downsizing
  // would churn allocations every regrid.
  {
    int nmb = pmy_pack->nmb_thispack;
    if (dmesh.extent_int(0) < nmb) {
      auto &ix = pmy_pack->pmesh->mb_indcs;
      int nc1 = ix.nx1 + 2*ix.ng;
      int nc2 = (ix.nx2 > 1) ? (ix.nx2 + 2*ix.ng) : 1;
      int nc3 = (ix.nx3 > 1) ? (ix.nx3 + 2*ix.ng) : 1;
      Kokkos::realloc(dmesh, nmb, nmeshaux, nc3, nc2, nc1);
    }
  }
  Zero();
  if (npar <= 0) return;

  auto &pr = prtcl_rdata_in;
  auto &pi = prtcl_idata_in;
  auto &dm = dmesh;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int gids = pmy_pack->gids;
  const int nmb_ = pmy_pack->nmb_thispack;
  const int n1 = dmesh.extent_int(4);
  const int n2 = dmesh.extent_int(3);
  const int n3 = dmesh.extent_int(2);
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto bad = nbad_;

  par_for("PMDepositMass", DevExeSpace(), 0, npar - 1,
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID, p) - gids;
      // Guard: a stale/foreign PGID (e.g. mid-regrid pathology) must not index out of
      // the pack. Count and skip rather than corrupt memory / SIGBUS.
      if (m < 0 || m >= nmb_) { Kokkos::atomic_add(&bad(0), 1); return; }
      Real mp = pr(IPM, p);

      // Cell volume (uniform Cartesian for now).
      Real dV = mbsize.d_view(m).dx1;
      if (multi_d) dV *= mbsize.d_view(m).dx2;
      if (three_d) dV *= mbsize.d_view(m).dx3;
      Real inv_dV = 1.0 / dV;

      // x-axis: always active (particles only in 2D/3D).
      Real xi1 = (pr(IPX, p) - mbsize.d_view(m).x1min) / mbsize.d_view(m).dx1 + is;
      // Guard the double->int casts: a non-finite or wildly out-of-block position (from
      // any upstream pathology, e.g. a diverging force) is undefined behaviour in the
      // cast and would index far outside dmesh. Skip + count instead.
      if (!(xi1 > -1.0e9 && xi1 < 1.0e9)) { Kokkos::atomic_add(&bad(0), 1); return; }
      int  i0  = static_cast<int>(xi1 - 1.0);
      Real fx  = (static_cast<Real>(i0) + 0.5) - xi1;

      // y-axis: active in 2D/3D.
      int  j0 = 0;
      Real fy = 0.0;
      if (multi_d) {
        Real xi2 = (pr(IPY, p) - mbsize.d_view(m).x2min) / mbsize.d_view(m).dx2 + js;
        if (!(xi2 > -1.0e9 && xi2 < 1.0e9)) { Kokkos::atomic_add(&bad(0), 1); return; }
        j0 = static_cast<int>(xi2 - 1.0);
        fy = (static_cast<Real>(j0) + 0.5) - xi2;
      }

      // z-axis: active in 3D only.
      int  k0 = 0;
      Real fz = 0.0;
      if (three_d) {
        Real xi3 = (pr(IPZ, p) - mbsize.d_view(m).x3min) / mbsize.d_view(m).dx3 + ks;
        if (!(xi3 > -1.0e9 && xi3 < 1.0e9)) { Kokkos::atomic_add(&bad(0), 1); return; }
        k0 = static_cast<int>(xi3 - 1.0);
        fz = (static_cast<Real>(k0) + 0.5) - xi3;
      }

      // Guard: the full 3-cell TSC footprint must lie within the array (interior +
      // ghosts). A correctly-binned particle always satisfies this (with ng >= 2);
      // a mis-binned one is counted and skipped.
      if (i0 < 0 || i0 + 2 >= n1 ||
          (multi_d && (j0 < 0 || j0 + 2 >= n2)) ||
          (three_d && (k0 < 0 || k0 + 2 >= n3))) {
        Kokkos::atomic_add(&bad(0), 1);
        return;
      }

      const int nkk = three_d ? 3 : 1;
      const int njj = multi_d ? 3 : 1;
      for (int kk = 0; kk < nkk; ++kk) {
        Real w3 = three_d ? TSCWeight(fz + static_cast<Real>(kk)) : 1.0;
        int  k_cell = three_d ? (k0 + kk) : 0;
        for (int jj = 0; jj < njj; ++jj) {
          Real w2 = multi_d ? TSCWeight(fy + static_cast<Real>(jj)) : 1.0;
          int  j_cell = multi_d ? (j0 + jj) : 0;
          for (int ii = 0; ii < 3; ++ii) {
            Real w1 = TSCWeight(fx + static_cast<Real>(ii));
            int  i_cell = i0 + ii;
            Kokkos::atomic_add(&dm(m, 0, k_cell, j_cell, i_cell),
                               mp * w1 * w2 * w3 * inv_dV);
          }
        }
      }
    });

  // Surface any guarded skips (rate-limited warning). A nonzero count means particles
  // with an invalid PGID or a non-finite/out-of-range position were excluded from the
  // deposit this stage -- the run continues, but mass is missing from the source.
  auto bad_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), nbad_);
  if (bad_h(0) > 0) {
    static int nwarn = 0;
    if (nwarn < 8 && global_variable::my_rank == 0) {
      std::cout << "### WARNING in ParticleMesh::DepositMass: " << bad_h(0)
                << " particle(s) skipped by index guards (invalid PGID or position); "
                << "particle state is likely corrupt (warning " << nwarn+1 << "/8)"
                << std::endl;
      ++nwarn;
    }
    Kokkos::deep_copy(nbad_, 0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::GatherGravity
//! \brief TSC gather of the gravitational acceleration (-grad phi) onto particles.
//!
//! For each particle p the same fractional-index logic as DepositMass selects the
//! 3x3x3 (3D) / 3x3 (2D) cloud of cells. At each cloud cell the cell-centered
//! acceleration is evaluated with the central-difference stencil a*(phi[m-1]-phi[m+1])
//! (a = 0.5/dx), identical to srcterms::SelfGravity, and TSC-weighted into the
//! particle's IPGX/IPGY/IPGZ slots. Mirrors Athena++ ParticleGravity.
void ParticleMesh::GatherGravity(const DvceArray5D<Real>& phi_in,
                                 DvceArray2D<Real>&        prtcl_rdata_io,
                                 const DvceArray2D<int>&   prtcl_idata_in,
                                 int npar) {
  if (npar <= 0) return;

  auto &pr = prtcl_rdata_io;
  auto &pi = prtcl_idata_in;
  auto ph  = phi_in;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int gids = pmy_pack->gids;
  const int nmb_ = pmy_pack->nmb_thispack;
  const int n1 = phi_in.extent_int(4);
  const int n2 = phi_in.extent_int(3);
  const int n3 = phi_in.extent_int(2);
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto bad = nbad_;

  par_for("PMGatherGravity", DevExeSpace(), 0, npar - 1,
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID, p) - gids;
      // Guards mirroring DepositMass: never index outside phi. A skipped particle
      // keeps zero acceleration this stage (counted; DepositMass warns).
      if (m < 0 || m >= nmb_) {
        Kokkos::atomic_add(&bad(0), 1);
        pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
        return;
      }

      // Central-difference prefactors a = 0.5/dx (matches srcterms SelfGravity).
      Real a1 = 0.5 / mbsize.d_view(m).dx1;
      Real a2 = multi_d ? 0.5 / mbsize.d_view(m).dx2 : 0.0;
      Real a3 = three_d ? 0.5 / mbsize.d_view(m).dx3 : 0.0;

      // x-axis: always active.
      Real xi1 = (pr(IPX, p) - mbsize.d_view(m).x1min) / mbsize.d_view(m).dx1 + is;
      if (!(xi1 > -1.0e9 && xi1 < 1.0e9)) {
        Kokkos::atomic_add(&bad(0), 1);
        pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
        return;
      }
      int  i0  = static_cast<int>(xi1 - 1.0);
      Real fx  = (static_cast<Real>(i0) + 0.5) - xi1;

      // y-axis: active in 2D/3D.
      int  j0 = 0;
      Real fy = 0.0;
      if (multi_d) {
        Real xi2 = (pr(IPY, p) - mbsize.d_view(m).x2min) / mbsize.d_view(m).dx2 + js;
        if (!(xi2 > -1.0e9 && xi2 < 1.0e9)) {
          Kokkos::atomic_add(&bad(0), 1);
          pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
          return;
        }
        j0 = static_cast<int>(xi2 - 1.0);
        fy = (static_cast<Real>(j0) + 0.5) - xi2;
      }

      // z-axis: active in 3D only.
      int  k0 = 0;
      Real fz = 0.0;
      if (three_d) {
        Real xi3 = (pr(IPZ, p) - mbsize.d_view(m).x3min) / mbsize.d_view(m).dx3 + ks;
        if (!(xi3 > -1.0e9 && xi3 < 1.0e9)) {
          Kokkos::atomic_add(&bad(0), 1);
          pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
          return;
        }
        k0 = static_cast<int>(xi3 - 1.0);
        fz = (static_cast<Real>(k0) + 0.5) - xi3;
      }

      // Guard: the TSC cloud [i0, i0+2] must lie within the array (matches the deposit
      // guard); beyond that the particle is genuinely mis-binned. Cloud cells whose
      // CENTRAL DIFFERENCE would poke one cell past the array edge are handled
      // per-cell below: that only happens for the extreme cell of the cloud when the
      // particle sits exactly on a MeshBlock face, where that cell's TSC weight
      // tends to zero -- so skipping just that cell is exact (weight-continuous),
      // whereas zeroing the whole particle's force would inject momentum.
      if (i0 < 0 || i0 + 2 >= n1 ||
          (multi_d && (j0 < 0 || j0 + 2 >= n2)) ||
          (three_d && (k0 < 0 || k0 + 2 >= n3))) {
        Kokkos::atomic_add(&bad(0), 1);
        pr(IPGX, p) = 0.0; pr(IPGY, p) = 0.0; pr(IPGZ, p) = 0.0;
        return;
      }

      Real gx = 0.0, gy = 0.0, gz = 0.0;
      const int nkk = three_d ? 3 : 1;
      const int njj = multi_d ? 3 : 1;
      for (int kk = 0; kk < nkk; ++kk) {
        Real w3 = three_d ? TSCWeight(fz + static_cast<Real>(kk)) : 1.0;
        int  kc = three_d ? (k0 + kk) : 0;
        if (three_d && (kc-1 < 0 || kc+1 >= n3)) continue;   // edge cell, weight ~0
        for (int jj = 0; jj < njj; ++jj) {
          Real w2 = multi_d ? TSCWeight(fy + static_cast<Real>(jj)) : 1.0;
          int  jc = multi_d ? (j0 + jj) : 0;
          if (multi_d && (jc-1 < 0 || jc+1 >= n2)) continue;  // edge cell, weight ~0
          for (int ii = 0; ii < 3; ++ii) {
            Real w1 = TSCWeight(fx + static_cast<Real>(ii));
            int  ic = i0 + ii;
            if (ic-1 < 0 || ic+1 >= n1) continue;             // edge cell, weight ~0
            Real w  = w1 * w2 * w3;
            gx += w * a1 * (ph(m, 0, kc, jc, ic-1) - ph(m, 0, kc, jc, ic+1));
            if (multi_d) gy += w * a2 * (ph(m, 0, kc, jc-1, ic) - ph(m, 0, kc, jc+1, ic));
            if (three_d) gz += w * a3 * (ph(m, 0, kc-1, jc, ic) - ph(m, 0, kc+1, jc, ic));
          }
        }
      }
      pr(IPGX, p) = gx;
      pr(IPGY, p) = gy;
      pr(IPGZ, p) = gz;
    });
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleMesh::FlushDepositBoundaries()
//! \brief Fold ghost-zone deposit contributions into neighbour interiors, then zero the
//! ghost spill (Phase 1c + AMR). On-rank neighbours at any refinement level.
//!
//! Each nonzero ghost cell resolves its owner geometrically: it searches this block's
//! neighbour list for the (unique) on-rank MeshBlock whose bounds contain the cell's
//! centre (wrapped by the domain length if periodic), then adds its density there
//! mass-conservatively:
//!   same level    : one coincident destination cell,      add rho            (dV ratio 1)
//!   fine -> coarse: one containing coarse cell,           add rho * dV_f/dV_c (= rho/2^d)
//!   coarse -> fine: the 2^d fine cells this cell covers,  add rho to each     (uniform
//!                   split of the mass rho*dV_c over 2^d cells of volume dV_f)
//! Geometric ownership handles faces, edges and corners in one code path -- including
//! coarser "interior edge" regions for which SetNeighbors leaves no dedicated slot
//! because the coarse face neighbour covers them. Cells owned by off-rank neighbours
//! (MPI, warned once) or lying outside the domain (physical boundaries) are dropped;
//! the trailing ghost-zero pass discards them either way. After all adds complete, a
//! second kernel zeroes every ghost cell so the spill is not double-counted when the
//! gravity source is assembled.
void ParticleMesh::FlushDepositBoundaries() {
  int nmb    = pmy_pack->nmb_thispack;
  int nvar   = dmesh.extent_int(1);

#if MPI_PARALLEL_ENABLED
  // One-time warning for the not-yet-implemented cross-rank path.
  static bool warned = false;
  if (!warned && global_variable::nranks > 1) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in ParticleMesh::FlushDepositBoundaries: cross-rank (MPI) "
                << "particle-mesh deposit flush is not yet implemented; deposits that spill "
                << "across a rank boundary are dropped." << std::endl;
    }
    warned = true;
  }
#endif

  const int my_rank = global_variable::my_rank;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &msize = pmy_pack->pmb->mb_size;
  auto dm = dmesh;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n1 = dmesh.extent_int(4);
  const int n2 = dmesh.extent_int(3);
  const int n3 = dmesh.extent_int(2);
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  // domain lengths, for wrapping ghost-cell positions across periodic boundaries
  auto &msz = pmy_pack->pmesh->mesh_size;
  const Real Lx1 = msz.x1max - msz.x1min;
  const Real Lx2 = msz.x2max - msz.x2min;
  const Real Lx3 = msz.x3max - msz.x3min;

  par_for("PMFlush", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int v, int k, int j, int i) {
    // ghost cells only
    if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) return;
    const Real rho = dm(m, v, k, j, i);
    if (rho == 0.0) return;                       // almost always: nothing deposited here

    // physical centre of this ghost cell
    const Real dxs = msize.d_view(m).dx1;
    const Real dys = msize.d_view(m).dx2;
    const Real dzs = msize.d_view(m).dx3;
    const Real xs = msize.d_view(m).x1min + (static_cast<Real>(i-is) + 0.5)*dxs;
    const Real ys = multi_d ?
                    msize.d_view(m).x2min + (static_cast<Real>(j-js) + 0.5)*dys : 0.0;
    const Real zs = three_d ?
                    msize.d_view(m).x3min + (static_cast<Real>(k-ks) + 0.5)*dzs : 0.0;
    const int mylev = mblev.d_view(m);

    // find the neighbour MeshBlock whose bounds contain this cell centre
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.d_view(m,n).gid < 0) continue;
      if (nghbr.d_view(m,n).rank != my_rank) continue;   // MPI path TODO (warned)
      const int dnb = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      if (dnb < 0 || dnb >= nmb) continue;               // safety

      const Real x1d = msize.d_view(dnb).x1min, x1u = msize.d_view(dnb).x1max;
      const Real x2d = msize.d_view(dnb).x2min, x2u = msize.d_view(dnb).x2max;
      const Real x3d = msize.d_view(dnb).x3min, x3u = msize.d_view(dnb).x3max;

      // wrap the cell position into the neighbour's frame (periodic domains)
      Real xc = xs, yc = ys, zc = zs;
      Real ctr = 0.5*(x1d + x1u);
      if (xc - ctr >  0.5*Lx1) xc -= Lx1;
      if (xc - ctr < -0.5*Lx1) xc += Lx1;
      if (multi_d) {
        ctr = 0.5*(x2d + x2u);
        if (yc - ctr >  0.5*Lx2) yc -= Lx2;
        if (yc - ctr < -0.5*Lx2) yc += Lx2;
      }
      if (three_d) {
        ctr = 0.5*(x3d + x3u);
        if (zc - ctr >  0.5*Lx3) zc -= Lx3;
        if (zc - ctr < -0.5*Lx3) zc += Lx3;
      }

      // containment (lower-inclusive, consistent with particle binning)
      if (xc < x1d || xc >= x1u) continue;
      if (multi_d && (yc < x2d || yc >= x2u)) continue;
      if (three_d && (zc < x3d || zc >= x3u)) continue;

      const Real dxd = msize.d_view(dnb).dx1;
      const Real dyd = msize.d_view(dnb).dx2;
      const Real dzd = msize.d_view(dnb).dx3;
      const bool dst_finer = (nghbr.d_view(m,n).lev > mylev);

      if (!dst_finer) {
        // same level (ratio 1) or coarser (ratio 1/2^d): single containing cell
        Real wvol = dxs/dxd;
        if (multi_d) wvol *= dys/dyd;
        if (three_d) wvol *= dzs/dzd;
        int it = is + static_cast<int>((xc - x1d)/dxd);
        int jt = multi_d ? js + static_cast<int>((yc - x2d)/dyd) : 0;
        int kt = three_d ? ks + static_cast<int>((zc - x3d)/dzd) : 0;
        // clamp against floating-point edge cases
        it = (it < is) ? is : (it > ie) ? ie : it;
        if (multi_d) jt = (jt < js) ? js : (jt > je) ? je : jt;
        if (three_d) kt = (kt < ks) ? ks : (kt > ke) ? ke : kt;
        Kokkos::atomic_add(&dm(dnb, v, kt, jt, it), rho*wvol);
      } else {
        // finer: split the mass uniformly over the 2 (per active dim) fine cells this
        // coarse cell covers; each receives Delta(rho_fine) = rho (dV ratio cancels)
        const int r1 = static_cast<int>(dxs/dxd + 0.5);
        const int r2 = multi_d ? static_cast<int>(dys/dyd + 0.5) : 1;
        const int r3 = three_d ? static_cast<int>(dzs/dzd + 0.5) : 1;
        Real wvol = dxs/dxd;
        if (multi_d) wvol *= dys/dyd;
        if (three_d) wvol *= dzs/dzd;
        const Real wsplit = wvol / (static_cast<Real>(r1*r2*r3));
        const int it0 = is + static_cast<int>((xc - 0.5*dxs - x1d)/dxd + 0.5);
        const int jt0 = multi_d ? js + static_cast<int>((yc - 0.5*dys - x2d)/dyd + 0.5) : 0;
        const int kt0 = three_d ? ks + static_cast<int>((zc - 0.5*dzs - x3d)/dzd + 0.5) : 0;
        for (int kk = 0; kk < r3; ++kk) {
          const int kt = three_d ? kt0 + kk : 0;
          if (three_d && (kt < ks || kt > ke)) continue;
          for (int jj = 0; jj < r2; ++jj) {
            const int jt = multi_d ? jt0 + jj : 0;
            if (multi_d && (jt < js || jt > je)) continue;
            for (int ii = 0; ii < r1; ++ii) {
              const int it = it0 + ii;
              if (it < is || it > ie) continue;
              Kokkos::atomic_add(&dm(dnb, v, kt, jt, it), rho*wsplit);
            }
          }
        }
      }
      break;   // owner found; each ghost cell is folded exactly once
    }
  });

  // Zero every ghost cell: its deposit has been flushed to the owning neighbour interior
  // (or, at a physical/off-rank boundary, dropped). Prevents double-counting.
  par_for("PMFlushZeroGhost", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int v, int k, int j, int i) {
    if (i < is || i > ie || j < js || j > je || k < ks || k > ke) {
      dm(m, v, k, j, i) = 0.0;
    }
  });
}

}  // namespace particles
