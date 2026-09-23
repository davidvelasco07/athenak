#ifndef HYDRO_WELL_BALANCE_HPP_
#define HYDRO_WELL_BALANCE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file well_balance.hpp
//! \brief Helpers for the well-balanced (WB) scheme (hydro_wb.cpp).
//!
//! The WB scheme evolves the perturbation U' = U - U_eq over a fixed equilibrium U_eq
//! supplied by the problem generator (Velasco-Romero & Teyssier 2025, MNRAS, sec. 2.2).
//! The equilibrium must be SINGLE-VALUED: a ghost cell of one MeshBlock must hold the
//! bit-identical equilibrium of the active cell of its neighbour, otherwise the face
//! equilibrium states differ across a block boundary and the scheme is neither
//! well-balanced there nor independent of the domain decomposition.  Coordinates built
//! from each block's own edges (LeftEdgeX(i, nx, mb_x1min, mb_x1max)) differ between
//! neighbours by an ulp, so FillEquilibriumCellAverages() builds every coordinate from
//! the GLOBAL, level-aware cell index instead, wrapping periodic directions.
//!
//! The problem generator supplies the equilibrium as a device functor
//!   struct MyEq { KOKKOS_INLINE_FUNCTION
//!                 void operator()(Real x1, Real x2, Real x3, Real w[]) const; };
//! that writes the primitive state (w[IDN], w[IVX..IVZ], and w[IEN] = internal energy
//! density for an ideal gas) at a point.  Cell averages use Gauss-Legendre quadrature
//! with kNGL points in each active direction.  The same helper should be used to set
//! the unperturbed part of the initial condition, so that U(t=0) - U_eq is exactly zero
//! wherever no perturbation is added.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "coordinates/cell_locations.hpp"

namespace wellbalance {

//----------------------------------------------------------------------------------------
//! \fn void AddFaceEquilibrium()
//! \brief Turn the reconstructed PERTURBATION states of face (m,k,j,i) into full states by
//! adding the single-valued face equilibrium to both sides, then apply the density and
//! internal-energy floors to the full states (never to the perturbation: a negative
//! rho' is legitimate).  `floors` mirrors the apply_floors of the non-WB reconstruction.
//! The stored face equilibrium is itself floored, so at U'=0 this returns it unchanged
//! and the Riemann solver sees exactly the state that produced f_eq.

KOKKOS_INLINE_FUNCTION
void AddFaceEquilibrium(const EOS_Data &eos, const bool floors, const int nhyd,
                        const int m, const int k, const int j, const int i,
                        const DvceArray5D<Real> &weqf,
                        const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr) {
  for (int n=0; n<nhyd; ++n) {
    wl(m,n,k,j,i) += weqf(m,n,k,j,i);
    wr(m,n,k,j,i) += weqf(m,n,k,j,i);
  }
  if (floors) {
    wl(m,IDN,k,j,i) = fmax(wl(m,IDN,k,j,i), eos.dfloor);
    wr(m,IDN,k,j,i) = fmax(wr(m,IDN,k,j,i), eos.dfloor);
    if (eos.is_ideal) {
      const Real efloor = eos.pfloor/(eos.gamma - 1.0);
      wl(m,IEN,k,j,i) = fmax(wl(m,IEN,k,j,i), efloor);
      wr(m,IEN,k,j,i) = fmax(wr(m,IEN,k,j,i), efloor);
    }
  }
}

constexpr int kNGL = 4;       // Gauss-Legendre points per active direction
constexpr int kNEqVars = 5;   // max number of primitive variables returned by the functor

// nodes and weights of the 4-point Gauss-Legendre rule on [-1,1]
KOKKOS_INLINE_FUNCTION
Real GLNode(const int q) {
  return (q == 0) ? -0.86113631159405257522 :
         (q == 1) ? -0.33998104358485626480 :
         (q == 2) ?  0.33998104358485626480 :
                     0.86113631159405257522;
}
KOKKOS_INLINE_FUNCTION
Real GLWeight(const int q) {
  return (q == 0 || q == 3) ? 0.34785484513745385737 : 0.65214515486254614263;
}

//----------------------------------------------------------------------------------------
//! \struct GlobalGeometry
//! \brief Device-copyable description of the global (level-aware) cell lattice, from
//! which every block computes bit-identical coordinates for the same physical cell.

struct GlobalGeometry {
  RegionSize ms;                  // mesh size (root grid)
  int nx1, nx2, nx3;              // cells per MeshBlock
  int is, js, ks;
  int mnx1, mnx2, mnx3;           // cells across the mesh at the root level
  int root_level;
  bool per1, per2, per3;          // periodic directions
  bool multi_d, three_d;
  DvceArray2D<int> loc;           // (m, {lx1,lx2,lx3,level}) for each block in the pack

  // edges [xl,xr] of cell `i` of block `m` along direction `dir` (1,2,3)
  KOKKOS_INLINE_FUNCTION
  void CellEdges(const int dir, const int m, const int i, Real &xl, Real &xr) const {
    int lx, nb, mn, ist;
    Real xmin, xmax;
    bool per;
    if (dir == 1) {
      lx = loc(m,0); nb = nx1; mn = mnx1; ist = is; xmin = ms.x1min; xmax = ms.x1max;
      per = per1;
    } else if (dir == 2) {
      lx = loc(m,1); nb = nx2; mn = mnx2; ist = js; xmin = ms.x2min; xmax = ms.x2max;
      per = per2;
    } else {
      lx = loc(m,2); nb = nx3; mn = mnx3; ist = ks; xmin = ms.x3min; xmax = ms.x3max;
      per = per3;
    }
    const int n = mn << (loc(m,3) - root_level);
    int gi = lx*nb + (i - ist);
    if (per) {gi = ((gi % n) + n) % n;}
    xl = LeftEdgeX(gi,   n, xmin, xmax);
    xr = LeftEdgeX(gi+1, n, xmin, xmax);
  }
};

//----------------------------------------------------------------------------------------
//! \fn GlobalGeometry MakeGlobalGeometry()
//! \brief Build the GlobalGeometry of the current MeshBlockPack (call after any mesh
//! refinement / load balancing, since block locations change).

inline GlobalGeometry MakeGlobalGeometry(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  auto &indcs = pm->mb_indcs;
  GlobalGeometry g;
  g.ms = pm->mesh_size;
  g.nx1 = indcs.nx1; g.nx2 = indcs.nx2; g.nx3 = indcs.nx3;
  g.is = indcs.is; g.js = indcs.js; g.ks = indcs.ks;
  g.mnx1 = pm->mesh_indcs.nx1; g.mnx2 = pm->mesh_indcs.nx2; g.mnx3 = pm->mesh_indcs.nx3;
  g.root_level = pm->root_level;
  g.per1 = (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic);
  g.per2 = (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic);
  g.per3 = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
  g.multi_d = pm->multi_d;
  g.three_d = pm->three_d;
  int nmb = pmbp->nmb_thispack;
  DualArray2D<int> loc("wb_loc", nmb, 4);
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[pmbp->pmb->mb_gid.h_view(m)];
    loc.h_view(m,0) = ll.lx1;
    loc.h_view(m,1) = ll.lx2;
    loc.h_view(m,2) = ll.lx3;
    loc.h_view(m,3) = ll.level;
  }
  loc.template modify<HostMemSpace>();
  loc.template sync<DevExeSpace>();
  g.loc = loc.d_view;
  return g;
}

//----------------------------------------------------------------------------------------
//! \fn void FillEquilibriumCellAverages()
//! \brief Fill w(m, 0..nvar-1, k,j,i) with the cell averages of the point functor f over
//! the FULL array of every MeshBlock in the pack (ghost zones included), using
//! Gauss-Legendre quadrature on the global cell lattice.  Inactive directions use the
//! single mid-plane point.

template <typename EqFn>
void FillEquilibriumCellAverages(MeshBlockPack *pmbp, DvceArray5D<Real> &w,
                                 const int nvar, const EqFn &f) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int n1m1 = indcs.nx1 + 2*indcs.ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng - 1) : 0;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const GlobalGeometry g = MakeGlobalGeometry(pmbp);
  const EqFn fn = f;
  const int nq2 = g.multi_d ? kNGL : 1;
  const int nq3 = g.three_d ? kNGL : 1;

  par_for("wb_eq_cellavg", DevExeSpace(), 0, nmb1, 0, n3m1, 0, n2m1, 0, n1m1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1l, x1r, x2l, x2r, x3l, x3r;
    g.CellEdges(1, m, i, x1l, x1r);
    g.CellEdges(2, m, j, x2l, x2r);
    g.CellEdges(3, m, k, x3l, x3r);
    const Real x1c = 0.5*(x1l + x1r), h1 = 0.5*(x1r - x1l);
    const Real x2c = 0.5*(x2l + x2r), h2 = 0.5*(x2r - x2l);
    const Real x3c = 0.5*(x3l + x3r), h3 = 0.5*(x3r - x3l);
    Real acc[kNEqVars], wpt[kNEqVars];
    for (int n=0; n<kNEqVars; ++n) {acc[n] = 0.0;}
    Real wsum = 0.0;
    for (int q3=0; q3<nq3; ++q3) {
      const Real x3 = (nq3 > 1) ? (x3c + h3*GLNode(q3)) : x3c;
      const Real w3 = (nq3 > 1) ? GLWeight(q3) : 1.0;
      for (int q2=0; q2<nq2; ++q2) {
        const Real x2 = (nq2 > 1) ? (x2c + h2*GLNode(q2)) : x2c;
        const Real w2 = (nq2 > 1) ? GLWeight(q2) : 1.0;
        for (int q1=0; q1<kNGL; ++q1) {
          const Real x1 = x1c + h1*GLNode(q1);
          const Real wq = GLWeight(q1)*w2*w3;
          for (int n=0; n<kNEqVars; ++n) {wpt[n] = 0.0;}
          fn(x1, x2, x3, wpt);
          for (int n=0; n<nvar; ++n) {acc[n] += wq*wpt[n];}
          wsum += wq;
        }
      }
    }
    for (int n=0; n<nvar; ++n) {w(m,n,k,j,i) = acc[n]/wsum;}
  });
  return;
}

}  // namespace wellbalance
#endif  // HYDRO_WELL_BALANCE_HPP_
