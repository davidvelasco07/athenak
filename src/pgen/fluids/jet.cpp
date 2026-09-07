//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file jet.cpp
//  \brief Problem generator for a nonrelativistic (M)HD jet injected through the
//  inner-X1 boundary into a uniform ambient medium.
//
//  Ported from the Athena++ jet.cpp problem generator.  The jet is a circular patch of
//  radius `rjet` centred on the transverse midpoint of the ix1 face; inside the patch the
//  ghost zones are held at a fixed inflow state, outside they are zeroth-order
//  extrapolated (outflow).  Set `ix1_bc = user` in <mesh>.
//
//  The stock parameters are the underdense Mach-10 jet: ambient d=1, p=6e-3, gamma=5/3
//  (c_s=0.1), jet d=0.1 at vx=1, so the beam is Mach 10 w.r.t. the ambient sound speed.
//  With bx=bxjet=0.1 the axial field gives beta = 2p/B^2 = 1.2.
//
//  DIV-B NOTE: the injected field is uniform and identical to the ambient field in the
//  shipped deck (bxjet == bx).  Bx is then constant along x1 across the inflow face, so
//  the ghost zones introduce no dBx/dx and the CT update starts (and stays) at
//  round-off div(B).  Choosing bxjet != bx puts a jump in Bx across the inflow face,
//  which is NOT divergence-free -- monitor `mhd_divb` if you do that deliberately.

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <iostream>
#include <string>

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

namespace {
// Jet + ambient state, filled in UserProblem and copied into a local (so that it is
// captured by value) before every device kernel in the boundary function.
struct JetState {
  Real d_amb, p_amb, vx_amb, vy_amb, vz_amb, bx_amb, by_amb, bz_amb;
  Real d_jet, p_jet, vx_jet, vy_jet, vz_jet, bx_jet, by_jet, bz_jet;
  Real r_jet, x2_0, x3_0, gm1;
  Real d_out, p_out;
  bool seed;             // put the beam state in the first active column at t=0
  int  outside;          // 0 = outflow copy, 1 = reservoir/fixed state
  bool is_ideal;
};
JetState jet;
}  // namespace

void JetBoundary(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for the jet problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // The boundary function must be enrolled on restarts too.
  user_bcs_func = JetBoundary;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool is_mhd = (pmbp->pmhd != nullptr);
  if (!is_mhd && pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "jet problem generator requires <hydro> or <mhd>" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ambient medium
  jet.d_amb  = pin->GetReal("problem", "d");
  jet.p_amb  = pin->GetReal("problem", "p");
  jet.vx_amb = pin->GetReal("problem", "vx");
  jet.vy_amb = pin->GetReal("problem", "vy");
  jet.vz_amb = pin->GetReal("problem", "vz");
  // jet beam
  jet.d_jet  = pin->GetReal("problem", "djet");
  jet.p_jet  = pin->GetReal("problem", "pjet");
  jet.vx_jet = pin->GetReal("problem", "vxjet");
  jet.vy_jet = pin->GetReal("problem", "vyjet");
  jet.vz_jet = pin->GetReal("problem", "vzjet");
  jet.r_jet  = pin->GetReal("problem", "rjet");
  // What the inlet face does OUTSIDE the nozzle.  References disagree, so it is a
  // parameter rather than a guess:
  //   outflow    zeroth-order copy from the first active cell (Athena++ jet.cpp)
  //   reservoir  a fixed state at rest -- Fu (2019) sec 4.3.4 states it outright
  //              for the Ha et al. jet: "(rho,u,p) = (5,0,0,0.4127) otherwise",
  //              i.e. the JET density, not the ambient
  //   ambient    a fixed state at rest at the ambient density
  // AthenaK computes the timestep from ACTIVE cells only.  At t=0 the beam exists
  // solely in the ghost zones, so the first dt is set by the ambient sound speed
  // while material enters at v_jet: the injected gas crosses
  // CFL*v_jet/c_ambient cells in one step.  For the Mach-80 jet at the paper's
  // CFL=0.4 that is ~10 cells, and no flux limiter can repair it -- the
  // Lax-Friedrichs positivity guarantee itself only holds for CFL <= 1/2.
  // Seeding the beam into the first active column makes the timestep see v_jet
  // from the start; that column would be overwritten by inflow within one step
  // anyway, so it costs nothing physically.
  jet.seed = pin->GetOrAddBoolean("problem", "seed_nozzle", false);
  std::string out_s = pin->GetOrAddString("problem", "inflow_outside", "outflow");
  if (out_s.compare("outflow") == 0) {
    jet.outside = 0;
  } else if (out_s.compare("reservoir") == 0 || out_s.compare("ambient") == 0) {
    jet.outside = 1;
    jet.d_out = (out_s.compare("reservoir") == 0) ? jet.d_jet : jet.d_amb;
    jet.p_out = (out_s.compare("reservoir") == 0) ? jet.p_jet : jet.p_amb;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<problem>/inflow_outside = '" << out_s
              << "' not implemented (outflow|reservoir|ambient)" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (is_mhd) {
    jet.bx_amb = pin->GetReal("problem", "bx");
    jet.by_amb = pin->GetReal("problem", "by");
    jet.bz_amb = pin->GetReal("problem", "bz");
    jet.bx_jet = pin->GetReal("problem", "bxjet");
    jet.by_jet = pin->GetReal("problem", "byjet");
    jet.bz_jet = pin->GetReal("problem", "bzjet");
  } else {
    jet.bx_amb = 0.0; jet.by_amb = 0.0; jet.bz_amb = 0.0;
    jet.bx_jet = 0.0; jet.by_jet = 0.0; jet.bz_jet = 0.0;
  }
  // jet axis = transverse centre of the domain
  jet.x2_0 = 0.5*(pmy_mesh_->mesh_size.x2max + pmy_mesh_->mesh_size.x2min);
  jet.x3_0 = 0.5*(pmy_mesh_->mesh_size.x3max + pmy_mesh_->mesh_size.x3min);

  EOS_Data &eos = (is_mhd) ? pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  jet.gm1 = eos.gamma - 1.0;
  jet.is_ideal = eos.is_ideal;

  if (restart) return;

  // capture variables for kernels
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  JetState j0 = jet;

  // uniform ambient medium, optionally with the beam seeded in the first column
  auto &u0 = (is_mhd) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto &indcs_ = pmy_mesh_->mb_indcs;
  auto &size_ = pmbp->pmb->mb_size;
  auto &mb_bcs_ = pmbp->pmb->mb_bcs;
  par_for("pgen_jet_amb", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real d = j0.d_amb, vx = j0.vx_amb, vy = j0.vy_amb, vz = j0.vz_amb, p = j0.p_amb;
    if (j0.seed && i == is &&
        mb_bcs_.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      int nx2 = indcs_.nx2, nx3 = indcs_.nx3;
      Real x2v = (nx2 > 1) ?
          CellCenterX(j-js, nx2, size_.d_view(m).x2min, size_.d_view(m).x2max) : j0.x2_0;
      Real x3v = (nx3 > 1) ?
          CellCenterX(k-ks, nx3, size_.d_view(m).x3min, size_.d_view(m).x3max) : j0.x3_0;
      if (sqrt(SQR(x2v - j0.x2_0) + SQR(x3v - j0.x3_0)) <= j0.r_jet) {
        d = j0.d_jet; vx = j0.vx_jet; vy = j0.vy_jet; vz = j0.vz_jet; p = j0.p_jet;
      }
    }
    u0(m,IDN,k,j,i) = d;
    u0(m,IM1,k,j,i) = d*vx;
    u0(m,IM2,k,j,i) = d*vy;
    u0(m,IM3,k,j,i) = d*vz;
    if (j0.is_ideal) {
      u0(m,IEN,k,j,i) = p/j0.gm1 + 0.5*d*(SQR(vx) + SQR(vy) + SQR(vz));
    }
  });

  // uniform ambient field, then add magnetic energy (separate kernel so that every
  // face of the block is set before the cell-centred field is formed)
  if (is_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_jet_b", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = j0.bx_amb;
      b0.x2f(m,k,j,i) = j0.by_amb;
      b0.x3f(m,k,j,i) = j0.bz_amb;
      if (i==ie) {b0.x1f(m,k,j,i+1) = j0.bx_amb;}
      if (j==je) {b0.x2f(m,k,j+1,i) = j0.by_amb;}
      if (k==ke) {b0.x3f(m,k+1,j,i) = j0.bz_amb;}
    });

    if (jet.is_ideal) {
      par_for("pgen_jet_e", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IEN,k,j,i) +=
            0.5*(SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1))) +
                 SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
                 SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i))));
      });
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void JetBoundary()
//  \brief Fixed jet inflow inside radius rjet on the inner-X1 face; outflow outside it.
//  Operates on the CONSERVED variables (AthenaK user BCs run before ConsToPrim).

void JetBoundary(Mesh *pm) {
  auto pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int &ng = indcs.ng;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  const bool is_mhd = (pmbp->pmhd != nullptr);
  JetState j0 = jet;

  auto &u0 = (is_mhd) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  par_for("jetbc_u", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) != BoundaryFlag::user) {return;}

    Real x2v = (nx2 > 1) ?
        CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) : j0.x2_0;
    Real x3v = (nx3 > 1) ?
        CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) : j0.x3_0;
    Real rad = sqrt(SQR(x2v - j0.x2_0) + SQR(x3v - j0.x3_0));

    if (rad <= j0.r_jet) {
      // fixed inflow: the beam state, with the AMBIENT field energy (the field is
      // uniform, so the injected magnetic energy is the same everywhere)
      Real ejet = 0.0;
      if (j0.is_ideal) {
        ejet = j0.p_jet/j0.gm1 +
            0.5*j0.d_jet*(SQR(j0.vx_jet) + SQR(j0.vy_jet) + SQR(j0.vz_jet)) +
            0.5*(SQR(j0.bx_jet) + SQR(j0.by_jet) + SQR(j0.bz_jet));
      }
      for (int i=0; i<ng; ++i) {
        u0(m,IDN,k,j,is-i-1) = j0.d_jet;
        u0(m,IM1,k,j,is-i-1) = j0.d_jet*j0.vx_jet;
        u0(m,IM2,k,j,is-i-1) = j0.d_jet*j0.vy_jet;
        u0(m,IM3,k,j,is-i-1) = j0.d_jet*j0.vz_jet;
        if (j0.is_ideal) {u0(m,IEN,k,j,is-i-1) = ejet;}
      }
    } else if (j0.outside == 1) {
      // fixed state at rest outside the beam (reservoir / ambient)
      Real eout = 0.0;
      if (j0.is_ideal) {
        eout = j0.p_out/j0.gm1 +
            0.5*(SQR(j0.bx_jet) + SQR(j0.by_jet) + SQR(j0.bz_jet));
      }
      for (int i=0; i<ng; ++i) {
        u0(m,IDN,k,j,is-i-1) = j0.d_out;
        u0(m,IM1,k,j,is-i-1) = 0.0;
        u0(m,IM2,k,j,is-i-1) = 0.0;
        u0(m,IM3,k,j,is-i-1) = 0.0;
        if (j0.is_ideal) {u0(m,IEN,k,j,is-i-1) = eout;}
      }
    } else {
      // outflow (zeroth-order extrapolation) outside the beam
      for (int i=0; i<ng; ++i) {
        u0(m,IDN,k,j,is-i-1) = u0(m,IDN,k,j,is);
        u0(m,IM1,k,j,is-i-1) = u0(m,IM1,k,j,is);
        u0(m,IM2,k,j,is-i-1) = u0(m,IM2,k,j,is);
        u0(m,IM3,k,j,is-i-1) = u0(m,IM3,k,j,is);
        if (j0.is_ideal) {u0(m,IEN,k,j,is-i-1) = u0(m,IEN,k,j,is);}
      }
    }
  });

  if (!is_mhd) return;

  auto &b0 = pmbp->pmhd->b0;
  par_for("jetbc_b", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) != BoundaryFlag::user) {return;}

    // Each face component must be tested for "inside the beam" at ITS OWN location.
    // Using the cell-centre radius for every component offsets the x2-face (and
    // x3-face) partition by half a cell, which is not mirror-symmetric about the jet
    // axis and drives an O(1) asymmetry within a few tens of time units -- even when
    // by_jet == by_amb, because the two branches (pin vs copy-from-interior) then
    // disagree on a half-cell-offset set of faces.
    Real x2v = (nx2 > 1) ?
        CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) : j0.x2_0;
    Real x3v = (nx3 > 1) ?
        CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) : j0.x3_0;
    Real x2f = (nx2 > 1) ?
        LeftEdgeX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) : j0.x2_0;
    Real x2fp1 = (nx2 > 1) ?
        LeftEdgeX(j+1-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) : j0.x2_0;
    Real x3f = (nx3 > 1) ?
        LeftEdgeX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) : j0.x3_0;
    Real x3fp1 = (nx3 > 1) ?
        LeftEdgeX(k+1-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) : j0.x3_0;

    // x1-faces sit at (x2v, x3v); x2-faces at (x2f, x3v); x3-faces at (x2v, x3f)
    Real r_x1 = sqrt(SQR(x2v - j0.x2_0) + SQR(x3v - j0.x3_0));
    Real r_x2 = sqrt(SQR(x2f - j0.x2_0) + SQR(x3v - j0.x3_0));
    Real r_x2p = sqrt(SQR(x2fp1 - j0.x2_0) + SQR(x3v - j0.x3_0));
    Real r_x3 = sqrt(SQR(x2v - j0.x2_0) + SQR(x3f - j0.x3_0));
    Real r_x3p = sqrt(SQR(x2v - j0.x2_0) + SQR(x3fp1 - j0.x3_0));

    for (int i=0; i<ng; ++i) {
      b0.x1f(m,k,j,is-i-1) = (r_x1 <= j0.r_jet) ? j0.bx_jet : b0.x1f(m,k,j,is);
      b0.x2f(m,k,j,is-i-1) = (r_x2 <= j0.r_jet) ? j0.by_jet : b0.x2f(m,k,j,is);
      b0.x3f(m,k,j,is-i-1) = (r_x3 <= j0.r_jet) ? j0.bz_jet : b0.x3f(m,k,j,is);
      if (j == n2-1) {
        b0.x2f(m,k,j+1,is-i-1) = (r_x2p <= j0.r_jet) ? j0.by_jet : b0.x2f(m,k,j+1,is);
      }
      if (k == n3-1) {
        b0.x3f(m,k+1,j,is-i-1) = (r_x3p <= j0.r_jet) ? j0.bz_jet : b0.x3f(m,k+1,j,is);
      }
    }
  });

  return;
}
