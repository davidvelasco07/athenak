//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_jet.cpp
//  \brief Astrophysical MHD jet with huge Mach number, injected through the inner-X2
//  boundary into a strongly magnetized ambient plasma.
//
//  Wu, Zhang & Shu (arXiv:2512.09116) Example 6.3, following Balsara and co-workers.
//  That review states the initial condition explicitly, which matters: the original
//  paper's density panel cannot be read quantitatively -- through its own printed
//  colourbar the undisturbed ambient comes out near 0.010 where the stated initial
//  condition fixes 0.1*gamma = 0.14.
//
//  Setup (gamma = 1.4):
//    ambient over [-1/2,1/2] x [0,3/2]:  (rho, v, B, p) = (0.1*gamma, 0, (0,2e5,0), 1)
//    jet at x in [-0.05,0.05], y = 0, injected in +y through the bottom boundary:
//                                     (rho, v, B, p) = (gamma, (0,1e6,0), (0,2e5,0), 1)
//    The computation uses the symmetric half-domain [0,1/2] x [0,3/2] on 200 x 600,
//    REFLECTING at x = 0 and outflow elsewhere.  Final time t = 1.8e-6.
//
//  This is an extreme test: with p = 1 and B = 2e5 the ambient plasma beta is
//  2p/B^2 = 5e-11, and the beam is sonic-Mach ~3e5.  Set `ix2_bc = user`.
//
//  DIV-B: the field is uniform (0, B0, 0) in BOTH the ambient and the beam, so the
//  injected By introduces no jump and the CT update starts and stays at round-off
//  div(B).  The nozzle carries no transverse field.

// C/C++ headers
#include <algorithm>
#include <cmath>
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
struct MJet {
  Real d_amb, p_amb, by_amb;
  Real d_jet, p_jet, vy_jet, by_jet;
  Real r_jet, x_axis, gm1;
};
MJet mj;
}  // namespace

void MHDJetBoundary(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_bcs_func = MHDJetBoundary;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mhd_jet problem generator requires <mhd>" << std::endl;
    exit(EXIT_FAILURE);
  }
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mhd_jet requires an ideal EOS" << std::endl;
    exit(EXIT_FAILURE);
  }

  mj.d_amb  = pin->GetOrAddReal("problem", "d_amb", 0.1*eos.gamma);
  mj.p_amb  = pin->GetOrAddReal("problem", "p_amb", 1.0);
  mj.by_amb = pin->GetOrAddReal("problem", "by_amb", 2.0e5);
  mj.d_jet  = pin->GetOrAddReal("problem", "d_jet", eos.gamma);
  mj.p_jet  = pin->GetOrAddReal("problem", "p_jet", 1.0);
  mj.vy_jet = pin->GetOrAddReal("problem", "vy_jet", 1.0e6);
  mj.by_jet = pin->GetOrAddReal("problem", "by_jet", 2.0e5);
  mj.r_jet  = pin->GetOrAddReal("problem", "r_jet", 0.05);
  // Distance is measured from the jet axis.  On the symmetric half-domain the axis
  // is the reflecting boundary itself, so this defaults to x1min rather than the
  // domain midpoint; set it to the midpoint to run the full width.
  mj.x_axis = pin->GetOrAddReal("problem", "x_axis", pmy_mesh_->mesh_size.x1min);
  mj.gm1 = eos.gamma - 1.0;

  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  MJet j0 = mj;

  // uniform ambient plasma at rest
  par_for("pgen_mjet_u", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = j0.d_amb;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
  });

  // uniform field along x2
  par_for("pgen_mjet_b", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = j0.by_amb;
    b0.x3f(m,k,j,i) = 0.0;
    if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
    if (j==je) {b0.x2f(m,k,j+1,i) = j0.by_amb;}
    if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
  });

  // total energy (separate kernel: every face must be set first)
  par_for("pgen_mjet_e", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) = j0.p_amb/j0.gm1 +
        (0.5/u0(m,IDN,k,j,i))*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
                               SQR(u0(m,IM3,k,j,i))) +
        0.5*(SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1))) +
             SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
             SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i))));
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHDJetBoundary()
//  \brief Nozzle on the inner-X2 face for |x - x_axis| < r_jet; outflow outside it.
//  Operates on conserved variables (AthenaK user BCs run before ConsToPrim).

void MHDJetBoundary(Mesh *pm) {
  auto pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int &ng = indcs.ng;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nx1 = indcs.nx1;
  int nmb = pmbp->nmb_thispack;
  MJet j0 = mj;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  par_for("mjetbc_u", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) != BoundaryFlag::user) {return;}
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    bool nozzle = (fabs(x1v - j0.x_axis) < j0.r_jet);

    if (nozzle) {
      Real ejet = j0.p_jet/j0.gm1 + 0.5*j0.d_jet*SQR(j0.vy_jet)
                + 0.5*SQR(j0.by_jet);
      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,js-j-1,i) = j0.d_jet;
        u0(m,IM1,k,js-j-1,i) = 0.0;
        u0(m,IM2,k,js-j-1,i) = j0.d_jet*j0.vy_jet;
        u0(m,IM3,k,js-j-1,i) = 0.0;
        u0(m,IEN,k,js-j-1,i) = ejet;
      }
    } else {
      for (int j=0; j<ng; ++j) {
        u0(m,IDN,k,js-j-1,i) = u0(m,IDN,k,js,i);
        u0(m,IM1,k,js-j-1,i) = u0(m,IM1,k,js,i);
        u0(m,IM2,k,js-j-1,i) = u0(m,IM2,k,js,i);
        u0(m,IM3,k,js-j-1,i) = u0(m,IM3,k,js,i);
        u0(m,IEN,k,js-j-1,i) = u0(m,IEN,k,js,i);
      }
    }
  });

  par_for("mjetbc_b", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) != BoundaryFlag::user) {return;}
    // Each face component is tested at ITS OWN location: the x1-faces sit half a
    // cell off the cell centre, and using the centre for all of them offsets the
    // nozzle partition asymmetrically about the axis.
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x1f = LeftEdgeX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x1fp = LeftEdgeX(i+1-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    bool nz_c  = (fabs(x1v  - j0.x_axis) < j0.r_jet);   // x2-, x3-faces
    bool nz_f  = (fabs(x1f  - j0.x_axis) < j0.r_jet);   // x1-faces
    bool nz_fp = (fabs(x1fp - j0.x_axis) < j0.r_jet);

    for (int j=0; j<ng; ++j) {
      b0.x1f(m,k,js-j-1,i) = nz_f  ? 0.0 : b0.x1f(m,k,js,i);
      b0.x2f(m,k,js-j-1,i) = nz_c  ? j0.by_jet : b0.x2f(m,k,js,i);
      b0.x3f(m,k,js-j-1,i) = nz_c  ? 0.0 : b0.x3f(m,k,js,i);
      if (i == n1-1) {
        b0.x1f(m,k,js-j-1,i+1) = nz_fp ? 0.0 : b0.x1f(m,k,js,i+1);
      }
      if (k == n3-1) {
        b0.x3f(m,k+1,js-j-1,i) = nz_c ? 0.0 : b0.x3f(m,k+1,js,i);
      }
    }
  });

  return;
}
