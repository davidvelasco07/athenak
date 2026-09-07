//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rotor.cpp
//  \brief Problem generator for the MHD rotor test.
//
//  Balsara & Spicer (1999); the standard form used by Toth (2000) and Stone et al.
//  (2008) section 5.5.  A dense disk of radius r0 rotates rigidly at angular velocity
//  omega inside a uniform, static, magnetized ambient medium at the same pressure.
//  The rotation winds up the initially uniform Bx into strong torsional Alfven waves
//  that launch into the ambient gas and squeeze the disk into an oblate shape.
//
//  This is a hard test for a high-order scheme for two reasons: the ambient plasma
//  beta is low (beta = 2 p0 / b0^2 ~ 0.35 at the standard parameters), so a small
//  overshoot in the total energy drives the gas pressure negative; and the rotor edge
//  is a genuine discontinuity in velocity.
//
//  A linear taper over r0 < r < r1 (Toth's f = (r1-r)/(r1-r0)) removes the worst of
//  the start-up transient from a sharp edge.  Set `taper=false` for the raw
//  discontinuous initial condition.
//
//  Standard parameters (Toth 2000 / Stone+ 2008), on [-0.5,0.5]^2 with gamma=1.4:
//      d0=10 (disk), damb=1, p0=1, omega=20 (=v0/r0 with v0=2), r0=0.1, r1=0.115,
//      b0=5/sqrt(4 pi) ~ 1.4104739588693909, tlim=0.15

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
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for the MHD rotor

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rotor problem generator requires <mhd>" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real d0    = pin->GetOrAddReal("problem", "d0", 10.0);
  Real damb  = pin->GetOrAddReal("problem", "damb", 1.0);
  Real p0    = pin->GetOrAddReal("problem", "p0", 1.0);
  Real omega = pin->GetOrAddReal("problem", "omega", 20.0);
  Real r0    = pin->GetOrAddReal("problem", "r0", 0.1);
  Real r1    = pin->GetOrAddReal("problem", "r1", 0.115);
  Real b0    = pin->GetOrAddReal("problem", "b0", 5.0/std::sqrt(4.0*M_PI));
  bool taper = pin->GetOrAddBoolean("problem", "taper", true);

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rotor problem generator requires an ideal EOS" << std::endl;
    exit(EXIT_FAILURE);
  }

  // rotor centre = centre of the domain
  Real xc = 0.5*(pmy_mesh_->mesh_size.x1max + pmy_mesh_->mesh_size.x1min);
  Real yc = 0.5*(pmy_mesh_->mesh_size.x2max + pmy_mesh_->mesh_size.x2min);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0f = pmbp->pmhd->b0;

  // density and momenta
  par_for("pgen_rotor", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    int nx1 = indcs.nx1, nx2 = indcs.nx2;
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real dx = x1v - xc, dy = x2v - yc;
    Real rad = sqrt(dx*dx + dy*dy);

    Real den, vx, vy;
    if (rad <= r0) {
      // rigidly rotating dense disk
      den = d0;
      vx  = -omega*dy;
      vy  =  omega*dx;
    } else if (taper && rad < r1) {
      // Toth (2000) linear taper: f=1 at r0, f=0 at r1.  The angular velocity of
      // the tapered annulus is omega*r0/rad so the AZIMUTHAL SPEED is continuous
      // with the disk edge.
      Real f = (r1 - rad)/(r1 - r0);
      den = damb + f*(d0 - damb);
      Real om = omega*r0/rad;
      vx  = -f*om*dy;
      vy  =  f*om*dx;
    } else {
      den = damb;
      vx  = 0.0;
      vy  = 0.0;
    }

    u0(m,IDN,k,j,i) = den;
    u0(m,IM1,k,j,i) = den*vx;
    u0(m,IM2,k,j,i) = den*vy;
    u0(m,IM3,k,j,i) = 0.0;
  });

  // uniform Bx = b0
  par_for("pgen_rotor_b", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0f.x1f(m,k,j,i) = b0;
    b0f.x2f(m,k,j,i) = 0.0;
    b0f.x3f(m,k,j,i) = 0.0;
    if (i==ie) {b0f.x1f(m,k,j,i+1) = b0;}
    if (j==je) {b0f.x2f(m,k,j+1,i) = 0.0;}
    if (k==ke) {b0f.x3f(m,k+1,j,i) = 0.0;}
  });

  // total energy: uniform pressure p0 everywhere, plus kinetic and magnetic.
  // Separate kernel so every face of the block is set before the cell-centred
  // field is formed.
  par_for("pgen_rotor_e", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) = p0/gm1 +
        (0.5/u0(m,IDN,k,j,i))*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
                               SQR(u0(m,IM3,k,j,i))) +
        0.5*(SQR(0.5*(b0f.x1f(m,k,j,i) + b0f.x1f(m,k,j,i+1))) +
             SQR(0.5*(b0f.x2f(m,k,j,i) + b0f.x2f(m,k,j+1,i))) +
             SQR(0.5*(b0f.x3f(m,k,j,i) + b0f.x3f(m,k+1,j,i))));
  });

  return;
}
