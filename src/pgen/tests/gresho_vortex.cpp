//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_wave.c
//! \brief Linear wave problem generator for 1D/2D/3D problems. Initializes both hydro and
//! MHD problems for non-relativistic and SR/GR relativistic flows, and for GRMHD in
//! dynamical spacetimes (dynGR)..
//!
//! Direction of the wavevector is set to be along the x? axis by using the
//! along_x? input flags, else it is automatically set along the grid diagonal in 2D/3D
//! See comments in Athena4.2 linear wave problem generator for more details.
//!
//! Errors in solution after an integer number of wave periods are automatically output
//! at end of calculation.

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \struct LinWaveVariables
//! \brief container for variables shared with vector potential and perturbation functions


namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;

} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::GreshoVortex()
//! \brief Sets initial conditions for linear wave tests

void ProblemGenerator::GreshoVortex(ParameterInput *pin, const bool restart) {
  // set linear wave errors function
  //pgen_final_func = GreshoVortexErrors;
  if (restart) return;
  bool use_mignone = pmy_mesh_->pmb_pack->phydro->use_mignone;

  // capture variables for kernels
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  Real Mach = pin->GetOrAddReal("problem","Mach",1.0);
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  Real U0 = pin->GetOrAddReal("problem","U0",1.0);
  Real T0 = pin->GetOrAddReal("problem","T0",1.0);
  Real dzp = pin->GetOrAddReal("problem","dzp",.2);
  int problem = pin->GetOrAddInteger("problem","problem",1);
  // Vortex center = global domain center
  Real xc = 0.5*(pin->GetReal("mesh","x1min") + pin->GetReal("mesh","x1max"));
  Real yc = 0.5*(pin->GetReal("mesh","x2min") + pin->GetReal("mesh","x2max"));
  Real zc = 0.5*(pin->GetReal("mesh","x3min") + pin->GetReal("mesh","x3max"));

  // Rotation angles for problem=4 (3D inclined Gresho vortex)
  // Follows same convention as linear_wave.cpp
  Real cos_a2 = 1.0, sin_a2 = 0.0, cos_a3 = 1.0, sin_a3 = 0.0;
  if (problem == 4) {
    bool along_x1 = pin->GetOrAddBoolean("problem","along_x1",false);
    bool along_x2 = pin->GetOrAddBoolean("problem","along_x2",false);
    bool along_x3 = pin->GetOrAddBoolean("problem","along_x3",false);
    Real x1size = pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min");
    Real x2size = pin->GetReal("mesh","x2max") - pin->GetReal("mesh","x2min");
    Real x3size = pin->GetReal("mesh","x3max") - pin->GetReal("mesh","x3min");

    // Default: vortex axis along grid diagonal
    if (pmy_mesh_->multi_d && !along_x1) {
      Real ang_3 = std::atan(x1size/x2size);
      sin_a3 = std::sin(ang_3);
      cos_a3 = std::cos(ang_3);
    }
    if (pmy_mesh_->three_d && !along_x1) {
      Real ang_2 = std::atan(0.5*(x1size*cos_a3 + x2size*sin_a3)/x3size);
      sin_a2 = std::sin(ang_2);
      cos_a2 = std::cos(ang_2);
    }
    if (along_x2) { cos_a3 = 0.0; sin_a3 = 1.0; cos_a2 = 1.0; sin_a2 = 0.0; }
    if (along_x3) { cos_a3 = 0.0; sin_a3 = 1.0; cos_a2 = 0.0; sin_a2 = 1.0; }
  }

  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real gamma_adi_red = eos.gamma / (eos.gamma - 1.0);

    // Calculate cell-centered primitive variables
    auto &w0 = pmbp->phydro->w0;
    auto &u0 = pmbp->phydro->u0;
    par_for("pgen_linwave1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),0,n3m1,0,n2m1,0,n1m1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &dx1 = size.d_view(m).dx1;
      int nx1 = indcs.nx1;
      
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &dx2 = size.d_view(m).dx2;
      int nx2 = indcs.nx2;
      
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real &dx3 = size.d_view(m).dx3;
      int nx3 = indcs.nx3;
      
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      if(problem == 1){
        // 2D Gresho vortex centered at domain midpoint
        Real xx = x1v - xc;
        Real yy = x2v - yc;
        Real r = std::sqrt(xx*xx + yy*yy);
        Real vphi;
        Real P0 = 1./((gm1 + 1.0)*Mach*Mach) - 0.5;
        w0(m,IDN,k,j,i) = 1;
        if (r < 0.2) {
          vphi = 5.0*r;
          w0(m,IEN,k,j,i) = (P0 + 25.0/2.0*r*r)/gm1;
        } else if (r < 0.4) {
          vphi = 2.0 - 5.0*r;
          w0(m,IEN,k,j,i) = (P0 +4.0*std::log(5.0*r) + 4  -20.0*r +25.0/2.0*r*r )/gm1;
        } else {
          vphi = 0.0;
          w0(m,IEN,k,j,i) = (P0 + 4.0*std::log(2.0)-2)/gm1;
        }
        if (r > 0) {
          w0(m,IVX,k,j,i) = -vphi * yy/r;
          w0(m,IVY,k,j,i) =  vphi * xx/r;
        } else {
          w0(m,IVX,k,j,i) = 0.0;
          w0(m,IVY,k,j,i) = 0.0;
        }
        w0(m,IVZ,k,j,i) =  0.0;
      }
      else if(problem == 2){
        // 2D Inviscid Taylor-Green vortex
        Real p0 = (rho0 * pow(U0 / Mach, 2.0)) / (gm1+1);
        w0(m,IDN,k,j,i) = rho0;
        w0(m,IVX,k,j,i) =  U0 * sin(x1v) * cos(x2v);
        w0(m,IVY,k,j,i) = -U0 * cos(x1v) * sin(x2v);
        w0(m,IVZ,k,j,i) =  0.0;
        w0(m,IEN,k,j,i) = p0/gm1 + 0.25*rho0*U0*U0*(cos(2*x1v) + cos(2*x2v));
      }
      else if(problem==3){
        // 2D Stationary Isentropic vortex
        Real beta = 5.0/(2.0*M_PI*sqrt(gm1+1))*exp(0.5);
        Real r2 = x1v*x1v + x2v*x2v;
        Real f = -0.5*r2;
        Real omega = beta*exp(f);
        Real deltaT = -0.5*(gm1*omega*omega);
        w0(m,IDN,k,j,i) = pow(1.0 + deltaT, 1.0/gm1);
        w0(m,IVX,k,j,i) =  - omega*x2v;
        w0(m,IVY,k,j,i) =    omega*x1v ;
        w0(m,IVZ,k,j,i) =  0.0;
        w0(m,IEN,k,j,i) = pow(1.0 + deltaT, gamma_adi_red)/(gm1+1)/gm1;
      }
      else if(problem==4){
        Real a3 = M_PI/16.0; // rotation angle for 3D inclined Gresho vortex
        Real cos_a3 = std::cos(a3);
        Real sin_a3 = std::sin(a3);
        // 3D inclined Gresho vortex
        // Transform grid coords to vortex frame (axis along rotated x')
        Real dx = x1v - xc;
        Real dy = x2v - yc;
        Real dz = x3v - zc;
        // x' and y' span the vortex plane (perpendicular to axis)
        Real xp =  dx*cos_a3 + dz*sin_a3;
        Real yp =  dy;
        Real zp = -dx*sin_a3 + dz*cos_a3;

        Real r = std::sqrt(xp*xp + yp*yp);

        Real vphi;
        Real P0 = 1./((gm1 + 1.0)*Mach*Mach) - 0.5;
        w0(m,IDN,k,j,i) = 1.0;
        Real h = std::abs(zp);
        if (r < 0.2 and  h < dzp) {
          vphi = 5.0*r*(dzp-h);
          w0(m,IEN,k,j,i) = (P0 + 25.0/2.0*r*r)/gm1;
        } else if (r < 0.4 and std::abs(zp) < dzp) {
          vphi = (2.0 - 5.0*r)*(dzp-h);
          w0(m,IEN,k,j,i) = (P0 + 4.0*std::log(5.0*r) + 4 - 20.0*r
                              + 25.0/2.0*r*r)/gm1;
        } else {
          vphi = 0.0;
          w0(m,IEN,k,j,i) = (P0 + 4.0*std::log(2.0) - 2.0)/gm1;
        }

        if (r > 0) {
          // Velocity in vortex frame: v_y' = -vphi*z'/r, v_z' = vphi*y'/r
          Real vxp = -vphi * yp/r;
          Real vyp =  vphi * xp/r;
          Real vzp = 0.0;
          // Inverse rotation to grid frame
          w0(m,IVX,k,j,i) =  vxp*cos_a3 - vzp*sin_a3;
          w0(m,IVY,k,j,i) =  vyp;
          w0(m,IVZ,k,j,i) =  vxp*sin_a3 + vzp*cos_a3;
        } else {
          w0(m,IVX,k,j,i) = 0.0;
          w0(m,IVY,k,j,i) = 0.0;
          w0(m,IVZ,k,j,i) = 0.0;
        }
      }
    });
    // "regular" GRHydro in stationary spacetimes
    //At this point we have cell-centered values for both conservative and primitive variables
    //We therefore need to compute the control volume average for both
    if (use_mignone) {
      // Primitive variables: volume average -> cell center
      pmbp->pcoord->DeAverageVolume(pmbp->phydro->w0, pmbp->phydro->w0_c);
      // Conservative variables from cell-centered primitive variables
      pmbp->phydro->peos->PrimToCons(pmbp->phydro->w0_c, pmbp->phydro->u0_c, 0, n1m1, 0, n2m1, 0, n3m1);
      // Conservative variables: cell center -> volume average
      pmbp->pcoord->AverageVolume(pmbp->phydro->u0_c, u0);
    } else {
      pmbp->phydro->peos->PrimToCons(w0, u0, 0, n1m1, 0, n2m1, 0, n3m1);
    }
  }  // End initialization Hydro variables
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void LinearWaveErrors_()
//! \brief Computes errors in linear wave solution by calling initialization function
//! again to compute initial condictions, and then calling generic error output function
//! that subtracts current solution from ICs, and outputs errors to file. Problem must be
//! run for an integer number of wave periods.

//void GreshoVortexErrors(ParameterInput *pin, Mesh *pm) {
//  // calculate reference solution by calling pgen again.  Solution stored in second
//  // register u1/b1 when flag is false.
//  set_initial_conditions = false;
//  pm->pgen->GreshoVortex(pin, false);
//  pm->pgen->OutputErrors(pin, pm);
//  return;
//}
