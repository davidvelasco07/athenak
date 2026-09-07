//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file current_sheet.cpp
//  \brief Problem generator for double Harris current sheet
//

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
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::CurrentSheet_()
//  \brief Sets initial conditions for double Harris current sheet

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // read global parameters
  Real d0 = pin->GetOrAddReal("problem", "d0", 1.0);
  Real ng = pin->GetOrAddReal("problem", "ng", 1.0);
  Real bb0 = pin->GetOrAddReal("problem", "b0", 1.0);
  Real a0 = pin->GetOrAddReal("problem", "a0", 1.0);
  Real bg = pin->GetOrAddReal("problem", "bg", 0.);
  Real x01 = pin->GetOrAddReal("problem", "x01", 3.0);
  Real epsb = pin->GetOrAddReal("problem", "epsb", 0.0);
  Real epsv = pin->GetOrAddReal("problem", "epsv", 0.0);
  Real kval = pin->GetOrAddReal("problem", "kval", 1.0);

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;

    auto &u0 = pmbp->phydro->u0;

    par_for("pgen_cs_hydro", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      // compute cell-centered conserved variables
      Real den = (d0/(pow(cosh((x1v+x01)/a0),2.0)) +
                  d0/(pow(cosh((x1v-x01)/a0),2.0)) + ng);
      // divergence-free velocity seed v = curl(psi zhat)
      Real vx = (epsv*sin(kval*x2v)*(exp(-1.0*pow((x1v+x01)/a0,2.0)) +
                 exp(-1.0*pow((x1v-x01)/a0,2.0))));
      Real vy = (epsv*-2.0*cos(kval*x2v)*
                 (exp(-1.0*pow((x1v+x01)/a0,2.0))*(x1v+x01) +
                  exp(-1.0*pow((x1v-x01)/a0,2.0))*(x1v-x01))/(kval*a0*a0));

      u0(m,IDN,k,j,i) = den;
      u0(m,IM1,k,j,i) = den*vx;
      u0(m,IM2,k,j,i) = den*vy;
      u0(m,IM3,k,j,i) = 0.0;

      if (eos.is_ideal) {
        // total energy = internal + kinetic
        u0(m,IEN,k,j,i) = p0*den/gm1 + 0.5*den*(SQR(vx) + SQR(vy));
      }
    });
  }  // End initialization Hydro variables

  // initialize MHD variables ------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;

    par_for("pgen_mhd", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      // compute cell-centered conserved variables
      Real den = (d0/(pow(cosh((x1v+x01)/a0),2.0)) +
                  d0/(pow(cosh((x1v-x01)/a0),2.0))+ng);
      // divergence-free velocity seed v = curl(psi zhat)
      Real vx = (epsv*sin(kval*x2v)*(exp(-1.0*pow((x1v+x01)/a0,2.0)) +
                 exp(-1.0*pow((x1v-x01)/a0,2.0))));
      Real vy = (epsv*-2.0*cos(kval*x2v)*
                 (exp(-1.0*pow((x1v+x01)/a0,2.0))*(x1v+x01) +
                  exp(-1.0*pow((x1v-x01)/a0,2.0))*(x1v-x01))/(kval*a0*a0));

      u0(m,IDN,k,j,i) = den;
      u0(m,IM1,k,j,i) = den*vx;
      u0(m,IM2,k,j,i) = den*vy;
      u0(m,IM3,k,j,i) = 0.0;

      // Compute face-centered fields
      Real x1f   = LeftEdgeX(i  -is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x1fp1 = LeftEdgeX(i+1-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2f   = LeftEdgeX(j  -js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x2fp1 = LeftEdgeX(j+1-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3f   = LeftEdgeX(k  -ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real x3fp1 = LeftEdgeX(k+1-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

      b0.x1f(m,k,j,i) = (bb0*epsb*sin(kval*x2v)*
                         (exp(-1.0*pow((x1f+x01)/a0,2.0)) +
                          exp(-1.0*pow((x1f-x01)/a0,2.0))));
      b0.x2f(m,k,j,i) = (bb0*(tanh((x1v+x01)/a0)) -
                         bb0*(tanh((x1v-x01)/a0)) -
                         bb0 -
                         (bb0*epsb*2.0*cos(kval*x2f)*
                          (exp(-1.*pow((x1v+x01)/a0,2.0))*(x1v+x01) +
                           exp(-1.*pow((x1v-x01)/a0,2.0))*(x1v-x01))/(kval*a0*a0)));
      b0.x3f(m,k,j,i) = bg;

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = (bb0*epsb*std::sin(kval*x2v)*
                             (exp(-1.0*pow((x1fp1+x01)/a0,2.0)) +
                              exp(-1.0*pow((x1fp1-x01)/a0,2.0))));
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = (bb0*(tanh((x1v+x01)/a0)) -
                             bb0*(tanh((x1v-x01)/a0)) -
                             bb0 -
                             (bb0*epsb*2.0*cos(kval*x2fp1)*
                              (exp(-1.*pow((x1v+x01)/a0,2.0))*(x1v+x01) +
                               exp(-1.*pow((x1v-x01)/a0,2.0))*(x1v-x01))/(kval*a0*a0)));
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = bg;
      }
    });

    // Initialize total energy.  Must be a separate kernel: the cell-centered field
    // needs every face of the block to have been set first.
    //
    // The gas pressure is set from transverse force balance against the unperturbed
    // reconnecting field By = bb0*(tanh((x+x01)/a0) - tanh((x-x01)/a0) - 1), i.e.
    //   p + By^2/2 = const   =>   p(x) = p0*ng + (bb0^2/2)*(sech^2_+ + sech^2_-)
    // so the double Harris sheet is an exact equilibrium for ANY d0 and gamma.  It
    // reduces to the uniform-temperature form p = p0*rho when d0 = gamma*bb0^2/2.
    // The uniform guide field bg and the epsb/epsv tearing seed are deliberately
    // excluded from the balance -- they are the perturbation.
    if (eos.is_ideal) {
      par_for("pgen_mhd_e", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real pgas = p0*ng + 0.5*SQR(bb0)*(1.0/SQR(cosh((x1v+x01)/a0)) +
                                          1.0/SQR(cosh((x1v-x01)/a0)));

        u0(m,IEN,k,j,i) = pgas/gm1 +
            (0.5/u0(m,IDN,k,j,i))*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
                                   SQR(u0(m,IM3,k,j,i))) +
            0.5*(SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1))) +
                 SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
                 SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i))));
      });
    }
  }  // End initialization MHD variables

  return;
}

