//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file slotted_cyl.cpp
//  \brief Slotted cylinder passive scalar advection problem generator for 2D/3D problems.

// C++ headers
#include <iostream>   // endl

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

namespace {

constexpr Real d0 = 1.0;

// Device-callable profile: all geometry is passed in (no host-namespace captures).
KOKKOS_INLINE_FUNCTION
Real SlottedCylinderProfile(Real x1, Real x2, Real center_x1, Real center_x2,
                            Real radius, Real s_width, Real s_height) {
  Real zx = x1 - center_x1;
  Real zy = x2 - center_x2;
  Real r = sqrt(SQR(zx) + SQR(zy));

  if (r > radius)
    return 0.0;
  if ((fabs(2*zx) < s_width) && (zy + radius < s_height) && (0 < zy + radius))
    return 0.0;
  return 1.0;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for the Zalesak slotted cylinder.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Slotted cylinder test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Capture by value for the device kernel (NVCC cannot read host-namespace globals).
  const Real radius = pin->GetOrAddReal("problem", "radius", 0.15);
  const Real center_x1 = pin->GetOrAddReal("problem", "center_x1", 0.50);
  const Real center_x2 = pin->GetOrAddReal("problem", "center_x2", 0.75);
  const Real omega = pin->GetOrAddReal("problem", "omega", 1.0);
  const Real omega_x1 = pin->GetOrAddReal("problem", "omega_x1", 0.50);
  const Real omega_x2 = pin->GetOrAddReal("problem", "omega_x2", 0.50);
  const Real s_width = pin->GetOrAddReal("problem", "s_width", 0.05);
  const Real s_height = pin->GetOrAddReal("problem", "s_height", 0.25);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2;
  int nscalars = pmbp->phydro->nscalars;
  int nhydro = pmbp->phydro->nhydro;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;

  par_for("pgen_slot_cyl", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = -d0*2.0*M_PI*omega*(x2v - omega_x2);
    u0(m,IM2,k,j,i) =  d0*2.0*M_PI*omega*(x1v - omega_x1);
    u0(m,IM3,k,j,i) = 0.0;

    Real cell_ave = SlottedCylinderProfile(x1v, x2v, center_x1, center_x2,
                                           radius, s_width, s_height);
    for (int n=nhydro; n<(nhydro+nscalars); ++n) {
      u0(m,n,k,j,i) = cell_ave*d0;
    }
  });

  return;
}
