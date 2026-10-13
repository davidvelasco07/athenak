//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gresho.cpp
//! \brief Low-Mach Gresho vortex (Gresho & Chan 1990; Miczek et al. 2015), in the form of
//! Velasco-Romero & Teyssier (2025, MNRAS, arXiv:2405.11063) sec. 3.1:
//!   rho = 1,  v_phi = 5r (r<0.2), 2-5r (0.2<r<0.4), 0 (r>0.4),
//!   p = 1/(gamma M^2) - 1/2 + { 25r^2/2 ; 25r^2/2 + 4(ln 5r + 1 - 5r) ; 4 ln 2 - 2 },
//! so v_phi/c_s peaks at exactly M (problem/mach) at r = 0.2, plus a uniform background
//! velocity problem/v0 along x1 (the paper uses v0 = 5 on the unit periodic box, i.e. five
//! crossings by t = 1).  The vortex is centred on the domain.  The state is set as
//! Gauss-Legendre cell averages of the CONSERVED variables, so the initial energy is
//! consistent with the averaged momentum.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "hydro/well_balance.hpp"
#include "pgen/pgen.hpp"

namespace {

struct GreshoState {         // conserved state at a point
  Real mach, v0, xc, yc, gamma;
  KOKKOS_INLINE_FUNCTION
  void operator()(const Real x1, const Real x2, const Real x3, Real u[]) const {
    const Real dx = x1 - xc, dy = x2 - yc;
    const Real r = sqrt(dx*dx + dy*dy);
    Real vphi, p = 1.0/(gamma*mach*mach) - 0.5;
    if (r < 0.2) {
      vphi = 5.0*r;
      p += 12.5*r*r;
    } else if (r < 0.4) {
      vphi = 2.0 - 5.0*r;
      p += 12.5*r*r + 4.0*(log(5.0*r) + 1.0 - 5.0*r);
    } else {
      vphi = 0.0;
      p += 4.0*log(2.0) - 2.0;
    }
    const Real vx = ((r > 0.0) ? -vphi*dy/r : 0.0) + v0;
    const Real vy = (r > 0.0) ? vphi*dx/r : 0.0;
    u[IDN] = 1.0;
    u[IM1] = vx;
    u[IM2] = vy;
    u[IM3] = 0.0;
    u[IEN] = p/(gamma - 1.0) + 0.5*(vx*vx + vy*vy);
  }
};

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Gresho()

void ProblemGenerator::Gresho(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || !pmbp->phydro->peos->eos_data.is_ideal ||
      !pmy_mesh_->two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gresho pgen requires 2D <hydro> with an ideal gas EOS" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &ms = pmy_mesh_->mesh_size;
  GreshoState gs;
  gs.mach = pin->GetOrAddReal("problem", "mach", 0.1);
  gs.v0 = pin->GetOrAddReal("problem", "v0", 0.0);
  gs.xc = 0.5*(ms.x1min + ms.x1max);
  gs.yc = 0.5*(ms.x2min + ms.x2max);
  gs.gamma = pmbp->phydro->peos->eos_data.gamma;
  wellbalance::FillEquilibriumCellAverages(pmbp, pmbp->phydro->u0, pmbp->phydro->nhydro,
                                           gs);
  return;
}
