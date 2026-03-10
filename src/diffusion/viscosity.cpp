//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.cpp
//  \brief Implements functions for Viscosity class. This includes isotropic shear
//  viscosity in a Newtonian fluid (in which stress is proportional to shear).
//  Viscosity may be added to Hydro and/or MHD independently.

#include <algorithm>
#include <limits>
#include <iostream>
#include <string> // string

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "viscosity.hpp"

//----------------------------------------------------------------------------------------
// ctor:
// Note first argument passes string ("hydro" or "mhd") denoting in wihch class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.

Viscosity::Viscosity(std::string block, MeshBlockPack *pp,
                     ParameterInput *pin) :
  pmy_pack(pp) {
  // Read coefficient of isotropic kinematic shear viscosity (must be present)
  nu_iso = pin->GetReal(block,"viscosity");

  // flag for 4th-order diffusive operators
  use_ho = pin->GetOrAddBoolean(block, "fourth_order_diff", false);

  // viscous timestep on MeshBlock(s) in this pack
  dtnew = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mb_size;
  Real fac;
  if (pp->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pp->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  for (int m=0; m<(pp->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/nu_iso);
    if (pp->pmesh->multi_d) {dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/nu_iso);}
    if (pp->pmesh->three_d) {dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/nu_iso);}
  }
}

//----------------------------------------------------------------------------------------
// Viscosity destructor

Viscosity::~Viscosity() {
}

//----------------------------------------------------------------------------------------
//! \fn void AddIsoViscousFlux
//  \brief Adds viscous fluxes to face-centered fluxes of conserved variables

void Viscosity::IsotropicViscousFlux(const DvceArray5D<Real> &w0, const Real nu_iso,
  const EOS_Data &eos, DvceFaceFld5D<Real> &flx) {
  if (use_ho) { FourthOrderIsotropicViscousFlux(w0, nu_iso, eos, flx); return; }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  int scr_level = 0;
  size_t scr_size = (ScrArray1D<Real>::shmem_size(ncells1)) * 3;
  auto flx1 = flx.x1f;

  par_for_outer("visc1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [2(dVx/dx)-(2/3)dVx/dx, dVy/dx, dVz/dx]
    par_for_inner(member, is, ie+1, [&](const int i) {
      fvx(i) = 4.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k,j,i-1))/(3.0*size.d_view(m).dx1);
      fvy(i) =     (w0(m,IVY,k,j,i) - w0(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      fvz(i) =     (w0(m,IVZ,k,j,i) - w0(m,IVZ,k,j,i-1))/size.d_view(m).dx1;
    });

    // In 2D/3D Add [(-2/3)dVy/dy, dVx/dy, 0]
    if (multi_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        fvx(i) -= ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k,j+1,i-1)) -
                   (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j-1,i-1)))/(6.0*size.d_view(m).dx2);
        fvy(i) += ((w0(m,IVX,k,j+1,i) + w0(m,IVX,k,j+1,i-1)) -
                   (w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j-1,i-1)))/(4.0*size.d_view(m).dx2);
      });
    }

    // In 3D Add [(-2/3)dVz/dz, 0,  dVx/dz]
    if (three_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        fvx(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j,i-1)) -
                   (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j,i-1)))/(6.0*size.d_view(m).dx3);
        fvz(i) += ((w0(m,IVX,k+1,j,i) + w0(m,IVX,k+1,j,i-1)) -
                   (w0(m,IVX,k-1,j,i) + w0(m,IVX,k-1,j,i-1)))/(4.0*size.d_view(m).dx3);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie+1, [&](const int i) {
      Real nud = 0.5*nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
      flx1(m,IVX,k,j,i) -= nud*fvx(i);
      flx1(m,IVY,k,j,i) -= nud*fvy(i);
      flx1(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx1(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto flx2 = flx.x2f;

  par_for_outer("visc2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [(dVx/dy+dVy/dx), 2(dVy/dy)-(2/3)(dVx/dx+dVy/dy), dVz/dy]
    par_for_inner(member, is, ie, [&](const int i) {
      fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k,j-1,i  ))/size.d_view(m).dx2 +
              ((w0(m,IVY,k,j,i+1) + w0(m,IVY,k,j-1,i+1)) -
               (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j-1,i-1)))/(4.0*size.d_view(m).dx1);
      fvy(i) = (w0(m,IVY,k,j,i) - w0(m,IVY,k,j-1,i))*4.0/(3.0*size.d_view(m).dx2) -
              ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k,j-1,i+1)) -
               (w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j-1,i-1)))/(6.0*size.d_view(m).dx1);
      fvz(i) = (w0(m,IVZ,k,j,i  ) - w0(m,IVZ,k,j-1,i  ))/size.d_view(m).dx2;
    });

    // In 3D Add [0, (-2/3)dVz/dz, dVy/dz]
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        fvy(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j-1,i)) -
                   (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j-1,i)))/(6.0*size.d_view(m).dx3);
        fvz(i) += ((w0(m,IVY,k+1,j,i) + w0(m,IVY,k+1,j-1,i)) -
                   (w0(m,IVY,k-1,j,i) + w0(m,IVY,k-1,j-1,i)))/(4.0*size.d_view(m).dx3);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real nud = 0.5*nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
      flx2(m,IVX,k,j,i) -= nud*fvx(i);
      flx2(m,IVY,k,j,i) -= nud*fvy(i);
      flx2(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx2(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto flx3 = flx.x3f;

  par_for_outer("visc3",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 2(dVz/dz)-(2/3)(dVx/dx+dVy/dy+dVz/dz)]
    par_for_inner(member, is, ie, [&](const int i) {
      fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k-1,j,i  ))/size.d_view(m).dx3 +
              ((w0(m,IVZ,k,j,i+1) + w0(m,IVZ,k-1,j,i+1)) -
               (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k-1,j,i-1)))/(4.0*size.d_view(m).dx1);
      fvy(i) = (w0(m,IVY,k,j,i  ) - w0(m,IVY,k-1,j,i  ))/size.d_view(m).dx3 +
              ((w0(m,IVZ,k,j+1,i) + w0(m,IVZ,k-1,j+1,i)) -
               (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k-1,j-1,i)))/(4.0*size.d_view(m).dx2);
      fvz(i) = (w0(m,IVZ,k,j,i) - w0(m,IVZ,k-1,j,i))*4.0/(3.0*size.d_view(m).dx3) -
              ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k-1,j,i+1)) -
               (w0(m,IVX,k,j,i-1) + w0(m,IVX,k-1,j,i-1)))/(6.0*size.d_view(m).dx1) -
              ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k-1,j+1,i)) -
               (w0(m,IVY,k,j-1,i) + w0(m,IVY,k-1,j-1,i)))/(6.0*size.d_view(m).dx2);
    });

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real nud = 0.5*nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
      flx3(m,IVX,k,j,i) -= nud*fvx(i);
      flx3(m,IVY,k,j,i) -= nud*fvy(i);
      flx3(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx3(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k-1,j,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k-1,j,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FourthOrderIsotropicViscousFlux
//  \brief Adds 4th-order viscous fluxes to face-centered fluxes of conserved variables.
//  Uses 4-point stencils for all derivatives:
//    normal:     (15(q_i - q_{i-1}) - (q_{i+1} - q_{i-2})) / (12*dx)
//    transverse: (-q_{j+2} + 8q_{j+1} - 8q_{j-1} + q_{j-2}) / (12*dy),  averaged at face

void Viscosity::FourthOrderIsotropicViscousFlux(const DvceArray5D<Real> &w0,
  const Real nu_iso, const EOS_Data &eos, DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  int scr_level = 0;
  size_t scr_size = (ScrArray1D<Real>::shmem_size(ncells1)) * 3;
  auto flx1 = flx.x1f;

  par_for_outer("visc4_1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // 4th-order normal derivatives [4/3 dVx/dx, dVy/dx, dVz/dx]
    par_for_inner(member, is, ie+1, [&](const int i) {
      Real idx1 = 1.0/size.d_view(m).dx1;
      fvx(i) = 4.0/3.0*(15.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k,j,i-1))
                           - (w0(m,IVX,k,j,i+1) - w0(m,IVX,k,j,i-2)))*(idx1/12.0);
      fvy(i) =          (15.0*(w0(m,IVY,k,j,i) - w0(m,IVY,k,j,i-1))
                            - (w0(m,IVY,k,j,i+1) - w0(m,IVY,k,j,i-2)))*(idx1/12.0);
      fvz(i) =          (15.0*(w0(m,IVZ,k,j,i) - w0(m,IVZ,k,j,i-1))
                            - (w0(m,IVZ,k,j,i+1) - w0(m,IVZ,k,j,i-2)))*(idx1/12.0);
    });

    // In 2D/3D: 4th-order transverse derivatives, averaged at face
    // fvx -= (2/3) dVy/dy;   fvy += dVx/dy
    if (multi_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real idy2 = 1.0/size.d_view(m).dx2;
        // dVy/dy at (i) and (i-1), 4-point stencil, then face-average
        Real dyVy_i   = (-w0(m,IVY,k,j+2,i  ) + 8.0*w0(m,IVY,k,j+1,i  )
                         -8.0*w0(m,IVY,k,j-1,i  ) +     w0(m,IVY,k,j-2,i  ))*(idy2/12.0);
        Real dyVy_im1 = (-w0(m,IVY,k,j+2,i-1) + 8.0*w0(m,IVY,k,j+1,i-1)
                         -8.0*w0(m,IVY,k,j-1,i-1) +     w0(m,IVY,k,j-2,i-1))*(idy2/12.0);
        fvx(i) -= (2.0/3.0)*0.5*(dyVy_i + dyVy_im1);

        // dVx/dy at (i) and (i-1), 4-point stencil, then face-average
        Real dyVx_i   = (-w0(m,IVX,k,j+2,i  ) + 8.0*w0(m,IVX,k,j+1,i  )
                         -8.0*w0(m,IVX,k,j-1,i  ) +     w0(m,IVX,k,j-2,i  ))*(idy2/12.0);
        Real dyVx_im1 = (-w0(m,IVX,k,j+2,i-1) + 8.0*w0(m,IVX,k,j+1,i-1)
                         -8.0*w0(m,IVX,k,j-1,i-1) +     w0(m,IVX,k,j-2,i-1))*(idy2/12.0);
        fvy(i) += 0.5*(dyVx_i + dyVx_im1);
      });
    }

    // In 3D: fvx -= (2/3) dVz/dz;   fvz += dVx/dz
    if (three_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real idz3 = 1.0/size.d_view(m).dx3;
        Real dzVz_i   = (-w0(m,IVZ,k+2,j,i  ) + 8.0*w0(m,IVZ,k+1,j,i  )
                         -8.0*w0(m,IVZ,k-1,j,i  ) +     w0(m,IVZ,k-2,j,i  ))*(idz3/12.0);
        Real dzVz_im1 = (-w0(m,IVZ,k+2,j,i-1) + 8.0*w0(m,IVZ,k+1,j,i-1)
                         -8.0*w0(m,IVZ,k-1,j,i-1) +     w0(m,IVZ,k-2,j,i-1))*(idz3/12.0);
        fvx(i) -= (2.0/3.0)*0.5*(dzVz_i + dzVz_im1);

        Real dzVx_i   = (-w0(m,IVX,k+2,j,i  ) + 8.0*w0(m,IVX,k+1,j,i  )
                         -8.0*w0(m,IVX,k-1,j,i  ) +     w0(m,IVX,k-2,j,i  ))*(idz3/12.0);
        Real dzVx_im1 = (-w0(m,IVX,k+2,j,i-1) + 8.0*w0(m,IVX,k+1,j,i-1)
                         -8.0*w0(m,IVX,k-1,j,i-1) +     w0(m,IVX,k-2,j,i-1))*(idz3/12.0);
        fvz(i) += 0.5*(dzVx_i + dzVx_im1);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie+1, [&](const int i) {
      // 4th-order face density and velocity
      Real rho_f = (7.0/12.0)*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1))
                 - (1.0/12.0)*(w0(m,IDN,k,j,i+1) + w0(m,IDN,k,j,i-2));
      Real nud = nu_iso*rho_f;
      flx1(m,IVX,k,j,i) -= nud*fvx(i);
      flx1(m,IVY,k,j,i) -= nud*fvy(i);
      flx1(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        Real vx_f = (7.0/12.0)*(w0(m,IVX,k,j,i) + w0(m,IVX,k,j,i-1))
                  - (1.0/12.0)*(w0(m,IVX,k,j,i+1) + w0(m,IVX,k,j,i-2));
        Real vy_f = (7.0/12.0)*(w0(m,IVY,k,j,i) + w0(m,IVY,k,j,i-1))
                  - (1.0/12.0)*(w0(m,IVY,k,j,i+1) + w0(m,IVY,k,j,i-2));
        Real vz_f = (7.0/12.0)*(w0(m,IVZ,k,j,i) + w0(m,IVZ,k,j,i-1))
                  - (1.0/12.0)*(w0(m,IVZ,k,j,i+1) + w0(m,IVZ,k,j,i-2));
        flx1(m,IEN,k,j,i) -= nud*(vx_f*fvx(i) + vy_f*fvy(i) + vz_f*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto flx2 = flx.x2f;

  par_for_outer("visc4_2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // 4th-order normal derivatives [(dVx/dy+dVy/dx), 4/3 dVy/dy, dVz/dy]
    par_for_inner(member, is, ie, [&](const int i) {
      Real idy2 = 1.0/size.d_view(m).dx2;
      // normal: 4th-order difference in y
      Real dVx = (15.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k,j-1,i))
                    - (w0(m,IVX,k,j+1,i) - w0(m,IVX,k,j-2,i)))*(idy2/12.0);
      Real dVy = (15.0*(w0(m,IVY,k,j,i) - w0(m,IVY,k,j-1,i))
                    - (w0(m,IVY,k,j+1,i) - w0(m,IVY,k,j-2,i)))*(idy2/12.0);
      Real dVz = (15.0*(w0(m,IVZ,k,j,i) - w0(m,IVZ,k,j-1,i))
                    - (w0(m,IVZ,k,j+1,i) - w0(m,IVZ,k,j-2,i)))*(idy2/12.0);

      // transverse: dVy/dx (divergence, -2/3) and dVx/dx (divergence, -2/3)
      Real idx1 = 1.0/size.d_view(m).dx1;
      Real dxVy_j   = (-w0(m,IVY,k,j  ,i+2) + 8.0*w0(m,IVY,k,j  ,i+1)
                       -8.0*w0(m,IVY,k,j  ,i-1) +     w0(m,IVY,k,j  ,i-2))*(idx1/12.0);
      Real dxVy_jm1 = (-w0(m,IVY,k,j-1,i+2) + 8.0*w0(m,IVY,k,j-1,i+1)
                       -8.0*w0(m,IVY,k,j-1,i-1) +     w0(m,IVY,k,j-1,i-2))*(idx1/12.0);
      Real dxVx_j   = (-w0(m,IVX,k,j  ,i+2) + 8.0*w0(m,IVX,k,j  ,i+1)
                       -8.0*w0(m,IVX,k,j  ,i-1) +     w0(m,IVX,k,j  ,i-2))*(idx1/12.0);
      Real dxVx_jm1 = (-w0(m,IVX,k,j-1,i+2) + 8.0*w0(m,IVX,k,j-1,i+1)
                       -8.0*w0(m,IVX,k,j-1,i-1) +     w0(m,IVX,k,j-1,i-2))*(idx1/12.0);

      fvx(i) = dVx + 0.5*(dxVy_j + dxVy_jm1);
      fvy(i) = 4.0/3.0*dVy - (2.0/3.0)*0.5*(dxVx_j + dxVx_jm1);
      fvz(i) = dVz;
    });

    // In 3D: fvy -= (2/3) dVz/dz;   fvz += dVy/dz
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        Real idz3 = 1.0/size.d_view(m).dx3;
        Real dzVz_j   = (-w0(m,IVZ,k+2,j  ,i) + 8.0*w0(m,IVZ,k+1,j  ,i)
                         -8.0*w0(m,IVZ,k-1,j  ,i) +     w0(m,IVZ,k-2,j  ,i))*(idz3/12.0);
        Real dzVz_jm1 = (-w0(m,IVZ,k+2,j-1,i) + 8.0*w0(m,IVZ,k+1,j-1,i)
                         -8.0*w0(m,IVZ,k-1,j-1,i) +     w0(m,IVZ,k-2,j-1,i))*(idz3/12.0);
        fvy(i) -= (2.0/3.0)*0.5*(dzVz_j + dzVz_jm1);

        Real dzVy_j   = (-w0(m,IVY,k+2,j  ,i) + 8.0*w0(m,IVY,k+1,j  ,i)
                         -8.0*w0(m,IVY,k-1,j  ,i) +     w0(m,IVY,k-2,j  ,i))*(idz3/12.0);
        Real dzVy_jm1 = (-w0(m,IVY,k+2,j-1,i) + 8.0*w0(m,IVY,k+1,j-1,i)
                         -8.0*w0(m,IVY,k-1,j-1,i) +     w0(m,IVY,k-2,j-1,i))*(idz3/12.0);
        fvz(i) += 0.5*(dzVy_j + dzVy_jm1);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real rho_f = (7.0/12.0)*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i))
                 - (1.0/12.0)*(w0(m,IDN,k,j+1,i) + w0(m,IDN,k,j-2,i));
      Real nud = nu_iso*rho_f;
      flx2(m,IVX,k,j,i) -= nud*fvx(i);
      flx2(m,IVY,k,j,i) -= nud*fvy(i);
      flx2(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        Real vx_f = (7.0/12.0)*(w0(m,IVX,k,j,i) + w0(m,IVX,k,j-1,i))
                  - (1.0/12.0)*(w0(m,IVX,k,j+1,i) + w0(m,IVX,k,j-2,i));
        Real vy_f = (7.0/12.0)*(w0(m,IVY,k,j,i) + w0(m,IVY,k,j-1,i))
                  - (1.0/12.0)*(w0(m,IVY,k,j+1,i) + w0(m,IVY,k,j-2,i));
        Real vz_f = (7.0/12.0)*(w0(m,IVZ,k,j,i) + w0(m,IVZ,k,j-1,i))
                  - (1.0/12.0)*(w0(m,IVZ,k,j+1,i) + w0(m,IVZ,k,j-2,i));
        flx2(m,IEN,k,j,i) -= nud*(vx_f*fvx(i) + vy_f*fvy(i) + vz_f*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto flx3 = flx.x3f;

  par_for_outer("visc4_3",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // 4th-order normal derivatives [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 4/3 dVz/dz]
    par_for_inner(member, is, ie, [&](const int i) {
      Real idz3 = 1.0/size.d_view(m).dx3;
      Real dVx = (15.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k-1,j,i))
                    - (w0(m,IVX,k+1,j,i) - w0(m,IVX,k-2,j,i)))*(idz3/12.0);
      Real dVy = (15.0*(w0(m,IVY,k,j,i) - w0(m,IVY,k-1,j,i))
                    - (w0(m,IVY,k+1,j,i) - w0(m,IVY,k-2,j,i)))*(idz3/12.0);
      Real dVz = (15.0*(w0(m,IVZ,k,j,i) - w0(m,IVZ,k-1,j,i))
                    - (w0(m,IVZ,k+1,j,i) - w0(m,IVZ,k-2,j,i)))*(idz3/12.0);

      Real idx1 = 1.0/size.d_view(m).dx1;
      Real dxVz_k   = (-w0(m,IVZ,k  ,j,i+2) + 8.0*w0(m,IVZ,k  ,j,i+1)
                       -8.0*w0(m,IVZ,k  ,j,i-1) +     w0(m,IVZ,k  ,j,i-2))*(idx1/12.0);
      Real dxVz_km1 = (-w0(m,IVZ,k-1,j,i+2) + 8.0*w0(m,IVZ,k-1,j,i+1)
                       -8.0*w0(m,IVZ,k-1,j,i-1) +     w0(m,IVZ,k-1,j,i-2))*(idx1/12.0);
      Real dxVx_k   = (-w0(m,IVX,k  ,j,i+2) + 8.0*w0(m,IVX,k  ,j,i+1)
                       -8.0*w0(m,IVX,k  ,j,i-1) +     w0(m,IVX,k  ,j,i-2))*(idx1/12.0);
      Real dxVx_km1 = (-w0(m,IVX,k-1,j,i+2) + 8.0*w0(m,IVX,k-1,j,i+1)
                       -8.0*w0(m,IVX,k-1,j,i-1) +     w0(m,IVX,k-1,j,i-2))*(idx1/12.0);

      Real idy2 = 1.0/size.d_view(m).dx2;
      Real dyVz_k   = (-w0(m,IVZ,k  ,j+2,i) + 8.0*w0(m,IVZ,k  ,j+1,i)
                       -8.0*w0(m,IVZ,k  ,j-1,i) +     w0(m,IVZ,k  ,j-2,i))*(idy2/12.0);
      Real dyVz_km1 = (-w0(m,IVZ,k-1,j+2,i) + 8.0*w0(m,IVZ,k-1,j+1,i)
                       -8.0*w0(m,IVZ,k-1,j-1,i) +     w0(m,IVZ,k-1,j-2,i))*(idy2/12.0);
      Real dyVy_k   = (-w0(m,IVY,k  ,j+2,i) + 8.0*w0(m,IVY,k  ,j+1,i)
                       -8.0*w0(m,IVY,k  ,j-1,i) +     w0(m,IVY,k  ,j-2,i))*(idy2/12.0);
      Real dyVy_km1 = (-w0(m,IVY,k-1,j+2,i) + 8.0*w0(m,IVY,k-1,j+1,i)
                       -8.0*w0(m,IVY,k-1,j-1,i) +     w0(m,IVY,k-1,j-2,i))*(idy2/12.0);

      fvx(i) = dVx + 0.5*(dxVz_k + dxVz_km1);
      fvy(i) = dVy + 0.5*(dyVz_k + dyVz_km1);
      fvz(i) = 4.0/3.0*dVz
             - (2.0/3.0)*0.5*(dxVx_k + dxVx_km1)
             - (2.0/3.0)*0.5*(dyVy_k + dyVy_km1);
    });

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real rho_f = (7.0/12.0)*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i))
                 - (1.0/12.0)*(w0(m,IDN,k+1,j,i) + w0(m,IDN,k-2,j,i));
      Real nud = nu_iso*rho_f;
      flx3(m,IVX,k,j,i) -= nud*fvx(i);
      flx3(m,IVY,k,j,i) -= nud*fvy(i);
      flx3(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        Real vx_f = (7.0/12.0)*(w0(m,IVX,k,j,i) + w0(m,IVX,k-1,j,i))
                  - (1.0/12.0)*(w0(m,IVX,k+1,j,i) + w0(m,IVX,k-2,j,i));
        Real vy_f = (7.0/12.0)*(w0(m,IVY,k,j,i) + w0(m,IVY,k-1,j,i))
                  - (1.0/12.0)*(w0(m,IVY,k+1,j,i) + w0(m,IVY,k-2,j,i));
        Real vz_f = (7.0/12.0)*(w0(m,IVZ,k,j,i) + w0(m,IVZ,k-1,j,i))
                  - (1.0/12.0)*(w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k-2,j,i));
        flx3(m,IEN,k,j,i) -= nud*(vx_f*fvx(i) + vy_f*fvy(i) + vz_f*fvz(i));
      }
    });
  });

  return;
}
