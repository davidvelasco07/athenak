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
#include "ho_diffusion_stencil.hpp"

//----------------------------------------------------------------------------------------
// ctor:
// Note first argument passes string ("hydro" or "mhd") denoting in wihch class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.

Viscosity::Viscosity(std::string block, MeshBlockPack *pp,
                     ParameterInput *pin) :
  pmy_pack(pp),
  ho_face_vel_("visc_ho_face_vel", 1, 4, 1, 1, 1),
  mignone_(pin->GetOrAddBoolean(block, "mignone", false)) {
  // Read coefficient of isotropic kinematic shear viscosity (must be present)
  nu_iso = pin->GetReal(block,"viscosity");

  // flag for 4th-order diffusive operators
  use_ho = pin->GetOrAddBoolean(block, "fourth_order_diff", false);

  if (use_ho) {
    int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
    auto &indcs = pp->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(ho_face_vel_.x1f, nmb, 4, ncells3, ncells2, ncells1);
    Kokkos::realloc(ho_face_vel_.x2f, nmb, 4, ncells3, ncells2, ncells1);
    Kokkos::realloc(ho_face_vel_.x3f, nmb, 4, ncells3, ncells2, ncells1);
  }

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
  if (use_ho) {
    FourthOrderIsotropicViscousFlux(w0, nu_iso, eos, flx);
    return;
  }
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
//! \fn void Viscosity::FillHoViscFaceVelocity
//  \brief Store (Vx,Vy,Vz) at each face using HoFaceValue along the face normal
//  (four nearest cell-centered samples). Transverse viscous terms use HoGrad
//  on these values along transverse directions.

void Viscosity::FillHoViscFaceVelocity(const DvceArray5D<Real> &w) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto vx1 = ho_face_vel_.x1f;
  auto vx2 = ho_face_vel_.x2f;
  auto vx3 = ho_face_vel_.x3f;

  // x1-face velocities: needed for x1 flux (including 1D ideal-gas energy term)
  // Extend into ghost zones (±2) in transverse directions so that HoGradT
  // can access face values at j±2 and k±2 for interior flux faces.
  int jl1 = multi_d ? js-2 : js, ju1 = multi_d ? je+2 : je;
  int kl1 = three_d ? ks-2 : ks, ku1 = three_d ? ke+2 : ke;
  par_for("visc4_fill_vx1", DevExeSpace(), 0, nmb1, kl1, ku1, jl1, ju1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for(int var=0; var<4; var++){
      vx1(m, var, k, j, i) = HoFaceValue(w(m, var, k, j, i-2), w(m, var, k, j, i-1),
          w(m, var, k, j, i), w(m, var, k, j, i+1));
    }
  });

  if (multi_d) {
    // Extend ±2 in x1 (transverse) and x3 (transverse) for HoGradT access
    int kl2 = three_d ? ks-2 : ks, ku2 = three_d ? ke+2 : ke;
    par_for("visc4_fill_vx2", DevExeSpace(), 0, nmb1, kl2, ku2, js, je+1, is-2, ie+2,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for(int var=0; var<4; var++){
        vx2(m, var, k, j, i) = HoFaceValue(w(m, var, k, j-2, i), w(m, var, k, j-1, i),
          w(m, var, k, j, i), w(m, var, k, j+1, i));
      }
    });
  }

  if (three_d) {
    // Extend ±2 in x1 and x2 (both transverse) for HoGradT access
    par_for("visc4_fill_vx3", DevExeSpace(), 0, nmb1, ks, ke+1, js-2, je+2, is-2, ie+2,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for(int var=0; var<4; var++){
        vx3(m, var, k, j, i) = HoFaceValue(w(m, var, k-2, j, i), w(m, var, k-1, j, i),
          w(m, var, k, j, i), w(m, var, k+1, j, i));
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FourthOrderIsotropicViscousFlux
//  \brief Adds 4th-order viscous fluxes to face-centered fluxes of conserved variables.
//  Normal derivatives use HoNormDir* on w; transverse terms use HoGrad on vx1/2/3.

void Viscosity::FourthOrderIsotropicViscousFlux(const DvceArray5D<Real> &w,
  const Real nu_iso, const EOS_Data &eos, DvceFaceFld5D<Real> &flx) {
  FillHoViscFaceVelocity(w);
  const bool use_pt = mignone_;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto vx1 = ho_face_vel_.x1f;
  auto vx2 = ho_face_vel_.x2f;
  auto vx3 = ho_face_vel_.x3f;

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

    // Normal derivatives [4/3 dVx/dx, dVy/dx, dVz/dx] — point or average stencil
    par_for_inner(member, is, ie+1, [&](const int i) {
      Real idx1 = 1.0/size.d_view(m).dx1;
      fvx(i) = 4.0/3.0*HoGradDir1(use_pt, w, m, IVX, k, j, i, idx1);
      fvy(i) =         HoGradDir1(use_pt, w, m, IVY, k, j, i, idx1);
      fvz(i) =         HoGradDir1(use_pt, w, m, IVZ, k, j, i, idx1);
    });

    // Transverse: HoGradT on vx1 along y / z
    if (multi_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real idy2 = 1.0/size.d_view(m).dx2;
        Real dyVy = HoGradT(use_pt,
            vx1(m, IVY, k, j-2, i), vx1(m, IVY, k, j-1, i),
            vx1(m, IVY, k, j+1, i), vx1(m, IVY, k, j+2, i), idy2);
        Real dyVx = HoGradT(use_pt,
            vx1(m, IVX, k, j-2, i), vx1(m, IVX, k, j-1, i),
            vx1(m, IVX, k, j+1, i), vx1(m, IVX, k, j+2, i), idy2);
        fvx(i) -= (2.0/3.0)*dyVy;
        fvy(i) += dyVx;
      });
    }

    if (three_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real idz3 = 1.0/size.d_view(m).dx3;
        Real dzVz = HoGradT(use_pt,
            vx1(m, IVZ, k-2, j, i), vx1(m, IVZ, k-1, j, i),
            vx1(m, IVZ, k+1, j, i), vx1(m, IVZ, k+2, j, i), idz3);
        Real dzVx = HoGradT(use_pt,
            vx1(m, IVX, k-2, j, i), vx1(m, IVX, k-1, j, i),
            vx1(m, IVX, k+1, j, i), vx1(m, IVX, k+2, j, i), idz3);
        fvx(i) -= (2.0/3.0)*dzVz;
        fvz(i) += dzVx;
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie+1, [&](const int i) {
      // 4th-order face density and velocity (w0 or w0_c per caller)
      Real rho_f =  vx1(m, IDN, k, j, i);
      Real nud = nu_iso*rho_f;
      flx1(m,IVX,k,j,i) -= nud*fvx(i);
      flx1(m,IVY,k,j,i) -= nud*fvy(i);
      flx1(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        Real vx_f = vx1(m, IVX, k, j, i);
        Real vy_f = vx1(m, IVY, k, j, i);
        Real vz_f = vx1(m, IVZ, k, j, i);
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

    // Normal derivatives [(dVx/dy+dVy/dx), 4/3 dVy/dy, dVz/dy]
    par_for_inner(member, is, ie, [&](const int i) {
      Real idy2 = 1.0/size.d_view(m).dx2;
      Real idx1 = 1.0/size.d_view(m).dx1;
      Real dVx = HoGradDir2(use_pt, w, m, IVX, k, j, i, idy2);
      Real dVy = HoGradDir2(use_pt, w, m, IVY, k, j, i, idy2);
      Real dVz = HoGradDir2(use_pt, w, m, IVZ, k, j, i, idy2);

      Real dxVy = HoGradT(use_pt,
          vx2(m, IVY, k, j, i-2), vx2(m, IVY, k, j, i-1),
          vx2(m, IVY, k, j, i+1), vx2(m, IVY, k, j, i+2), idx1);
      Real dxVx = HoGradT(use_pt,
          vx2(m, IVX, k, j, i-2), vx2(m, IVX, k, j, i-1),
          vx2(m, IVX, k, j, i+1), vx2(m, IVX, k, j, i+2), idx1);
      fvx(i) = dVx + dxVy;
      fvy(i) = 4.0/3.0*dVy - (2.0/3.0)*dxVx;
      fvz(i) = dVz;
    });

    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        Real idz3 = 1.0/size.d_view(m).dx3;
        Real dzVz = HoGradT(use_pt,
            vx2(m, IVZ, k-2, j, i), vx2(m, IVZ, k-1, j, i),
            vx2(m, IVZ, k+1, j, i), vx2(m, IVZ, k+2, j, i), idz3);
        Real dzVy = HoGradT(use_pt,
            vx2(m, IVY, k-2, j, i), vx2(m, IVY, k-1, j, i),
            vx2(m, IVY, k+1, j, i), vx2(m, IVY, k+2, j, i), idz3);
        fvy(i) -= (2.0/3.0)*dzVz;
        fvz(i) += dzVy;
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real rho_f = vx2(m, IDN, k, j, i);
      Real nud = nu_iso*rho_f;
      flx2(m,IVX,k,j,i) -= nud*fvx(i);
      flx2(m,IVY,k,j,i) -= nud*fvy(i);
      flx2(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        Real vx_f = vx2(m, IVX, k, j, i);
        Real vy_f = vx2(m, IVY, k, j, i);
        Real vz_f = vx2(m, IVZ, k, j, i);
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

    // Normal derivatives [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 4/3 dVz/dz]
    par_for_inner(member, is, ie, [&](const int i) {
      Real idz3 = 1.0/size.d_view(m).dx3;
      Real idx1 = 1.0/size.d_view(m).dx1;
      Real idy2 = 1.0/size.d_view(m).dx2;
      Real dVx = HoGradDir3(use_pt, w, m, IVX, k, j, i, idz3);
      Real dVy = HoGradDir3(use_pt, w, m, IVY, k, j, i, idz3);
      Real dVz = HoGradDir3(use_pt, w, m, IVZ, k, j, i, idz3);

      Real dxVz = HoGradT(use_pt,
          vx3(m, IVZ, k, j, i-2), vx3(m, IVZ, k, j, i-1),
          vx3(m, IVZ, k, j, i+1), vx3(m, IVZ, k, j, i+2), idx1);
      Real dxVx = HoGradT(use_pt,
          vx3(m, IVX, k, j, i-2), vx3(m, IVX, k, j, i-1),
          vx3(m, IVX, k, j, i+1), vx3(m, IVX, k, j, i+2), idx1);
      Real dyVz = HoGradT(use_pt,
          vx3(m, IVZ, k, j-2, i), vx3(m, IVZ, k, j-1, i),
          vx3(m, IVZ, k, j+1, i), vx3(m, IVZ, k, j+2, i), idy2);
      Real dyVy = HoGradT(use_pt,
          vx3(m, IVY, k, j-2, i), vx3(m, IVY, k, j-1, i),
          vx3(m, IVY, k, j+1, i), vx3(m, IVY, k, j+2, i), idy2);

      fvx(i) = dVx + dxVz;
      fvy(i) = dVy + dyVz;
      fvz(i) = 4.0/3.0*dVz - (2.0/3.0)*dxVx - (2.0/3.0)*dyVy;
    });

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real rho_f = vx3(m, IDN, k, j, i);
      Real nud = nu_iso*rho_f;
      flx3(m,IVX,k,j,i) -= nud*fvx(i);
      flx3(m,IVY,k,j,i) -= nud*fvy(i);
      flx3(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        Real vx_f = vx3(m, IVX, k, j, i);
        Real vy_f = vx3(m, IVY, k, j, i);
        Real vz_f = vx3(m, IVZ, k, j, i);
        flx3(m,IEN,k,j,i) -= nud*(vx_f*fvx(i) + vy_f*fvy(i) + vz_f*fvz(i));
      }
    });
  });

  return;
}
