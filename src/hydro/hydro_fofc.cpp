//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fofc.cpp
//! \brief Implements functions for first-order flux correction (FOFC) algorithm.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/rsolvers/llf_hyd_singlestate.hpp"
#include "hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn WBFaceLLF<ivx>()
//! \brief Well-balanced first-order LLF flux at face (m,k,j,i) in direction ivx:
//! donor-cell perturbation states on both sides plus the face equilibrium, minus the
//! LLF flux of the
//! face equilibrium itself (evaluated in registers in the same kernel, so it cancels
//! exactly at U'=0).  Vector components are permuted so vx is the face-normal one.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void WBFaceLLF(const EOS_Data &eos, const int m, const int k, const int j, const int i,
               const DvceArray5D<Real> &wp, const DvceArray5D<Real> &weqf,
               const DvceArray5D<Real> &flx) {
  constexpr int di = (ivx == IVX) ? 1 : 0;
  constexpr int dj = (ivx == IVY) ? 1 : 0;
  constexpr int dk = (ivx == IVZ) ? 1 : 0;
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;
  HydPrim1D we, wl, wr;
  we.d  = weqf(m,IDN,k,j,i);
  we.vx = weqf(m,ivx,k,j,i);
  we.vy = weqf(m,ivy,k,j,i);
  we.vz = weqf(m,ivz,k,j,i);
  wl.d  = wp(m,IDN,k-dk,j-dj,i-di) + we.d;
  wl.vx = wp(m,ivx,k-dk,j-dj,i-di) + we.vx;
  wl.vy = wp(m,ivy,k-dk,j-dj,i-di) + we.vy;
  wl.vz = wp(m,ivz,k-dk,j-dj,i-di) + we.vz;
  wr.d  = wp(m,IDN,k,j,i) + we.d;
  wr.vx = wp(m,ivx,k,j,i) + we.vx;
  wr.vy = wp(m,ivy,k,j,i) + we.vy;
  wr.vz = wp(m,ivz,k,j,i) + we.vz;
  if (eos.is_ideal) {
    we.e = weqf(m,IEN,k,j,i);
    wl.e = wp(m,IEN,k-dk,j-dj,i-di) + we.e;
    wr.e = wp(m,IEN,k,j,i) + we.e;
  }
  HydCons1D f, feq;
  SingleStateLLF_Hyd(wl, wr, eos, f);
  SingleStateLLF_Hyd(we, we, eos, feq);
  flx(m,IDN,k,j,i) = f.d  - feq.d;
  flx(m,ivx,k,j,i) = f.mx - feq.mx;
  flx(m,ivy,k,j,i) = f.my - feq.my;
  flx(m,ivz,k,j,i) = f.mz - feq.mz;
  if (eos.is_ideal) {flx(m,IEN,k,j,i) = f.e - feq.e;}
}
//----------------------------------------------------------------------------------------
//! \fn void Hydro::FOFC
//! \brief Implements first-order flux-correction (FOFC) algorithm for Hydro.  First an
//! estimate of the updated conserved variables is made. This estimate is then used to
//! flag any cell where floors will be required during the conversion to primitives. Then
//! the fluxes on the faces of flagged cells are replaced with first-order LLF fluxes.
//! Often this is enough to prevent floors from being needed. The FOFC infrastructure is
//! also exploited for BH excision. If a cell is about the horizon, FOFC is automatically
//! triggered (without estimating updated conserved variables).

void Hydro::FOFC(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;

  // Well-balanced scheme: candidate and fluxes are perturbations; PAD on the full state,
  // first-order replacement fluxes through the face equilibrium (WBFaceLLF).  Scalar
  // fluxes are left as computed by the base scheme, as in the default path.
  if (well_balanced) {
    Real &gam0 = pdriver->gam0[stage-1];
    Real &gam1 = pdriver->gam1[stage-1];
    Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
    int &nhyd_ = nhydro;
    auto &u0_ = u0;
    auto &u1_ = u1;
    auto &utest_ = utest;
    int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-1, ju = je+1; }
    if (three_d) { kl = ks-1, ku = ke+1; }
    par_for("FOFC-newu-wb", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size.d_view(m).dx1;
      Real dtodx2 = beta_dt/size.d_view(m).dx2;
      Real dtodx3 = beta_dt/size.d_view(m).dx3;
      for (int n=0; n<nhyd_; ++n) {
        Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
        if (multi_d) divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
        if (three_d) divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
      }
    });
    peos->ConsToPrimWB(utest_, u_eq, w0, true, il, iu, jl, ju, kl, ku);

    auto &eos = peos->eos_data;
    auto &fofc_ = fofc;
    auto &wp_ = w_pert;
    auto weqf1 = w_eq_face.x1f, weqf2 = w_eq_face.x2f, weqf3 = w_eq_face.x3f;
    par_for("FOFC-flx-wb", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      if (fofc_(m,k,j,i)) {
        WBFaceLLF<IVX>(eos, m, k, j, i,   wp_, weqf1, flx1);
        WBFaceLLF<IVX>(eos, m, k, j, i+1, wp_, weqf1, flx1);
        if (multi_d) {
          WBFaceLLF<IVY>(eos, m, k, j,   i, wp_, weqf2, flx2);
          WBFaceLLF<IVY>(eos, m, k, j+1, i, wp_, weqf2, flx2);
        }
        if (three_d) {
          WBFaceLLF<IVZ>(eos, m, k,   j, i, wp_, weqf3, flx3);
          WBFaceLLF<IVZ>(eos, m, k+1, j, i, wp_, weqf3, flx3);
        }
      }
    });
    return;
  }

  if (use_fofc) {
    Real &gam0 = pdriver->gam0[stage-1];
    Real &gam1 = pdriver->gam1[stage-1];
    Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

    int &nhyd_ = nhydro;
    auto &u0_ = u0;
    auto &u1_ = u1;
    auto &utest_ = utest;

    // Index bounds
    int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-1, ju = je+1; }
    if (three_d) { kl = ks-1, ku = ke+1; }

    // Estimate updated conserved variables and cell-centered fields
    par_for("FOFC-newu", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtodx1 = beta_dt/size.d_view(m).dx1;
      Real dtodx2 = beta_dt/size.d_view(m).dx2;
      Real dtodx3 = beta_dt/size.d_view(m).dx3;

      // Estimate conserved variables
      for (int n=0; n<nhyd_; ++n) {
        Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
        if (multi_d) {
          divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
        }
        if (three_d) {
          divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
        }
        utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
      }
    });

    // Test whether conversion to primitives requires floors
    // Note b0 and w0 passed to function, but not used/changed.
    peos->ConsToPrim(utest_, w0, true, il, iu, jl, ju, kl, ku);
  }

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = peos->eos_data;
  auto &use_fofc_ = use_fofc;
  auto &fofc_ = fofc;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &w0_ = w0;

  // Index bounds
  int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-1, ju = je+1; }
  if (three_d) { kl = ks-1, ku = ke+1; }

  // Now replace fluxes with first-order LLF fluxes for any cell where floors needed (if
  // using FOFC) and/or for any cell about the excision (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise) { fofc_excision = excision_flux_(m,k,j,i); }
    }

    // Apply FOFC
    if (fofc_flag || fofc_excision) {
      // replace x1-flux at i
      // load left state
      HydPrim1D wim1;
      wim1.d  = w0_(m,IDN,k,j,i-1);
      wim1.vx = w0_(m,IVX,k,j,i-1);
      wim1.vy = w0_(m,IVY,k,j,i-1);
      wim1.vz = w0_(m,IVZ,k,j,i-1);
      if (eos.is_ideal) {wim1.e  = w0_(m,IEN,k,j,i-1);}

      // load right state
      HydPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.is_ideal) {wi.e = w0_(m,IEN,k,j,i);}

      // compute new 1st-order LLF flux
      HydCons1D flux;
      if (is_gr) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = LeftEdgeX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
        SingleStateLLF_GRHyd(wim1, wi, x1v, x2v, x3v, IVX, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRHyd(wim1, wi, eos, flux);
      } else {
        SingleStateLLF_Hyd(wim1, wi, eos, flux);
      }

      // store 1st-order fluxes
      flx1(m,IDN,k,j,i) = flux.d;
      flx1(m,IM1,k,j,i) = flux.mx;
      flx1(m,IM2,k,j,i) = flux.my;
      flx1(m,IM3,k,j,i) = flux.mz;
      if (eos.is_ideal) {flx1(m,IEN,k,j,i) = flux.e;}

      // replace x1-flux at i+1
      // load right state (left state just wi from above)
      HydPrim1D wip1;
      wip1.d  = w0_(m,IDN,k,j,i+1);
      wip1.vx = w0_(m,IVX,k,j,i+1);
      wip1.vy = w0_(m,IVY,k,j,i+1);
      wip1.vz = w0_(m,IVZ,k,j,i+1);
      if (eos.is_ideal) {wip1.e = w0_(m,IEN,k,j,i+1);}

      // compute new 1st-order LLF flux
      if (is_gr) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = LeftEdgeX(i+1-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
        SingleStateLLF_GRHyd(wi, wip1, x1v, x2v, x3v, IVX, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRHyd(wi, wip1, eos, flux);
      } else {
        SingleStateLLF_Hyd(wi, wip1, eos, flux);
      }

      // store 1st-order fluxes
      flx1(m,IDN,k,j,i+1) = flux.d;
      flx1(m,IM1,k,j,i+1) = flux.mx;
      flx1(m,IM2,k,j,i+1) = flux.my;
      flx1(m,IM3,k,j,i+1) = flux.mz;
      if (eos.is_ideal) {flx1(m,IEN,k,j,i+1) = flux.e;}

      if (multi_d) {
        // replace x2-flux at j
        // load left state, permutting components of vectors
        HydPrim1D wjm1;
        wjm1.d  = w0_(m,IDN,k,j-1,i);
        wjm1.vx = w0_(m,IVY,k,j-1,i);
        wjm1.vy = w0_(m,IVZ,k,j-1,i);
        wjm1.vz = w0_(m,IVX,k,j-1,i);
        if (eos.is_ideal) {wjm1.e = w0_(m,IEN,k,j-1,i);}

        // load right state, permutting components of vectors
        HydPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.is_ideal) {wj.e = w0_(m,IEN,k,j,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wjm1, wj, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wjm1, wj, eos, flux);
        } else {
          SingleStateLLF_Hyd(wjm1, wj, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx2(m,IDN,k,j,i) = flux.d;
        flx2(m,IM2,k,j,i) = flux.mx;
        flx2(m,IM3,k,j,i) = flux.my;
        flx2(m,IM1,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j,i) = flux.e;}

        // replace x2-flux at j+1
        // load left state, permutting components of vectors (just wj from above)
        // load right state, permutting components of vectors
        HydPrim1D wjp1;
        wjp1.d  = w0_(m,IDN,k,j+1,i);
        wjp1.vx = w0_(m,IVY,k,j+1,i);
        wjp1.vy = w0_(m,IVZ,k,j+1,i);
        wjp1.vz = w0_(m,IVX,k,j+1,i);
        if (eos.is_ideal) {wjp1.e = w0_(m,IEN,k,j+1,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j+1-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wj, wjp1, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wj, wjp1, eos, flux);
        } else {
          SingleStateLLF_Hyd(wj, wjp1, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx2(m,IDN,k,j+1,i) = flux.d;
        flx2(m,IM2,k,j+1,i) = flux.mx;
        flx2(m,IM3,k,j+1,i) = flux.my;
        flx2(m,IM1,k,j+1,i) = flux.mz;
        if (eos.is_ideal) {flx2(m,IEN,k,j+1,i) = flux.e;}
      }

      if (three_d) {
        // replace x3-flux at k
        // load left state, permutting components of vectors
        HydPrim1D wkm1;
        wkm1.d  = w0_(m,IDN,k-1,j,i);
        wkm1.vx = w0_(m,IVZ,k-1,j,i);
        wkm1.vy = w0_(m,IVX,k-1,j,i);
        wkm1.vz = w0_(m,IVY,k-1,j,i);
        if (eos.is_ideal) {wkm1.e = w0_(m,IEN,k-1,j,i);}

        // load right state, permutting components of vectors
        HydPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.is_ideal) {wk.e = w0_(m,IEN,k,j,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wkm1, wk, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wkm1, wk, eos, flux);
        } else {
          SingleStateLLF_Hyd(wkm1, wk, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx3(m,IDN,k,j,i) = flux.d;
        flx3(m,IM3,k,j,i) = flux.mx;
        flx3(m,IM1,k,j,i) = flux.my;
        flx3(m,IM2,k,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k,j,i) = flux.e;}

        // replace x3-flux at k+1
        // load left state, permutting components of vectors (just wk from above)
        // load right state, permutting components of vectors
        HydPrim1D wkp1;
        wkp1.d  = w0_(m,IDN,k+1,j,i);
        wkp1.vx = w0_(m,IVZ,k+1,j,i);
        wkp1.vy = w0_(m,IVX,k+1,j,i);
        wkp1.vz = w0_(m,IVY,k+1,j,i);
        if (eos.is_ideal) {wkp1.e = w0_(m,IEN,k+1,j,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wk, wkp1, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wk, wkp1, eos, flux);
        } else {
          SingleStateLLF_Hyd(wk, wkp1, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx3(m,IDN,k+1,j,i) = flux.d;
        flx3(m,IM3,k+1,j,i) = flux.mx;
        flx3(m,IM1,k+1,j,i) = flux.my;
        flx3(m,IM2,k+1,j,i) = flux.mz;
        if (eos.is_ideal) {flx3(m,IEN,k+1,j,i) = flux.e;}
      }

      // reset FOFC flag (do not reset excision flag)
      if (use_fofc_ && fofc_flag) { fofc_(m,k,j,i) = false; }
    }
  });

  return;
}

} // namespace hydro
