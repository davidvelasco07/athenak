//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid.cpp
//! \brief implementation of the functions commonly used in Multigrid

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>    // memset, memcpy
#include <iostream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>    // setprecision
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../coordinates/cell_locations.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/nghbr_index.hpp"
#include "../parameter_input.hpp"
#include "multigrid.hpp"

//namespace multigrid{ // NOLINT (build/namespace)
//----------------------------------------------------------------------------------------
//! \fn Multigrid::Multigrid(MultigridDriver *pmd, MeshBlock *pmb, int nghost)
//  \brief Multigrid constructor

Multigrid::Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host):
  pmy_driver_(pmd), pmy_pack_(pmbp), pmy_mesh_(pmd->pmy_mesh_), ngh_(nghost),
  nvar_(pmd->nvar_), defscale_(1.0), on_host_(on_host)  {
  if(pmy_pack_ != nullptr) {
    //Meshblock levels
    indcs_ = pmy_mesh_->mb_indcs;
    nmmb_  = pmy_pack_->nmb_thispack;
    nmmbx1_ = pmy_mesh_->nmb_rootx1;
    nmmbx2_ = pmy_mesh_->nmb_rootx2;
    nmmbx3_ = pmy_mesh_->nmb_rootx3;
    if (global_variable::my_rank == 0) {
      std::cout<< "Number of MeshBlocks in the pack: " << nmmb_ << std::endl;
      std::cout<< "MeshBlock size: "
               << indcs_.nx1 << " x " << indcs_.nx2 << " x " << indcs_.nx3 << std::endl;
    }
    if (indcs_.nx1 != indcs_.nx2 || indcs_.nx1 != indcs_.nx3) {
      std::cout << "### FATAL ERROR in Multigrid::Multigrid" << std::endl
         << "The Multigrid solver requires logically cubic MeshBlock." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
     }
    
     // initialize loc/size from the first meshblock in the pack (needs to be addpated for AMR)
    loc_ = pmy_pack_->pmesh->lloc_eachmb[0];
    size_ = pmy_pack_->pmb->mb_size.h_view(0);
  } else {
    //Root levels
    indcs_.nx1 = pmy_mesh_->nmb_rootx1;
    indcs_.nx2 = pmy_mesh_->nmb_rootx2;
    indcs_.nx3 = pmy_mesh_->nmb_rootx3;
    size_ = pmy_mesh_->mesh_size;
    nmmbx1_ = 1;
    nmmbx2_ = 1;
    nmmbx3_ = 1;
    // Root grid should be a single meshblock
    nmmb_ = 1;
    loc_  = pmy_mesh_->lloc_eachmb[0];
  }

  rdx_ = (size_.x1max-size_.x1min)/static_cast<Real>(indcs_.nx1);
  rdy_ = (size_.x2max-size_.x2min)/static_cast<Real>(indcs_.nx2);
  rdz_ = (size_.x3max-size_.x3min)/static_cast<Real>(indcs_.nx3);

  block_rdx_ = DualArray1D<Real>("block_rdx", nmmb_);
  Kokkos::realloc(fc_childx_, nmmb_);
  Kokkos::realloc(fc_childy_, nmmb_);
  Kokkos::realloc(fc_childz_, nmmb_);
  {
    auto brdx_h = block_rdx_.h_view;
    if (pmy_pack_ != nullptr) {
      auto &mb_size = pmy_pack_->pmb->mb_size;
      Real rnx1 = static_cast<Real>(indcs_.nx1);
      for (int m = 0; m < nmmb_; ++m) {
        brdx_h(m) = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min) / rnx1;
      }
    } else {
      brdx_h(0) = rdx_;
    }
    Kokkos::deep_copy(block_rdx_.d_view, brdx_h);
  }

  nlevel_ = 0;
  if (pmy_pack_ == nullptr) { 
    // Root grid levels
    int nbx = 0, nby = 0, nbz = 0;
    for (int l = 0; l < 20; l++) {
      if (indcs_.nx1%(1<<l) == 0 && indcs_.nx2%(1<<l) == 0 && indcs_.nx3%(1<<l) == 0) {
        nbx = indcs_.nx1/(1<<l), nby = indcs_.nx2/(1<<l), nbz = indcs_.nx3/(1<<l);
        nlevel_ = l+1;
      }
    }
    int nmaxr = std::max(nbx, std::max(nby, nbz));
    if (global_variable::my_rank == 0) {
      std::cout<< "Multigrid root grid levels: " << nlevel_ << std::endl;
    }
    // int nminr=std::min(nbx, std::min(nby, nbz)); // unused variable
    if (nmaxr != 1 && global_variable::my_rank == 0) {
      std::cout
          << "### Warning in Multigrid::Multigrid" << std::endl
          << "The root grid can not be reduced to a single cell." << std::endl
          << "Multigrid should still work, but this is not the"
          << " most efficient configuration"
          << " as the coarsest level is not solved exactly but iteratively." << std::endl;
    }
    if (nbx*nby*nbz>100 && global_variable::my_rank==0) {
      std::cout << "### Warning in Multigrid::Multigrid" << std::endl
                << "The degrees of freedom on the coarsest level is very large: "
                << nbx << " x " << nby << " x " << nbz << " = " << nbx*nby*nbz<< std::endl
                << "Multigrid should still work, but this is not efficient configuration "
                << "as the coarsest level solver costs considerably." << std::endl
                << "We recommend to reconsider grid configuration." << std::endl;
    }
  } else {
    // MeshBlock levels
    for (int l = 0; l < 20; l++) {
      if ((1<<l) == indcs_.nx1) {
        nlevel_=l+1;
        break;
      }
    }
    if (nlevel_ == 0) {
      std::cout << "### FATAL ERROR in Multigrid::Multigrid" << std::endl
          << "The MeshBlock size must be power of two." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
    }
  }

  current_level_ = nlevel_-1;

  // allocate arrays
  u_ = new DualArray5D<Real>[nlevel_];
  src_ = new DualArray5D<Real>[nlevel_];
  def_ = new DualArray5D<Real>[nlevel_];
  coeff_ = new DualArray5D<Real>[nlevel_];
  matrix_ = new DualArray5D<Real>[nlevel_];
  uold_ = new DualArray5D<Real>[nlevel_];

  for (int l = 0; l < nlevel_; l++) {
    int ll=nlevel_-1-l;
    int ncx=(indcs_.nx1>>ll)+2*ngh_;
    int ncy=(indcs_.nx2>>ll)+2*ngh_;
    int ncz=(indcs_.nx3>>ll)+2*ngh_;
    Kokkos::realloc(u_[l]  , nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(src_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(def_[l], nmmb_, nvar_, ncz, ncy, ncx);

    if (!((pmy_pack_ != nullptr) && (l == nlevel_-1)))
      Kokkos::realloc(uold_[l], nmmb_, nvar_, ncz, ncy, ncx);

    ncx=(indcs_.nx1>>(ll+1))+2*ngh_;
    ncy=(indcs_.nx2>>(ll+1))+2*ngh_;
    ncz=(indcs_.nx3>>(ll+1))+2*ngh_;

  }

}


//----------------------------------------------------------------------------------------
//! \fn Multigrid::~Multigrid
//! \brief Multigrid destroctor

Multigrid::~Multigrid() {
  delete [] u_;
  delete [] src_;
  delete [] def_;
  delete [] uold_;
  delete [] coeff_;
  delete [] matrix_;
  delete [] coord_;
  delete [] ccoord_;
}



//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ReallocateForAMR()
//! \brief Reallocate MG arrays when the number of MeshBlocks changes (AMR)

void Multigrid::UpdateBlockDx() {
  if (pmy_pack_ == nullptr) return;
  auto &mb_size = pmy_pack_->pmb->mb_size;
  Real rnx1 = static_cast<Real>(indcs_.nx1);

  // Refresh reference block size and cell widths from the first MeshBlock
  size_ = mb_size.h_view(0);
  rdx_ = (size_.x1max - size_.x1min) / static_cast<Real>(indcs_.nx1);
  rdy_ = (size_.x2max - size_.x2min) / static_cast<Real>(indcs_.nx2);
  rdz_ = (size_.x3max - size_.x3min) / static_cast<Real>(indcs_.nx3);

  auto brdx_h = block_rdx_.h_view;
  for (int m = 0; m < nmmb_; ++m) {
    brdx_h(m) = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min) / rnx1;
  }
  Kokkos::deep_copy(block_rdx_.d_view, brdx_h);

  // Compute per-block child octant positions for FC ghost fills
  auto &mbgid = pmy_pack_->pmb->mb_gid;
  auto *lloc = pmy_mesh_->lloc_eachmb;
  int root_level = pmy_mesh_->root_level;
  auto cx_h = Kokkos::create_mirror_view(fc_childx_);
  auto cy_h = Kokkos::create_mirror_view(fc_childy_);
  auto cz_h = Kokkos::create_mirror_view(fc_childz_);
  for (int m = 0; m < nmmb_; ++m) {
    int gid = mbgid.h_view(m);
    LogicalLocation &loc = lloc[gid];
    cx_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx1 & 1) : 0;
    cy_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx2 & 1) : 0;
    cz_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx3 & 1) : 0;
  }
  Kokkos::deep_copy(fc_childx_, cx_h);
  Kokkos::deep_copy(fc_childy_, cy_h);
  Kokkos::deep_copy(fc_childz_, cz_h);
}

void Multigrid::ReallocateForAMR() {
  if (pmy_pack_ == nullptr) return;
  int new_nmmb = pmy_pack_->nmb_thispack;
  if (new_nmmb == nmmb_) return;
  nmmb_ = new_nmmb;

  Kokkos::realloc(block_rdx_, nmmb_);
  Kokkos::realloc(fc_childx_, nmmb_);
  Kokkos::realloc(fc_childy_, nmmb_);
  Kokkos::realloc(fc_childz_, nmmb_);
  UpdateBlockDx();

  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_ - 1 - l;
    int ncx = (indcs_.nx1 >> ll) + 2 * ngh_;
    int ncy = (indcs_.nx2 >> ll) + 2 * ngh_;
    int ncz = (indcs_.nx3 >> ll) + 2 * ngh_;
    Kokkos::realloc(u_[l],   nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(src_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(def_[l], nmmb_, nvar_, ncz, ncy, ncx);
    if (l != nlevel_ - 1)
      Kokkos::realloc(uold_[l], nmmb_, nvar_, ncz, ncy, ncx);
  }

}


//! \fn void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh)
//! \brief Fill the inital guess in the active zone of the finest level

void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh) {
  auto &dst = u_[nlevel_-1].d_view;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + indcs_.nx1 - 1; je = js + indcs_.nx2 - 1; ke = ks + indcs_.nx3 - 1;

  const int lns = ns;
  const int lks = ks, ljs = js, lis = is, lngh = ngh;

  par_for("Multigrid::LoadFinestData", DevExeSpace(),0, nmmb_-1,
          0, nvar_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int nsrc = lns + v;
    const int k = mk - lks + lngh;
    const int j = mj - ljs + lngh;
    const int i = mi - lis + lngh;
    dst(m, v, mk, mj, mi) = src(m, nsrc, k, j, i);
  });

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::LoadSource(const DvceArray5D<Real> &src, int ns, int ngh,
//!                                Real fac)
//! \brief Fill the source in the active zone of the finest level

void Multigrid::LoadSource(const DvceArray5D<Real> &src, int ns, int ngh, Real fac) {
  auto &dst = src_[nlevel_-1].d_view;
  int sngh = std::min(ngh_, ngh);
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_ - sngh;
  ie = is + indcs_.nx1 + 2*sngh - 1;
  je = js + indcs_.nx2 + 2*sngh - 1;
  ke = ks + indcs_.nx3 + 2*sngh - 1;

  // local copies for device lambda capture
  const Real lfac = fac;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int src_off = ngh - ngh_;

  par_for("Multigrid::LoadSource", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int nsrc = ns + v;
    const int k = mk + src_off;
    const int j = mj + src_off;
    const int i = mi + src_off;
    if (lfac == (Real)1.0) {
      dst(m, v, mk, mj, mi) = src(m, nsrc, k, j, i);
    } else {
      dst(m, v, mk, mj, mi) = src(m, nsrc, k, j, i) * lfac;
    }
  });

  current_level_ = nlevel_-1;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::LoadCoefficients(const DvceArray5D<Real> &coeff, int ngh)
//! \brief Load coefficients of the diffusion and source terms

void Multigrid::LoadCoefficients(const DvceArray5D<Real> &coeff, int ngh) {
  auto &cm = coeff_[nlevel_-1].d_view;
  int is, ie, js, je, ks, ke;
  is = js = ks = 0;
  ie = indcs_.nx1 + 2*ngh_ - 1; je = indcs_.nx2 + 2*ngh_ - 1; ke = indcs_.nx3 + 2*ngh_ - 1;

  // copy locals for device lambda capture
  const int coeff_off = ngh - ngh_;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = ncoeff_ - 1;

  auto cm_ = cm;
  auto coeff_ = coeff;

  par_for("Multigrid::LoadCoefficients", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int k = mk + coeff_off;
    const int j = mj + coeff_off;
    const int i = mi + coeff_off;
    cm_(m, v, mk, mj, mi) = coeff_(m, v, k, j, i);
  });

  return;
}



//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ApplyMask()
//  \brief Apply the user-defined source mask function on the finest level

void Multigrid::ApplyMask() {
  Real mask_r = pmy_driver_->mask_radius_;
  if (mask_r <= 0.0) return;

  int ngh = ngh_;
  int nx1 = indcs_.nx1, nx2 = indcs_.nx2, nx3 = indcs_.nx3;
  int nmb = nmmb_;
  Real mask_r2 = mask_r * mask_r;
  Real ox = pmy_driver_->mask_origin_[0];
  Real oy = pmy_driver_->mask_origin_[1];
  Real oz = pmy_driver_->mask_origin_[2];

  auto src = src_[nlevel_-1].d_view;
  auto &mb_size = pmy_pack_->pmb->mb_size;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;

  par_for("Multigrid::ApplyMask", DevExeSpace(), 0, nmb-1,
          ngh, ngh+nx3-1, ngh, ngh+nx2-1, ngh, ngh+nx1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x = CellCenterX(i - ngh, nx1, mb_size.d_view(m).x1min, mb_size.d_view(m).x1max);
    Real y = CellCenterX(j - ngh, nx2, mb_size.d_view(m).x2min, mb_size.d_view(m).x2max);
    Real z = CellCenterX(k - ngh, nx3, mb_size.d_view(m).x3min, mb_size.d_view(m).x3max);
    Real r2 = (x-ox)*(x-ox) + (y-oy)*(y-oy) + (z-oz)*(z-oz);
    if (r2 > mask_r2) {
      src(m, 0, k, j, i) = 0.0;
    }
  });
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictCoefficients()
//! \brief restrict coefficients within a Multigrid object

void Multigrid::RestrictCoefficients() {
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  if (on_host_) {
    for (int lev = nlevel_ - 1; lev > 0; lev--) {
      int ll = nlevel_ - lev;
      ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
      Restrict(coeff_[lev-1].h_view, coeff_[lev].h_view, ncoeff_,
               is, ie, js, je, ks, ke, false);
    }
  } else {
    for (int lev = nlevel_ - 1; lev > 0; lev--) {
      int ll = nlevel_ - lev;
      ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
      Restrict(coeff_[lev-1].d_view, coeff_[lev].d_view, ncoeff_,
               is, ie, js, je, ks, ke, false);
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh)
//! \brief Set the result, including the ghost zone

void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh) {
  const auto &src = u_[nlevel_-1].d_view;
  int sngh = std::min(ngh_,ngh);

  if (ns == 0 && ngh_ == ngh && nvar_ == 1
      && src.extent(0) == dst.extent(0)
      && src.extent(2) == dst.extent(2)
      && src.extent(3) == dst.extent(3)
      && src.extent(4) == dst.extent(4)) {
    Kokkos::deep_copy(dst, src);
  } else {
    int is, ie, js, je, ks, ke;
    is = js = ks = ngh_ - sngh;
    ie = indcs_.nx1 + ngh_ + sngh - 1;
    je = indcs_.nx2 + ngh_ + sngh - 1;
    ke = indcs_.nx3 + ngh_ + sngh - 1;

    const int dst_off = ngh - ngh_;

    par_for("Multigrid::RetrieveResult", DevExeSpace(),
            0, nmmb_-1, 0, nvar_-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
      const int ndst = ns + v;
      const int k = mk + dst_off;
      const int j = mj + dst_off;
      const int i = mi + dst_off;
      dst(m, ndst, k, j, i) = src(m, v, mk, mj, mi);
    });
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RetrieveDefect(DvceArray5D<Real> &dst, int ns, int ngh)
//! \brief Set the defect, including the ghost zone

void Multigrid::RetrieveDefect(DvceArray5D<Real> &dst, int ns, int ngh) {
  const auto &src = def_[nlevel_-1].d_view;
  int sngh = std::min(ngh_,ngh);
  int ie = indcs_.nx1 + ngh_ + sngh - 1;
  int je = indcs_.nx2 + ngh_ + sngh - 1;
  int ke = indcs_.nx3 + ngh_ + sngh - 1;

  // local copies for device lambda capture
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int mk0 = ngh_ - sngh, mk1 = ke;
  const int mj0 = ngh_ - sngh, mj1 = je;
  const int mi0 = ngh_ - sngh, mi1 = ie;
  const Real scale = defscale_;
  const int dst_off = ngh - ngh_;

  auto dst_ = dst;
  auto src_ = src;

  par_for("Multigrid::RetrieveDefect", DevExeSpace(),
          m0, m1, v0, v1, mk0, mk1, mj0, mj1, mi0, mi1,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int ndst = ns + v;
    const int k = mk + dst_off;
    const int j = mj + dst_off;
    const int i = mi + dst_off;
    dst_(m, ndst, k, j, i) = src_(m, v, mk, mj, mi) * scale;
  });

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ZeroClearData()
//! \brief Clear the data array with zero

void Multigrid::ZeroClearData() {
  if (on_host_) {
    Kokkos::deep_copy(u_[current_level_].h_view, 0.0);
  } else {
    Kokkos::deep_copy(u_[current_level_].d_view, 0.0);
  }
}

void Multigrid::CopySourceToData() {
  if (on_host_) {
    Kokkos::deep_copy(u_[nlevel_-1].h_view, src_[nlevel_-1].h_view);
  } else {
    Kokkos::deep_copy(u_[nlevel_-1].d_view, src_[nlevel_-1].d_view);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictPack()
//! \brief Restrict the defect to the source

void Multigrid::RestrictPack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  CalculateDefectPack();
  is=js=ks= ngh_;
  ie = is + (indcs_.nx1>>ll) - 1;
  je = js + (indcs_.nx2>>ll) - 1;
  ke = ks + (indcs_.nx3>>ll) - 1;
  if (on_host_) {
    Restrict(src_[current_level_-1].h_view, def_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
    Restrict(u_[current_level_-1].h_view, u_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
  } else {
    Restrict(src_[current_level_-1].d_view, def_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
    Restrict(u_[current_level_-1].d_view, u_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
  }
  current_level_--;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictSourcePack()
//! \brief Restrict the source (and solution) without forming defect

void Multigrid::RestrictSourcePack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks= ngh_;
  ie = is+(indcs_.nx1>>ll) - 1;
  je = js+(indcs_.nx2>>ll) - 1;
  ke = ks+(indcs_.nx3>>ll) - 1;
  if (on_host_) {
    Restrict(src_[current_level_-1].h_view, src_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
  } else {
    Restrict(src_[current_level_-1].d_view, src_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
  }
  current_level_--;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrectPack()
//! \brief Prolongate the potential using tri-linear interpolation

void Multigrid::ProlongateAndCorrectPack() {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1;
  je=js+(indcs_.nx2>>ll)-1;
  ke=ks+(indcs_.nx3>>ll)-1;

  ComputeCorrection();

  if (on_host_) {
    ProlongateAndCorrect(u_[current_level_+1].h_view, u_[current_level_].h_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  } else {
    ProlongateAndCorrect(u_[current_level_+1].d_view, u_[current_level_].d_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  }

  current_level_++;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongatePack()
//! \brief Prolongate the solution for FMG (direct overwrite, always tricubic)

void Multigrid::FMGProlongatePack() {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1;
  je=js+(indcs_.nx2>>ll)-1;
  ke=ks+(indcs_.nx3>>ll)-1;

  if (on_host_) {
    FMGProlongate(u_[current_level_+1].h_view, u_[current_level_].h_view,
                  is, ie, js, je, ks, ke, ngh_, ngh_, ngh_);
  } else {
    FMGProlongate(u_[current_level_+1].d_view, u_[current_level_].d_view,
                  is, ie, js, je, ks, ke, ngh_, ngh_, ngh_);
  }

  current_level_++;
}


//----------------------------------------------------------------------------------------
//! \fn  void Multigrid::SmoothPack(int color)
//! \brief Apply Smoother on the Pack


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::SetFromRootGrid(bool folddata)
//! \brief Load the data from the root grid or octets (Athena++ style per-cell octets)

void Multigrid::SetFromRootGrid(bool folddata) {
  current_level_ = 0;
  auto dst_h = u_[current_level_].h_view;
  auto odst_h = uold_[current_level_].h_view;

  auto src_h = pmy_driver_->GetRootData_h();
  auto osrc_h = pmy_driver_->GetRootOldData_h();
  int padding = pmy_mesh_->gids_eachrank[global_variable::my_rank];

  for (int m = 0; m < nmmb_; ++m) {
    auto loc = pmy_mesh_->lloc_eachmb[m + padding];
    int lev = loc.level - pmy_driver_->locrootlevel_;
    if (lev == 0) {
      int ci = static_cast<int>(loc.lx1);
      int cj = static_cast<int>(loc.lx2);
      int ck = static_cast<int>(loc.lx3);
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k <= 2*ngh_; ++k) {
          for (int j = 0; j <= 2*ngh_; ++j) {
            for (int i = 0; i <= 2*ngh_; ++i) {
              dst_h(m, v, k, j, i) = src_h(0, v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = osrc_h(0, v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    } else {
      LogicalLocation oloc;
      oloc.lx1 = (loc.lx1 >> 1);
      oloc.lx2 = (loc.lx2 >> 1);
      oloc.lx3 = (loc.lx3 >> 1);
      oloc.level = loc.level - 1;
      int olev = oloc.level - pmy_driver_->locrootlevel_;
      int oid = pmy_driver_->octetmap_[olev][oloc];
      int ci = (static_cast<int>(loc.lx1) & 1);
      int cj = (static_cast<int>(loc.lx2) & 1);
      int ck = (static_cast<int>(loc.lx3) & 1);
      const MGOctet &oct = pmy_driver_->octets_[olev][oid];
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k <= 2*ngh_; ++k) {
          for (int j = 0; j <= 2*ngh_; ++j) {
            for (int i = 0; i <= 2*ngh_; ++i) {
              dst_h(m, v, k, j, i) = oct.U(v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = oct.Uold(v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    }
  }
  u_[current_level_].template modify<HostExeSpace>();
  u_[current_level_].template sync<DevExeSpace>();
  if (folddata) {
    uold_[current_level_].template modify<HostExeSpace>();
    uold_[current_level_].template sync<DevExeSpace>();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n)
//! \brief calculate the residual norm

Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n) {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
  Real dx = rdx_ * static_cast<Real>(1 << ll);
  Real dy = rdy_ * static_cast<Real>(1 << ll);
  Real dz = rdz_ * static_cast<Real>(1 << ll);
  Real dV = dx * dy * dz;
  CalculateDefectPack();

  Real norm = 0.0;

  if (on_host_) {
    auto &def = def_[current_level_].h_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::DefectNorm_Linf",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_max) {
          local_max = std::max(local_max, std::abs(def(m, v, k, j, i)));
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::DefectNorm_L1",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          local_sum += std::abs(def(m, v, k, j, i));
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real val = def(m, v, k, j, i);
          local_sum += val * val;
        }, Kokkos::Sum<Real>(norm));
    }
  } else {
    auto &def = def_[current_level_].d_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::DefectNorm_Linf",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_max) {
          local_max = std::max(local_max, std::abs(def(m, v, k, j, i)));
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::DefectNorm_L1",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          local_sum += std::abs(def(m, v, k, j, i));
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real val = def(m, v, k, j, i);
          local_sum += val * val;
        }, Kokkos::Sum<Real>(norm));
    }
  }
  norm *= defscale_;
  return norm;
}

//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateAverage(MGVariable type)
//! \brief Calculate volume-weighted average of variable 0 on current level

Real Multigrid::CalculateAverage(MGVariable type) {
  int ll = nlevel_ - 1 - current_level_;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + (indcs_.nx1 >> ll) - 1;
  je = js + (indcs_.nx2 >> ll) - 1;
  ke = ks + (indcs_.nx3 >> ll) - 1;
  int ll_l = ll;

  Real sum = 0.0;
  if (on_host_) {
    auto data = (type == MGVariable::src) ? src_[current_level_].h_view
                                          : u_[current_level_].h_view;
    auto brdx = block_rdx_.h_view;
    Kokkos::parallel_reduce("MG::Average",
      Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                            {nmmb_, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &local_sum) {
        Real dx_m = brdx(m) * static_cast<Real>(1 << ll_l);
        Real dV_m = dx_m * dx_m * dx_m;
        local_sum += data(m, 0, k, j, i) * dV_m;
      }, Kokkos::Sum<Real>(sum));
  } else {
    auto data = (type == MGVariable::src) ? src_[current_level_].d_view
                                          : u_[current_level_].d_view;
    auto brdx = block_rdx_.d_view;
    Kokkos::parallel_reduce("MG::Average",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                           {nmmb_, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &local_sum) {
        Real dx_m = brdx(m) * static_cast<Real>(1 << ll_l);
        Real dV_m = dx_m * dx_m * dx_m;
        local_sum += data(m, 0, k, j, i) * dV_m;
      }, Kokkos::Sum<Real>(sum));
  }

  Real volume = 0.0;
  {
    Real nx = static_cast<Real>(indcs_.nx1);
    for (int m = 0; m < nmmb_; ++m) {
      Real len = block_rdx_.h_view(m) * nx;
      volume += len * len * len;
    }
  }

  #if MPI_PARALLEL_ENABLED
  Real global_sum = 0.0;
  Real global_volume = 0.0;
  MPI_Allreduce(&sum, &global_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&volume, &global_volume, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sum = global_sum;
  volume = global_volume;
  #endif

  return (volume > 0.0) ? (sum / volume) : 0.0;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateTotal(MGVariable type, int n)
//! \brief calculate the sum of the array (type: 0=src, 1=u)

Real Multigrid::CalculateTotal(MGVariable type, int n) {
  //DvceArray5D<Real> &src =
  //                  (type == MGVariable::src) ? src_[current_level_] : u_[current_level_];
  //int ll = nlevel_ - 1 - current_level_;
  //Real s=0.0;
  //int is, ie, js, je, ks, ke;
  //is=js=ks=ngh_;
  //ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
  //Real dx=rdx_*static_cast<Real>(1<<ll), dy=rdy_*static_cast<Real>(1<<ll),
  //     dz=rdz_*static_cast<Real>(1<<ll);
  //for (int k=ks; k<=ke; ++k) {
  //  for (int j=js; j<=je; ++j) {
  //    for (int i=is; i<=ie; ++i)
  //      s+=src(n,k,j,i);
  //  }
  //}
  //return s*dx*dy*dz;
  return 0.0;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::SubtractAverage(MGVariable type, int v, Real ave)
//! \brief subtract the average value (type: 0=src, 1=u)

void Multigrid::SubtractAverage(MGVariable type, int n, Real ave) {
  int is, ie, js, je, ks, ke;
  is = js = ks = 0;
  ie = is + indcs_.nx1 + 2*ngh_ - 1;
  je = js + indcs_.nx2 + 2*ngh_ - 1;
  ke = ks + indcs_.nx3 + 2*ngh_ - 1;

  const int m0 = 0, m1 = nmmb_ - 1;
  const int vn = n;
  const Real lave = ave;

  if (on_host_) {
    auto dst = (type == MGVariable::src) ? src_[nlevel_-1].h_view
                                         : u_[nlevel_-1].h_view;
    par_for("Multigrid::SubtractAverage", HostExeSpace(),
            m0, m1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
      dst(m, vn, mk, mj, mi) -= lave;
    });
  } else {
    auto dst = (type == MGVariable::src) ? src_[nlevel_-1].d_view
                                         : u_[nlevel_-1].d_view;
    par_for("Multigrid::SubtractAverage", DevExeSpace(),
            m0, m1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
      dst(m, vn, mk, mj, mi) -= lave;
    });
  }
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::StoreOldData()
//! \brief store the old u data in the uold array

void Multigrid::StoreOldData() {
  if (on_host_) {
    Kokkos::deep_copy(HostExeSpace(), uold_[current_level_].h_view,
                      u_[current_level_].h_view);
  } else {
    Kokkos::deep_copy(DevExeSpace(), uold_[current_level_].d_view,
                      u_[current_level_].d_view);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::Restrict(...)
//  \brief Actual implementation of restriction (templated on view type)

template <typename ViewType>
void Multigrid::Restrict(ViewType &dst, const ViewType &src,
                int nvar, int i0, int i1, int j0, int j1, int k0, int k1, bool th) {

  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar - 1;
  const int ngh = ngh_;
                
  par_for("Multigrid::Restrict", ExeSpace(),
          m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int fk = 2*k - ngh;
    const int fj = 2*j - ngh;
    const int fi = 2*i - ngh;
    dst(m, v, k, j, i) = 0.125 * (
        src(m, v, fk,   fj,   fi)   + src(m, v, fk,   fj,   fi+1)
      + src(m, v, fk,   fj+1, fi)   + src(m, v, fk,   fj+1, fi+1)
      + src(m, v, fk+1, fj,   fi)   + src(m, v, fk+1, fj,   fi+1)
      + src(m, v, fk+1, fj+1, fi)   + src(m, v, fk+1, fj+1, fi+1));
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ComputeCorrection(DvceArray5D<Real> &correction, int level)
//! \brief Compute the correction as u_[level] - uold_[level]

void Multigrid::ComputeCorrection() {
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  int ll = nlevel_ - 1 - current_level_;
  int is = 0, ie = is + (indcs_.nx1 >> ll) + 2*ngh_ -1;
  int js = 0, je = js + (indcs_.nx2 >> ll) + 2*ngh_ -1;
  int ks = 0, ke = ks + (indcs_.nx3 >> ll) + 2*ngh_ -1;

  if (on_host_) {
    auto u = u_[current_level_].h_view;
    auto uold = uold_[current_level_].h_view;
    par_for("Multigrid::ComputeCorrection", HostExeSpace(),
            m0, m1, v0, v1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) -= uold(m, v, k, j, i);
    });
  } else {
    auto u = u_[current_level_].d_view;
    auto uold = uold_[current_level_].d_view;
    par_for("Multigrid::ComputeCorrection", DevExeSpace(),
            m0, m1, v0, v1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) -= uold(m, v, k, j, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrect(...)
//! \brief Actual implementation of prolongation and correction (templated on view type)

template <typename ViewType>
void Multigrid::ProlongateAndCorrect(ViewType &dst, const ViewType &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th) {

  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  const int ll = pmy_driver_->fprolongation_; // copy host flag for capture

  auto dst_ = dst;
  auto src_ = src;

  if (ll == 1) { // tricubic
    par_for("Multigrid::ProlongateAndCorrect_tricubic", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      // For brevity: local references to src entries
      // compute and add to 8 target cells as in original implementation
      dst_(m,v,fk  ,fj  ,fi  ) += (
        + 125.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
        + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
        + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
        -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
        +  45.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )-  27.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;

      dst_(m,v,fk,  fj,  fi+1) += (
        -  75.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )+ 125.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
        +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
        + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        -  27.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;

      dst_(m,v,fk  ,fj+1,fi  ) += (
        -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
        + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        + 125.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        +  45.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )-  27.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
        -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;

      dst_(m,v,fk,  fj+1,fi+1) += (
        +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
        -  75.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )+ 125.*src_(m,v,k-1,j+1,i+1)
        + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
        -  27.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
        + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;

      dst_(m,v,fk+1,fj,  fi  ) += (
        -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
        +  45.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )-  27.*src_(m,v,k-1,j+1,i+1)
        + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
        + 125.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
        + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;

      dst_(m,v,fk+1,fj,  fi+1) += (
        +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
        + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        -  27.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        -  75.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )+ 125.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
        +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;

      dst_(m,v,fk+1,fj+1,fi  ) += (
        +  45.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )-  27.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
        -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
        + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        + 125.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;

      dst_(m,v,fk+1,fj+1,fi+1) += (
        -  27.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
        + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
        + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
        +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
        -  75.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )+ 125.*src_(m,v,k+1,j+1,i+1)
      ) / 32768.0;  
    });
  } else { // trilinear
    par_for("Multigrid::ProlongateAndCorrect_trilinear", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      dst_(m,v,fk  ,fj  ,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk  ,fj  ,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk+1,fj+1,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j+1,i+1)));
      dst_(m,v,fk+1,fj+1,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j+1,i+1)));
    });
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongate(...)
//! \brief FMG prolongation: direct overwrite (=) with tricubic interpolation.
//! Unlike ProlongateAndCorrect (+=), this overwrites the destination array.

template <typename ViewType>
void Multigrid::FMGProlongate(ViewType &dst, const ViewType &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl) {

  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  auto dst_ = dst;
  auto src_ = src;

  par_for("Multigrid::FMGProlongate", ExeSpace(),
          m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int fk = 2*(k-kl) + fkl;
    const int fj = 2*(j-jl) + fjl;
    const int fi = 2*(i-il) + fil;

    dst_(m,v,fk  ,fj  ,fi  ) = (
      + 125.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
      + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
      + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )-  27.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk,  fj,  fi+1) = (
      -  75.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )+ 125.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
      + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      -  27.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk  ,fj+1,fi  ) = (
      -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
      + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      + 125.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )-  27.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk,  fj+1,fi+1) = (
      +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )+ 125.*src_(m,v,k-1,j+1,i+1)
      + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
      -  27.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
      + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj,  fi  ) = (
      -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )-  27.*src_(m,v,k-1,j+1,i+1)
      + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
      + 125.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
      + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj,  fi+1) = (
      +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
      + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      -  27.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )+ 125.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj+1,fi  ) = (
      +  45.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )-  27.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
      + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      + 125.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj+1,fi+1) = (
      -  27.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
      + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
      + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )+ 125.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;
  });
  return;
}


//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::MultigridBoundaryValues()
//! \brief Constructor for multigrid boundary values object
//----------------------------------------------------------------------------------------

MultigridBoundaryValues::MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin, bool coarse, Multigrid *pmg) 
  :
   MeshBoundaryValuesCC(pmbp, pin, coarse), pmy_mg(pmg) {
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::RemapIndicesForMG()
//! \brief Remap isame/icoar/ifine indices from hydro coordinates (ng ghost cells) to MG
//! coordinates (ngh_ ghost cells). Must be called AFTER InitializeBuffers.

void MultigridBoundaryValues::RemapIndicesForMG() {
  int ng  = pmy_pack->pmesh->mb_indcs.ng;
  int ngh = pmy_mg->GetGhostCells();
  int nx1 = pmy_pack->pmesh->mb_indcs.nx1;
  int nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  int nx3 = pmy_pack->pmesh->mb_indcs.nx3;
  int nnghbr = pmy_pack->pmb->nnghbr;

  if (ng != ngh) {
    int is_h = ng, ie_h = ng + nx1 - 1;
    int js_h = ng, je_h = ng + nx2 - 1;
    int ks_h = ng, ke_h = ng + nx3 - 1;
    int is_m = ngh, ie_m = ngh + nx1 - 1;
    int js_m = ngh, je_m = ngh + nx2 - 1;
    int ks_m = ngh, ke_m = ngh + nx3 - 1;
    int ng1_m = ngh - 1;

    auto remap_send = [](int &lo, int &hi,
                         int s_h, int e_h, int s_m, int e_m, int ng1) {
      if (lo == s_h && hi == e_h) { lo = s_m; hi = e_m; }
      else if (lo > s_h)          { lo = e_m - ng1; hi = e_m; }
      else                        { lo = s_m; hi = s_m + ng1; }
    };
    auto remap_recv = [](int &lo, int &hi,
                         int s_h, int e_h, int s_m, int e_m, int ng_m) {
      if (lo >= s_h && hi <= e_h) { lo = s_m; hi = e_m; }
      else if (lo > e_h)          { lo = e_m + 1; hi = e_m + ng_m; }
      else                        { lo = s_m - ng_m; hi = s_m - 1; }
    };

    for (int n = 0; n < nnghbr; ++n) {
      auto &si = sendbuf[n].isame[0];
      remap_send(si.bis, si.bie, is_h, ie_h, is_m, ie_m, ng1_m);
      remap_send(si.bjs, si.bje, js_h, je_h, js_m, je_m, ng1_m);
      remap_send(si.bks, si.bke, ks_h, ke_h, ks_m, ke_m, ng1_m);
      sendbuf[n].isame_ndat = (si.bie-si.bis+1)*(si.bje-si.bjs+1)*(si.bke-si.bks+1);

      auto &ri = recvbuf[n].isame[0];
      remap_recv(ri.bis, ri.bie, is_h, ie_h, is_m, ie_m, ngh);
      remap_recv(ri.bjs, ri.bje, js_h, je_h, js_m, je_m, ngh);
      remap_recv(ri.bks, ri.bke, ks_h, ke_h, ks_m, ke_m, ngh);
      recvbuf[n].isame_ndat = (ri.bie-ri.bis+1)*(ri.bje-ri.bjs+1)*(ri.bke-ri.bks+1);
    }
  }

  // Recompute icoar/ifine indices from scratch using MG mesh parameters.
  // These are needed for inter-level (fine-coarse) boundary communication.
  int ng1_m = ngh - 1;
  int cnx1 = nx1 / 2, cnx2 = nx2 / 2, cnx3 = nx3 / 2;
  int is_m = ngh, ie_m = ngh + nx1 - 1;
  int js_m = ngh, je_m = ngh + nx2 - 1;
  int ks_m = ngh, ke_m = ngh + nx3 - 1;
  int cis_m = ngh, cie_m = ngh + cnx1 - 1;
  int cjs_m = ngh, cje_m = ngh + cnx2 - 1;
  int cks_m = ngh, cke_m = ngh + cnx3 - 1;

  // Recover (ox1,ox2,ox3,f1,f2) from the buffer index n and recompute icoar/ifine.
  // We iterate the same way InitializeBuffers does.
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  auto compute_send_icoar = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3) {
    auto &ic = buf.icoar[0];
    ic.bis = (ox1 > 0) ? (cie_m - ng1_m) : cis_m;
    ic.bie = (ox1 < 0) ? (cis_m + ng1_m) : cie_m;
    ic.bjs = (ox2 > 0) ? (cje_m - ng1_m) : cjs_m;
    ic.bje = (ox2 < 0) ? (cjs_m + ng1_m) : cje_m;
    ic.bks = (ox3 > 0) ? (cke_m - ng1_m) : cks_m;
    ic.bke = (ox3 < 0) ? (cks_m + ng1_m) : cke_m;
    buf.icoar_ndat = (ic.bie-ic.bis+1)*(ic.bje-ic.bjs+1)*(ic.bke-ic.bks+1);
  };

  auto compute_send_ifine = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3, int f1, int f2) {
    auto &ifn = buf.ifine[0];
    ifn.bis = (ox1 > 0) ? (ie_m - ng1_m) : is_m;
    ifn.bie = (ox1 < 0) ? (is_m + ng1_m) : ie_m;
    ifn.bjs = (ox2 > 0) ? (je_m - ng1_m) : js_m;
    ifn.bje = (ox2 < 0) ? (js_m + ng1_m) : je_m;
    ifn.bks = (ox3 > 0) ? (ke_m - ng1_m) : ks_m;
    ifn.bke = (ox3 < 0) ? (ks_m + ng1_m) : ke_m;
    if (ox1 == 0) {
      if (f1 == 1) { ifn.bis += cnx1 - ngh; }
      else         { ifn.bie -= cnx1 - ngh; }
    }
    if (ox2 == 0 && nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) { ifn.bjs += cnx2 - ngh; }
        else         { ifn.bje -= cnx2 - ngh; }
      } else {
        if (f2 == 1) { ifn.bjs += cnx2 - ngh; }
        else         { ifn.bje -= cnx2 - ngh; }
      }
    }
    if (ox3 == 0 && nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) { ifn.bks += cnx3 - ngh; }
        else         { ifn.bke -= cnx3 - ngh; }
      } else {
        if (f2 == 1) { ifn.bks += cnx3 - ngh; }
        else         { ifn.bke -= cnx3 - ngh; }
      }
    }
    buf.ifine_ndat = (ifn.bie-ifn.bis+1)*(ifn.bje-ifn.bjs+1)*(ifn.bke-ifn.bks+1);
  };

  auto compute_recv_icoar = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3, int f1, int f2) {
    auto &ic = buf.icoar[0];
    if (ox1 == 0) {
      ic.bis = cis_m; ic.bie = cie_m;
      if (f1 == 0) { ic.bie += ngh; } else { ic.bis -= ngh; }
    } else if (ox1 > 0) {
      ic.bis = cie_m + 1; ic.bie = cie_m + ngh;
    } else {
      ic.bis = cis_m - ngh; ic.bie = cis_m - 1;
    }
    if (ox2 == 0) {
      ic.bjs = cjs_m; ic.bje = cje_m;
      if (nx2 > 1) {
        if (ox1 != 0) {
          if (f1 == 0) { ic.bje += ngh; } else { ic.bjs -= ngh; }
        } else {
          if (f2 == 0) { ic.bje += ngh; } else { ic.bjs -= ngh; }
        }
      }
    } else if (ox2 > 0) {
      ic.bjs = cje_m + 1; ic.bje = cje_m + ngh;
    } else {
      ic.bjs = cjs_m - ngh; ic.bje = cjs_m - 1;
    }
    if (ox3 == 0) {
      ic.bks = cks_m; ic.bke = cke_m;
      if (nx3 > 1) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 0) { ic.bke += ngh; } else { ic.bks -= ngh; }
        } else {
          if (f2 == 0) { ic.bke += ngh; } else { ic.bks -= ngh; }
        }
      }
    } else if (ox3 > 0) {
      ic.bks = cke_m + 1; ic.bke = cke_m + ngh;
    } else {
      ic.bks = cks_m - ngh; ic.bke = cks_m - 1;
    }
    buf.icoar_ndat = (ic.bie-ic.bis+1)*(ic.bje-ic.bjs+1)*(ic.bke-ic.bks+1);
  };

  auto compute_recv_ifine = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3, int f1, int f2) {
    auto &ifn = buf.ifine[0];
    if (ox1 == 0) {
      ifn.bis = is_m; ifn.bie = ie_m;
      if (f1 == 1) { ifn.bis += cnx1; } else { ifn.bie -= cnx1; }
    } else if (ox1 > 0) {
      ifn.bis = ie_m + 1; ifn.bie = ie_m + ngh;
    } else {
      ifn.bis = is_m - ngh; ifn.bie = is_m - 1;
    }
    if (ox2 == 0) {
      ifn.bjs = js_m; ifn.bje = je_m;
      if (nx2 > 1) {
        if (ox1 != 0) {
          if (f1 == 1) { ifn.bjs += cnx2; } else { ifn.bje -= cnx2; }
        } else {
          if (f2 == 1) { ifn.bjs += cnx2; } else { ifn.bje -= cnx2; }
        }
      }
    } else if (ox2 > 0) {
      ifn.bjs = je_m + 1; ifn.bje = je_m + ngh;
    } else {
      ifn.bjs = js_m - ngh; ifn.bje = js_m - 1;
    }
    if (ox3 == 0) {
      ifn.bks = ks_m; ifn.bke = ke_m;
      if (nx3 > 1) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 1) { ifn.bks += cnx3; } else { ifn.bke -= cnx3; }
        } else {
          if (f2 == 1) { ifn.bks += cnx3; } else { ifn.bke -= cnx3; }
        }
      }
    } else if (ox3 > 0) {
      ifn.bks = ke_m + 1; ifn.bke = ke_m + ngh;
    } else {
      ifn.bks = ks_m - ngh; ifn.bke = ks_m - 1;
    }
    buf.ifine_ndat = (ifn.bie-ifn.bis+1)*(ifn.bje-ifn.bjs+1)*(ifn.bke-ifn.bks+1);
  };

  // Iterate over all buffer directions (mirrors InitializeBuffers order)
  // x1 faces
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy=0; fy<nfy; fy++) {
        int idx = NeighborIndex(n,0,0,fy,fz);
        compute_send_icoar(sendbuf[idx], n, 0, 0);
        compute_send_ifine(sendbuf[idx], n, 0, 0, fy, fz);
        compute_recv_icoar(recvbuf[idx], n, 0, 0, fy, fz);
        compute_recv_ifine(recvbuf[idx], n, 0, 0, fy, fz);
      }
    }
  }
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,m,0,fx,fz);
          compute_send_icoar(sendbuf[idx], 0, m, 0);
          compute_send_ifine(sendbuf[idx], 0, m, 0, fx, fz);
          compute_recv_icoar(recvbuf[idx], 0, m, 0, fx, fz);
          compute_recv_ifine(recvbuf[idx], 0, m, 0, fx, fz);
        }
      }
    }
    // x1x2 edges
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int idx = NeighborIndex(n,m,0,fz,0);
          compute_send_icoar(sendbuf[idx], n, m, 0);
          compute_send_ifine(sendbuf[idx], n, m, 0, fz, 0);
          compute_recv_icoar(recvbuf[idx], n, m, 0, fz, 0);
          compute_recv_ifine(recvbuf[idx], n, m, 0, fz, 0);
        }
      }
    }
  }
  if (pmy_pack->pmesh->three_d) {
    // x3 faces
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,0,l,fx,fy);
          compute_send_icoar(sendbuf[idx], 0, 0, l);
          compute_send_ifine(sendbuf[idx], 0, 0, l, fx, fy);
          compute_recv_icoar(recvbuf[idx], 0, 0, l, fx, fy);
          compute_recv_ifine(recvbuf[idx], 0, 0, l, fx, fy);
        }
      }
    }
    // x3x1 edges
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int idx = NeighborIndex(n,0,l,fy,0);
          compute_send_icoar(sendbuf[idx], n, 0, l);
          compute_send_ifine(sendbuf[idx], n, 0, l, fy, 0);
          compute_recv_icoar(recvbuf[idx], n, 0, l, fy, 0);
          compute_recv_ifine(recvbuf[idx], n, 0, l, fy, 0);
        }
      }
    }
    // x2x3 edges
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,m,l,fx,0);
          compute_send_icoar(sendbuf[idx], 0, m, l);
          compute_send_ifine(sendbuf[idx], 0, m, l, fx, 0);
          compute_recv_icoar(recvbuf[idx], 0, m, l, fx, 0);
          compute_recv_ifine(recvbuf[idx], 0, m, l, fx, 0);
        }
      }
    }
    // corners
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int idx = NeighborIndex(n,m,l,0,0);
          compute_send_icoar(sendbuf[idx], n, m, l);
          compute_send_ifine(sendbuf[idx], n, m, l, 0, 0);
          compute_recv_icoar(recvbuf[idx], n, m, l, 0, 0);
          compute_recv_ifine(recvbuf[idx], n, m, l, 0, 0);
        }
      }
    }
  }

  int nvar = pmy_mg->nvar_;
  int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  if (pmy_pack->pmesh->multilevel) {
    for (int n = 0; n < nnghbr; ++n) {
      int smax = std::max(sendbuf[n].isame_ndat,
                   std::max(sendbuf[n].icoar_ndat, sendbuf[n].ifine_ndat));
      if (nvar * smax > sendbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(sendbuf[n].vars, nmb, nvar * smax);
      }
      int rmax = std::max(recvbuf[n].isame_ndat,
                   std::max(recvbuf[n].icoar_ndat, recvbuf[n].ifine_ndat));
      if (nvar * rmax > recvbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(recvbuf[n].vars, nmb, nvar * rmax);
      }
    }

    int cbnx3 = cnx3 + 2*ngh;
    int cbnx2 = cnx2 + 2*ngh;
    int cbnx1 = cnx1 + 2*ngh;
    Kokkos::realloc(coarse_buf_, nmb, nvar, cbnx3, cbnx2, cbnx1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::ComputePerLevelIndices()
//! \brief Pre-compute isame/icoar/ifine send and recv indices for every MG level and
//! every neighbor direction.  This replaces the fragile runtime shift logic in
//! PackAndSendMG / RecvAndUnpackMG with exact, pre-computed values.  Uses the same index
//! formulas as buffs_cc.cpp but parameterized by MG ngh and the per-level cell count.
//! Must be called AFTER InitializeBuffers (so the 56 buffer slots exist).

void MultigridBoundaryValues::ComputePerLevelIndices() {
  int ngh    = pmy_mg->GetGhostCells();
  int ng1    = ngh - 1;
  int nx_max = pmy_mg->GetSize();            // finest-level cell count per direction
  int nlevel = pmy_mg->GetNumberOfLevels();

  bool md = pmy_pack->pmesh->multi_d;
  bool td = pmy_pack->pmesh->three_d;
  bool ml = pmy_pack->pmesh->multilevel;

  int nfx = ml ? 2 : 1;
  int nfy = (ml && md) ? 2 : 1;
  int nfz = (ml && td) ? 2 : 1;

  // Helper lambdas that mirror buffs_cc.cpp formulas but with MG parameters.
  // ncells = active cells in this direction at the current MG level.
  auto compute_send = [&](MGPerLevelIndcs &out,
                          int ox1, int ox2, int ox3, int f1, int f2,
                          int ncells) {
    int is_m  = ngh, ie_m  = ngh + ncells - 1;
    int js_m  = ngh, je_m  = ngh + ncells - 1;
    int ks_m  = ngh, ke_m  = ngh + ncells - 1;
    int cnx   = ncells / 2;
    int cis_m = ngh, cie_m = ngh + cnx - 1;
    int cjs_m = ngh, cje_m = ngh + cnx - 1;
    int cks_m = ngh, cke_m = ngh + cnx - 1;

    // -- isame (same-level send) --
    if (f1 == 0 && f2 == 0) {
      auto &s = out.isame;
      s.bis = (ox1 > 0) ? (ie_m - ng1) : is_m;
      s.bie = (ox1 < 0) ? (is_m + ng1) : ie_m;
      s.bjs = (ox2 > 0) ? (je_m - ng1) : js_m;
      s.bje = (ox2 < 0) ? (js_m + ng1) : je_m;
      s.bks = (ox3 > 0) ? (ke_m - ng1) : ks_m;
      s.bke = (ox3 < 0) ? (ks_m + ng1) : ke_m;
      out.isame_ndat = (s.bie-s.bis+1)*(s.bje-s.bjs+1)*(s.bke-s.bks+1);
    }

    // -- icoar (send to coarser) --
    // Face neighbors (nface==1): indices point to coarse_buf_ ghost cells
    // where face-aligned 2x2 averages are stored by FillCoarseMG.
    // Edge/corner neighbors (nface>1): indices point to coarse_buf_ interior
    // where volume averages are stored by FillCoarseMG.
    {
      auto &c = out.icoar;
      int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
      if (nface == 1) {
        c.bis = (ox1 > 0) ? (cie_m + 1)   : (ox1 < 0) ? (cis_m - ngh) : cis_m;
        c.bie = (ox1 > 0) ? (cie_m + ngh)  : (ox1 < 0) ? (cis_m - 1)  : cie_m;
        c.bjs = (ox2 > 0) ? (cje_m + 1)   : (ox2 < 0) ? (cjs_m - ngh) : cjs_m;
        c.bje = (ox2 > 0) ? (cje_m + ngh)  : (ox2 < 0) ? (cjs_m - 1)  : cje_m;
        c.bks = (ox3 > 0) ? (cke_m + 1)   : (ox3 < 0) ? (cks_m - ngh) : cks_m;
        c.bke = (ox3 > 0) ? (cke_m + ngh)  : (ox3 < 0) ? (cks_m - 1)  : cke_m;
      } else {
        c.bis = (ox1 > 0) ? (cie_m - ng1) : cis_m;
        c.bie = (ox1 < 0) ? (cis_m + ng1) : cie_m;
        c.bjs = (ox2 > 0) ? (cje_m - ng1) : cjs_m;
        c.bje = (ox2 < 0) ? (cjs_m + ng1) : cje_m;
        c.bks = (ox3 > 0) ? (cke_m - ng1) : cks_m;
        c.bke = (ox3 < 0) ? (cks_m + ng1) : cke_m;
      }
      out.icoar_ndat = (c.bie-c.bis+1)*(c.bje-c.bjs+1)*(c.bke-c.bks+1);
    }

    // -- ifine (send to finer) --
    {
      auto &f = out.ifine;
      f.bis = (ox1 > 0) ? (ie_m - ng1) : is_m;
      f.bie = (ox1 < 0) ? (is_m + ng1) : ie_m;
      f.bjs = (ox2 > 0) ? (je_m - ng1) : js_m;
      f.bje = (ox2 < 0) ? (js_m + ng1) : je_m;
      f.bks = (ox3 > 0) ? (ke_m - ng1) : ks_m;
      f.bke = (ox3 < 0) ? (ks_m + ng1) : ke_m;
      if (ox1 == 0) {
        if (f1 == 1) { f.bis += cnx - ngh; }
        else         { f.bie -= cnx - ngh; }
      }
      if (ox2 == 0 && md) {
        if (ox1 != 0) {
          if (f1 == 1) { f.bjs += cnx - ngh; }
          else         { f.bje -= cnx - ngh; }
        } else {
          if (f2 == 1) { f.bjs += cnx - ngh; }
          else         { f.bje -= cnx - ngh; }
        }
      }
      if (ox3 == 0 && td) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 1) { f.bks += cnx - ngh; }
          else         { f.bke -= cnx - ngh; }
        } else {
          if (f2 == 1) { f.bks += cnx - ngh; }
          else         { f.bke -= cnx - ngh; }
        }
      }
      out.ifine_ndat = (f.bie-f.bis+1)*(f.bje-f.bjs+1)*(f.bke-f.bks+1);
    }
  };

  auto compute_recv = [&](MGPerLevelIndcs &out,
                          int ox1, int ox2, int ox3, int f1, int f2,
                          int ncells) {
    int is_m  = ngh, ie_m  = ngh + ncells - 1;
    int js_m  = ngh, je_m  = ngh + ncells - 1;
    int ks_m  = ngh, ke_m  = ngh + ncells - 1;
    int cnx   = ncells / 2;
    int cis_m = ngh, cie_m = ngh + cnx - 1;
    int cjs_m = ngh, cje_m = ngh + cnx - 1;
    int cks_m = ngh, cke_m = ngh + cnx - 1;

    // -- isame (same-level recv) --
    if (f1 == 0 && f2 == 0) {
      auto &s = out.isame;
      if (ox1 == 0)      { s.bis = is_m;      s.bie = ie_m; }
      else if (ox1 > 0)  { s.bis = ie_m + 1;  s.bie = ie_m + ngh; }
      else               { s.bis = is_m - ngh; s.bie = is_m - 1; }
      if (ox2 == 0)      { s.bjs = js_m;      s.bje = je_m; }
      else if (ox2 > 0)  { s.bjs = je_m + 1;  s.bje = je_m + ngh; }
      else               { s.bjs = js_m - ngh; s.bje = js_m - 1; }
      if (ox3 == 0)      { s.bks = ks_m;      s.bke = ke_m; }
      else if (ox3 > 0)  { s.bks = ke_m + 1;  s.bke = ke_m + ngh; }
      else               { s.bks = ks_m - ngh; s.bke = ks_m - 1; }
      out.isame_ndat = (s.bie-s.bis+1)*(s.bje-s.bjs+1)*(s.bke-s.bks+1);
    }

    // -- icoar (recv from coarser, matches send-to-finer) --
    {
      auto &c = out.icoar;
      if (ox1 == 0)      { c.bis = cis_m;      c.bie = cie_m;
                           if (f1 == 0) { c.bie += ngh; } else { c.bis -= ngh; } }
      else if (ox1 > 0)  { c.bis = cie_m + 1;  c.bie = cie_m + ngh; }
      else               { c.bis = cis_m - ngh; c.bie = cis_m - 1; }

      if (ox2 == 0) {
        c.bjs = cjs_m; c.bje = cje_m;
        if (md) {
          if (ox1 != 0) {
            if (f1 == 0) { c.bje += ngh; } else { c.bjs -= ngh; }
          } else {
            if (f2 == 0) { c.bje += ngh; } else { c.bjs -= ngh; }
          }
        }
      } else if (ox2 > 0)  { c.bjs = cje_m + 1;  c.bje = cje_m + ngh; }
      else                  { c.bjs = cjs_m - ngh; c.bje = cjs_m - 1; }

      if (ox3 == 0) {
        c.bks = cks_m; c.bke = cke_m;
        if (td) {
          if (ox1 != 0 && ox2 != 0) {
            if (f1 == 0) { c.bke += ngh; } else { c.bks -= ngh; }
          } else {
            if (f2 == 0) { c.bke += ngh; } else { c.bks -= ngh; }
          }
        }
      } else if (ox3 > 0)  { c.bks = cke_m + 1;  c.bke = cke_m + ngh; }
      else                  { c.bks = cks_m - ngh; c.bke = cks_m - 1; }
      out.icoar_ndat = (c.bie-c.bis+1)*(c.bje-c.bjs+1)*(c.bke-c.bks+1);
    }

    // -- ifine (recv from finer, matches send-to-coarser) --
    {
      auto &fn = out.ifine;
      if (ox1 == 0) {
        fn.bis = is_m; fn.bie = ie_m;
        if (f1 == 1) { fn.bis += cnx; } else { fn.bie -= cnx; }
      } else if (ox1 > 0) { fn.bis = ie_m + 1;  fn.bie = ie_m + ngh; }
      else                 { fn.bis = is_m - ngh; fn.bie = is_m - 1; }

      if (ox2 == 0) {
        fn.bjs = js_m; fn.bje = je_m;
        if (md) {
          if (ox1 != 0) {
            if (f1 == 1) { fn.bjs += cnx; } else { fn.bje -= cnx; }
          } else {
            if (f2 == 1) { fn.bjs += cnx; } else { fn.bje -= cnx; }
          }
        }
      } else if (ox2 > 0) { fn.bjs = je_m + 1;  fn.bje = je_m + ngh; }
      else                 { fn.bjs = js_m - ngh; fn.bje = js_m - 1; }

      if (ox3 == 0) {
        fn.bks = ks_m; fn.bke = ke_m;
        if (td) {
          if (ox1 != 0 && ox2 != 0) {
            if (f1 == 1) { fn.bks += cnx; } else { fn.bke -= cnx; }
          } else {
            if (f2 == 1) { fn.bks += cnx; } else { fn.bke -= cnx; }
          }
        }
      } else if (ox3 > 0) { fn.bks = ke_m + 1;  fn.bke = ke_m + ngh; }
      else                 { fn.bks = ks_m - ngh; fn.bke = ks_m - 1; }
      out.ifine_ndat = (fn.bie-fn.bis+1)*(fn.bje-fn.bjs+1)*(fn.bke-fn.bks+1);
    }
  };

  // Fill indices for each MG level and each neighbor direction.
  for (int lev = 0; lev < nlevel && lev < kMaxMGLevels; ++lev) {
    int shift = nlevel - 1 - lev;
    int ncells = nx_max >> shift;
    if (ncells < 1) ncells = 1;

    // x1 faces
    for (int n = -1; n <= 1; n += 2) {
      for (int fz = 0; fz < nfz; fz++) {
        for (int fy = 0; fy < nfy; fy++) {
          int idx = NeighborIndex(n, 0, 0, fy, fz);
          compute_send(send_mg_indcs_[idx][lev], n, 0, 0, fy, fz, ncells);
          compute_recv(recv_mg_indcs_[idx][lev], n, 0, 0, fy, fz, ncells);
        }
      }
    }
    if (md) {
      // x2 faces
      for (int m = -1; m <= 1; m += 2) {
        for (int fz = 0; fz < nfz; fz++) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, m, 0, fx, fz);
            compute_send(send_mg_indcs_[idx][lev], 0, m, 0, fx, fz, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, m, 0, fx, fz, ncells);
          }
        }
      }
      // x1x2 edges
      for (int m = -1; m <= 1; m += 2) {
        for (int n = -1; n <= 1; n += 2) {
          for (int fz = 0; fz < nfz; fz++) {
            int idx = NeighborIndex(n, m, 0, fz, 0);
            compute_send(send_mg_indcs_[idx][lev], n, m, 0, fz, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, m, 0, fz, 0, ncells);
          }
        }
      }
    }
    if (td) {
      // x3 faces
      for (int l = -1; l <= 1; l += 2) {
        for (int fy = 0; fy < nfy; fy++) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, 0, l, fx, fy);
            compute_send(send_mg_indcs_[idx][lev], 0, 0, l, fx, fy, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, 0, l, fx, fy, ncells);
          }
        }
      }
      // x3x1 edges
      for (int l = -1; l <= 1; l += 2) {
        for (int n = -1; n <= 1; n += 2) {
          for (int fy = 0; fy < nfy; fy++) {
            int idx = NeighborIndex(n, 0, l, fy, 0);
            compute_send(send_mg_indcs_[idx][lev], n, 0, l, fy, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, 0, l, fy, 0, ncells);
          }
        }
      }
      // x2x3 edges
      for (int l = -1; l <= 1; l += 2) {
        for (int m = -1; m <= 1; m += 2) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, m, l, fx, 0);
            compute_send(send_mg_indcs_[idx][lev], 0, m, l, fx, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, m, l, fx, 0, ncells);
          }
        }
      }
      // corners
      for (int l = -1; l <= 1; l += 2) {
        for (int m = -1; m <= 1; m += 2) {
          for (int n = -1; n <= 1; n += 2) {
            int idx = NeighborIndex(n, m, l, 0, 0);
            compute_send(send_mg_indcs_[idx][lev], n, m, l, 0, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, m, l, 0, 0, ncells);
          }
        }
      }
    }
  }

  int nvar = pmy_mg->nvar_;
  int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  int nnghbr = pmy_pack->pmb->nnghbr;
  int finest = nlevel - 1;

  if (pmy_pack->pmesh->multilevel) {
    for (int n = 0; n < nnghbr; ++n) {
      int smax = std::max(send_mg_indcs_[n][finest].isame_ndat,
                   std::max(send_mg_indcs_[n][finest].icoar_ndat,
                            send_mg_indcs_[n][finest].ifine_ndat));
      if (nvar * smax > sendbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(sendbuf[n].vars, nmb, nvar * smax);
      }
      int rmax = std::max(recv_mg_indcs_[n][finest].isame_ndat,
                   std::max(recv_mg_indcs_[n][finest].icoar_ndat,
                            recv_mg_indcs_[n][finest].ifine_ndat));
      if (nvar * rmax > recvbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(recvbuf[n].vars, nmb, nvar * rmax);
      }
    }

    int cnx_f = nx_max / 2;
    int cbn = cnx_f + 2*ngh;
    Kokkos::realloc(coarse_buf_, nmb, nvar, cbn, cbn, cbn);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::FillCoarseMG()
//! \brief Restrict MG data interior into coarse_buf_ interior so that the prolongation
//! kernel has gradient context from the block's own data.

void MultigridBoundaryValues::FillCoarseMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return;
  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;
  if (ncells < 2) return;
  int cnc = ncells / 2;
  int nmb = pmy_pack->nmb_thispack;
  auto cbuf = coarse_buf_;

  // Volume-average restriction: fill coarse_buf_ interior
  Kokkos::parallel_for("FillCoarseMG",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, DevExeSpace>(
      {0, 0, 0, 0}, {nmb * nvar, cnc, cnc, cnc}),
    KOKKOS_LAMBDA(const int mv, const int ck, const int cj, const int ci) {
      int m = mv / nvar;
      int v = mv - m * nvar;
      int fi = ngh + 2*ci;
      int fj = ngh + 2*cj;
      int fk = ngh + 2*ck;
      cbuf(m, v, ngh + ck, ngh + cj, ngh + ci) = 0.125 * (
        u(m,v,fk,  fj,  fi) + u(m,v,fk,  fj,  fi+1) +
        u(m,v,fk,  fj+1,fi) + u(m,v,fk,  fj+1,fi+1) +
        u(m,v,fk+1,fj,  fi) + u(m,v,fk+1,fj,  fi+1) +
        u(m,v,fk+1,fj+1,fi) + u(m,v,fk+1,fj+1,fi+1));
    });

  if (!(pmy_pack->pmesh->multilevel)) return;

  // Face-aligned restriction: store 2x2 face averages in coarse_buf_ ghost cells.
  // These are consumed by PackAndSendMG for fine-to-coarse face sends.
  // face_id: 0=x-left, 1=x-right, 2=y-left, 3=y-right, 4=z-left, 5=z-right
  Kokkos::parallel_for("FillCoarseMG_faces",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, DevExeSpace>(
      {0, 0, 0, 0}, {nmb * nvar, 6, cnc, cnc}),
    KOKKOS_LAMBDA(const int mv, const int face, const int c1, const int c0) {
      int m = mv / nvar;
      int v = mv - m * nvar;
      if (face < 2) {
        int fj = ngh + 2*c0;
        int fk = ngh + 2*c1;
        int fi = (face == 0) ? ngh : ngh + ncells - 1;
        int ci = (face == 0) ? ngh - 1 : ngh + cnc;
        cbuf(m, v, ngh + c1, ngh + c0, ci) = 0.25 * (
          u(m,v,fk,  fj,  fi) + u(m,v,fk,  fj+1,fi) +
          u(m,v,fk+1,fj,  fi) + u(m,v,fk+1,fj+1,fi));
      } else if (face < 4) {
        int fi = ngh + 2*c0;
        int fk = ngh + 2*c1;
        int fj = (face == 2) ? ngh : ngh + ncells - 1;
        int cj = (face == 2) ? ngh - 1 : ngh + cnc;
        cbuf(m, v, ngh + c1, cj, ngh + c0) = 0.25 * (
          u(m,v,fk,  fj,fi  ) + u(m,v,fk,  fj,fi+1) +
          u(m,v,fk+1,fj,fi  ) + u(m,v,fk+1,fj,fi+1));
      } else {
        int fi = ngh + 2*c0;
        int fj = ngh + 2*c1;
        int fk = (face == 4) ? ngh : ngh + ncells - 1;
        int ck = (face == 4) ? ngh - 1 : ngh + cnc;
        cbuf(m, v, ck, ngh + c1, ngh + c0) = 0.25 * (
          u(m,v,fk,fj,  fi  ) + u(m,v,fk,fj,  fi+1) +
          u(m,v,fk,fj+1,fi  ) + u(m,v,fk,fj+1,fi+1));
      }
    });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::ProlongateFCMG()
//! \brief Prolongate from coarse_buf_ to fine ghost cells using the same flux-conserving
//! formulas as FillFineCoarseMGGhosts.  For face neighbors at coarser level, uses
//! gradient-based prolongation.  For edge/corner neighbors at coarser level, uses simple
//! injection.  For finer neighbors, restriction was already done inline in unpack.

TaskStatus MultigridBoundaryValues::ProlongateFCMG(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;
  if (ncells < 2) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr_d = pmy_pack->pmb->nghbr.d_view;
  auto mblev_d = pmy_pack->pmb->mb_lev.d_view;
  auto mbgid_d = pmy_pack->pmb->mb_gid.d_view;
  auto fc_cx = pmy_mg->fc_childx_;
  auto fc_cy = pmy_mg->fc_childy_;
  auto fc_cz = pmy_mg->fc_childz_;
  auto cbuf = coarse_buf_;

  int nvar_l = nvar;
  int ngh_l = ngh;
  int ncells_l = ncells;
  int half = ncells / 2;
  constexpr Real ot = 1.0/3.0;

  Kokkos::parallel_for("ProlongateFCMG",
    Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m) {
      int m_lev = mblev_d(m);
      int child_x = fc_cx(m);
      int child_y = fc_cy(m);
      int child_z = fc_cz(m);

      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
            int f2_max = (nface == 1) ? 1 : 0;
            int f1_max = (nface <= 2) ? 1 : 0;

            for (int f2 = 0; f2 <= f2_max; ++f2) {
              for (int f1 = 0; f1 <= f1_max; ++f1) {
                int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
                if (n < 0 || n >= 56) continue;
                if (nghbr_d(m, n).gid < 0) continue;
                int nlev = nghbr_d(m, n).lev;

                // From finer face neighbor: apply FC correction.
                // Ghost cells already contain restricted face avg from unpack.
                if (nlev > m_lev && nface == 1) {
                  int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                  int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                  int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (ox1 != 0) { sub_y = f1; sub_z = f2; }
                  else if (ox2 != 0) { sub_x = f1; sub_z = f2; }
                  else { sub_x = f1; sub_y = f2; }

                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;              gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l;  gie = ngh_l+ncells_l+ngh_l-1; }
                  else { gis = ngh_l+sub_x*half; gie = ngh_l+sub_x*half+half-1; }
                  if (ox2 < 0)      { gjs = 0;              gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l;  gje = ngh_l+ncells_l+ngh_l-1; }
                  else { gjs = ngh_l+sub_y*half; gje = ngh_l+sub_y*half+half-1; }
                  if (ox3 < 0)      { gks = 0;              gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l;  gke = ngh_l+ncells_l+ngh_l-1; }
                  else { gks = ngh_l+sub_z*half; gke = ngh_l+sub_z*half+half-1; }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          Real avg = u(m,v,gk,gj,gi);
                          u(m,v,gk,gj,gi) = ot*(4.0*avg
                              - u(m,v,gk+ok,gj+oj,gi+oi));
                        }
                      }
                    }
                  }
                  continue;
                }

                if (nlev >= m_lev) continue;  // skip same-level and remaining finer

                // Face neighbor from coarser: flux-conserving prolongation
                // from coarse_buf_ into fine ghost cells of u
                if (nface == 1) {
                  if (ox1 != 0) {
                    int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int si  = (ox1 < 0) ? ngh_l - 1 : ngh_l + half;
                    int sj0 = ngh_l;
                    int sk0 = ngh_l;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int sj = sj0; sj < sj0 + half; ++sj) {
                          int fj = ngh_l + 2*(sj - sj0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = cbuf(m,v,sk,sj,si);
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+half) ? sj+1 : sj;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+half) ? sk+1 : sk;
                          Real gy = 0.125*(cbuf(m,v,sk,sjp,si)-cbuf(m,v,sk,sjm,si));
                          Real gz = 0.125*(cbuf(m,v,skp,sj,si)-cbuf(m,v,skm,sj,si));
                          u(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u(m,v,fk  ,fj  ,fi));
                          u(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u(m,v,fk  ,fj+1,fi));
                          u(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u(m,v,fk+1,fj  ,fi));
                          u(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u(m,v,fk+1,fj+1,fi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sj  = (ox2 < 0) ? ngh_l - 1 : ngh_l + half;
                    int si0 = ngh_l;
                    int sk0 = ngh_l;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = cbuf(m,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+half) ? si+1 : si;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+half) ? sk+1 : sk;
                          Real gx = 0.125*(cbuf(m,v,sk,sj,sip)-cbuf(m,v,sk,sj,sim));
                          Real gz = 0.125*(cbuf(m,v,skp,sj,si)-cbuf(m,v,skm,sj,si));
                          u(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u(m,v,fk  ,fj,fi  ));
                          u(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u(m,v,fk  ,fj,fi+1));
                          u(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u(m,v,fk+1,fj,fi  ));
                          u(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u(m,v,fk+1,fj,fi+1));
                        }
                      }
                    }
                  } else {
                    int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sk  = (ox3 < 0) ? ngh_l - 1 : ngh_l + half;
                    int si0 = ngh_l;
                    int sj0 = ngh_l;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fj = ngh_l + 2*(sj - sj0);
                          Real cc = cbuf(m,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+half) ? si+1 : si;
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+half) ? sj+1 : sj;
                          Real gx = 0.125*(cbuf(m,v,sk,sj,sip)-cbuf(m,v,sk,sj,sim));
                          Real gy = 0.125*(cbuf(m,v,sk,sjp,si)-cbuf(m,v,sk,sjm,si));
                          u(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u(m,v,fk,fj  ,fi  ));
                          u(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u(m,v,fk,fj  ,fi+1));
                          u(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u(m,v,fk,fj+1,fi  ));
                          u(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u(m,v,fk,fj+1,fi+1));
                        }
                      }
                    }
                  }
                } else {
                  // Edge/corner from coarser: simple injection from coarse_buf_
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;             gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l; gie = ngh_l+ncells_l+ngh_l-1; }
                  else              { gis = ngh_l;           gie = ngh_l + ncells_l - 1; }
                  if (ox2 < 0)      { gjs = 0;             gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l; gje = ngh_l+ncells_l+ngh_l-1; }
                  else              { gjs = ngh_l;           gje = ngh_l + ncells_l - 1; }
                  if (ox3 < 0)      { gks = 0;             gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l; gke = ngh_l+ncells_l+ngh_l-1; }
                  else              { gks = ngh_l;           gke = ngh_l + ncells_l - 1; }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int ci, cj, ck;
                          if (ox1 < 0)      ci = ngh_l - 1;
                          else if (ox1 > 0) ci = ngh_l + half;
                          else              ci = ngh_l + (gi - ngh_l)/2;
                          if (ox2 < 0)      cj = ngh_l - 1;
                          else if (ox2 > 0) cj = ngh_l + half;
                          else              cj = ngh_l + (gj - ngh_l)/2;
                          if (ox3 < 0)      ck = ngh_l - 1;
                          else if (ox3 > 0) ck = ngh_l + half;
                          else              ck = ngh_l + (gk - ngh_l)/2;

                          u(m, v, gk, gj, gi) = cbuf(m, v, ck, cj, ci);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts()
//! \brief Fill ghost cells at fine-coarse boundaries.
//! Faces use flux-conserving prolongation/restriction matching Athena++ formulas.
//! Edges and corners use simple injection/restriction. Same-rank only.

TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;

  if (ncells < 1) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int my_rank = global_variable::my_rank;
  auto nghbr_d = pmy_pack->pmb->nghbr.d_view;
  auto mblev_d = pmy_pack->pmb->mb_lev.d_view;
  auto mbgid_d = pmy_pack->pmb->mb_gid.d_view;

#ifndef NDEBUG
  {
    static bool warned_cross_rank_fc = false;
    if (!warned_cross_rank_fc) {
      auto &nghbr_h = pmy_pack->pmb->nghbr;
      auto &mblev_h = pmy_pack->pmb->mb_lev;
      for (int m = 0; m < nmb && !warned_cross_rank_fc; ++m) {
        for (int n = 0; n < nnghbr && !warned_cross_rank_fc; ++n) {
          if (nghbr_h.h_view(m,n).gid >= 0
              && nghbr_h.h_view(m,n).lev != mblev_h.h_view(m)
              && nghbr_h.h_view(m,n).rank != my_rank) {
            std::cout << "### MG WARNING: cross-rank fine-coarse neighbor detected "
                      << "(m=" << m << " n=" << n << " rank=" << nghbr_h.h_view(m,n).rank
                      << "). FillFineCoarseMGGhosts skips cross-rank neighbors."
                      << std::endl;
            warned_cross_rank_fc = true;
          }
        }
      }
    }
  }
#endif
  auto fc_cx = pmy_mg->fc_childx_;
  auto fc_cy = pmy_mg->fc_childy_;
  auto fc_cz = pmy_mg->fc_childz_;

  int nmb_l = nmb;
  int nnghbr_l = nnghbr;
  int my_rank_l = my_rank;
  int nvar_l = nvar;
  int ngh_l = ngh;
  int ncells_l = ncells;
  constexpr Real ot = 1.0/3.0;

  Kokkos::parallel_for("FillFCMGGhosts",
    Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m) {
      int m_lev = mblev_d(m);
      int child_x = fc_cx(m);
      int child_y = fc_cy(m);
      int child_z = fc_cz(m);
      int half = ncells_l / 2;

      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
            int f2_max = (nface == 1) ? 1 : 0;
            int f1_max = (nface <= 2) ? 1 : 0;

            for (int f2 = 0; f2 <= f2_max; ++f2) {
              for (int f1 = 0; f1 <= f1_max; ++f1) {
                int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
                if (n < 0 || n >= nnghbr_l) continue;
                if (nghbr_d(m, n).gid < 0) continue;

                int nlev = nghbr_d(m, n).lev;
                if (nlev == m_lev) continue;
                if (nghbr_d(m, n).rank != my_rank_l) continue;

                int dm = nghbr_d(m, n).gid - mbgid_d(0);
                if (dm < 0 || dm >= nmb_l) continue;

                if (nlev < m_lev && nface == 1) {
                  // Coarser neighbor, face: flux-conserving prolongation
                  if (ox1 != 0) {
                    int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int si  = (ox1 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int sj0 = ngh_l + child_y * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int sj = sj0; sj < sj0 + half; ++sj) {
                          int fj = ngh_l + 2*(sj - sj0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real gy = 0.125*(u(dm,v,sk,sjp,si)-u(dm,v,sk,sjm,si));
                          Real gz = 0.125*(u(dm,v,skp,sj,si)-u(dm,v,skm,sj,si));
                          u(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u(m,v,fk  ,fj  ,fi));
                          u(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u(m,v,fk  ,fj+1,fi));
                          u(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u(m,v,fk+1,fj  ,fi));
                          u(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u(m,v,fk+1,fj+1,fi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sj  = (ox2 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real gx = 0.125*(u(dm,v,sk,sj,sip)-u(dm,v,sk,sj,sim));
                          Real gz = 0.125*(u(dm,v,skp,sj,si)-u(dm,v,skm,sj,si));
                          u(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u(m,v,fk  ,fj,fi  ));
                          u(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u(m,v,fk  ,fj,fi+1));
                          u(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u(m,v,fk+1,fj,fi  ));
                          u(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u(m,v,fk+1,fj,fi+1));
                        }
                      }
                    }
                  } else {
                    int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sk  = (ox3 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sj0 = ngh_l + child_y * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fj = ngh_l + 2*(sj - sj0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          Real gx = 0.125*(u(dm,v,sk,sj,sip)-u(dm,v,sk,sj,sim));
                          Real gy = 0.125*(u(dm,v,sk,sjp,si)-u(dm,v,sk,sjm,si));
                          u(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u(m,v,fk,fj  ,fi  ));
                          u(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u(m,v,fk,fj  ,fi+1));
                          u(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u(m,v,fk,fj+1,fi  ));
                          u(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u(m,v,fk,fj+1,fi+1));
                        }
                      }
                    }
                  }

                } else if (nlev > m_lev && nface == 1) {
                  // Finer neighbor, face: flux-conserving restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (ox1 != 0) { sub_y = f1; sub_z = f2; }
                  if (ox2 != 0) { sub_x = f1; sub_z = f2; }
                  if (ox3 != 0) { sub_x = f1; sub_y = f2; }

                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;             gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l; gie = ngh_l+ncells_l+ngh_l-1; }
                  else { gis = ngh_l+sub_x*half; gie = ngh_l+sub_x*half+half-1; }
                  if (ox2 < 0)      { gjs = 0;             gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l; gje = ngh_l+ncells_l+ngh_l-1; }
                  else { gjs = ngh_l+sub_y*half; gje = ngh_l+sub_y*half+half-1; }
                  if (ox3 < 0)      { gks = 0;             gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l; gke = ngh_l+ncells_l+ngh_l-1; }
                  else { gks = ngh_l+sub_z*half; gke = ngh_l+sub_z*half+half-1; }

                  int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                  int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                  int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                  if (ox1 != 0) {
                    int fi = (ox1 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gj = gjs; gj <= gje; ++gj) {
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = 0.25*(u(dm,v,fk0,fj0,fi)+u(dm,v,fk0,fj0+1,fi)
                                           +u(dm,v,fk0+1,fj0,fi)+u(dm,v,fk0+1,fj0+1,fi));
                          u(m,v,gk,gj,gis) = ot*(4.0*favg - u(m,v,gk+ok,gj+oj,gis+oi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fj = (ox2 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = 0.25*(u(dm,v,fk0,fj,fi0)+u(dm,v,fk0,fj,fi0+1)
                                           +u(dm,v,fk0+1,fj,fi0)+u(dm,v,fk0+1,fj,fi0+1));
                          u(m,v,gk,gjs,gi) = ot*(4.0*favg - u(m,v,gk+ok,gjs+oj,gi+oi));
                        }
                      }
                    }
                  } else {
                    int fk = (ox3 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          Real favg = 0.25*(u(dm,v,fk,fj0,fi0)+u(dm,v,fk,fj0,fi0+1)
                                           +u(dm,v,fk,fj0+1,fi0)+u(dm,v,fk,fj0+1,fi0+1));
                          u(m,v,gks,gj,gi) = ot*(4.0*favg - u(m,v,gks+ok,gj+oj,gi+oi));
                        }
                      }
                    }
                  }

                } else if (nlev < m_lev) {
                  // Coarser neighbor, edge/corner: simple injection
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;             gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l; gie = ngh_l+ncells_l+ngh_l-1; }
                  else              { gis = ngh_l;           gie = ngh_l + ncells_l - 1; }
                  if (ox2 < 0)      { gjs = 0;             gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l; gje = ngh_l+ncells_l+ngh_l-1; }
                  else              { gjs = ngh_l;           gje = ngh_l + ncells_l - 1; }
                  if (ox3 < 0)      { gks = 0;             gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l; gke = ngh_l+ncells_l+ngh_l-1; }
                  else              { gks = ngh_l;           gke = ngh_l + ncells_l - 1; }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int si, sj, sk;
                          if (ox1 < 0)      si = ngh_l + ncells_l - 1;
                          else if (ox1 > 0) si = ngh_l;
                          else si = ngh_l + child_x*half + (gi - ngh_l)/2;
                          if (ox2 < 0)      sj = ngh_l + ncells_l - 1;
                          else if (ox2 > 0) sj = ngh_l;
                          else sj = ngh_l + child_y*half + (gj - ngh_l)/2;
                          if (ox3 < 0)      sk = ngh_l + ncells_l - 1;
                          else if (ox3 > 0) sk = ngh_l;
                          else sk = ngh_l + child_z*half + (gk - ngh_l)/2;

                          u(m, v, gk, gj, gi) = u(dm, v, sk, sj, si);
                        }
                      }
                    }
                  }

                } else {
                  // Finer neighbor, edge/corner: simple restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (nface == 2) {
                    if (ox1 == 0) sub_x = f1;
                    if (ox2 == 0) sub_y = f1;
                    if (ox3 == 0) sub_z = f1;
                  }
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;             gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l; gie = ngh_l+ncells_l+ngh_l-1; }
                  else { gis = ngh_l+sub_x*half; gie = ngh_l+sub_x*half+half-1; }
                  if (ox2 < 0)      { gjs = 0;             gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l; gje = ngh_l+ncells_l+ngh_l-1; }
                  else { gjs = ngh_l+sub_y*half; gje = ngh_l+sub_y*half+half-1; }
                  if (ox3 < 0)      { gks = 0;             gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l; gke = ngh_l+ncells_l+ngh_l-1; }
                  else { gks = ngh_l+sub_z*half; gke = ngh_l+sub_z*half+half-1; }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0, fi1, fj0, fj1, fk0, fk1;
                          if (ox1 < 0) {
                            fi0 = ngh_l+ncells_l-2; fi1 = ngh_l+ncells_l-1;
                          } else if (ox1 > 0) {
                            fi0 = ngh_l; fi1 = ngh_l + 1;
                          } else {
                            fi0 = ngh_l+2*(gi-(ngh_l+sub_x*half)); fi1 = fi0+1;
                          }
                          if (ox2 < 0) {
                            fj0 = ngh_l+ncells_l-2; fj1 = ngh_l+ncells_l-1;
                          } else if (ox2 > 0) {
                            fj0 = ngh_l; fj1 = ngh_l + 1;
                          } else {
                            fj0 = ngh_l+2*(gj-(ngh_l+sub_y*half)); fj1 = fj0+1;
                          }
                          if (ox3 < 0) {
                            fk0 = ngh_l+ncells_l-2; fk1 = ngh_l+ncells_l-1;
                          } else if (ox3 > 0) {
                            fk0 = ngh_l; fk1 = ngh_l + 1;
                          } else {
                            fk0 = ngh_l+2*(gk-(ngh_l+sub_z*half)); fk1 = fk0+1;
                          }
                          u(m, v, gk, gj, gi) = 0.125 * (
                            u(dm,v,fk0,fj0,fi0) + u(dm,v,fk0,fj0,fi1) +
                            u(dm,v,fk0,fj1,fi0) + u(dm,v,fk0,fj1,fi1) +
                            u(dm,v,fk1,fj0,fi0) + u(dm,v,fk1,fj0,fi1) +
                            u(dm,v,fk1,fj1,fi0) + u(dm,v,fk1,fj1,fi1));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
  });

#if MPI_PARALLEL_ENABLED
  // Cross-rank fine-coarse ghost fill via synchronous MPI.
  // Each rank sends its block's data at the current MG level for every cross-rank
  // FC neighbor pair, then applies prolongation/restriction using the received data.
  {
    auto &nghbr_h = pmy_pack->pmb->nghbr;
    auto &mblev_h = pmy_pack->pmb->mb_lev;

    struct FCPair {
      int m, n, ox1, ox2, ox3, f1, f2, nface;
      int remote_rank, remote_gid, nlev, m_lev;
    };
    std::vector<FCPair> pairs;
    for (int m = 0; m < nmb; ++m) {
      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nf_ = (ox1!=0?1:0)+(ox2!=0?1:0)+(ox3!=0?1:0);
            int f2_max = (nf_ == 1) ? 1 : 0;
            int f1_max = (nf_ <= 2) ? 1 : 0;
            for (int f2_ = 0; f2_ <= f2_max; ++f2_) {
              for (int f1_ = 0; f1_ <= f1_max; ++f1_) {
                int nn = NeighborIndex(ox1, ox2, ox3, f1_, f2_);
                if (nn < 0 || nn >= nnghbr) continue;
                if (nghbr_h.h_view(m,nn).gid < 0) continue;
                int nlev_n = nghbr_h.h_view(m,nn).lev;
                if (nlev_n == mblev_h.h_view(m)) continue;
                if (nghbr_h.h_view(m,nn).rank == my_rank) continue;
                FCPair p;
                p.m = m; p.n = nn;
                p.ox1 = ox1; p.ox2 = ox2; p.ox3 = ox3;
                p.f1 = f1_; p.f2 = f2_;
                p.nface = (ox1!=0?1:0)+(ox2!=0?1:0)+(ox3!=0?1:0);
                p.remote_rank = nghbr_h.h_view(m,nn).rank;
                p.remote_gid = nghbr_h.h_view(m,nn).gid;
                p.nlev = nlev_n;
                p.m_lev = mblev_h.h_view(m);
                pairs.push_back(p);
              }
            }
          }
        }
      }
    }

    if (!pairs.empty()) {
      Kokkos::fence();
      int ntot = ncells + 2*ngh;
      int blk_sz = nvar * ntot * ntot * ntot;

      int np = static_cast<int>(pairs.size());
      std::vector<std::vector<Real>> sdata(np), rdata(np);
      std::vector<MPI_Request> sreqs(np, MPI_REQUEST_NULL);
      std::vector<MPI_Request> rreqs(np, MPI_REQUEST_NULL);

      for (int p = 0; p < np; ++p) {
        rdata[p].resize(blk_sz);
        int tag = CreateBvals_MPI_Tag(pairs[p].m, pairs[p].n + 64);
        MPI_Irecv(rdata[p].data(), blk_sz, MPI_ATHENA_REAL,
                  pairs[p].remote_rank, tag, comm_vars, &rreqs[p]);
      }
      for (int p = 0; p < np; ++p) {
        sdata[p].resize(blk_sz);
        int bm = pairs[p].m;
        int idx = 0;
        for (int v = 0; v < nvar; ++v)
          for (int k = 0; k < ntot; ++k)
            for (int j = 0; j < ntot; ++j)
              for (int i = 0; i < ntot; ++i)
                sdata[p][idx++] = u(bm, v, k, j, i);
        int dn = nghbr_h.h_view(bm, pairs[p].n).dest;
        int rlid = pairs[p].remote_gid
                   - pmy_pack->pmesh->gids_eachrank[pairs[p].remote_rank];
        int stag = CreateBvals_MPI_Tag(rlid, dn + 64);
        MPI_Isend(sdata[p].data(), blk_sz, MPI_ATHENA_REAL,
                  pairs[p].remote_rank, stag, comm_vars, &sreqs[p]);
      }

      MPI_Waitall(np, rreqs.data(), MPI_STATUSES_IGNORE);

      constexpr Real ot_h = 1.0/3.0;
      for (int p = 0; p < np; ++p) {
        auto &pr = pairs[p];
        int ml = pr.m;
        // Compute child offsets from global LogicalLocation data
        int gid_ml = pmy_pack->pmb->mb_gid.h_view(ml);
        LogicalLocation &loc_ml = pmy_pack->pmesh->lloc_eachmb[gid_ml];
        int root_level = pmy_pack->pmesh->root_level;
        int cx = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx1 & 1) : 0;
        int cy = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx2 & 1) : 0;
        int cz = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx3 & 1) : 0;
        int hl = ncells / 2;
        int nl = pr.nlev, mlev = pr.m_lev, nf = pr.nface;
        int ox1=pr.ox1, ox2=pr.ox2, ox3=pr.ox3, f1_=pr.f1, f2_=pr.f2;

        auto R = [&](int v, int k, int j, int i) -> Real {
          return rdata[p][((v*ntot + k)*ntot + j)*ntot + i];
        };

        if (nl < mlev && nf == 1) {
          if (ox1 != 0) {
            int fig=(ox1<0)?ngh-1:ngh+ncells;
            int fi=(ox1<0)?ngh:ngh+ncells-1;
            int si=(ox1<0)?ngh+ncells-1:ngh;
            int sj0=ngh+cy*hl, sk0=ngh+cz*hl;
            for (int v=0;v<nvar;++v)
              for (int sk=sk0;sk<sk0+hl;++sk)
                for (int sj=sj0;sj<sj0+hl;++sj) {
                  int fj=ngh+2*(sj-sj0), fk=ngh+2*(sk-sk0);
                  Real cc=R(v,sk,sj,si);
                  int sjm=(sj>ngh)?sj-1:sj, sjp=(sj<ngh+ncells-1)?sj+1:sj;
                  int skm=(sk>ngh)?sk-1:sk, skp=(sk<ngh+ncells-1)?sk+1:sk;
                  Real gy=0.125*(R(v,sk,sjp,si)-R(v,sk,sjm,si));
                  Real gz=0.125*(R(v,skp,sj,si)-R(v,skm,sj,si));
                  u(ml,v,fk,fj,fig)=ot_h*(2.0*(cc-gy-gz)+u(ml,v,fk,fj,fi));
                  u(ml,v,fk,fj+1,fig)=ot_h*(2.0*(cc+gy-gz)+u(ml,v,fk,fj+1,fi));
                  u(ml,v,fk+1,fj,fig)=ot_h*(2.0*(cc-gy+gz)+u(ml,v,fk+1,fj,fi));
                  u(ml,v,fk+1,fj+1,fig)=ot_h*(2.0*(cc+gy+gz)+u(ml,v,fk+1,fj+1,fi));
                }
          } else if (ox2 != 0) {
            int fjg=(ox2<0)?ngh-1:ngh+ncells;
            int fj=(ox2<0)?ngh:ngh+ncells-1;
            int sj=(ox2<0)?ngh+ncells-1:ngh;
            int si0=ngh+cx*hl, sk0=ngh+cz*hl;
            for (int v=0;v<nvar;++v)
              for (int sk=sk0;sk<sk0+hl;++sk)
                for (int si=si0;si<si0+hl;++si) {
                  int fi=ngh+2*(si-si0), fk=ngh+2*(sk-sk0);
                  Real cc=R(v,sk,sj,si);
                  int sim=(si>ngh)?si-1:si, sip=(si<ngh+ncells-1)?si+1:si;
                  int skm=(sk>ngh)?sk-1:sk, skp=(sk<ngh+ncells-1)?sk+1:sk;
                  Real gx=0.125*(R(v,sk,sj,sip)-R(v,sk,sj,sim));
                  Real gz=0.125*(R(v,skp,sj,si)-R(v,skm,sj,si));
                  u(ml,v,fk,fjg,fi)=ot_h*(2.0*(cc-gx-gz)+u(ml,v,fk,fj,fi));
                  u(ml,v,fk,fjg,fi+1)=ot_h*(2.0*(cc+gx-gz)+u(ml,v,fk,fj,fi+1));
                  u(ml,v,fk+1,fjg,fi)=ot_h*(2.0*(cc-gx+gz)+u(ml,v,fk+1,fj,fi));
                  u(ml,v,fk+1,fjg,fi+1)=ot_h*(2.0*(cc+gx+gz)+u(ml,v,fk+1,fj,fi+1));
                }
          } else {
            int fkg=(ox3<0)?ngh-1:ngh+ncells;
            int fk=(ox3<0)?ngh:ngh+ncells-1;
            int sk=(ox3<0)?ngh+ncells-1:ngh;
            int si0=ngh+cx*hl, sj0=ngh+cy*hl;
            for (int v=0;v<nvar;++v)
              for (int sj=sj0;sj<sj0+hl;++sj)
                for (int si=si0;si<si0+hl;++si) {
                  int fi=ngh+2*(si-si0), fj=ngh+2*(sj-sj0);
                  Real cc=R(v,sk,sj,si);
                  int sim=(si>ngh)?si-1:si, sip=(si<ngh+ncells-1)?si+1:si;
                  int sjm=(sj>ngh)?sj-1:sj, sjp=(sj<ngh+ncells-1)?sj+1:sj;
                  Real gx=0.125*(R(v,sk,sj,sip)-R(v,sk,sj,sim));
                  Real gy=0.125*(R(v,sk,sjp,si)-R(v,sk,sjm,si));
                  u(ml,v,fkg,fj,fi)=ot_h*(2.0*(cc-gx-gy)+u(ml,v,fk,fj,fi));
                  u(ml,v,fkg,fj,fi+1)=ot_h*(2.0*(cc+gx-gy)+u(ml,v,fk,fj,fi+1));
                  u(ml,v,fkg,fj+1,fi)=ot_h*(2.0*(cc-gx+gy)+u(ml,v,fk,fj+1,fi));
                  u(ml,v,fkg,fj+1,fi+1)=ot_h*(2.0*(cc+gx+gy)+u(ml,v,fk,fj+1,fi+1));
                }
          }
        } else if (nl > mlev && nf == 1) {
          int sx=0,sy=0,sz=0;
          if (ox1!=0){sy=f1_;sz=f2_;} if (ox2!=0){sx=f1_;sz=f2_;}
          if (ox3!=0){sx=f1_;sy=f2_;}
          int gis,gie,gjs,gje,gks,gke;
          if (ox1<0){gis=0;gie=ngh-1;}
          else if(ox1>0){gis=ngh+ncells;gie=ngh+ncells+ngh-1;}
          else{gis=ngh+sx*hl;gie=ngh+sx*hl+hl-1;}
          if (ox2<0){gjs=0;gje=ngh-1;}
          else if(ox2>0){gjs=ngh+ncells;gje=ngh+ncells+ngh-1;}
          else{gjs=ngh+sy*hl;gje=ngh+sy*hl+hl-1;}
          if (ox3<0){gks=0;gke=ngh-1;}
          else if(ox3>0){gks=ngh+ncells;gke=ngh+ncells+ngh-1;}
          else{gks=ngh+sz*hl;gke=ngh+sz*hl+hl-1;}
          int oi=(ox1<0)?1:(ox1>0)?-1:0;
          int oj=(ox2<0)?1:(ox2>0)?-1:0;
          int ok=(ox3<0)?1:(ox3>0)?-1:0;
          if (ox1!=0) {
            int fi=(ox1>0)?ngh:ngh+ncells-1;
            for (int v=0;v<nvar;++v)
              for (int gk=gks;gk<=gke;++gk)
                for (int gj=gjs;gj<=gje;++gj){
                  int fj0=ngh+2*(gj-(ngh+sy*hl));
                  int fk0=ngh+2*(gk-(ngh+sz*hl));
                  Real fa=0.25*(R(v,fk0,fj0,fi)+R(v,fk0,fj0+1,fi)
                               +R(v,fk0+1,fj0,fi)+R(v,fk0+1,fj0+1,fi));
                  u(ml,v,gk,gj,gis)=ot_h*(4.0*fa-u(ml,v,gk+ok,gj+oj,gis+oi));
                }
          } else if (ox2!=0) {
            int fj=(ox2>0)?ngh:ngh+ncells-1;
            for (int v=0;v<nvar;++v)
              for (int gk=gks;gk<=gke;++gk)
                for (int gi=gis;gi<=gie;++gi){
                  int fi0=ngh+2*(gi-(ngh+sx*hl));
                  int fk0=ngh+2*(gk-(ngh+sz*hl));
                  Real fa=0.25*(R(v,fk0,fj,fi0)+R(v,fk0,fj,fi0+1)
                               +R(v,fk0+1,fj,fi0)+R(v,fk0+1,fj,fi0+1));
                  u(ml,v,gk,gjs,gi)=ot_h*(4.0*fa-u(ml,v,gk+ok,gjs+oj,gi+oi));
                }
          } else {
            int fk=(ox3>0)?ngh:ngh+ncells-1;
            for (int v=0;v<nvar;++v)
              for (int gj=gjs;gj<=gje;++gj)
                for (int gi=gis;gi<=gie;++gi){
                  int fi0=ngh+2*(gi-(ngh+sx*hl));
                  int fj0=ngh+2*(gj-(ngh+sy*hl));
                  Real fa=0.25*(R(v,fk,fj0,fi0)+R(v,fk,fj0,fi0+1)
                               +R(v,fk,fj0+1,fi0)+R(v,fk,fj0+1,fi0+1));
                  u(ml,v,gks,gj,gi)=ot_h*(4.0*fa-u(ml,v,gks+ok,gj+oj,gi+oi));
                }
          }
        } else if (nl < mlev) {
          int gis,gie,gjs,gje,gks,gke;
          if (ox1<0){gis=0;gie=ngh-1;}
          else if(ox1>0){gis=ngh+ncells;gie=ngh+ncells+ngh-1;}
          else{gis=ngh;gie=ngh+ncells-1;}
          if (ox2<0){gjs=0;gje=ngh-1;}
          else if(ox2>0){gjs=ngh+ncells;gje=ngh+ncells+ngh-1;}
          else{gjs=ngh;gje=ngh+ncells-1;}
          if (ox3<0){gks=0;gke=ngh-1;}
          else if(ox3>0){gks=ngh+ncells;gke=ngh+ncells+ngh-1;}
          else{gks=ngh;gke=ngh+ncells-1;}
          for (int v=0;v<nvar;++v)
            for (int gk=gks;gk<=gke;++gk)
              for (int gj=gjs;gj<=gje;++gj)
                for (int gi=gis;gi<=gie;++gi){
                  int si,sj,sk;
                  if (ox1<0)si=ngh+ncells-1; else if(ox1>0)si=ngh;
                  else si=ngh+cx*hl+(gi-ngh)/2;
                  if (ox2<0)sj=ngh+ncells-1; else if(ox2>0)sj=ngh;
                  else sj=ngh+cy*hl+(gj-ngh)/2;
                  if (ox3<0)sk=ngh+ncells-1; else if(ox3>0)sk=ngh;
                  else sk=ngh+cz*hl+(gk-ngh)/2;
                  u(ml,v,gk,gj,gi)=R(v,sk,sj,si);
                }
        } else {
          int sx=0,sy=0,sz=0;
          if (nf==2){if(ox1==0)sx=f1_;if(ox2==0)sy=f1_;if(ox3==0)sz=f1_;}
          int gis,gie,gjs,gje,gks,gke;
          if (ox1<0){gis=0;gie=ngh-1;}
          else if(ox1>0){gis=ngh+ncells;gie=ngh+ncells+ngh-1;}
          else{gis=ngh+sx*hl;gie=ngh+sx*hl+hl-1;}
          if (ox2<0){gjs=0;gje=ngh-1;}
          else if(ox2>0){gjs=ngh+ncells;gje=ngh+ncells+ngh-1;}
          else{gjs=ngh+sy*hl;gje=ngh+sy*hl+hl-1;}
          if (ox3<0){gks=0;gke=ngh-1;}
          else if(ox3>0){gks=ngh+ncells;gke=ngh+ncells+ngh-1;}
          else{gks=ngh+sz*hl;gke=ngh+sz*hl+hl-1;}
          for (int v=0;v<nvar;++v)
            for (int gk=gks;gk<=gke;++gk)
              for (int gj=gjs;gj<=gje;++gj)
                for (int gi=gis;gi<=gie;++gi){
                  int fi0,fi1,fj0,fj1,fk0,fk1;
                  if (ox1<0){fi0=ngh+ncells-2;fi1=ngh+ncells-1;}
                  else if(ox1>0){fi0=ngh;fi1=ngh+1;}
                  else{fi0=ngh+2*(gi-(ngh+sx*hl));fi1=fi0+1;}
                  if (ox2<0){fj0=ngh+ncells-2;fj1=ngh+ncells-1;}
                  else if(ox2>0){fj0=ngh;fj1=ngh+1;}
                  else{fj0=ngh+2*(gj-(ngh+sy*hl));fj1=fj0+1;}
                  if (ox3<0){fk0=ngh+ncells-2;fk1=ngh+ncells-1;}
                  else if(ox3>0){fk0=ngh;fk1=ngh+1;}
                  else{fk0=ngh+2*(gk-(ngh+sz*hl));fk1=fk0+1;}
                  u(ml,v,gk,gj,gi)=0.125*(
                    R(v,fk0,fj0,fi0)+R(v,fk0,fj0,fi1)+
                    R(v,fk0,fj1,fi0)+R(v,fk0,fj1,fi1)+
                    R(v,fk1,fj0,fi0)+R(v,fk1,fj0,fi1)+
                    R(v,fk1,fj1,fi0)+R(v,fk1,fj1,fi1));
                }
        }
      }
      MPI_Waitall(np, sreqs.data(), MPI_STATUSES_IGNORE);
    }
  }
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::PackAndSend()
//! \brief Pack restricted fluxes of multigrid variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors. Adapts to different block sizes per level.

TaskStatus MultigridBoundaryValues::PackAndSendMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = u.extent_int(1);

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  int lev_ = pmy_mg->GetCurrentLevel();
  int nlev_total = pmy_mg->GetNumberOfLevels();
  int shift_ps = nlev_total - 1 - lev_;
  int ncells_ps = pmy_mg->GetSize() >> shift_ps;
  bool skip_fc_this_level = (ncells_ps < 2);
  auto &smgi = send_mg_indcs_;
  auto cbuf = coarse_buf_;

#if MPI_PARALLEL_ENABLED
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0
          && nghbr.h_view(m,n).rank != my_rank) {
        int nlev_h = nghbr.h_view(m,n).lev;
        int mlev_h = mblev.h_view(m);
        bool is_fc = (nlev_h != mlev_h);
        if (is_fc && skip_fc_this_level) continue;
        MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
      }
    }
  }
#endif

  {
  int nmnv = nmb * nnghbr * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("PackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank() / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
    const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

    if (nghbr.d_view(m, n).gid < 0) {
      tmember.team_barrier();
      return;
    }

    int nlev = nghbr.d_view(m, n).lev;
    int mlev = mblev.d_view(m);

    bool is_fc = (nlev != mlev);
    if (is_fc && skip_fc_this_level) {
      tmember.team_barrier();
      return;
    }

    int il, iu, jl, ju, kl, ku;
    bool is_coarser = (nlev < mlev);

    if (nlev == mlev) {
      il = smgi[n][lev_].isame.bis; iu = smgi[n][lev_].isame.bie;
      jl = smgi[n][lev_].isame.bjs; ju = smgi[n][lev_].isame.bje;
      kl = smgi[n][lev_].isame.bks; ku = smgi[n][lev_].isame.bke;
    } else if (is_coarser) {
      il = smgi[n][lev_].icoar.bis; iu = smgi[n][lev_].icoar.bie;
      jl = smgi[n][lev_].icoar.bjs; ju = smgi[n][lev_].icoar.bje;
      kl = smgi[n][lev_].icoar.bks; ku = smgi[n][lev_].icoar.bke;
    } else {
      il = smgi[n][lev_].ifine.bis; iu = smgi[n][lev_].ifine.bie;
      jl = smgi[n][lev_].ifine.bjs; ju = smgi[n][lev_].ifine.bje;
      kl = smgi[n][lev_].ifine.bks; ku = smgi[n][lev_].ifine.bke;
    }

    int ni = iu - il + 1;
    int nj = ju - jl + 1;
    int nk = ku - kl + 1;
    int nkj = nk * nj;

    int dm = nghbr.d_view(m, n).gid - mbgid.d_view(0);
    int dn = nghbr.d_view(m, n).dest;

    if (is_coarser) {
      // Restricted data is pre-computed in coarse_buf_ by FillCoarseMG:
      //   face neighbors  -> face-aligned 2x2 avg in ghost cells
      //   edge/corner     -> volume 2x2x2 avg in interior cells
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
        if (nghbr.d_view(m, n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = cbuf(m, v, k, j, i);
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = cbuf(m, v, k, j, i);
          });
        }
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        if (nghbr.d_view(m, n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = u(m, v, k, j, i);
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = u(m, v, k, j, i);
          });
        }
      });
    }
    tmember.team_barrier();
  });
  }

  #if MPI_PARALLEL_ENABLED
  bool has_cross_rank = false;
  for (int m=0; m<nmb && !has_cross_rank; ++m) {
    for (int n=0; n<nnghbr && !has_cross_rank; ++n) {
      if (nghbr.h_view(m,n).gid >= 0 && nghbr.h_view(m,n).rank != my_rank) {
        int nlev_h = nghbr.h_view(m,n).lev;
        int mlev_h = mblev.h_view(m);
        bool is_fc_h = (nlev_h != mlev_h);
        if (is_fc_h && skip_fc_this_level) continue;
        has_cross_rank = true;
      }
    }
  }
  if (has_cross_rank) Kokkos::fence();

  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid < 0) continue;
      int nlev = nghbr.h_view(m,n).lev;
      int mlev = pmy_pack->pmb->mb_lev.h_view(m);
      bool is_fc_mpi = (nlev != mlev);
      if (is_fc_mpi && skip_fc_this_level) continue;
      {
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          int data_size;
          if (nlev < mlev) {
            data_size = nvar * send_mg_indcs_[n][lev_].icoar_ndat;
          } else if (nlev == mlev) {
            data_size = nvar * send_mg_indcs_[n][lev_].isame_ndat;
          } else {
            data_size = nvar * send_mg_indcs_[n][lev_].ifine_ndat;
          }

          MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);

          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);
          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          pmy_mg->pmy_driver_->mg_timers_.msg_count++;
          pmy_mg->pmy_driver_->mg_timers_.bytes_sent +=
              data_size * static_cast<int64_t>(sizeof(Real));
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValuesCC::RecvAndUnpackMG()
//! \brief Receive and unpack cell-centered multigrid variables.
//! Handles ghost-cell filling at each multigrid level independently.

TaskStatus MultigridBoundaryValues::RecvAndUnpackMG(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  int shift_ru = pmy_mg->GetNumberOfLevels() - 1 - pmy_mg->GetCurrentLevel();
  int ncells_ru = pmy_mg->GetSize() >> shift_ru;
  bool skip_fc_this_level = (ncells_ru < 2);

  #if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  bool bflag = false;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0 && nghbr.h_view(m,n).rank != global_variable::my_rank) {
        int nlev_h = nghbr.h_view(m,n).lev;
        int mlev_h = mblev.h_view(m);
        bool is_fc_h = (nlev_h != mlev_h);
        if (is_fc_h && skip_fc_this_level) continue;
        {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "MPI error in testing non-blocking receives"
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }
          if (!static_cast<bool>(test)) {
            bflag = true;
          }
        }
      }
    }
  }
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack
  int nvar = u.extent_int(1);
  int ngh = pmy_mg->GetGhostCells();
  int lev_ = pmy_mg->GetCurrentLevel();
  auto cbuf = coarse_buf_;
  auto &rmgi = recv_mg_indcs_;

  {
  int nmnv = nmb * nnghbr * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("UnpackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank() / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
    const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

    if (nghbr.d_view(m, n).gid < 0) {
      tmember.team_barrier();
      return;
    }

    int nlev = nghbr.d_view(m, n).lev;
    int mlev = mblev.d_view(m);

    bool is_fc = (nlev != mlev);
    if (is_fc && skip_fc_this_level) {
      tmember.team_barrier();
      return;
    }

    bool from_coarser = (nlev < mlev);

    int il, iu, jl, ju, kl, ku;

    if (nlev == mlev) {
      il = rmgi[n][lev_].isame.bis; iu = rmgi[n][lev_].isame.bie;
      jl = rmgi[n][lev_].isame.bjs; ju = rmgi[n][lev_].isame.bje;
      kl = rmgi[n][lev_].isame.bks; ku = rmgi[n][lev_].isame.bke;
    } else if (from_coarser) {
      il = rmgi[n][lev_].icoar.bis; iu = rmgi[n][lev_].icoar.bie;
      jl = rmgi[n][lev_].icoar.bjs; ju = rmgi[n][lev_].icoar.bje;
      kl = rmgi[n][lev_].icoar.bks; ku = rmgi[n][lev_].icoar.bke;
    } else {
      il = rmgi[n][lev_].ifine.bis; iu = rmgi[n][lev_].ifine.bie;
      jl = rmgi[n][lev_].ifine.bjs; ju = rmgi[n][lev_].ifine.bje;
      kl = rmgi[n][lev_].ifine.bks; ku = rmgi[n][lev_].ifine.bke;
    }

    int ni = iu - il + 1;
    int nj = ju - jl + 1;
    int nk = ku - kl + 1;
    int nkj = nk * nj;

    if (from_coarser) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        [&](const int i) {
          cbuf(m, v, k, j, i) = rbuf[n].vars(m,
              (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
        });
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        [&](const int i) {
          u(m, v, k, j, i) = rbuf[n].vars(m,
              (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
        });
      });
    }
    tmember.team_barrier();
  });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus MultigridBoundaryValues::InitRecvMG(const int nvars) {
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int lev_ = pmy_mg->GetCurrentLevel();
  int shift_ir = pmy_mg->GetNumberOfLevels() - 1 - lev_;
  int ncells_ir = pmy_mg->GetSize() >> shift_ir;
  bool skip_fc_ir = (ncells_ir < 2);

  // Initialize communications of variables
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        int nlev = nghbr.h_view(m,n).lev;
        int mlev = mblev.h_view(m);
        bool is_fc_ir = (nlev != mlev);
        if (is_fc_ir && skip_fc_ir) continue;
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          int data_size;
          if (nlev < mlev) {
            data_size = nvars * recv_mg_indcs_[n][lev_].icoar_ndat;
          } else if (nlev == mlev) {
            data_size = nvars * recv_mg_indcs_[n][lev_].isame_ndat;
          } else {
            data_size = nvars * recv_mg_indcs_[n][lev_].ifine_ndat;
          }

          auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, Kokkos::ALL);

          MPI_Wait(&(recvbuf[n].vars_req[m]), MPI_STATUS_IGNORE);

          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(recvbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

void Multigrid::PrintActiveRegion(const DvceArray5D<Real> &u_in) {
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_in);
  int ll = nlevel_ - 1 - current_level_;
  int ngh = ngh_;  // number of ghost cells
  
  int is = ngh, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh, ke = ks + (indcs_.nx3 >> ll) - 1;
  std::cout<<"nrbx1="<<nmmbx1_<<", nrbx2="<<nmmbx2_<<", nrbx3="<<nmmbx3_<<std::endl;  
  std::cout << "Active region at level " << current_level_ << " (nx=" << (indcs_.nx1 >> ll) << ")\n";
  std::cout << "Range: i=[" << is << "," << ie << "], j=[" << js << "," << je 
            << "], k=[" << ks << "," << ke << "]\n";
  std::cout << "[";
  for (int mz = 0; mz < nmmbx3_/global_variable::nranks; ++mz) {
  for (int k = ks; k <= ks+((ke-ks)/(3-global_variable::nranks)); ++k) {
        std::cout << "[";
        for (int my=0; my < nmmbx2_; ++my) {
          for (int j = js; j <= je; ++j) {
            std::cout << "[";
            for (int mx= 0; mx < nmmbx1_; ++mx) {
              for (int i = is; i <= ie; ++i){
                std::cout << std::setprecision(3) << u_h(mx+my*2+mz*4, 0, k, j, i) << ", ";
              }
            }
            std::cout << "],";
            std::cout << "\n";
          }
        }
        std::cout << "],";
        std::cout << "\n";
    }
  }
  std::cout << "]";
  return;
}

void Multigrid::PrintAll(const DvceArray5D<Real> &u_in) {
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_in);
  int ll = nlevel_ - 1 - current_level_;
  int ngh = ngh_;  // number of ghost cells
  
  int is = 2, ie = is + (indcs_.nx1 >> ll) +2 *ngh - 5;
  int js = 2, je = js + (indcs_.nx2 >> ll) +2 *ngh - 5;
  int ks = 2, ke = ks + (indcs_.nx3 >> ll) +2 *ngh - 5;
  //std::cout<<"nrbx1="<<nmmbx1_<<", nrbx2="<<nmmbx2_<<", nrbx3="<<nmmbx3_<<std::endl;  
  //std::cout << "Whole domain at level " << current_level_ << " (nx=" << (indcs_.nx1 >> ll) << ")\n";
  //std::cout << "Range: i=[" << is << "," << ie << "], j=[" << js << "," << je 
  //          << "], k=[" << ks << "," << ke << "]\n";
  for (int mz = 0; mz < nmmbx3_; ++mz) {
  for (int k = ks+mz; k <= ke+(1-nmmbx3_)+mz; ++k) {
        for (int my=0; my < nmmbx2_; ++my) {
          for (int j = js+my; j <= je+(1-nmmbx2_)+my; ++j) {
            for (int mx= 0; mx < nmmbx1_; ++mx) {
              for (int i = is+mx; i <= ie+(1-nmmbx1_)+mx; ++i){
                std::cout << std::setprecision(3) << u_h(mx+my*2+mz*4, 0, k, j, i) << ", ";
              }
            }
            std::cout << "],";
            std::cout << "\n";
          }
        }
        std::cout << "],";
        std::cout << "\n";
    }
  }
  return;
}