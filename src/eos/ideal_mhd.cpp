//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic mhd

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"

//----------------------------------------------------------------------------------------
//! \fn FaceToCenterInterp()
//! \brief Interpolate face-centered B to cell center using high-order reconstruction.
//! Treats face values as "cell averages" in a dual grid, then reconstructs to the
//! midpoint (cell center of the primal grid).  This ensures consistency with the
//! finite-volume reconstruction used elsewhere in the code.

KOKKOS_INLINE_FUNCTION
Real FaceToCenterInterp(const ReconstructionMethod recon,
                        const Real &bf_im2, const Real &bf_im1, const Real &bf_i,
                        const Real &bf_ip1, const Real &bf_ip2, const Real &bf_ip3) {
  // bf values are face-centered at positions i-3/2, i-1/2, i+1/2, i+3/2, i+5/2, i+7/2
  // (mapped to face indices i-2, i-1, i, i+1, i+2, i+3)
  // Cell center i is the midpoint between faces i (pos i-1/2) and i+1 (pos i+1/2)
  // Reconstruction from face "cells" to the "face" at cell center i:
  //   qL: left state at cell center from WENOZ centered on face i
  //   qR: right state at cell center from WENOZ centered on face i+1
  // Cell center i is midpoint between face i (bf_i at x_{i-1/2}) and
  // face i+1 (bf_ip1 at x_{i+1/2}).  Reconstruction from dual grid of face values:
  //   Left state at cell center: ql_ip1 from recon centered on face i
  //   Right state at cell center: qr_{i+1} from recon centered on face i+1
  Real qL, qR, dum;
  switch (recon) {
    case ReconstructionMethod::dc:
      return 0.5*(bf_i + bf_ip1);
    case ReconstructionMethod::plm:
      PLM(bf_im1, bf_i, bf_ip1, qL, dum);    // center face i → qL at cell center
      PLM(bf_i, bf_ip1, bf_ip2, dum, qR);    // center face i+1 → qR at cell center
      return 0.5*(qL + qR);
    case ReconstructionMethod::ppm4:
    case ReconstructionMethod::ppmx:
      PPM4(bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, qL, dum);  // center face i → qL at cc
      PPM4(bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3, dum, qR);  // center face i+1 → qR at cc
      return 0.5*(qL + qR);
    case ReconstructionMethod::wenoz:
      WENOZ(bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, qL, dum);  // center face i → qL at cc
      WENOZ(bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3, dum, qR);  // center face i+1 → qR at cc
      return 0.5*(qL + qR);
  }
  return 0.5*(bf_i + bf_ip1);  // fallback
}

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealMHD::IdealMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.sigma_max = pin->GetOrAddReal("mhd","sigma_max",(FLT_MAX));  // sigma ceiling
}

//----------------------------------------------------------------------------------------
//! \!fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.

void IdealMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto recon = pmy_pack->pmhd->recon_method;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  Kokkos::parallel_reduce("mhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    } else {
      // High-order face-to-center interpolation using the same reconstruction
      // method as the flux computation.  Treats face-averaged B values as "cell
      // averages" in a dual grid, consistent with the FV framework.
      // Falls back to 2nd-order at domain boundaries where stencil doesn't fit.
      if (i > il+1 && i < iu-1) {
        u.bx = FaceToCenterInterp(recon,
          b.x1f(m,k,j,i-2), b.x1f(m,k,j,i-1), b.x1f(m,k,j,i),
          b.x1f(m,k,j,i+1), b.x1f(m,k,j,i+2), b.x1f(m,k,j,i+3));
      } else if (i > il && i < iu) {
        u.bx = (7.0/12.0)*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1))
             - (1.0/12.0)*(b.x1f(m,k,j,i-1) + b.x1f(m,k,j,i+2));
      } else {
        u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      }
      if (j > jl+1 && j < ju-1) {
        u.by = FaceToCenterInterp(recon,
          b.x2f(m,k,j-2,i), b.x2f(m,k,j-1,i), b.x2f(m,k,j,i),
          b.x2f(m,k,j+1,i), b.x2f(m,k,j+2,i), b.x2f(m,k,j+3,i));
      } else if (j > jl && j < ju) {
        u.by = (7.0/12.0)*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i))
             - (1.0/12.0)*(b.x2f(m,k,j-1,i) + b.x2f(m,k,j+2,i));
      } else {
        u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      }
      if (k > kl+1 && k < ku-1) {
        u.bz = FaceToCenterInterp(recon,
          b.x3f(m,k-2,j,i), b.x3f(m,k-1,j,i), b.x3f(m,k,j,i),
          b.x3f(m,k+1,j,i), b.x3f(m,k+2,j,i), b.x3f(m,k+3,j,i));
      } else if (k > kl && k < ku) {
        u.bz = (7.0/12.0)*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i))
             - (1.0/12.0)*(b.x3f(m,k-1,j,i) + b.x3f(m,k+2,j,i));
      } else {
        u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
      }
    }

    // call c2p function
    // (inline function in ideal_c2p_mhd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used);

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || tfloor_used) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      // update counter, reset conserved if floor was hit
      if (dfloor_used) {
        cons(m,IDN,k,j,i) = u.d;
        sumd++;
      }
      if (efloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sume++;
      }
      if (tfloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sumt++;
      }
      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;
      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        // apply scalar floor
        if (cons(m,n,k,j,i) < 0.0) {
          cons(m,n,k,j,i) = 0.0;
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void PrimToCons()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.  Does not change cell- or face-centered magnetic fields.

void IdealMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons, const int il, const int iu,
                          const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  par_for("mhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // load single state primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealMHD(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any), always stored at end of cons and prim arrays.
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
