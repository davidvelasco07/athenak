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
#include "eos/mhd_bcc.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealMHD::IdealMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
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
  ConsToPrimImpl<false>(cons, cons, b, prim, bcc, only_testfloors,
                        il, iu, jl, ju, kl, ku);
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrimWB()
//! \brief Well-balanced version: cons holds the perturbation U' and ueq the equilibrium;
//! the inversion is done on U'+U_eq (in registers) with the FULL face field b.  Where a
//! floor fires, the corrected full state is written back to cons as a perturbation.

void IdealMHD::ConsToPrimWB(DvceArray5D<Real> &cons, const DvceArray5D<Real> &ueq,
                            const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim,
                            DvceArray5D<Real> &bcc, const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  ConsToPrimImpl<true>(cons, ueq, b, prim, bcc, only_testfloors, il, iu, jl, ju, kl, ku);
}

template <bool WB>
void IdealMHD::ConsToPrimImpl(DvceArray5D<Real> &cons, const DvceArray5D<Real> &ueq,
                              const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim,
                              DvceArray5D<Real> &bcc, const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto recon = pmy_pack->pmhd->recon_method;
  const bool ho_bcc = (pmy_pack->pmhd->emf_method != MHD_EMF::ct_contact);

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  // MOOD touched-cell mask (see EquationOfState::c2p_mask): skip untouched cells
  const bool masked_ = use_c2p_mask && only_testfloors;
  auto mask_ = c2p_mask;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  Kokkos::parallel_reduce("mhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;
    if (masked_ && !mask_(m,k,j,i)) return;

    // load single state conserved variables (full state = perturbation + equilibrium
    // with the well-balanced scheme)
    MHDCons1D u;
    // plain `if` on the template constant: CUDA extended lambdas may not first-capture a
    // variable (cons, ueq) inside an `if constexpr` branch; the dead branch is still removed
    if (WB) {
      u.d  = cons(m,IDN,k,j,i) + ueq(m,IDN,k,j,i);
      u.mx = cons(m,IM1,k,j,i) + ueq(m,IM1,k,j,i);
      u.my = cons(m,IM2,k,j,i) + ueq(m,IM2,k,j,i);
      u.mz = cons(m,IM3,k,j,i) + ueq(m,IM3,k,j,i);
      u.e  = cons(m,IEN,k,j,i) + ueq(m,IEN,k,j,i);
    } else {
      u.d  = cons(m,IDN,k,j,i);
      u.mx = cons(m,IM1,k,j,i);
      u.my = cons(m,IM2,k,j,i);
      u.mz = cons(m,IM3,k,j,i);
      u.e  = cons(m,IEN,k,j,i);
    }

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    } else if (!ho_bcc) {
      // standard 2nd-order face-to-center average (ct_contact EMF path)
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    } else {
      u.bx = mhd_bcc::CellCenteredComponent(recon, b.x1f, m,k,j,i, 0,0,1, il,iu);
      u.by = mhd_bcc::CellCenteredComponent(recon, b.x2f, m,k,j,i, 0,1,0, jl,ju);
      u.bz = mhd_bcc::CellCenteredComponent(recon, b.x3f, m,k,j,i, 1,0,0, kl,ku);
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
      if (WB) {
        if (dfloor_used) {
          cons(m,IDN,k,j,i) = u.d - ueq(m,IDN,k,j,i);
          sumd++;
        }
        if (efloor_used) {
          cons(m,IEN,k,j,i) = u.e - ueq(m,IEN,k,j,i);
          sume++;
        }
        if (tfloor_used) {
          cons(m,IEN,k,j,i) = u.e - ueq(m,IEN,k,j,i);
          sumt++;
        }
      } else {
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

template void IdealMHD::ConsToPrimImpl<false>(DvceArray5D<Real>&,
    const DvceArray5D<Real>&, const DvceFaceFld4D<Real>&, DvceArray5D<Real>&,
    DvceArray5D<Real>&, const bool, const int, const int, const int, const int,
    const int, const int);
template void IdealMHD::ConsToPrimImpl<true>(DvceArray5D<Real>&,
    const DvceArray5D<Real>&, const DvceFaceFld4D<Real>&, DvceArray5D<Real>&,
    DvceArray5D<Real>&, const bool, const int, const int, const int, const int,
    const int, const int);

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
