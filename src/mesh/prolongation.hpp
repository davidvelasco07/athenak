#ifndef MESH_PROLONGATION_HPP_
#define MESH_PROLONGATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prolongation.hpp
//! \brief prolongation operators for cell-centered and face-centered variables,
//! implemented as inline functions so they can be used both in Bval and AMR functions.

#include "z4c/z4c.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProlongCC()
//! \brief 2nd-order (piecewise-linear) prolongation operator for cell-centered variables

KOKKOS_INLINE_FUNCTION
void ProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi,
               const bool multi_d, const bool three_d,
               const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  // calculate x1-gradient using the min-mod limiter
  Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
  Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  // calculate x2-gradient using the min-mod limiter
  Real dvar2 = 0.0;
  if (multi_d) {
    dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
    dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // calculate x1-gradient using the min-mod limiter
  Real dvar3 = 0.0;
  if (three_d) {
    dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
    dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // interpolate to the finer grid
  a(m,v,fk,fj,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
  a(m,v,fk,fj,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
  if (multi_d) {
    a(m,v,fk,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
    a(m,v,fk,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
  }
  if (three_d) {
    a(m,v,fk+1,fj  ,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj  ,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongHydroCC()
//! \brief Velocity-bounded, conservative prolongation of the hydro conserved variables of
//! coarse cell (k,j,i) into its 2^dim children.
//!
//! ProlongCC() limits each conserved variable independently, so a child can receive a low
//! density from the density slope but nearly the parent's momentum, i.e. a velocity far
//! outside the range of its coarse neighbours. At a steep density peak this empties the
//! child on its first update (rho -> 0, v -> 1e38). Here density (and, for an ideal EOS, the
//! internal energy density) use the same min-mod slopes as ProlongCC, while the velocity is
//! prolongated with min-mod slopes of the coarse VELOCITY field. A uniform per-parent shift
//! of the child velocities (and of the child total energies) then restores exact momentum
//! (and energy) conservation. If that would leave a child with non-positive density or
//! internal energy, the parent state is copied into every child (piecewise constant).
//! Passive scalars are prolongated with ProlongCC().

KOKKOS_INLINE_FUNCTION
Real ProlongMinmodQuarter(const Real dl, const Real dr) {
  // same limited slope, scaled to the child offset, as used in ProlongCC()
  return 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
}

KOKKOS_INLINE_FUNCTION
void ProlongHydroCC(const int m, const int k, const int j, const int i,
                    const int fk, const int fj, const int fi,
                    const bool multi_d, const bool three_d,
                    const int nhydro, const int nscalars, const bool is_ideal,
                    const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  const int nc2 = multi_d ? 2 : 1;
  const int nc3 = three_d ? 2 : 1;
  const int nchild = 2*nc2*nc3;

  // coarse primitive-like fields at the parent (index 0) and its 6 face neighbours
  // (1:-i 2:+i 3:-j 4:+j 5:-k 6:+k)
  const int ok[7] = {0, 0, 0, 0, 0, -1, 1};
  const int oj[7] = {0, 0, 0, -1, 1, 0, 0};
  const int oi[7] = {0, -1, 1, 0, 0, 0, 0};
  const int nstencil = three_d ? 7 : (multi_d ? 5 : 3);
  Real d[7], vel[3][7], eint[7];
  for (int s=0; s<nstencil; ++s) {
    const int kk = k + ok[s], jj = j + oj[s], ii = i + oi[s];
    d[s] = ca(m,IDN,kk,jj,ii);
    Real ke = 0.0;
    for (int n=0; n<3; ++n) {
      vel[n][s] = ca(m,IM1+n,kk,jj,ii)/d[s];
      ke += 0.5*ca(m,IM1+n,kk,jj,ii)*vel[n][s];
    }
    eint[s] = is_ideal ? (ca(m,IEN,kk,jj,ii) - ke) : 0.0;
  }

  // limited slopes (already scaled to the child offset) of each field
  auto slope = [&](const Real *f, const int dir) -> Real {
    return ProlongMinmodQuarter(f[0] - f[2*dir+1], f[2*dir+2] - f[0]);
  };
  Real sd[3] = {slope(d, 0), multi_d ? slope(d, 1) : 0.0, three_d ? slope(d, 2) : 0.0};
  Real se[3] = {0.0, 0.0, 0.0};
  if (is_ideal) {
    se[0] = slope(eint, 0);
    se[1] = multi_d ? slope(eint, 1) : 0.0;
    se[2] = three_d ? slope(eint, 2) : 0.0;
  }
  Real sv[3][3];
  for (int n=0; n<3; ++n) {
    sv[n][0] = slope(vel[n], 0);
    sv[n][1] = multi_d ? slope(vel[n], 1) : 0.0;
    sv[n][2] = three_d ? slope(vel[n], 2) : 0.0;
  }

  // child states before the conservation correction
  Real rc[8], vc[3][8], ec[8];
  Real sum_rv[3] = {0.0, 0.0, 0.0};
  int c = 0;
  for (int ck=0; ck<nc3; ++ck) {
    for (int cj=0; cj<nc2; ++cj) {
      for (int ci=0; ci<2; ++ci) {
        const Real s1 = (ci == 0) ? -1.0 : 1.0;
        const Real s2 = (cj == 0) ? -1.0 : 1.0;
        const Real s3 = (ck == 0) ? -1.0 : 1.0;
        rc[c] = d[0] + s1*sd[0] + s2*sd[1] + s3*sd[2];
        ec[c] = eint[0] + s1*se[0] + s2*se[1] + s3*se[2];
        for (int n=0; n<3; ++n) {
          vc[n][c] = vel[n][0] + s1*sv[n][0] + s2*sv[n][1] + s3*sv[n][2];
          sum_rv[n] += rc[c]*vc[n][c];
        }
        ++c;
      }
    }
  }

  // uniform velocity shift restoring sum(child momentum) = nchild * parent momentum
  // (sum of child densities = nchild * parent density, since the slopes are symmetric)
  Real dv[3];
  for (int n=0; n<3; ++n) {
    dv[n] = (nchild*ca(m,IM1+n,k,j,i) - sum_rv[n])/(nchild*d[0]);
  }
  bool ok_state = (d[0] > 0.0);
  Real etot[8];
  Real de = 0.0;
  for (int cc=0; cc<nchild; ++cc) {
    ok_state = ok_state && (rc[cc] > 0.0);
    if (is_ideal) {
      Real ke = 0.0;
      for (int n=0; n<3; ++n) {
        const Real v = vc[n][cc] + dv[n];
        ke += 0.5*rc[cc]*v*v;
      }
      etot[cc] = ec[cc] + ke;
      de += etot[cc];
    }
  }
  if (is_ideal) {
    // uniform energy shift restoring sum(child total energy) = nchild * parent energy
    de = (nchild*ca(m,IEN,k,j,i) - de)/nchild;
    for (int cc=0; cc<nchild; ++cc) {ok_state = ok_state && (ec[cc] + de > 0.0);}
  }

  c = 0;
  for (int ck=0; ck<nc3; ++ck) {
    for (int cj=0; cj<nc2; ++cj) {
      for (int ci=0; ci<2; ++ci) {
        if (ok_state) {
          a(m,IDN,fk+ck,fj+cj,fi+ci) = rc[c];
          for (int n=0; n<3; ++n) {
            a(m,IM1+n,fk+ck,fj+cj,fi+ci) = rc[c]*(vc[n][c] + dv[n]);
          }
          if (is_ideal) {a(m,IEN,fk+ck,fj+cj,fi+ci) = etot[c] + de;}
        } else {
          // fall back to piecewise-constant prolongation of the whole hydro state
          for (int v=0; v<nhydro; ++v) {a(m,v,fk+ck,fj+cj,fi+ci) = ca(m,v,k,j,i);}
        }
        ++c;
      }
    }
  }

  for (int v=nhydro; v<nhydro+nscalars; ++v) {
    ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX1Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X1-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX1Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d, const bool three_d,
                   const DvceArray4D<Real> &cbx1f, const DvceArray4D<Real> &bx1f) {
  // Prolongate b.x1f (v=0) by interpolating in x2/x3
  Real dvar2 = 0.0;
  if (multi_d) {
    Real dl = cbx1f(m,k,j  ,i) - cbx1f(m,k,j-1,i);
    Real dr = cbx1f(m,k,j+1,i) - cbx1f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  Real dvar3 = 0.0;
  if (three_d) {
    Real dl = cbx1f(m,k  ,j,i) - cbx1f(m,k-1,j,i);
    Real dr = cbx1f(m,k+1,j,i) - cbx1f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx1f(m,fk,fj,fi) = cbx1f(m,k,j,i) - dvar2 - dvar3;
  if (multi_d) {
    bx1f(m,fk,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 - dvar3;
  }
  if (three_d) {
    bx1f(m,fk+1,fj  ,fi) = cbx1f(m,k,j,i) - dvar2 + dvar3;
    bx1f(m,fk+1,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX2Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X2-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX2Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool three_d,
                   const DvceArray4D<Real> &cbx2f, const DvceArray4D<Real> &bx2f) {
  // Prolongate b.x2f (v=1) by interpolating in x1/x3
  Real dl = cbx2f(m,k,j,i  ) - cbx2f(m,k,j,i-1);
  Real dr = cbx2f(m,k,j,i+1) - cbx2f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar3 = 0.0;
  if (three_d) {
    dl = cbx2f(m,k  ,j,i) - cbx2f(m,k-1,j,i);
    dr = cbx2f(m,k+1,j,i) - cbx2f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx2f(m,fk  ,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 - dvar3;
  bx2f(m,fk  ,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 - dvar3;
  if (three_d) {
    bx2f(m,fk+1,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 + dvar3;
    bx2f(m,fk+1,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX3Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X3-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX3Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d,
                   const DvceArray4D<Real> &cbx3f, const DvceArray4D<Real> &bx3f) {
  // Prolongate b.x3f (v=2) by interpolating in x1/x2
  Real dl = cbx3f(m,k,j,i  ) - cbx3f(m,k,j,i-1);
  Real dr = cbx3f(m,k,j,i+1) - cbx3f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar2 = 0.0;
  if (multi_d) {
    dl = cbx3f(m,k,j  ,i) - cbx3f(m,k,j-1,i);
    dr = cbx3f(m,k,j+1,i) - cbx3f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx3f(m,fk,fj  ,fi  ) = cbx3f(m,k,j,i) - dvar1 - dvar2;
  bx3f(m,fk,fj  ,fi+1) = cbx3f(m,k,j,i) + dvar1 - dvar2;
  if (multi_d) {
    bx3f(m,fk,fj+1,fi  ) = cbx3f(m,k,j,i) - dvar1 + dvar2;
    bx3f(m,fk,fj+1,fi+1) = cbx3f(m,k,j,i) + dvar1 + dvar2;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongInternalFC()
//! \brief 2nd-order prolongation operator for face-centered variables on internal edges
//! of new fine cells within one coarse cell using divergence-preserving interpolation
//! scheme of Toth & Roe, JCP 180, 736 (2002).

KOKKOS_INLINE_FUNCTION
void ProlongFCInternal(const int m, const int fk, const int fj, const int fi,
                       const bool three_d, const DvceFaceFld4D<Real> &b) {
  // Prolongate internal fields in 3D
  if (three_d) {
    Real Uxx  = 0.0, Vyy  = 0.0, Wzz  = 0.0;
    Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
    for (int jj=0; jj<2; jj++) {
      int jsgn = 2*jj - 1;
      int fjj  = fj + jj, fjp = fj + 2*jj;
      for (int ii=0; ii<2; ii++) {
        int isgn = 2*ii - 1;
        int fii = fi + ii, fip = fi + 2*ii;
        Uxx += isgn*(jsgn*(b.x2f(m,fk  ,fjp,fii) + b.x2f(m,fk+1,fjp,fii)) +
                          (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)));

        Vyy += jsgn*(     (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)) +
                     isgn*(b.x1f(m,fk  ,fjj,fip) + b.x1f(m,fk+1,fjj,fip)));

        Wzz +=       isgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip)) +
                     jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));

        Uxyz += isgn*jsgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip));
        Vxyz += isgn*jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));
        Wxyz += isgn*jsgn*(b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii));
      }
    }
    Uxx *= 0.125;  Vyy *= 0.125;  Wzz *= 0.125;
    Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

    b.x1f(m,fk  ,fj  ,fi+1) = 0.5*(b.x1f(m,fk  ,fj  ,fi  ) + b.x1f(m,fk  ,fj  ,fi+2))
                            + Uxx - Vxyz - Wxyz;
    b.x1f(m,fk  ,fj+1,fi+1) = 0.5*(b.x1f(m,fk  ,fj+1,fi  ) + b.x1f(m,fk  ,fj+1,fi+2))
                            + Uxx - Vxyz + Wxyz;
    b.x1f(m,fk+1,fj  ,fi+1) = 0.5*(b.x1f(m,fk+1,fj  ,fi  ) + b.x1f(m,fk+1,fj  ,fi+2))
                            + Uxx + Vxyz - Wxyz;
    b.x1f(m,fk+1,fj+1,fi+1) = 0.5*(b.x1f(m,fk+1,fj+1,fi  ) + b.x1f(m,fk+1,fj+1,fi+2))
                            + Uxx + Vxyz + Wxyz;
    b.x2f(m,fk  ,fj+1,fi  ) = 0.5*(b.x2f(m,fk  ,fj  ,fi  ) + b.x2f(m,fk  ,fj+2,fi  ))
                            + Vyy - Uxyz - Wxyz;
    b.x2f(m,fk  ,fj+1,fi+1) = 0.5*(b.x2f(m,fk  ,fj  ,fi+1) + b.x2f(m,fk  ,fj+2,fi+1))
                            + Vyy - Uxyz + Wxyz;
    b.x2f(m,fk+1,fj+1,fi  ) = 0.5*(b.x2f(m,fk+1,fj  ,fi  ) + b.x2f(m,fk+1,fj+2,fi  ))
                            + Vyy + Uxyz - Wxyz;
    b.x2f(m,fk+1,fj+1,fi+1) = 0.5*(b.x2f(m,fk+1,fj  ,fi+1) + b.x2f(m,fk+1,fj+2,fi+1))
                            + Vyy + Uxyz + Wxyz;
    b.x3f(m,fk+1,fj  ,fi  ) = 0.5*(b.x3f(m,fk+2,fj  ,fi  ) + b.x3f(m,fk  ,fj  ,fi  ))
                            + Wzz - Uxyz - Vxyz;
    b.x3f(m,fk+1,fj  ,fi+1) = 0.5*(b.x3f(m,fk+2,fj  ,fi+1) + b.x3f(m,fk  ,fj  ,fi+1))
                            + Wzz - Uxyz + Vxyz;
    b.x3f(m,fk+1,fj+1,fi  ) = 0.5*(b.x3f(m,fk+2,fj+1,fi  ) + b.x3f(m,fk  ,fj+1,fi  ))
                            + Wzz + Uxyz - Vxyz;
    b.x3f(m,fk+1,fj+1,fi+1) = 0.5*(b.x3f(m,fk+2,fj+1,fi+1) + b.x3f(m,fk  ,fj+1,fi+1))
                            + Wzz + Uxyz + Vxyz;

  // Prolongate internal fields in 2D
  } else {
    Real tmp1 = 0.25*(b.x2f(m,fk,fj+2,fi+1) - b.x2f(m,fk,fj,  fi+1)
                    - b.x2f(m,fk,fj+2,fi  ) + b.x2f(m,fk,fj,  fi  ));
    Real tmp2 = 0.25*(b.x1f(m,fk,fj,  fi  ) - b.x1f(m,fk,fj,  fi+2)
                    - b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2));
    b.x1f(m,fk,fj  ,fi+1) = 0.5*(b.x1f(m,fk,fj,  fi  ) + b.x1f(m,fk,fj,  fi+2)) + tmp1;
    b.x1f(m,fk,fj+1,fi+1) = 0.5*(b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2)) + tmp1;
    b.x2f(m,fk,fj+1,fi  ) = 0.5*(b.x2f(m,fk,fj,  fi  ) + b.x2f(m,fk,fj+2,fi  )) + tmp2;
    b.x2f(m,fk,fj+1,fi+1) = 0.5*(b.x2f(m,fk,fj,  fi+1) + b.x2f(m,fk,fj+2,fi+1)) + tmp2;
  }
  return;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real ProlongInterpolation(const int m, const int v, int k, int j, int i,
                            const int nx1, const int nx2, const int nx3,
                            const bool offsetk, const bool offsetj, const bool offseti,
                        const DvceArray5D<Real> &ca, const DualArray3D<Real> &weights) {
  // interpolated value at new grid point
  Real ivals = 0;

  for (int kk=0; kk<NGHOST+1; kk++) {
    for (int jj=0; jj<NGHOST+1; jj++) {
      for (int ii=0; ii<NGHOST+1; ii++) {
        int wghti = (offseti) ? NGHOST-ii : ii;
        int wghtj = (offsetj) ? NGHOST-jj : jj;
        int wghtk = (offsetk) ? NGHOST-kk : kk;
        ivals += weights.d_view(wghtk,wghtj,wghti)*ca(m,v,
                    k-NGHOST/2+kk,j-NGHOST/2+jj,i-NGHOST/2+ii);
      }
    }
  }

  return ivals;
}

//----------------------------------------------------------------------------------------
//! \fn HighOrderProlongCC()
//! \brief high-order prolongation operator for cell-centered variables

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void HighOrderProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi, const int nx1, const int nx2,
               const int nx3, const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a,
               const DualArray3D<Real> &weights) {
  // stencil size for interpolator
  a(m,v,fk  ,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false,false, ca, weights);
  a(m,v,fk  ,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false, true, ca, weights);
  a(m,v,fk  ,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true,false, ca, weights);
  a(m,v,fk  ,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true, true, ca, weights);
  a(m,v,fk+1,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false,false, ca, weights);
  a(m,v,fk+1,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false, true, ca, weights);
  a(m,v,fk+1,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true,false, ca, weights);
  a(m,v,fk+1,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true, true, ca, weights);
  return;
}

#endif // MESH_PROLONGATION_HPP_

