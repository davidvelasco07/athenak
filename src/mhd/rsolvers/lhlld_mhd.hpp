//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lhlld_mhd.hpp
//! \brief Low-dissipation HLLD (LHLLD) Riemann solver for ideal-gas MHD, Minoshima &
//! Miyoshi (2021, JCP 446, 110639; arXiv:2108.04991), ported from the authors' MLAU
//! reference code (github.com/minoshim/MLAU, include/mhd_flux.h calc_flux_lhlld).
//!
//! Two modifications relative to standard HLLD:
//!  (1) low-Mach ("quasi-all-speed"): the velocity-jump dissipation in the star total
//!      pressure is scaled by phi = mcc*(2-mcc), mcc = min(1, c_conv/c_fast), so it
//!      vanishes as the Mach number -> 0 (accurate low-speed flows);
//!  (2) carbuncle control: the pressure-jump term in the contact speed S_M is scaled by
//!      theta^4, a shock sensor comparing the NORMAL velocity jump to the TRANSVERSE
//!      velocity compression (sdet) of the two adjacent cells (multidimensional
//!      dissipation).  theta=1 (standard) at 1-D normal shocks and in smooth flow.
//!
//! Per-face implementation for the split-kernel path.  ct_contact EMF only (no UCT
//! coefficients produced).  sdet(m,d,k,j,i) = min(v_d(+1)-v_d, v_d-v_d(-1)) per direction
//! d (unused directions filled with a large value so the transverse min ignores them).

#ifndef MHD_RSOLVERS_LHLLD_MHD_HPP_
#define MHD_RSOLVERS_LHLLD_MHD_HPP_

#include <math.h>
#include "coordinates/cell_locations.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn LHLLD<ivx>()
template <int ivx>
KOKKOS_INLINE_FUNCTION
void LHLLD(const EOS_Data &eos,
           const int m, const int k, const int j, const int i,
           const int is, const int js, const int ks,
           const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
           const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
           const DvceArray4D<Real> &bx,
           const DvceArray5D<Real> &flx,
           const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez,
           const DvceArray5D<Real> &sdet) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;
  // sdet direction slots: 0=x,1=y,2=z.  normal = (ivx-IVX); transverse two are cyclic.
  constexpr int nd  = (ivx-IVX);
  constexpr int td1 = ((ivx-IVX)+1)%3;
  constexpr int td2 = ((ivx-IVX)+2)%3;
  // normal-direction cell offset for the "left" adjacent cell of this face
  constexpr int odk = (nd==2) ? 1 : 0;
  constexpr int odj = (nd==1) ? 1 : 0;
  constexpr int odi = (nd==0) ? 1 : 0;
  const Real EPS = 1.0e-40;

  // Load L/R primitive states (ideal gas)
  Real rol = wl(m,IDN,k,j,i);
  Real vnl = wl(m,ivx,k,j,i), vtl = wl(m,ivy,k,j,i), vul = wl(m,ivz,k,j,i);
  Real btl = bl(m,iby,k,j,i), bul = bl(m,ibz,k,j,i);
  Real prl = eos.IdealGasPressure(wl(m,IEN,k,j,i));
  Real ror = wr(m,IDN,k,j,i);
  Real vnr = wr(m,ivx,k,j,i), vtr = wr(m,ivy,k,j,i), vur = wr(m,ivz,k,j,i);
  Real btr = br(m,iby,k,j,i), bur = br(m,ibz,k,j,i);
  Real prr = eos.IdealGasPressure(wr(m,IEN,k,j,i));
  Real bnc = bx(m,k,j,i);
  Real gamma = eos.gamma;
  Real gammam1i = 1.0/(gamma-1.0);
  Real bnc2 = bnc*bnc;
  Real sgn = (bnc > 0.0) ? 1.0 : -1.0;

  // shock-detection inputs: normal velocity jump, transverse velocity compression
  Real dv_n = vnr - vnl;   // reconstructed normal jump across the face
  Real dv_t = fmin(fmin(sdet(m,td1,k,j,i), sdet(m,td1,k-odk,j-odj,i-odi)),
                   fmin(sdet(m,td2,k,j,i), sdet(m,td2,k-odk,j-odj,i-odi)));

  // Left/right derived quantities
  Real roli=1.0/rol;
  Real vl2=vnl*vnl+vtl*vtl+vul*vul;
  Real pml=0.5*(btl*btl+bul*bul);
  Real ptl=prl+pml;
  Real enl=gammam1i*prl+pml+0.5*rol*vl2;
  Real vbl=vtl*btl+vul*bul;
  Real rori=1.0/ror;
  Real vr2=vnr*vnr+vtr*vtr+vur*vur;
  Real pmr=0.5*(btr*btr+bur*bur);
  Real ptr=prr+pmr;
  Real enr=gammam1i*prr+pmr+0.5*ror*vr2;
  Real vbr=vtr*btr+vur*bur;

  // Fast (and convective) wave speeds
  Real cl2=gamma*prl*roli, cr2=gamma*prr*rori;
  Real cal2=bnc2*roli, car2=bnc2*rori;
  Real cbl2=cl2+cal2+2.0*pml*roli;
  Real cbr2=cr2+car2+2.0*pmr*rori;
  Real cfl2=0.5*(cbl2+sqrt(fabs(cbl2*cbl2-4.0*cl2*cal2)));
  Real cfr2=0.5*(cbr2+sqrt(fabs(cbr2*cbr2-4.0*cr2*car2)));
  Real cbl2c=vl2+cal2+2.0*pml*roli;   // sound speed -> convective speed
  Real cbr2c=vr2+car2+2.0*pmr*rori;
  Real ccl2=0.5*(cbl2c+sqrt(fabs(cbl2c*cbl2c-4.0*vl2*cal2)));
  Real ccr2=0.5*(cbr2c+sqrt(fabs(cbr2c*cbr2c-4.0*vr2*car2)));
  Real cmax=sqrt(fmax(cfl2,cfr2));
  Real sl=fmin(0.0,fmin(vnl,vnr)-cmax);
  Real sr=fmax(0.0,fmax(vnl,vnr)+cmax);

  Real slvl=sl-vnl, srvr=sr-vnr;
  Real rslvl=rol*slvl, rsrvr=ror*srvr;
  Real drsvi=1.0/(rsrvr-rslvl);

  // (2) carbuncle shock sensor theta^4
  Real theta=fmin(1.0,(cmax-fmin(dv_n,0.0))/(cmax-fmin(dv_t,0.0)));
  theta*=theta; theta*=theta;
  Real vnc=(rsrvr*vnr-rslvl*vnl-theta*(ptr-ptl))*drsvi;    // contact speed S_M

  // (1) low-Mach factor phi = mcc*(2-mcc) on the velocity-jump term of the star pressure
  Real mcc=fmin(1.0,sqrt(fmax(ccl2,ccr2))/cmax);
  Real ptc=(rsrvr*ptl-rslvl*ptr+(mcc*(2.0-mcc))*rsrvr*rslvl*(vnr-vnl))*drsvi;

  // Outer states of the Riemann fan
  Real slvc=sl-vnc, srvc=sr-vnc;
  Real ro2l=rslvl/slvc, ro2r=rsrvr/srvc;
  Real rhdl=rslvl*slvc-bnc2, rhdr=rsrvr*srvc-bnc2;
  Real vt2l,vu2l,bt2l,bu2l,vt2r,vu2r,bt2r,bu2r;
  if (fabs(rhdl) > EPS) {
    Real rhdli=1.0/rhdl, rhnvl=(vnl-vnc)*bnc, rhnbl=rslvl*slvl-bnc2;
    vt2l=vtl+rhnvl*rhdli*btl; vu2l=vul+rhnvl*rhdli*bul;
    bt2l=rhnbl*rhdli*btl;     bu2l=rhnbl*rhdli*bul;
  } else { vt2l=vtl; vu2l=vul; bt2l=btl; bu2l=bul; }
  if (fabs(rhdr) > EPS) {
    Real rhdri=1.0/rhdr, rhnvr=(vnr-vnc)*bnc, rhnbr=rsrvr*srvr-bnc2;
    vt2r=vtr+rhnvr*rhdri*btr; vu2r=vur+rhnvr*rhdri*bur;
    bt2r=rhnbr*rhdri*btr;     bu2r=rhnbr*rhdri*bur;
  } else { vt2r=vtr; vu2r=vur; bt2r=btr; bu2r=bur; }
  Real vb2l=vt2l*bt2l+vu2l*bu2l, vb2r=vt2r*bt2r+vu2r*bu2r;
  Real en2l=(slvl*enl-ptl*vnl+ptc*vnc+bnc*(vbl-vb2l))/slvc;
  Real en2r=(srvr*enr-ptr*vnr+ptc*vnc+bnc*(vbr-vb2r))/srvc;

  // Inner (rotational) states
  Real rro2l=sqrt(ro2l), rro2r=sqrt(ro2r), rro2i=1.0/(rro2r+rro2l);
  Real vt3m=(rro2r*vt2r+rro2l*vt2l+(bt2r-bt2l)*sgn)*rro2i;
  Real vu3m=(rro2r*vu2r+rro2l*vu2l+(bu2r-bu2l)*sgn)*rro2i;
  Real bt3m=(rro2l*bt2r+rro2r*bt2l+rro2r*rro2l*(vt2r-vt2l)*sgn)*rro2i;
  Real bu3m=(rro2l*bu2r+rro2r*bu2l+rro2r*rro2l*(vu2r-vu2l)*sgn)*rro2i;
  Real vb3m=vt3m*bt3m+vu3m*bu3m;
  Real en3l=en2l-rro2l*(vb2l-vb3m)*sgn;
  Real en3r=en2r+rro2r*(vb2r-vb3m)*sgn;

  // Select the state at the interface (vnc>0 side, inside the Alfven speeds)
  Real hl=(vnc > 0.0)?1.0:0.0, hr=1.0-hl;
  Real h2l=(vnc-fabs(bnc)/rro2l > 0.0)?1.0:0.0, h3l=(1.0-h2l)*hl;
  Real h2r=(vnc+fabs(bnc)/rro2r > 0.0)?0.0:1.0, h3r=(1.0-h2r)*hr;
  Real rou=ro2l*hl+ro2r*hr;
  Real vtu=vt2l*h2l+vt3m*(h3l+h3r)+vt2r*h2r;
  Real vuu=vu2l*h2l+vu3m*(h3l+h3r)+vu2r*h2r;
  Real btu=bt2l*h2l+bt3m*(h3l+h3r)+bt2r*h2r;
  Real buu=bu2l*h2l+bu3m*(h3l+h3r)+bu2r*h2r;
  Real enu=en2l*h2l+en3l*h3l+en3r*h3r+en2r*h2r;

  // Fluxes
  Real fro=rou*vnc;
  Real fmn=rou*vnc*vnc+ptc-bnc2*0.5;
  Real fmt=rou*vtu*vnc-bnc*btu;
  Real fmu=rou*vuu*vnc-bnc*buu;
  Real fbt=btu*vnc-bnc*vtu;
  Real fbu=buu*vnc-bnc*vuu;
  Real fen=(enu+ptc)*vnc-bnc*(vtu*btu+vuu*buu);

  flx(m,IDN,k,j,i) = fro;
  flx(m,ivx,k,j,i) = fmn;
  flx(m,ivy,k,j,i) = fmt;
  flx(m,ivz,k,j,i) = fmu;
  flx(m,IEN,k,j,i) = fen;
  ey(m,k,j,i) = -fbt;
  ez(m,k,j,i) =  fbu;
}

} // namespace mhd
#endif // MHD_RSOLVERS_LHLLD_MHD_HPP_
