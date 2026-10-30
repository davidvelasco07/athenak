
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include "eos/ideal_c2p_mhd.hpp"
#include "coordinates/cartesian_ks.hpp"

int main(int argc, char **argv) {
 Kokkos::initialize(argc,argv);
 int states=0, legacy_rejections=0, component_checks=0, magnitude_checks=0;
 double max_error=0;
 {
  mhd::MoodNadPassCtx c{};
  c.utest_=DvceArray5D<Real>("u",1,5,1,1,1);
  c.bcctest_=DvceArray5D<Real>("b",1,3,1,1,1);
  c.trial_velocity_=DvceArray5D<Real>("v",1,3,1,1,1);
  c.bmn=DvceArray5D<Real>("min",1,8,1,1,1);
  c.bmx=DvceArray5D<Real>("max",1,8,1,1,1);
  c.npass_he=2;c.npass_hb=5;c.npass=8;c.s_v=5;
  c.atol=1e-8;c.eps0=0;c.use_sed=false;
  EOS_Data eos{};eos.gamma_max=1e10;eos.sigma_max=1e20;
  eos.dfloor=1e-14;eos.pfloor=1e-14;eos.tfloor=0;eos.sfloor=0;eos.is_ideal=true;
  for(int metric=0;metric<4;++metric) for(double gam:{4./3.,5./3.})
  for(double rho:{0.01,1.,10.}) for(double b:{0.,0.1,3.})
  for(double speed:{0.,0.2,1.,3.}) {
   eos.gamma=gam;Real gl[4][4],gu[4][4];
   const double xyz[4][3]={{8,2,1},{8,2,1},{3,1,2},{20,-5,1}};
   ComputeMetricAndInverse(xyz[metric][0],xyz[metric][1],xyz[metric][2],metric==0,0.9375,gl,gu);
   MHDPrim1D w{};w.d=rho;w.e=0.3*rho;w.vx=speed;w.vy=-0.3*speed;w.vz=0.2*speed;
   w.bx=b;w.by=-0.7*b;w.bz=0.4*b;
   auto recover=[&](const MHDPrim1D &p) {
    HydCons1D h;SingleP2C_IdealGRMHD(gl,gu,p,gam,h);
    MHDCons1D u{};u.d=h.d;u.e=h.e;u.mx=h.mx;u.my=h.my;u.mz=h.mz;u.bx=p.bx;u.by=p.by;u.bz=p.bz;
    MHDCons1D sr;Real s2,b2,rpar;TransformToSRMHD(u,gl,gu,s2,b2,rpar,sr);
    HydPrim1D q{};bool df=false,ef=false,fail=false;int it=0;
    SingleC2P_IdealSRMHD(sr,eos,s2,b2,rpar,q,df,ef,fail,it);
    assert(!df && !ef && !fail);
    c.utest_(0,IDN,0,0,0)=u.d;c.utest_(0,IEN,0,0,0)=u.e;
    c.utest_(0,IM1,0,0,0)=u.mx;c.utest_(0,IM2,0,0,0)=u.my;c.utest_(0,IM3,0,0,0)=u.mz;
    c.bcctest_(0,0,0,0,0)=u.bx;c.bcctest_(0,1,0,0,0)=u.by;c.bcctest_(0,2,0,0,0)=u.bz;
    c.trial_velocity_(0,0,0,0,0)=q.vx;c.trial_velocity_(0,1,0,0,0)=q.vy;c.trial_velocity_(0,2,0,0,0)=q.vz;
    return q;
   };
   auto q=recover(w);
   const double ww[5]={w.d,w.e,w.vx,w.vy,w.vz},qq[5]={q.d,q.e,q.vx,q.vy,q.vz};
   for(int v=0;v<5;++v){double err=std::abs(qq[v]-ww[v])/std::max(1.,std::abs(ww[v]));max_error=std::max(max_error,err);assert(err<1e-7);}
   for(int v=0;v<3;++v){c.bmn(0,5+v,0,0,0)=ww[2+v]-1e-7;c.bmx(0,5+v,0,0,0)=ww[2+v]+1e-7;}
   c.nad_vmag=false;c.npass=8;c.gr_velocity=true;
   for(int v=0;v<3;++v){assert(!c.Test(0,5+v,0,0,0,1e-8,c.utest_(0,IDN,0,0,0)));++component_checks;}
   c.gr_velocity=false;
   for(int v=0;v<3;++v)legacy_rejections+=c.Test(0,5+v,0,0,0,1e-8,c.utest_(0,IDN,0,0,0));
   c.gr_velocity=true;auto changed=w;changed.vx+=0.25;recover(changed);
   assert(c.Test(0,5,0,0,0,1e-8,c.utest_(0,IDN,0,0,0)));++component_checks;
   recover(w);c.nad_vmag=true;c.npass=6;double mag=sqrt(w.vx*w.vx+w.vy*w.vy+w.vz*w.vz);
   c.bmn(0,5,0,0,0)=mag-1e-7;c.bmx(0,5,0,0,0)=mag+1e-7;
   assert(!c.Test(0,5,0,0,0,1e-8,c.utest_(0,IDN,0,0,0)));++magnitude_checks;
   changed=w;changed.vx+=0.5;recover(changed);assert(c.Test(0,5,0,0,0,1e-8,c.utest_(0,IDN,0,0,0)));++magnitude_checks;
   ++states;
  }
  assert(legacy_rejections>0);
 }
 Kokkos::finalize();
 std::cout<<std::setprecision(17)<<"{\"passed\":true,\"states\":"<<states<<",\"component_checks\":"<<component_checks<<",\"magnitude_checks\":"<<magnitude_checks<<",\"old_formula_rejections_for_identical_state\":"<<legacy_rejections<<",\"maximum_scaled_roundtrip_error\":"<<max_error<<"}\n";
}
