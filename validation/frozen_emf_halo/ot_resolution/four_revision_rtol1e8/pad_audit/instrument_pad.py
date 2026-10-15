"""Add diagnostics to an isolated source copy; leave the numerical update unchanged."""
from pathlib import Path
import sys

p=Path(sys.argv[1])/'src/eos/ideal_mhd.cpp'
s=p.read_text()
anchor='  int nfloord_=0, nfloore_=0, nfloort_=0;'
assert s.count(anchor)==1
s=s.replace(anchor,'''  const auto audit_q = pmy_pack->pmesh->mb_indcs;
  const int audit_cycle = pmy_pack->pmesh->ncycle;
  const Real audit_time = pmy_pack->pmesh->time;
  auto audit_levels = pmy_pack->pmhd->fb_level;
'''+anchor)
anchor='    SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used);'
assert s.count(anchor)==1
s=s.replace(anchor,'''    const auto audit_u = u;
'''+anchor+'''
    if (!only_testfloors && efloor_used && !WB) {
      const bool interior = i>=audit_q.is && i<=audit_q.ie &&
                            j>=audit_q.js && j<=audit_q.je &&
                            k>=audit_q.ks && k<=audit_q.ke;
      const Real ba1 = 0.5*(b.x1f(m,k,j,i)+b.x1f(m,k,j,i+1));
      const Real ba2 = 0.5*(b.x2f(m,k,j,i)+b.x2f(m,k,j+1,i));
      const Real ba3 = 0.5*(b.x3f(m,k,j,i)+b.x3f(m,k+1,j,i));
      const Real ek = 0.5*(SQR(audit_u.mx)+SQR(audit_u.my)+SQR(audit_u.mz))/audit_u.d;
      const Real eho = audit_u.e-ek-0.5*(SQR(audit_u.bx)+SQR(audit_u.by)+SQR(audit_u.bz));
      const Real eavg = audit_u.e-ek-0.5*(SQR(ba1)+SQR(ba2)+SQR(ba3));
      printf("PAD_AUDIT cycle=%d time=%.17e m=%d k=%d j=%d i=%d interior=%d level=%d e_ho=%.17e e_avg=%.17e rho=%.17e\\n",
             audit_cycle,audit_time,m,k,j,i,static_cast<int>(interior),
             audit_levels(m,k,j,i),eho,eavg,audit_u.d);
    }
''')
p.write_text(s)
