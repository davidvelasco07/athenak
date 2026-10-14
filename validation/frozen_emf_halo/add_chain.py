from pathlib import Path
p=Path(__file__).parent/'source/src/mhd/mhd_mood.cpp'
s=p.read_text()
old='  const int audit_seed = mood_test_seed;'
new=old+'''
  // seed <= -3 enables an adversarial detector: after the first seed, every
  // cell whose candidate changed from the frozen base is rejected. This tests
  // the full dependency cone without relying on a shock to trigger every pass.
  DvceArray5D<Real> audit_base_u, audit_base_b;
  if (audit_seed <= -3) {
    audit_base_u=DvceArray5D<Real>("audit_base_u",utest_.layout());
    audit_base_b=DvceArray5D<Real>("audit_base_b",bcctest_.layout());
  }
'''
assert s.count(old)==1;s=s.replace(old,new)
old='    int ndemoted = 0, ndemoted_int = 0;'
new='''    if (audit_seed <= -3 && rev == 1) {
      Kokkos::deep_copy(audit_base_u,utest_);
      Kokkos::deep_copy(audit_base_b,bcctest_);
    }
'''+old
assert s.count(old)==1;s=s.replace(old,new)
old='        trouble=(audit_seed>=0 && rev==1 && (gz*audit_ny+gy)*audit_nx+gx==audit_seed);'
new='''        const int seed_cell=(audit_seed <= -3) ? -audit_seed-3 : audit_seed;
        trouble=(seed_cell>=0 && rev==1 && (gz*audit_ny+gy)*audit_nx+gx==seed_cell);
        if (audit_seed <= -3 && rev > 1) {
          for (int n=0;n<nmhd_;++n) {
            trouble=trouble || (utest_(m,n,k,j,i)!=audit_base_u(m,n,k,j,i));
          }
          for (int n=0;n<3;++n) {
            trouble=trouble || (bcctest_(m,n,k,j,i)!=audit_base_b(m,n,k,j,i));
          }
        }'''
assert s.count(old)==1;s=s.replace(old,new)
p.write_text(s)
