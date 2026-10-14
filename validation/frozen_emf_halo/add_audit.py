from pathlib import Path
import re
r=Path(__file__).parent/'source'
p=r/'src/outputs/binary.cpp'
# Preserve doubles in the experiment's field outputs; the existing reader handles both.
p.write_text(re.sub(r'\bfloat\b','double',p.read_text()))
p=r/'src/mhd/mhd.hpp'
s=p.read_text().replace('  int mood_test_seed = -1;', '  int mood_test_seed = -1;\n  bool mood_test_dump = false;')
p.write_text(s)
p=r/'src/mhd/mhd.cpp'
s=p.read_text().replace('    mood_test_seed = pin->GetOrAddInteger', '    mood_test_dump = pin->GetOrAddBoolean("mhd", "mood_test_dump", false);\n    mood_test_seed = pin->GetOrAddInteger')
p.write_text(s)
p=r/'src/mhd/mhd_mood.cpp'
s=p.read_text().replace('#include <type_traits>', '#include <type_traits>\n#include <fstream>\n#include "globals.hpp"')
point='  int ndemoted_total = 0;'
audit='''  const int audit_seed = mood_test_seed;
  const int audit_nx = pmy_pack->pmesh->mesh_indcs.nx1;
  const int audit_ny = pmy_pack->pmesh->mesh_indcs.nx2;
  const int audit_nz = pmy_pack->pmesh->mesh_indcs.nx3;
  const Real audit_xmin = pmy_pack->pmesh->mesh_size.x1min;
  const Real audit_ymin = pmy_pack->pmesh->mesh_size.x2min;
  const Real audit_zmin = pmy_pack->pmesh->mesh_size.x3min;
  auto audit_dump = [&](int rev, int h) {
    if (!mood_test_dump) return;
    auto U=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),utest_);
    auto B=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bcctest_);
    auto L=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),fb_level_);
    std::string name="audit_c"+std::to_string(pmy_pack->pmesh->ncycle)
      +"_s"+std::to_string(stage)+"_r"+std::to_string(rev)
      +"_rank"+std::to_string(global_variable::my_rank)+".dat";
    std::ofstream out(name,std::ios::binary);
    for (int m=0;m<nmb;++m) {
      auto loc=pmy_pack->pmesh->lloc_eachmb[pmy_pack->pmb->mb_gid.h_view(m)];
      for (int k=three_d?ks-h:ks;k<=(three_d?ke+h:ks);++k)
      for (int j=js-h;j<=je+h;++j) for (int i=is-h;i<=ie+h;++i) {
        double row[13];
        row[0]=loc.lx1*indcs_.nx1+i-is;
        row[1]=loc.lx2*indcs_.nx2+j-js;
        row[2]=loc.lx3*indcs_.nx3+k-ks;
        for (int n=0;n<5;++n) row[3+n]=U(m,n,k,j,i);
        for (int n=0;n<3;++n) row[8+n]=B(m,n,k,j,i);
        row[11]=L(m,k,j,i);
        row[12]=(i>=is&&i<=ie&&j>=js&&j<=je&&k>=ks&&k<=ke)?1:0;
        out.write(reinterpret_cast<const char*>(row),sizeof(row));
      }
    }
  };
'''
assert s.count(point)==1
s=s.replace(point,audit+point)
point='      fofc_(m,k,j,i) = trouble;'
assert s.count(point)==1
s=s.replace(point,'''      // Test-only forced one-cell repair, or -2 to suppress all repairs.
      if (audit_seed != -1) {
        int gx=static_cast<int>(round((size_.d_view(m).x1min-audit_xmin)/size_.d_view(m).dx1))+i-is;
        int gy=static_cast<int>(round((size_.d_view(m).x2min-audit_ymin)/size_.d_view(m).dx2))+j-js;
        int gz=three_d ? static_cast<int>(round((size_.d_view(m).x3min-audit_zmin)/size_.d_view(m).dx3))+k-ks : 0;
        gx=(gx%audit_nx+audit_nx)%audit_nx;
        gy=(gy%audit_ny+audit_ny)%audit_ny;
        gz=(gz%audit_nz+audit_nz)%audit_nz;
        trouble=(audit_seed>=0 && rev==1 && (gz*audit_ny+gy)*audit_nx+gx==audit_seed);
      }
'''+point)
point='    if (ndemoted == 0) break;'
assert s.count(point)==1
s=s.replace(point,'    audit_dump(rev, dh);\n'+point)
p.write_text(s)
print('Added stage/halo snapshots and double-precision field outputs')
