from pathlib import Path

root = Path(__file__).parent / 'source'
def edit(name, old, new):
    p = root / name
    s = p.read_text()
    assert s.count(old) == 1, (name, old[:100], s.count(old))
    p.write_text(s.replace(old, new))

edit('src/mhd/mhd.hpp', '  int mood_halo0;', '''  int mood_halo0;
  // Experimental frozen-EMF halo audit, opt-in; uniform Newtonian ideal MHD only.
  bool mood_frozen_emf = false;
  int mood_test_halo = -1;
  int mood_test_seed = -1;''')
edit('src/mhd/mhd.cpp', '    mood_sed = pin->GetOrAddBoolean', '''    mood_frozen_emf = pin->GetOrAddBoolean("mhd", "mood_frozen_emf", false);
    mood_test_halo = pin->GetOrAddInteger("mhd", "mood_test_halo", -1);
    mood_test_seed = pin->GetOrAddInteger("mhd", "mood_test_seed", -1);
    mood_sed = pin->GetOrAddBoolean''')
edit('src/mhd/mhd.cpp', '      mood_halo0 = h_f*mood_max_revs;', '''      mood_halo0 = h_f*mood_max_revs;
      if (mood_frozen_emf) {
        if (!is_uct || !mood_edge_flag || mood_touched_only ||
            !pmy_pack->pmesh->multi_d || pmy_pack->pmesh->multilevel ||
            pmy_pack->pcoord->is_special_relativistic ||
            pmy_pack->pcoord->is_general_relativistic) {
          std::cerr << "Frozen EMF audit requires uniform Newtonian multidimensional UCT,"
                    << " mood_edge=flag and mood_touched_only=false\\n";
          std::exit(EXIT_FAILURE);
        }
        mood_halo0 = (mood_test_halo >= 0) ? mood_test_halo : mood_max_revs;
      }''')
edit('src/mhd/mhd.cpp', '      auto &indcs = pmy_pack->pmesh->mb_indcs;\n      if (indcs.ng < ng_need)', '''      // Audit floor protects array accesses. Decomposition consistency may need
      // more cells because face-to-center interpolation loses validity near ghosts.
      if (mood_frozen_emf) {
        ng_need = mood_halo0 + std::max(3, ReconEdgeReach(recon_method));
      }
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < ng_need)''')
edit('src/mhd/mhd_tasks.cpp', '  CornerE(pdrive, stage);', '''  if (!mood_frozen_emf) CornerE(pdrive, stage);''')

p = root/'src/mhd/mhd_mood.cpp'
s = p.read_text()
start = s.index('    // x1 faces\n', s.index('// (4) revise every face'))
end = s.index('\n  }\n\n  pmy_pack->pmesh->ecounter.nmood', start)
revision = s[start:end]
# Reuse the actual fallback Riemann solve to precompute complete tier EMFs.
for axis, letter in [(1,'i'),(2,'j'),(3,'k')]:
    loc = {'i':('m,k,j,i-1','m,k,j,i'),
           'j':('m,k,j-1,i','m,k,j,i'),
           'k':('m,k-1,j,i','m,k,j,i')}[letter]
    revision = revision.replace(f'if (fofc_({loc[0]}) || fofc_({loc[1]}))',
                                f'if (force_tier > 0 || fofc_({loc[0]}) || fofc_({loc[1]}))')
revision = revision.replace('int tier = (lv_l > lv_r) ? lv_l : lv_r;',
    'int tier = (force_tier > 0) ? force_tier : ((lv_l > lv_r) ? lv_l : lv_r);')
# Forced tier: face-normal range stays at dh, transverse data cover the edge stencil.
revision = revision.replace('kl, ku, jl, ju, is-dh, ie+1+dh,',
    'ks-de3, ke+de3, js-de2, je+de2, is-dh, ie+1+dh,')
revision = revision.replace('kl, ku, js-dh, je+1+dh, il, iu,',
    'ks-de3, ke+de3, js-dh, je+1+dh, is-de, ie+de,')
revision = revision.replace('ks-dh, ke+1+dh, jl, ju, il, iu,',
    'ks-dh, ke+1+dh, js-de2, je+de2, is-de, ie+de,')
lambda_code = '''  auto revise_faces = [&](const int dh, const int force_tier) {
    const int extra = (force_tier > 0) ? ((force_tier >= n_fb) ? 1 : 2) : 0;
    const int de = dh + extra;
    const int de2 = multi_d ? de : 0, de3 = three_d ? de : 0;
''' + revision + '\n  };\n'
s = s[:start] + '''    revise_faces(dh, 0);
    if (mood_frozen_emf) select_frozen_edges(dh);
''' + s[end:]
insert = s.index('  int ndemoted_total = 0;')
prep = '''
  // Reference implementation for the halo experiment, deliberately caching whole
  // tiers. No base-order reconstruction runs after the initial candidate EMFs.
  DvceEdgeFld4D<Real> ehigh("audit_high",1,1,1,1);
  DvceEdgeFld4D<Real> eplm("audit_plm",1,1,1,1);
  DvceEdgeFld4D<Real> edc("audit_dc",1,1,1,1);
  auto copy_edges = [&](DvceEdgeFld4D<Real> &dst, const DvceEdgeFld4D<Real> &src) {
    Kokkos::deep_copy(dst.x1e, src.x1e);
    Kokkos::deep_copy(dst.x2e, src.x2e);
    Kokkos::deep_copy(dst.x3e, src.x3e);
  };
  if (mood_frozen_emf) {
    if (wb) { std::cerr << "Frozen EMF audit does not support well balance\\n"; std::exit(1); }
    ehigh = DvceEdgeFld4D<Real>("audit_high",nmb,ncells3,ncells2,ncells1);
    eplm = DvceEdgeFld4D<Real>("audit_plm",nmb,ncells3,ncells2,ncells1);
    edc = DvceEdgeFld4D<Real>("audit_dc",nmb,ncells3,ncells2,ncells1);
    const int h = mood_halo0, h2 = multi_d ? h : 0, h3 = three_d ? h : 0;
    auto compose = [&]() {
      ComposeCornerEMF(is-h,ie+h+1,js-h2,je+h2+1,ks-h3,
                       three_d ? ke+h3+1 : ks, false);
    };
    compose();
    copy_edges(ehigh, efld);
    auto clone = [](const auto &v) {
      using V = typename std::decay<decltype(v)>::type;
      V saved("audit_saved", v.layout());
      Kokkos::deep_copy(saved,v);
      return saved;
    };
    auto f1save=clone(flx1), f2save=clone(flx2), f3save=clone(flx3);
    auto e31save=clone(e3x1_), e21save=clone(e2x1_), e12save=clone(e1x2_);
    auto e32save=clone(e3x2_), e23save=clone(e2x3_), e13save=clone(e1x3_);
    auto base_recon = recon_method;
    if (n_fb > 1) {
      revise_faces(h, 1);
      recon_method = ReconstructionMethod::plm;
      compose();
      copy_edges(eplm, efld);
    }
    revise_faces(h, n_fb);
    recon_method = ReconstructionMethod::dc;
    compose();
    copy_edges(edc, efld);
    recon_method = base_recon;
    copy_edges(efld, ehigh);
    Kokkos::deep_copy(flx1,f1save); Kokkos::deep_copy(flx2,f2save);
    Kokkos::deep_copy(flx3,f3save);
    Kokkos::deep_copy(e3x1_,e31save); Kokkos::deep_copy(e2x1_,e21save);
    Kokkos::deep_copy(e1x2_,e12save); Kokkos::deep_copy(e3x2_,e32save);
    Kokkos::deep_copy(e2x3_,e23save); Kokkos::deep_copy(e1x3_,e13save);
  }
  auto select_frozen_edges = [&](const int h) {
    auto E1=efld.x1e, E2=efld.x2e, E3=efld.x3e;
    auto P1=eplm.x1e, P2=eplm.x2e, P3=eplm.x3e;
    auto D1=edc.x1e, D2=edc.x2e, D3=edc.x3e;
    const bool td=three_d;
    par_for("audit_select_edges", DevExeSpace(), 0,nmb1,
            td ? ks-h : ks, td ? ke+h+1 : ks, js-h,je+h+1,is-h,ie+h+1,
    KOKKOS_LAMBDA(const int m,const int k,const int j,const int i) {
      int t3=0;
      for (int dj=-1;dj<=0;++dj) for (int di=-1;di<=0;++di) {
        const int t=fb_level_(m,k,j+dj,i+di); t3=(t>t3)?t:t3;
      }
      if (t3>0) E3(m,k,j,i)=(t3>=n_fb)?D3(m,k,j,i):P3(m,k,j,i);
      if (td) {
        int t1=0,t2=0;
        for (int dk=-1;dk<=0;++dk) for (int dj=-1;dj<=0;++dj) {
          const int t=fb_level_(m,k+dk,j+dj,i); t1=(t>t1)?t:t1;
        }
        for (int dk=-1;dk<=0;++dk) for (int di=-1;di<=0;++di) {
          const int t=fb_level_(m,k+dk,j,i+di); t2=(t>t2)?t:t2;
        }
        if (t1>0) E1(m,k,j,i)=(t1>=n_fb)?D1(m,k,j,i):P1(m,k,j,i);
        if (t2>0) E2(m,k,j,i)=(t2>=n_fb)?D2(m,k,j,i):P2(m,k,j,i);
      } else {
        int t1=fb_level_(m,ks,j-1,i), t=fb_level_(m,ks,j,i);
        t1=(t>t1)?t:t1;
        int t2=fb_level_(m,ks,j,i-1); t2=(t>t2)?t:t2;
        if (t1>0) {
          E1(m,ks,j,i)=(t1>=n_fb)?D1(m,ks,j,i):P1(m,ks,j,i);
          E1(m,ke+1,j,i)=E1(m,ks,j,i);
        }
        if (t2>0) {
          E2(m,ks,j,i)=(t2>=n_fb)?D2(m,ks,j,i):P2(m,ks,j,i);
          E2(m,ke+1,j,i)=E2(m,ks,j,i);
        }
      }
    });
  };
'''
s = s[:insert] + lambda_code + prep + s[insert:]
s = s.replace('const int h_f = mood_halo0/nrevs;',
              'const int h_f = mood_frozen_emf ? 1 : mood_halo0/nrevs;')
s = s.replace('const int dh = h_f*(nrevs - rev + 1);',
              'const int dh = mood_frozen_emf ? std::max(0, mood_halo0-rev+1) : h_f*(nrevs-rev+1);')
s = s.replace('      ComposeCornerEMF(il, iu+1, jl, ju+1, kl, (three_d ? ku+1 : ks), masked);',
              '      if (!mood_frozen_emf) ComposeCornerEMF(il, iu+1, jl, ju+1, kl, (three_d ? ku+1 : ks), masked);')
s = s.replace('#include <limits>', '#include <limits>\n#include <type_traits>')
p.write_text(s)
print('Created opt-in frozen-EMF experiment')
