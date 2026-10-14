from physics_suite import *
jobs=[]
for order in (4,6,8):
 for nx in (32,64):
  s=setparam(deck(order=order,ng=15,nx=nx,blocks=2,case='wave',tlim=1.0),'mhd','mood_frozen_emf','false')
  jobs.append((f'wave_legacy_fv{order}_n{nx}',s))
# The existing validation manifest turns velocity NAD off for linear waves.
for mode in ['legacy','frozen']:
 for nx in (32,64):
  s=deck(order=8,ng=15 if mode=='legacy' else 8,nx=nx,blocks=2,case='wave',tlim=1.0)
  s=setparam(s,'mhd','mood_frozen_emf','false' if mode=='legacy' else 'true')
  s=setparam(s,'mhd','mood_nad_v','off')
  jobs.append((f'wave_voff_{mode}_n{nx}',s))
with ThreadPoolExecutor(max_workers=3) as pool:runs=list(pool.map(lambda x:execute(*x),jobs))
assert all(r['code']==0 for r in runs),runs
errors={}
for p in RUNS.glob('wave_*'):
 f=p/'halo-errs.dat'
 if f.exists():errors[p.name]=float(f.read_text().splitlines()[-1].split()[4])
(W/'accuracy-results.json').write_text(json.dumps({'runs':runs,'rms_l1':errors},indent=2))
print(errors)
