"""Validate cell-local rcfl, frozen halos, and unchanged relative-mode arithmetic."""
from pathlib import Path
import json, re
import numpy as np
import halo_test as h
import physics_suite as p
W=Path(__file__).resolve().parent
results={'runs':[], 'comparisons':{}, 'probes':{}, 'metrics':{}}
def same(name,c):
    assert c and all(x['max_abs']==0 and x['time_equal'] for x in c.values()),(name,c)
    results['comparisons'][name]=c

def probe(name,directory):
    files=list(directory.glob('rcfl_probe_rank*.dat'));assert files,name
    rows=np.concatenate([np.fromfile(f,dtype=np.float64).reshape(-1,13) for f in files])
    rows=rows[rows[:,0]>=0];assert len(rows),name
    speed=np.max(np.abs(rows[:,1:4]),axis=1)
    cs2=rows[:,10]*(rows[:,10]-1)*rows[:,5]/rows[:,4]
    speed+=rows[:,9]*np.sqrt(np.maximum(cs2+rows[:,6]/rows[:,4],0))
    factors=np.minimum(1,rows[:,8]*speed/rows[:,7])
    expected=rows[:,12]*factors
    assert np.allclose(rows[:,0],expected,rtol=2.e-15,atol=0),(name,np.max(abs(rows[:,0]-expected)))
    global_expected=rows[:,12]*min(1,rows[0,8]*speed.max()/rows[:,7].min())
    results['probes'][name]={'cells':len(rows),'max_error':float(np.max(abs(rows[:,0]-expected))),
      'factor_min':float(factors.min()),'factor_max':float(factors.max()),
      'different_from_global':int(np.count_nonzero(rows[:,0]!=global_expected))}

# Naturally triggered fallback, at the validated minimal ghost counts.
for order,dim,steps in [(4,2,8),(6,2,8),(8,2,8),(8,3,2)]:
    stem=f'rcfl_fv{order}_{dim}d';names=[]
    for label,blocks,ng in [('ref',1,14),('min',2,max(order,5))]:
        name=f'{stem}_{label}';names.append(name)
        r=h.run(name,order=order,ng=ng,blocks=blocks,dim=dim,steps=steps,scale='rcfl',rtol=1.e-4)
        results['runs'].append(r);assert r['code']==0,r
        probe(name,h.RUNS/name)
    c=h.compare(*names)
    if dim==2:
        assert c['fields_exact'] and c['audits_exact'],c
    else:
        # Two-step 3D also has last-bit field differences in the relative control.
        # Require identical decisions and round-off-level fields, not bitwise identity.
        assert all(v['time_equal'] and v['finite'] and v['max_abs']<1.e-14 for v in c['fields'].values()),c
        assert all(v['flag_differences']==0 and v['max_abs']<1.e-14 for v in c['audits'].values()),c
    results['comparisons'][stem]=c
name='rcfl_fv8_mpi4'
r=h.run(name,ng=8,steps=8,mpi=4,scale='rcfl',rtol=1.e-4)
results['runs'].append(r);assert r['code']==0,r
c=h.compare('rcfl_fv8_2d_ref',name);assert c['fields_exact'] and c['audits_exact'],c
results['comparisons'][name]=c;probe(name,h.RUNS/name)

# Variable-speed MHD and zero-initial-speed hydro exercise local and zero factors.
for case in ['ot','hydro']:
    mod='hydro' if case=='hydro' else 'mhd'
    for label,blocks in [('ref',1),('split',2)]:
        name=f'rcfl_{case}_{label}'
        s=p.deck(order=4 if case=='hydro' else 8,ng=8,blocks=blocks,case=case,tlim=.05)
        s=p.setparam(s,mod,'mood_nad_scale','rcfl')
        # Check restored rcfl rtol default, not just explicit input.
        s=re.sub(r'(?m)^mood_rtol\s*=.*\n','',s)
        r=p.execute(name,s);results['runs'].append(r);assert r['code']==0,r
        probe(name,p.RUNS/name);results['metrics'][name]=p.metrics(name)
    same(case,p.compare(f'rcfl_{case}_ref',f'rcfl_{case}_split'))
assert results['probes']['rcfl_ot_ref']['different_from_global']>0
assert results['probes']['rcfl_hydro_ref']['factor_max']==0

# Explicit old global modes remain rejected in both modules.
for case,mod in [('ot','mhd'),('hydro','hydro')]:
    for mode in ('gcfl','grange','gdu'):
        name=f'rcfl_reject_{mod}_{mode}'
        r=p.execute(name,p.setparam(p.deck(case=case),mod,'mood_nad_scale',mode))
        results['runs'].append(r)
        assert r['code']!=0 and 'is unsupported' in (p.RUNS/name/'run.log').read_text()
    old=W/'athena-before-rcfl'
    if old.exists():
        for label,exe in [('before',old),('after',p.EXE)]:
            name=f'rcfl_relative_{case}_{label}'
            r=p.execute(name,p.deck(order=4,ng=8,case=case,tlim=.005,steps=8),exe=exe)
            results['runs'].append(r);assert r['code']==0,r
        same('relative_'+case,p.compare(f'rcfl_relative_{case}_before',f'rcfl_relative_{case}_after'))
(W/'local-rcfl-results.json').write_text(json.dumps(results,indent=2))
print('PASS local rcfl validation',flush=True)
