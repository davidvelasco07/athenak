from halo_test import *
from concurrent.futures import ThreadPoolExecutor
jobs=[];pairs=[]
for order in (4,6,8):
    for rev in (1,2,3,4):
        base=f'fv{order}_r{rev}'
        ref=base+'_ref';test=base+'_min'
        jobs += [(ref,dict(order=order,revisions=rev,ng=14,blocks=1)),
                 (test,dict(order=order,revisions=rev,ng=max(order,rev+3)))]
        pairs.append((ref,test))
    for dim,steps,label in [(3,1,'3d'),(2,8,'long')]:
        ref=f'fv{order}_{label}_ref';test=f'fv{order}_{label}_min'
        jobs += [(ref,dict(order=order,dim=dim,steps=steps,ng=14,blocks=1)),
                 (test,dict(order=order,dim=dim,steps=steps,ng=max(order,5)))]
        pairs.append((ref,test))
for rev in (1,2,3,4):
    seed=-(32*64+32-rev)-3
    ref=f'chain_r{rev}_ref';test=f'chain_r{rev}_min';short=f'chain_r{rev}_short'
    args=dict(order=4,revisions=rev,seed=seed)
    jobs += [(ref,dict(**args,ng=14,blocks=1)),(test,dict(**args,ng=max(4,rev+3))),
             (short,dict(**args,ng=14,halo=rev-1))]
    pairs += [(ref,test),(ref,short)]
def work(job):
    n,a=job
    r=run(n,**a)
    assert r['code']==0,r
    return r
with ThreadPoolExecutor(max_workers=3) as pool: runs=list(pool.map(work,jobs))
comps=[]
for ref,test in pairs:
    c=compare(ref,test);comps.append(c)
    print('COMPARE',test,c['fields_exact'],c['audits_exact'],flush=True)
(W/'max-sweep-results.json').write_text(json.dumps(dict(runs=runs,comparisons=comps),indent=2))
for c in comps:
    if c['test'].endswith('_short'): assert not c['fields_exact'],c
    else: assert c['fields_exact'] and c['audits_exact'],c
