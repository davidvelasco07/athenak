from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from halo_test import run,compare,RUNS

jobs=[]
for rev in (1,2,3,4):
    seed=-(32*64+32-rev)-3
    for tag,blocks,halo in [('ref',1,None),('full',2,None),('short',2,rev-1)]:
        jobs.append((f'chain_r{rev}_{tag}',dict(order=8,ng=16,revisions=rev,
            seed=seed,blocks=blocks,halo=halo)))
for order in (4,6,8):
    for tag,blocks,ng in [('ref',1,16),('min',2,order+2),('one_less',2,order+1)]:
        jobs.append((f'long_fv{order}_{tag}',dict(order=order,ng=ng,blocks=blocks,steps=8)))
for order in (4,6):
    for tag,blocks,ng in [('ref',1,14),('min',2,order+2),('one_less',2,order+1)]:
        jobs.append((f'fv{order}_3d_{tag}',dict(order=order,ng=ng,blocks=blocks,dim=3)))
jobs.append(('fv8_3d_one_less',dict(order=8,ng=9,blocks=2,dim=3)))

def work(job):
    name,args=job
    if (RUNS/name/'run_result.json').exists():
        return json.loads((RUNS/name/'run_result.json').read_text())
    return run(name,**args)
with ThreadPoolExecutor(max_workers=2) as pool:
    runs=list(pool.map(work,jobs))
comparisons=[]
for r in runs:
    if r['code'] or r['blocks']==1: continue
    name=r['name']
    if name.startswith('chain_'): ref=f"chain_r{r['revisions']}_ref"
    elif name.startswith('long_'): ref=f"long_fv{r['order']}_ref"
    else: ref=f"fv{r['order']}_3d_ref"
    c=compare(ref,name); comparisons.append(c)
    print('COMPARE',name,'fields',c['fields_exact'],'all_halo_candidates',c['audits_exact'],flush=True)
(Path(__file__).parent/'extended.json').write_text(json.dumps(dict(runs=runs,comparisons=comparisons),indent=2))
