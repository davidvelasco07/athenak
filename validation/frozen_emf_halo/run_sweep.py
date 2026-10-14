from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from halo_test import run, compare, RUNS

jobs=[]
for order in (4,6,8):
    jobs.append((f'fv{order}_ref',dict(order=order,ng=16,blocks=1)))
    for ng in range(order+1,order+4):
        jobs.append((f'fv{order}_ng{ng}',dict(order=order,ng=ng)))
    for rev in (1,3,4):
        jobs.append((f'fv{order}_r{rev}_ref',dict(order=order,revisions=rev,ng=16,blocks=1)))
        for ng in sorted({max(order+1,order+rev-1),order+rev}):
            jobs.append((f'fv{order}_r{rev}_ng{ng}',dict(order=order,revisions=rev,ng=ng)))
    jobs.append((f'fv{order}_short_halo',dict(order=order,ng=16,halo=1)))

def work(job):
    name,args=job
    if (RUNS/name/'run_result.json').exists():
        return json.loads((RUNS/name/'run_result.json').read_text())
    return run(name,**args)
with ThreadPoolExecutor(max_workers=3) as pool:
    runs=list(pool.map(work,jobs))
comparisons=[]
for r in runs:
    if r['code'] or r['blocks']==1: continue
    ref=f"fv{r['order']}_ref" if r['revisions']==2 else f"fv{r['order']}_r{r['revisions']}_ref"
    c=compare(ref,r['name']); comparisons.append(c)
    print('COMPARE',r['name'],'fields',c['fields_exact'],'all_halo_candidates',c['audits_exact'],flush=True)
out=Path(__file__).parent/'sweep.json'
out.write_text(json.dumps(dict(runs=runs,comparisons=comparisons),indent=2))
