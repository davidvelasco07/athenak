"""Focused controls for magnetic volume-average detection; no halo optimization."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

base=Path('/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution')
root=base/'volume-average-validation'
source=base/'pad-audit/source'
runner=base/'pad-corrected-resolution/run_study.py'
spec=importlib.util.spec_from_file_location('study',runner)
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sys.path.insert(0,str(source/'vis/python'))
import bin_convert as reader
used=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
assert used<50,f'GPU1 busy: {used}MiB'
summary=[]
for name,order,nx,block in [('fv4_n128_b64',4,128,64),('fv4_n128_b32',4,128,32),
                            ('fv6_n128_b64',6,128,64),('fv8_n128_b64',8,128,64),
                            ('fv4_n512_b64',4,512,64)]:
    subprocess.run([sys.executable,str(runner),'--source',str(source),
        '--exe',str(root/'athena-volume-average'),'--output',str(root/name),
        '--resolutions',str(nx),'--orders',str(order),'--block',str(block),
        '--revisions','4','--rtol','1e-8','--resume'],check=True)
    r=json.loads((root/name/'results.json').read_text())['runs'][0]
    summary.append({'case':name,**{k:v for k,v in r.items() if k not in ('history','snapshots')}})
    study.write_json(root/'checks.json',summary)

left=root/'fv4_n128_b32/ot_fv4_n128/bin'
right=root/'fv4_n128_b64/ot_fv4_n128/bin'
checks=[]
for f in sorted(left.glob('*.bin')):
    a,aa=study.assemble(f,reader);b,bb=study.assemble(right/f.name,reader)
    checks.append({'file':f.name,'time':float(a['time']),
        'identical':bool(a['time']==b['time'] and aa.keys()==bb.keys() and
                         all(np.array_equal(aa[v],bb[v]) for v in aa))})
study.write_json(root/'blocks32_vs64.json',{'all_identical':bool(checks) and all(c['identical'] for c in checks),'files':checks})
print('VOLUME_AVERAGE_CASES_COMPLETE',json.dumps([{'case':r['case'],'clean':r['clean']} for r in summary]),flush=True)

for dims in (2,3):
    run=root/f'memcheck_{dims}d';run.mkdir(exist_ok=True)
    deck=study.input_deck(8,32,32,.002,revisions=4,rtol=1.e-8)
    if dims==3:deck=deck.replace('nx3 = 1\n','nx3 = 32\n')
    (run/'input.athinput').write_text(deck)
    with (run/'stdout.txt').open('w') as out:
        result=subprocess.run(['compute-sanitizer','--tool','memcheck','--error-exitcode','99',
                               str(root/'athena-volume-average'),'-i','input.athinput'],
                               cwd=run,stdout=out,stderr=subprocess.STDOUT,timeout=300)
    log=(run/'stdout.txt').read_text()
    assert result.returncode==0 and 'ERROR SUMMARY: 0 errors' in log,(dims,result.returncode)
    print('MEMCHECK_PASS',dims,flush=True)
