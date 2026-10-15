"""Focused GPU checks for production-consistent magnetic PAD."""
import json
import os
from pathlib import Path
import subprocess
import sys

base=Path('/home/velasco/codex-validation/local-rcfl-gpu-20261014')
audit=Path('/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution/pad-audit')
root=audit/'fixed-validation'
exe=audit/'build/src/athena'
runner=Path('/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution/four-revision-rtol1e-8/run_study_atomic.py')
os.environ['CUDA_VISIBLE_DEVICES']='0'
used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
assert used<20,f'GPU0 busy: {used}MiB'
cases=[('fv4_n512',4,512,64),('fv4_n128_b64',4,128,64),
       ('fv4_n128_b32',4,128,32),('fv6_n128',6,128,64),('fv8_n128',8,128,64)]
summary=[]
for name,order,nx,block in cases:
    reach=order//2;ng=reach*4+max(3,reach)
    subprocess.run([sys.executable,str(runner),'--source',str(base/'source'),
        '--exe',str(exe),'--output',str(root/name),'--resolutions',str(nx),
        '--orders',str(order),'--block',str(block),'--revisions','4',
        '--nghost',str(ng),'--rtol','1e-8','--resume'],check=True)
    r=json.loads((root/name/'results.json').read_text())['runs'][0]
    summary.append({'case':name,'nghost':ng,'clean':r['clean'],'code':r['code'],
                    'energy_floors':r.get('energy_floors'),
                    'pad_rejections':r.get('pad_rejections')})
    (root/'checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert r['clean'],f'Corrected case failed: {name}'

sys.path.insert(0,str(base/'source/vis/python'))
import bin_convert as reader
sys.path.insert(0,str(runner.parent))
from run_study_atomic import assemble,input_deck
import numpy as np

def compare(left,right,initial_only=False):
    checks=[]
    for f in sorted((left/'bin').glob('*.bin')):
        if initial_only and not f.name.endswith('.00000.bin'):continue
        a,aa=assemble(f,reader);b,bb=assemble(right/'bin'/f.name,reader)
        same=a['time']==b['time'] and aa.keys()==bb.keys() and all(np.array_equal(aa[v],bb[v]) for v in aa)
        checks.append({'file':f.name,'time':float(a['time']),'identical':bool(same)})
    return {'all_identical':all(c['identical'] for c in checks),'files':checks}

comparisons={
 'production_initial_state':compare(root/'baseline_old_ng11/ot_fv4_n512',root/'fv4_n512/ot_fv4_n512',True),
 'blocks32_vs64':compare(root/'fv4_n128_b32/ot_fv4_n128',root/'fv4_n128_b64/ot_fv4_n128')}
(root/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
assert all(c['all_identical'] for c in comparisons.values()),'Field comparison failed'

for dims in (2,3):
    run=root/f'memcheck_{dims}d';run.mkdir(exist_ok=True)
    deck=input_deck(8,32,32,.002,revisions=4,rtol=1.e-8,nghost=20)
    if dims==3:deck=deck.replace('nx3 = 1\n','nx3 = 32\n')
    (run/'input.athinput').write_text(deck)
    with (run/'stdout.txt').open('w') as output:
        subprocess.run(['compute-sanitizer','--tool','memcheck','--error-exitcode','99',
                        str(exe),'-i','input.athinput'],cwd=run,stdout=output,
                       stderr=subprocess.STDOUT,check=True)
    log=(run/'stdout.txt').read_text()
    assert 'ERROR SUMMARY: 0 errors' in log
print('PAD_CORRECTION_VALIDATED',flush=True)
