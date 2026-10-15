"""Compare current OT fields with the earlier correct fixed-count implementation."""
import json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'ot-scripts'));from run_study import assemble,write_json
sys.path.insert(0,str(root/'source/vis/python'));import bin_convert as reader
old=Path('/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution/cascade-halo-validation/minimum')
records=[]
for order,nx in [(4,128),(6,128),(8,128),(4,512)]:
 new=root/f'ot-resolution-early-stop/ot_fv{order}_n{nx}';ref=old/f'fv{order}_n{nx}'
 if not (new/'result.json').exists():continue
 checks=[]
 for p in sorted((new/'bin').glob('*.bin')):
  others=sorted((ref/'bin').glob('*'+p.name[p.name.index('.mhd_'):]))
  assert len(others)==1,(p,others)
  md,a=assemble(p,reader);mr,b=assemble(others[0],reader)
  assert md['time']==mr['time'],(p,md['time'],mr['time'])
  for k in a:
   checks.append({'file':p.name,'variable':k,'equal':bool(np.array_equal(a[k],b[k])),'max_abs_difference':float(np.max(np.abs(a[k]-b[k])))})
 records.append({'order':order,'nx':nx,'reference':str(ref),'current':str(new),'arrays_compared':len(checks),'all_bitwise_equal':all(c['equal'] for c in checks),'checks':checks})
write_json(root/'ot-resolution-early-stop/reference-comparison.json',{'comparisons':records})
print([(r['order'],r['nx'],r['arrays_compared'],r['all_bitwise_equal']) for r in records],flush=True)
