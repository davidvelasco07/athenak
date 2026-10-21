"""Compare every intensity and moment against the original full-angle SC beam solve."""
from array import array
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
exe=Path(sys.argv[1]).resolve(); deck=Path(sys.argv[2]).resolve()
case=sys.argv[3]
variants={
 'uniform':[],
 'tiles':['nr_radiation/tile_size=8'],
 'rectangular':['meshblock/nx2=8'],
 'vertical':['problem/ihor1=-1','problem/ihor2=-1','problem/ivert1=8','problem/ivert2=23'],
}
reference=None; report=[]
with tempfile.TemporaryDirectory(prefix='beam-integration-') as temp:
 for mode in ('dense','selected','forced','sparse'):
  work=Path(temp)/mode; work.mkdir()
  args=[str(exe),'-i',str(deck),'mesh/nx1=64','mesh/nx2=32',
        'problem/ihor1=16','problem/ihor2=47',f'nr_radiation/intensity_storage={mode}',
        *variants[case]]
  run=subprocess.run(args,cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                     text=True,timeout=180)
  if run.returncode:
   print(run.stdout);raise SystemExit('Beam integration failed: '+mode)
  record=json.loads((work/'beam.json').read_text())
  assert record['converged'],record
  values=array('d');values.frombytes((work/'beam_state.bin').read_bytes())
  assert all(__import__('math').isfinite(v) for v in values)
  if reference is None:reference=values;expected_sweeps=record['sweeps']
  assert len(values)==len(reference)
  error=max(abs(a-b) for a,b in zip(values,reference))
  # Float builds have 32-bit SC input/state; the file deliberately stores doubles.
  assert error<2e-6,(mode,error)
  assert record['sweeps']==expected_sweeps,(mode,record)
  record['max_intensity_moment_error']=error;report.append(record)
 print(json.dumps(report))
