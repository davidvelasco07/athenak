"""Integration checks against AthenaK's existing dense passive scalars."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
exe=Path(sys.argv[1]).resolve()
input_file=Path(sys.argv[2]).resolve()
case=sys.argv[3]
options={
 'forward':['time/nlim=12','passive_tracers/capacity=512'],
 'reverse':['time/nlim=24','problem/vx=-0.7','problem/vy=-0.35','time/integrator=rk3'],
 'dc':['time/nlim=12','hydro/reconstruct=dc','time/integrator=rk1'],
 'reject_rk4':['time/nlim=12','time/integrator=rk4'],
 'forced':['time/nlim=12','passive_tracers/storage=forced'],
 'dense':['time/nlim=12','passive_tracers/storage=dense'],
 'native':['time/nlim=12','passive_tracers/storage=native'],
 'oneblock':['mesh/nx1=32','mesh/nx2=32','meshblock/nx1=32','meshblock/nx2=32',
             'time/nlim=12','problem/radius=0.1','passive_tracers/capacity=8'],
 'one_d':['mesh/nx2=1','meshblock/nx2=1','passive_tracers/capacity=64','time/nlim=16'],
 'three_d':['mesh/nx1=32','mesh/nx2=32','mesh/nx3=16','meshblock/nx1=16',
            'meshblock/nx2=16','meshblock/nx3=8','mesh/ix3_bc=periodic',
            'mesh/ox3_bc=periodic','time/nlim=8','problem/radius=0.14',
            'passive_tracers/capacity=64'],
}
with tempfile.TemporaryDirectory(prefix='tracer-integration-') as work:
 result=subprocess.run([str(exe),'-i',str(input_file),*options[case]],cwd=work,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
 if case == 'reject_rk4':
  assert result.returncode and 'Unsupported tracer integrator' in result.stdout,result.stdout
  print('Unsupported non-SSP RK4 rejected explicitly')
  raise SystemExit(0)
 if result.returncode:
  print(result.stdout);raise SystemExit('Tracer integration failed: '+case)
 report=json.loads((Path(work)/'tracers.json').read_text())
 assert report['compare_native'],report
 assert report['cycles']>0,report
 print(json.dumps(report))
