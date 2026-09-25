#!/usr/bin/env python3
"""Small end-to-end GMTF AMR checks against analytic Jeans thresholds.

Usage: check_gmtf_amr.py EXE PILOT_INPUT WORK [--mpi-prefix 'mpirun ... -n 2']
Does not validate a production cloud's fragmentation or long-term conservation.
"""
import argparse,json,re,shlex,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('exe',type=Path);p.add_argument('input',type=Path)
p.add_argument('work',type=Path);p.add_argument('--mpi-prefix',default='');a=p.parse_args()
exe=a.exe.resolve();a.work.mkdir(parents=True,exist_ok=True)
text=a.input.read_text().replace('<problem>','<problem>\nnsink_seed = 0\nsink_mass_cv = 10').replace('<output1>','<output1>\ndcycle = 0')
inp=(a.work/'test.athinput').resolve();inp.write_text(text)
common=['mesh/nx1=32','mesh/nx2=32','mesh/nx3=32','meshblock/nx1=16',
 'meshblock/nx2=16','meshblock/nx3=16','mesh_refinement/num_levels=2',
 'mesh_refinement/max_nmb_per_rank=64','time/nlim=2','time/ndiag=1',
 'problem/Mach=1e-8','problem/nhigh=4','problem/sfe_term=-1',
 'particles/creation=false','particles/accretion=false','particles/merging=false',
 'output1/dcycle=1','output2/dt=0','output3/dt=0','output5/dt=0']
results={}
def run(name,opts,blocks,prefix=''):
 d=a.work/name;d.mkdir(exist_ok=False)
 cmd=shlex.split(prefix)+[str(exe),'-i',str(inp)]+common+opts
 with (d/'stdout.log').open('w') as f:r=subprocess.run(cmd,cwd=d,stdout=f,stderr=subprocess.STDOUT,timeout=120)
 assert r.returncode==0,(name,(d/'stdout.log').read_text()[-1500:])
 h=np.loadtxt(next(d.glob('*.user.hst')));assert np.isfinite(h).all()
 assert (int(h[-1,12])==blocks if blocks is not None else h[-1,12]>8),(name,h[-1])
 assert abs(h[-1,2]+h[-1,6]-(h[0,2]+h[0,6]))<1e-8,(name,'mass')
 assert np.max(np.abs(h[:,9:12]-h[0,9:12]))<1e-8,(name,'momentum')
 results[name]={'blocks':int(h[-1,12]),'mass_error':float(h[-1,2]+h[-1,6]-h[0,2]-h[0,6]),'time':float(h[-1,0])}
 return h
# rho=1, cs=1, G=pi, dx=1/8 -> Jeans length/dx=8.
run('jeans7_stays',['problem/njeans=7'],8)
run('jeans9_refines',['problem/njeans=9'],64)
# Increasing G by4 halves the Jeans length: threshold is now4, not8.
run('gravity4_jeans3_stays',['gravity/four_pi_G=157.91367041742973','problem/njeans=3'],8)
run('gravity4_jeans5_refines',['gravity/four_pi_G=157.91367041742973','problem/njeans=5'],64)
seed=['problem/njeans=1','problem/nsink_seed=1','problem/sink_mass_cv=1e-8']
run('sink_protection',seed+['problem/sink_protect=true'],None)
run('jeans_only',seed+['problem/sink_protect=false'],8)
if a.mpi_prefix:
 run('jeans9_mpi',['problem/njeans=9'],64,a.mpi_prefix)
 run('sink_protection_mpi',seed+['problem/sink_protect=true'],results['sink_protection']['blocks'],a.mpi_prefix)
# Strong gravity would exceed the LP creation threshold on the base grid, but
# finest-only creation must reject it before the end-of-step refinement occurs.
d=a.work/'creation_guard';d.mkdir()
cmd=[str(exe),'-i',str(inp)]+common+['time/nlim=1','gravity/four_pi_G=1e6',
 'particles/creation=true','particles/accretion=true','particles/creation_finest_only=true']
with (d/'stdout.log').open('w') as f:r=subprocess.run(cmd,cwd=d,stdout=f,stderr=subprocess.STDOUT,timeout=120)
assert r.returncode==0,(d/'stdout.log').read_text()[-1000:]
h=np.loadtxt(next(d.glob('*.user.hst')));assert np.max(h[:,7])==0,'coarse-level sink created'
results['creation_guard']={'sinks':int(h[-1,7])}
(a.work/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
