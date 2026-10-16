"""Preserve a real GR R8 blast while removing unproductive revision rounds."""
from pathlib import Path
import collections,hashlib,json,os,re,shutil,subprocess,sys,time
import numpy as np
root=Path(__file__).resolve().parent;oldroot=Path('/scratch/lustre/velasco/codex-validation/early-stop-20261015')
base=Path('/scratch/lustre/velasco/codex-validation/stress-abea4ac2-20261015');sys.path.insert(0,str(base));from run_stress import read_vtk
reference=oldroot/'results/grmhd_r8';out=root/'gpu/grmhd_r8';out.mkdir(parents=True,exist_ok=True);shutil.copy2(reference/'run.athinput',out/'run.athinput')
exe=root/'build-blast/src/athena';env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
with (out/'run.log').open('w') as f:p=subprocess.run([str(exe),'-i','run.athinput'],cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
text=(out/'run.log').read_text();assert p.returncode==0,text[-2500:]
n=0
for path in sorted((reference/'vtk').glob('*.vtk')):
 a,ma=read_vtk(path);b,mb=read_vtk(out/'vtk'/path.name);assert ma==mb
 for var in a:assert np.array_equal(a[var],b[var]),(path.name,var,float(np.max(abs(a[var]-b[var]))));n+=1
traces=[dict(re.findall(r'(\w+)=(\w+)',line)) for line in text.splitlines() if line.startswith('MOOD ')]
old=(reference/'run.log').read_text();previous=[dict(re.findall(r'(\w+)=(\w+)',line)) for line in old.splitlines() if line.startswith('MOOD ')]
assert len(traces)==len(previous)==855
name=next(out.glob('*.log'));name=next(p for p in out.glob('*.log') if p.name!='run.log')
newlog=np.loadtxt(name,ndmin=2);oldlog=np.loadtxt(reference/name.name,ndmin=2)
physical_new=newlog[:,1:6].sum(axis=0).astype(int).tolist();physical_old=oldlog[:,1:6].sum(axis=0).astype(int).tolist();assert physical_new==physical_old,(physical_new,physical_old)
newrev=[int(r['revisions']) for r in traces];oldrev=[int(r['revisions']) for r in previous]
assert all(a<=b for a,b in zip(newrev,oldrev))
record={'case':'grmhd_r8','arrays_compared':n,'all_bitwise_equal':True,'new_executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'old_executable_sha256':hashlib.sha256((oldroot/'build-prod/src/athena').read_bytes()).hexdigest(),'physical_eos_counts_new':physical_new,'physical_eos_counts_old':physical_old,'stages':len(traces),'mean_revisions_new':sum(newrev)/len(newrev),'mean_revisions_old':sum(oldrev)/len(oldrev),'status_counts':dict(collections.Counter(r['status'] for r in traces)),'revision_histogram':dict(collections.Counter(newrev)),'traces':traces,'old_traces':previous}
(out/'comparison.json').write_text(json.dumps(record,indent=2)+'\n');print('REAL_GR_BLAST_PASS',json.dumps({k:v for k,v in record.items() if k not in ('traces','old_traces')}),flush=True)
