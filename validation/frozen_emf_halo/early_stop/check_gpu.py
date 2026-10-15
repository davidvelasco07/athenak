"""GPU field-equivalence and actual-revision controls against the preserved stress study."""
from pathlib import Path
import argparse,collections,hashlib,json,os,re,subprocess,sys,time
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
base=a.baseline.resolve();exe=a.exe.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(base));from run_stress import suite,read_vtk,analyze
plan=json.loads((base/'plan.json').read_text());env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';records=[];comparisons=[]
def save(): (root/'results.json').write_text(json.dumps({'runs':records,'comparisons':comparisons},indent=2))
def run(name,case='blast_mhd_lowbeta',revs=4,mpi=1,extra=None,target=None):
 j=next(j for j in plan if j['case']==case and j['scheme']=='fv8_fb');out=root/name;out.mkdir(exist_ok=True)
 ov=[f'mhd/mood_max_revs={revs}','mhd/mood_diagnostics=true']+(extra or [])
 suite.apply_athinput_overrides(base/'results/stress'/j['name']/'run.athinput',out/'run.athinput',ov)
 cmd=(['mpiexec','--mca','pml','ob1','--mca','btl','self,vader,tcp','-n',str(mpi)] if mpi>1 else [])+[str(exe),'-i',str(out/'run.athinput'),'-d',str(out)]
 with (out/'run.log').open('w') as log:r=subprocess.run(cmd,cwd=exe.parent,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 text=(out/'run.log').read_text();assert r.returncode==0,(name,text[-3000:])
 h=analyze({**j,'outdir':str(out),'target':j['target'] if target is None else target});assert h['healthy'],(name,h)
 rounds=[{'cycle':int(c),'stage':int(s),'revisions':int(n),'checks':int(k),'status':status} for c,s,n,k,status in re.findall(r'MOOD cycle=(\d+) stage=(\d+) revisions=(\d+) checks=(\d+) status=(\w+)',text)]
 assert len(rounds)==3*h['cycles'] and all(x['checks']==x['revisions']+1 and x['revisions']<=revs for x in rounds)
 record={'name':name,'case':case,'nghost':5,'max_revisions':revs,'mpi':mpi,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'command':cmd,'health':h,'rounds':rounds,'revision_histogram':dict(sorted(collections.Counter(x['revisions'] for x in rounds).items())),'status_histogram':dict(collections.Counter(x['status'] for x in rounds)),'mean_revisions':sum(x['revisions'] for x in rounds)/len(rounds)}
 records.append(record);save();print('GPU_PASS',name,record['revision_histogram'],record['status_histogram'],flush=True)
 return out
def compare(ref,test):
 count=0
 for p in sorted((ref/'vtk').glob('*.vtk')):
  x,mx=read_vtk(p);y,my=read_vtk(test/'vtk'/p.name);assert mx==my
  assert x.keys()==y.keys()
  for v in x:assert np.array_equal(x[v],y[v]),(ref,test,p.name,v,float(np.max(abs(x[v]-y[v]))));count+=1
 assert count>=20,(ref,test,count)
 comparisons.append({'reference':str(ref),'test':str(test),'saved_float32_arrays':count,'all_identical':True});save();print('GPU_IDENTICAL',test.name,count,flush=True)
uniform=['problem/prat=1','problem/drat=1','problem/pi_amb=1','problem/b_amb=0','mhd/gamma=2','mesh/nx1=32','mesh/nx2=32','meshblock/nx1=16','meshblock/nx2=16','time/tlim=0.001']
p=run('uniform_mpi1',revs=8,extra=uniform,target=.001);q=run('uniform_mpi4',revs=8,mpi=4,extra=uniform,target=.001);compare(p,q)
assert all(x['revisions']==0 for r in records for x in r['rounds'])
for case in ('blast_mhd_lowbeta','blast_srmhd_uct','blast_grmhd_uct'):
 p=run(case+'_r4',case=case);j=next(j for j in plan if j['case']==case and j['scheme']=='fv8_fb');compare(base/'results/stress'/j['name'],p)
p=run('grmhd_r8',case='blast_grmhd_uct',revs=8);compare(base/'revision-control/grmhd_fv8_r8',p)
extra=['mesh/nx1=64','mesh/nx2=64','meshblock/nx1=16','meshblock/nx2=16','time/tlim=0.001']
p=run('active_mpi1',revs=8,extra=extra,target=.001);q=run('active_mpi4',revs=8,mpi=4,extra=extra,target=.001);compare(p,q)
assert any(x['revisions']>0 for x in records[-1]['rounds'])
print('GPU_EARLY_STOP_VALIDATION_FINISHED',flush=True)
