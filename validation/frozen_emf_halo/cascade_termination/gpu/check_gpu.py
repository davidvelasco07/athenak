"""Short CUDA controls for global counters, integer exchange and unchanged fields."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
import numpy as np
root=Path(__file__).resolve().parent;source=root/'source'
sys.path.insert(0,str(source/'validation/frozen_emf_halo/ot_resolution'));from run_study import input_deck,assemble
sys.path.insert(0,str(source/'vis/python'));import bin_convert as reader
exe=root/'build/src/athena';oldot=Path('/scratch/lustre/velasco/codex-validation/early-stop-20261015/build-ot/src/athena');oldbuiltin=Path('/scratch/lustre/velasco/codex-validation/stress-abea4ac2-20261015/build_builtin/src/athena')
assert hashlib.sha256(oldot.read_bytes()).hexdigest()=='87f247504b87e4e6ed2a457a583640face97c6f9fff9bc9a44e04bfca0d6ee5f'
runs=[];comparisons=[];outroot=root/'gpu';outroot.mkdir(exist_ok=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
def save(): (outroot/'results.json').write_text(json.dumps({'runs':runs,'comparisons':comparisons,'interval_end_unix':time.time(),'overlaps_pinned_ot_study':True},indent=2)+'\n')
def run(name,order=8,mpi=1,uniform=False,regime='nr',reference=False):
 out=outroot/name;out.mkdir(exist_ok=True)
 deck=input_deck(order,32 if uniform else 64,16,.001 if uniform else .01,solver='hlld' if regime=='nr' else 'hlle',revisions=8,rtol=1.e-8,diagnostics=True)
 if uniform or regime!='nr':
  start=deck.index('<problem>');end=deck.index('<output1>',start);p='<problem>\npgen_name = shock_tube\nshock_dir = 1\nxshock = 0\n'
  for side in ('l','r'):
   d=1. if uniform or side=='l' else .5
   p+=f'd{side} = {d}\nu{side} = 0\nv{side} = 0\nw{side} = 0\np{side} = {d}\nbx{side} = 0.5\nby{side} = {0.5 if uniform or side=="l" else -0.5}\nbz{side} = 0.2\n'
  deck=deck[:start]+p+deck[end:]
 if regime!='nr':deck+=f'\n<coord>\nspecial_rel = {str(regime=="sr").lower()}\ngeneral_rel = {str(regime=="gr").lower()}\nminkowski = true\n'
 (out/'input.athinput').write_text(deck)
 binary=(oldot if regime=='nr' else oldbuiltin) if reference else exe
 cmd=(['mpiexec','--mca','pml','ob1','--mca','btl','self,vader,tcp','-n',str(mpi)] if mpi>1 else [])+[str(binary),'-i','input.athinput']
 with (out/'stdout.txt').open('w') as f:r=subprocess.run(cmd,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 text=(out/'stdout.txt').read_text();assert r.returncode==0,(name,text[-2500:])
 traces=[dict(re.findall(r'(\w+)=(\w+)',line)) for line in text.splitlines() if line.startswith('MOOD ')]
 if not reference:
  assert traces and all(int(v['checks'])==int(v['revisions'])+1 for v in traces),name
  if uniform:assert all(int(v['revisions'])==0 and v['status']=='converged' for v in traces),name
 rec={'name':name,'order':order,'mpi':mpi,'regime':regime,'uniform':uniform,'reference':reference,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'traces':traces}
 runs.append(rec);save();print('GPU_PASS',name,flush=True);return out
def compare(a,b,trace=False):
 n=0
 for f in sorted((a/'bin').glob('*.bin')):
  ma,aa=assemble(f,reader);mb,bb=assemble(b/'bin'/f.name,reader)
  assert ma['time']==mb['time'] and aa.keys()==bb.keys()
  for v in aa:assert np.array_equal(aa[v],bb[v]),(a.name,b.name,f.name,v,float(np.max(abs(aa[v]-bb[v]))));n+=1
 assert n==30,(a,b,n)
 if trace:assert next(r['traces'] for r in runs if r['name']==a.name)==next(r['traces'] for r in runs if r['name']==b.name)
 comparisons.append({'reference':a.name,'test':b.name,'arrays_compared':n,'bitwise_equal':True,'global_traces_equal':trace});save()
a=run('uniform_p1',uniform=True);b=run('uniform_p4',uniform=True,mpi=4);compare(a,b,trace=True)
for order in (4,6,8):
 a=run(f'ot_fv{order}_p1',order=order);b=run(f'old_ot_fv{order}',order=order,reference=True);compare(b,a)
 if order==8:
  b=run('ot_fv8_p4',mpi=4);compare(a,b,trace=True)
for regime in ('sr','gr'):
 a=run(regime,regime=regime);b=run('old_'+regime,regime=regime,reference=True);compare(b,a)
print('ALL_GPU_CASCADE_CHECKS_PASS',len(runs),sum(v['arrays_compared'] for v in comparisons),flush=True)
