"""Production CPU/MPI checks for globally coordinated early termination."""
from pathlib import Path
import importlib.util,json,re,subprocess,sys,time
import numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);repo=args.repo.resolve()
spec=importlib.util.spec_from_file_location('study',repo/'validation/frozen_emf_halo/ot_resolution/run_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sys.path.insert(0,str(repo/'vis/python'));import bin_convert as reader
exe=args.exe.resolve();fixed=args.reference.resolve();assert exe.is_file() and fixed.is_file()
runs=[];comparisons=[]
def execute(name,order=8,revs=8,mpi=1,uniform=False,dim=2,regime='nr',reference=False):
 out=root/name;out.mkdir(exist_ok=True)
 deck=study.input_deck(order,32 if uniform else 64,16,.001 if uniform else .01,solver='hlld' if regime=='nr' else 'hlle',revisions=revs,rtol=1e-8,nghost=order//2+1)
 deck=deck.replace('mood_touched_only = false','mood_touched_only = false\nmood_diagnostics = true')
 if uniform or regime!='nr':
  a=deck.index('<problem>');b=deck.index('<output1>',a)
  p='<problem>\npgen_name = shock_tube\nshock_dir = 1\nxshock = 0\n'
  for suffix in ('l','r'):
   rho=1. if uniform or suffix=='l' else .5
   p+=f'd{suffix} = {rho}\nu{suffix} = 0\nv{suffix} = 0\nw{suffix} = 0\np{suffix} = {rho}\nbx{suffix} = 0.5\nby{suffix} = {0.5 if uniform or suffix=="l" else -0.5}\nbz{suffix} = 0.2\n'
  deck=deck[:a]+p+deck[b:]
 if regime!='nr':deck+=f'\n<coord>\nspecial_rel = {str(regime=="sr").lower()}\ngeneral_rel = {str(regime=="gr").lower()}\nminkowski = true\n'
 if dim==3:
  a=deck.index('<meshblock>');deck=deck[:a].replace('nx3 = 1','nx3 = 16')+deck[a:].replace('nx3 = 1','nx3 = 8',1)
 (out/'input.athinput').write_text(deck)
 args=(['mpiexec','-n',str(mpi)] if mpi>1 else [])+[str(fixed if reference else exe),'-i','input.athinput']
 with (out/'stdout.txt').open('w') as f:r=subprocess.run(args,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 log=(out/'stdout.txt').read_text()
 rounds=[{'cycle':int(c),'stage':int(s),'revisions':int(n),'checks':int(k),'status':v} for c,s,n,k,v in re.findall(r'MOOD cycle=(\d+) stage=(\d+) revisions=(\d+) checks=(\d+) status=(\w+)',log)]
 assert r.returncode==0,(name,log[-2500:])
 if not reference:
  assert rounds and all(x['checks']==x['revisions']+1 and x['revisions']<=revs for x in rounds),name
  if uniform:assert all(x['revisions']==0 and x['status']=='converged' for x in rounds),(name,rounds)
 rec={'name':name,'mpi':mpi,'order':order,'max_revisions':revs,'regime':regime,'dim':dim,'uniform':uniform,'reference':reference,'rounds':rounds}
 runs.append(rec);write();print('PASS',name,'revisions',sorted(set(x['revisions'] for x in rounds)),flush=True)
 return out
def write(): (root/'cpu-results.json').write_text(json.dumps({'runs':runs,'comparisons':comparisons},indent=2))
def compare(a,b):
 n=0
 for p in sorted((a/'bin').glob('*.bin')):
  ma,aa=study.assemble(p,reader);mb,bb=study.assemble(b/'bin'/p.name,reader)
  assert ma['time']==mb['time'] and aa.keys()==bb.keys()
  for v in aa:assert np.array_equal(aa[v],bb[v]),(a.name,b.name,p.name,v,float(np.max(abs(aa[v]-bb[v]))));n+=1
 assert n>=8,(a,b,n)
 comparisons.append({'reference':a.name,'test':b.name,'arrays_compared':n,'all_identical':True});write()
# Quiet ranks can terminate before even one correction, including self-neighbor cases.
a=execute('uniform_mpi1',uniform=True);b=execute('uniform_mpi4',uniform=True,mpi=4);compare(a,b)
for order in (4,6,8):
 a=execute(f'ot_fv{order}_mpi1',order);b=execute(f'ot_fv{order}_mpi4',order,mpi=4);compare(a,b)
a=execute('ot_fv8_3d_mpi1',dim=3);b=execute('ot_fv8_3d_mpi4',dim=3,mpi=4);compare(a,b)
for regime in ('sr','gr'):
 a=execute(regime+'_mpi1',regime=regime);b=execute(regime+'_mpi4',regime=regime,mpi=4);compare(a,b)
print('EARLY_STOP_CPU_MPI_PASS',flush=True)
# Numerical reference built independently from the unmodified published source.
for order in (4,6,8):
 a=execute(f'fixed_ot_fv{order}',order,reference=True);compare(a,root/f'ot_fv{order}_mpi1')
print('EARLY_STOP_FIXED_REFERENCE_PASS',flush=True)
