from pathlib import Path
import importlib.util,subprocess,json,os,sys
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Compare double candidate snapshots from audit.patch, with short-exchange negative controls.')
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--exe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
src=args.repo.resolve();root=args.output.resolve()
if root.exists():parser.error('output must be a fresh directory')
root.mkdir(parents=True)
spec=importlib.util.spec_from_file_location('study',src/'validation/frozen_emf_halo/ot_resolution/run_study.py');study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
exe=args.exe.resolve();outroot=root
results=[]
def execute(order,dim,blocks,mpi,short=False):
 name=f'fv{order}_{dim}d_b{blocks}_mpi{mpi}'+('_short' if short else '')
 p=outroot/name;p.mkdir(exist_ok=True)
 ng=max(order,order//2*4+max(3,order//2)) if blocks==1 else order//2+1
 deck=study.input_deck(order,32,32//blocks,1.,revisions=4,rtol=1.e-8,nghost=ng).replace('nlim = -1','nlim = 1')
 if dim==3:
  deck=deck.replace('nx3 = 1\n','nx3 = 32\n');a=deck.index('<meshblock>');b=deck.index('<time>',a)
  deck=deck[:a]+deck[a:b].replace('nx3 = 32',f'nx3 = {32//blocks}')+deck[b:]
 (p/'input.athinput').write_text(deck)
 env=os.environ.copy()
 if short:env['MOOD_AUDIT_SHORT_FLAGS']='1'
 cmd=(['mpiexec','-n',str(mpi)] if mpi>1 else [])+[str(exe),'-i','input.athinput']
 with (p/'stdout.txt').open('w') as log:r=subprocess.run(cmd,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 assert r.returncode==0,(name,(p/'stdout.txt').read_text()[-2000:])
 print(name,'ran',flush=True)
 return p

def arrays(p,key):
 a=np.concatenate([np.fromfile(f,dtype=np.float64).reshape(-1,12) for f in p.glob(key+'_rank*.dat')])
 return a[np.lexsort((a[:,0],a[:,1],a[:,2]))]

def compare(ref,test,negative=False):
 files=[]
 for f in sorted(ref.glob('audit_*_rank0.dat')):
  key=f.name.split('_rank')[0];a=arrays(ref,key);b=arrays(test,key)
  assert a.shape==b.shape and np.array_equal(a[:,:3],b[:,:3]),(key,a.shape,b.shape)
  error=float(np.max(abs(a[:,3:11]-b[:,3:11])))
  files.append({'key':key,'max_abs':error,'levels_identical':bool(np.array_equal(a[:,11],b[:,11])),
                'candidate_identical':bool(np.array_equal(a[:,3:11],b[:,3:11]))})
 assert len(files)==12,(ref,test,len(files))
 result={'reference':ref.name,'test':test.name,'negative':negative,'files':files,'max_abs':max(f['max_abs'] for f in files),
         'levels_identical':all(f['levels_identical'] for f in files),'candidates_identical':all(f['candidate_identical'] for f in files)}
 if negative:assert result['max_abs']>1.e-12,result
 else:assert result['levels_identical'] and result['max_abs']<1.e-13,result
 results.append(result);study.write_json(outroot/'results.json',results)
 print(test.name,result['max_abs'],result['candidates_identical'],flush=True)

for dim in (2,3):
 for order in (4,6,8):
  ref=execute(order,dim,1,1)
  test=execute(order,dim,4,4)
  compare(ref,test)
  if order==8:
   short=execute(order,dim,4,4,True)
   compare(ref,short,True)
print('FORCED_CASCADE_CONTROLS_PASS',flush=True)
