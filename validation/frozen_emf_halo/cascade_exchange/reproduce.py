from pathlib import Path
import importlib.util,json,subprocess,sys,time
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Compare integer cascade halos with a pre-exchange CPU executable.')
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--exe',type=Path,required=True)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=args.repo.resolve();root=args.output.resolve()
if root.exists():parser.error('output must be a fresh directory')
root.mkdir(parents=True)
spec=importlib.util.spec_from_file_location('study',source/'validation/frozen_emf_halo/ot_resolution/run_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sys.path.insert(0,str(source/'vis/python'));import bin_convert as reader
old=args.reference.resolve();new=args.exe.resolve()
runs=[];comparisons=[]
def execute(name,exe,order,revs,ng,nx,block,dim=2,mpi=1,solver='hlld'):
 p=root/name;p.mkdir(exist_ok=True)
 text=study.input_deck(order,nx,block,.01 if dim==2 else .002,solver=solver,revisions=revs,rtol=1.e-8,nghost=ng)
 if dim==3:text=text.replace('nx3 = 1\n',f'nx3 = {nx}\n').replace(f'<meshblock>\nnx1 = {block}\nnx2 = {block}\nnx3 = {nx}\n',f'<meshblock>\nnx1 = {block}\nnx2 = {block}\nnx3 = {block}\n')
 # Replace the meshblock nx3 explicitly; the original input places blocks last.
 if dim==3:
  a=text.index('<meshblock>');b=text.index('<time>',a)
  text=text[:a]+text[a:b].replace(f'nx3 = {nx}',f'nx3 = {block}')+text[b:]
 (p/'input.athinput').write_text(text)
 args=([str(exe)] if mpi==1 else ['mpiexec','-n',str(mpi),str(exe)])+['-i','input.athinput']
 start=time.monotonic()
 with (p/'stdout.txt').open('w') as out:status=subprocess.run(args,cwd=p,stdout=out,stderr=subprocess.STDOUT,timeout=300)
 result={'case':name,'code':status.returncode,'wall_seconds':time.monotonic()-start,'order':order,'revisions':revs,'nghost':ng,'dim':dim,'mpi':mpi,
         **study.analyze(p,nx,.01 if dim==2 else .002,reader)}
 result['clean']=result['clean'] and status.returncode==0
 runs.append(result);study.write_json(root/'minimum-results.json',{'runs':runs,'comparisons':comparisons})
 print(name,result['code'],result['clean'],flush=True)
 assert result['clean'],result
 return p

def compare(ref,test):
 files=[]
 for f in sorted((ref/'bin').glob('*.bin')):
  a,aa=study.assemble(f,reader);b,bb=study.assemble(test/'bin'/f.name,reader)
  files.append({'file':f.name,'same_time':a['time']==b['time'],
       'max_abs':max(float(np.max(abs(aa[v]-bb[v]))) for v in aa),
       'identical':bool(aa.keys()==bb.keys() and all(np.array_equal(aa[v],bb[v]) for v in aa))})
 result={'reference':ref.name,'test':test.name,'files':files,'all_identical':bool(files) and all(c['same_time'] and c['identical'] for c in files)}
 comparisons.append(result);study.write_json(root/'minimum-results.json',{'runs':runs,'comparisons':comparisons})
 assert result['all_identical'],result

for revs in (1,2,3,4):
 for order in (4,6,8):
  ref=execute(f'old_fv{order}_r{revs}',old,order,revs,max(order,order//2*revs+max(3,order//2)),64,64)
  for mpi in (1,4):
   p=execute(f'min_fv{order}_r{revs}_mpi{mpi}',new,order,revs,order//2+1,64,32,mpi=mpi)
   compare(ref,p)
for order in (4,6,8):
 ref=execute(f'old_3d_fv{order}',old,order,4,max(order,order//2*4+max(3,order//2)),32,32,dim=3)
 p=execute(f'min_3d_fv{order}',new,order,4,order//2+1,32,16,dim=3,mpi=4)
 compare(ref,p)
for solver in ('llf','hlle','lhlld'):
 for order in (4,6,8):
  ng=order//2+1+(solver=='lhlld')
  ref=execute(f'solver_{solver}_fv{order}_old',old,order,4,max(order,order//2*4+max(3,order//2)),64,64,solver=solver)
  p=execute(f'solver_{solver}_fv{order}_minimum',new,order,4,ng,64,32,mpi=4,solver=solver)
  compare(ref,p)
print('MINIMUM_HALOS_AND_SOLVERS_PASS',flush=True)
