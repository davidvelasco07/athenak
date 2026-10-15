import importlib.util,json,os,subprocess
from pathlib import Path
base=Path('/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution')
root=base/'cascade-halo-validation/minimum';runner=base/'pad-corrected-resolution/run_study.py'
spec=importlib.util.spec_from_file_location('study',runner);study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env['OMPI_MCA_opal_cuda_support']='false'
results=[]
for dim in (2,3):
 for solver,ng in [('hlld',5),('lhlld',6)]:
  p=root/f'memcheck_{dim}d_{solver}';p.mkdir(exist_ok=True)
  deck=study.input_deck(8,32,32,.002,revisions=4,rtol=1.e-8,nghost=ng,solver=solver)
  if solver=='lhlld':deck=deck.replace('emf = uct_hll\n','emf = uct_hlld\n')
  if dim==3:deck=deck.replace('nx3 = 1\n','nx3 = 32\n')
  (p/'input.athinput').write_text(deck)
  cmd=['mpiexec','--mca','pml','ob1','--mca','btl','self,vader,tcp','-n','1',
       'compute-sanitizer','--target-processes','application-only','--tool','memcheck',
       '--launch-timeout','20','--error-exitcode','99',str(root/'athena-cascade-flags'),'-i','input.athinput']
  with (p/'stdout.txt').open('w') as log:r=subprocess.run(cmd,cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
  result={'dim':dim,'solver':solver,'nghost':ng,'code':r.returncode,
          'zero_errors':'ERROR SUMMARY: 0 errors' in (p/'stdout.txt').read_text(),
          'mpi_cuda_support':False,'mpi_ranks':1,'target_processes':'application-only'}
  results.append(result);study.write_json(root/'memcheck-results.json',results)
  print(result,flush=True);assert r.returncode==0 and result['zero_errors'],result
print('MINIMUM_MEMCHECKS_PASS',flush=True)
