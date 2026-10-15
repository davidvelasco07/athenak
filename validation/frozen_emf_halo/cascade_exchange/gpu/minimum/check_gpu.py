import importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
base=Path('/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution')
root=base/'cascade-halo-validation/minimum';source=base/'pad-audit/source';oldroot=base/'volume-average-validation'
runner=base/'pad-corrected-resolution/run_study.py'
spec=importlib.util.spec_from_file_location('study',runner);study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sys.path.insert(0,str(source/'vis/python'));import bin_convert as reader
exe=root/'athena-cascade-flags';summary=[];comparisons=[]
used=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
assert used<50,f'GPU1 busy: {used}MiB'
for name,order,nx,block,mpi in [('fv4_n128',4,128,32,1),('fv6_n128',6,128,32,1),
                              ('fv8_n128',8,128,32,1),('fv4_n512',4,512,64,1)]:
 p=root/name;p.mkdir(exist_ok=True)
 (p/'input.athinput').write_text(study.input_deck(order,nx,block,1.,revisions=4,rtol=1.e-8,nghost=order//2+1))
 cmd=([str(exe)] if mpi==1 else ['mpiexec','-n',str(mpi),str(exe)])+['-i','input.athinput']
 started=time.monotonic()
 with (p/'stdout.txt').open('w') as out:
  proc=subprocess.run(cmd,cwd=p,stdout=out,stderr=subprocess.STDOUT,timeout=1800)
 r={'case':name,'order':order,'nx':nx,'nghost':order//2+1,'mpi':mpi,'code':proc.returncode,
    'wall_seconds':time.monotonic()-started,**study.analyze(p,nx,1.,reader)}
 r['clean']=r['clean'] and proc.returncode==0;summary.append(r)
 study.write_json(root/'results.json',{'runs':summary,'comparisons':comparisons})
 print(name,r['clean'],r.get('solver_seconds'),flush=True)
 assert r['clean'],r
 ref=oldroot/f'fv{order}_n{nx}_b64'/f'ot_fv{order}_n{nx}'
 files=[]
 for f in sorted((ref/'bin').glob('*.bin')):
  a,aa=study.assemble(f,reader);b,bb=study.assemble(p/'bin'/f.name,reader)
  files.append({'file':f.name,'time':float(a['time']),'identical':bool(a['time']==b['time'] and aa.keys()==bb.keys() and all(np.array_equal(aa[v],bb[v]) for v in aa)),
                'max_abs':max(float(np.max(abs(aa[v]-bb[v]))) for v in aa)})
 c={'case':name,'reference':str(ref),'files':files,'all_identical':bool(files) and all(f['identical'] for f in files)}
 comparisons.append(c);study.write_json(root/'results.json',{'runs':summary,'comparisons':comparisons})
 assert c['all_identical'],c
mpi_controls=[]
for ranks in (1,4):
 p=root/f'fv8_short_mpi{ranks}';p.mkdir(exist_ok=True)
 deck=study.input_deck(8,64,16,1.,revisions=4,rtol=1.e-8,nghost=5).replace('nlim = -1','nlim = 8')
 (p/'input.athinput').write_text(deck)
 cmd=(['mpiexec','-n',str(ranks)] if ranks>1 else [])+[str(exe),'-i','input.athinput']
 with (p/'stdout.txt').open('w') as log:proc=subprocess.run(cmd,cwd=p,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 r={'case':p.name,'mpi':ranks,'code':proc.returncode,**study.analyze(p,64,1.,reader)}
 r['step_control_successful']=bool(proc.returncode==0 and r['cycles']==8 and r['finite'] and r['positive'] and all(r.get(k,0)==0 for k in ('density_floors','energy_floors','temperature_floors','velocity_ceilings','c2p_failures')))
 mpi_controls.append(r);assert r['step_control_successful'],r
 print('GPU_MPI_STEP_CONTROL',ranks,'PASS',flush=True)
left=root/'fv8_short_mpi1';right=root/'fv8_short_mpi4';files=[]
for f in sorted((left/'bin').glob('*.bin')):
 a,aa=study.assemble(f,reader);b,bb=study.assemble(right/'bin'/f.name,reader)
 files.append({'file':f.name,'identical':bool(a['time']==b['time'] and all(np.array_equal(aa[v],bb[v]) for v in aa))})
c={'case':'fv8_short_mpi4','files':files,'all_identical':bool(files) and all(f['identical'] for f in files)}
comparisons.append(c);assert c['all_identical'],c
study.write_json(root/'results.json',{'runs':summary,'comparisons':comparisons,'mpi_controls':mpi_controls})
for dim in (2,3):
 for solver,ng in [('hlld',5),('lhlld',6)]:
  p=root/f'memcheck_{dim}d_{solver}';p.mkdir(exist_ok=True)
  deck=study.input_deck(8,32,32,.002,revisions=4,rtol=1.e-8,nghost=ng,solver=solver)
  if solver=='lhlld':deck=deck.replace('emf = uct_hll\n','emf = uct_hlld\n')
  if dim==3:deck=deck.replace('nx3 = 1\n','nx3 = 32\n')
  (p/'input.athinput').write_text(deck)
  with (p/'stdout.txt').open('w') as log:
   proc=subprocess.run(['compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(exe),'-i','input.athinput'],cwd=p,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  assert proc.returncode==0 and 'ERROR SUMMARY: 0 errors' in (p/'stdout.txt').read_text(),(dim,solver,proc.returncode)
  print('MEMCHECK_PASS',dim,solver,flush=True)
print('MINIMUM_GPU_CASES_AND_COMPARISONS_PASS',flush=True)
