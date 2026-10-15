"""Two independent GPU workers; immutable case outputs and incremental records."""
import argparse,concurrent.futures,hashlib,json,os,queue,shutil,subprocess,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args()
root=args.root.resolve()
source=root/'source';study=root/'ot-resolution-early-stop';study.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'ot-scripts'));import run_study as ot
sys.path.insert(0,str(source/'vis/python'));import bin_convert as reader
exe=root/'build-ot/src/athena'
provenance=json.loads((root/'ot-build.json').read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest()==provenance['executable_sha256']
report={'source_commit':provenance['source_commit'],'executable_sha256':provenance['executable_sha256'],
 'config':{'resolutions':'64,128,256,512,1024,2048,4096','orders':'4,6,8','block':64,'tlim':1.,'max_revisions':4,'rtol':1.e-8,'cfl':.25,'solver':'hlld','emf':'uct_hlld','frozen':True,'nad':'rcfl','mood_diagnostics':True,'continue_after_floor':True,'nghost':{'4':3,'6':4,'8':5},'gpus':[0,1],'timeout_seconds':172800},'runs':[]}
jobs=queue.Queue()
for nx in (64,128,256,512,1024,2048,4096):
 for order in (4,6,8):jobs.put((nx,order))
ot.write_json(study/'results.json',report)
shutil.copy2(root/'ot-build.json',study/'build-provenance.json')
def worker(gpu):
 while True:
  try:nx,order=jobs.get_nowait()
  except queue.Empty:return
  name=f'ot_fv{order}_n{nx}';run=study/name;run.mkdir(exist_ok=True)
  resultfile=run/'result.json'
  if resultfile.exists():
   result=json.loads(resultfile.read_text());yield result;continue
  (run/'input.athinput').write_text(ot.input_deck(order,nx,64,1.,revisions=4,rtol=1.e-8,diagnostics=True))
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
  t0=time.monotonic();reason=None
  print('START',name,'GPU',gpu,flush=True)
  with (run/'stdout.txt').open('w') as log:
   proc=subprocess.Popen([str(exe),'-i','input.athinput'],cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
   ot.write_json(run/'running.json',{'pid':proc.pid,'gpu':gpu,'started_unix':time.time(),'executable':str(exe)})
   try:code=proc.wait(timeout=172800)
   except subprocess.TimeoutExpired:
    reason='wall_timeout';proc.terminate()
    try:code=proc.wait(timeout=15)
    except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
  result={'name':name,'order':order,'nx':nx,'nghost':max(3,order//2+1),'gpu':gpu,'code':code,'stop_reason':reason,'wall_seconds':time.monotonic()-t0}
  try:result.update(ot.analyze(run,nx,1.,reader))
  except Exception as exc:result.update(clean=False,analysis_error=repr(exc))
  result['clean']=bool(result.get('clean',False) and code==0)
  result=ot.finite_json(result);ot.write_json(resultfile,result)
  print('DONE',json.dumps({k:v for k,v in result.items() if k not in ('snapshots','history')}),flush=True)
  yield result
# The two worker iterators each own one GPU; only the main thread writes aggregate JSON.
messages=queue.Queue()
def consume(gpu):
 try:
  for result in worker(gpu):messages.put(result)
 except BaseException as exc:messages.put({'worker_error':repr(exc),'gpu':gpu})
 finally:messages.put(None)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(consume,gpu) for gpu in (0,1)];done=0
 while done<2:
  result=messages.get()
  if result is None:done+=1;continue
  if 'worker_error' in result:
   report.setdefault('worker_errors',[]).append(result)
  else:report['runs'].append(result);report['runs'].sort(key=lambda r:(r['nx'],r['order']))
  ot.write_json(study/'results.json',report)
 for future in futures:future.result()
print('STUDY_COMPLETE',len(report['runs']),flush=True)
