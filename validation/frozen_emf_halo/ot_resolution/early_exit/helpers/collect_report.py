"""Collect report updates for this finite 21-case GPU batch, then exit."""
import hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
output=root/'ot-fallback-resolution-study';output.mkdir(exist_ok=True)
work=Path(__file__).resolve().parent
remote='/scratch/lustre/velasco/codex-validation/early-stop-20261015'
socket='/private/tmp/codex-ot-early-stop-20261015.sock'
options=['-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+socket,'-o','BatchMode=yes']
last=None;deadline=time.monotonic()+4*24*3600
while time.monotonic()<deadline:
 try:
  result=subprocess.run(['ssh',*options,'apollo','cat '+remote+'/ot-report-status.json'],capture_output=True,text=True,timeout=120,check=True)
  status=json.loads(result.stdout)
  if status['archive_sha256']!=last:
   archive=work/'current-report.tar.gz'
   subprocess.run(['scp',*options,'apollo:'+remote+'/ot-current-report.tar.gz',str(archive)],check=True,timeout=300)
   if hashlib.sha256(archive.read_bytes()).hexdigest()!=status['archive_sha256']:
    print('Archive changed during collection; retrying',flush=True);time.sleep(10);continue
   staging=work/'staging';shutil.rmtree(staging,ignore_errors=True);staging.mkdir()
   with tarfile.open(archive,'r:gz') as tar:tar.extractall(staging,filter='data')
   for source in (staging/'report').rglob('*'):
    if not source.is_file():continue
    target=output/source.relative_to(staging/'report');target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+'.collect-tmp');shutil.copy2(source,tmp);tmp.replace(target)
   (output/'collector-status.json').write_text(json.dumps(status,indent=2)+'\n')
   last=status['archive_sha256']
   print('REPORT_COLLECTED',status['recorded_cases'],'/',status['planned_cases'],time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),flush=True)
  if status.get('done'):
   print('COLLECTION_FINISHED',flush=True);break
 except Exception as exc:print('Collection retry:',repr(exc),flush=True)
 time.sleep(60)
else:print('Collection deadline reached; remote results remain preserved',flush=True)
