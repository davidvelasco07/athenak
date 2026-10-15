"""Update saved report after case completions; exits once this finite study ends."""
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent;study=root/'ot-resolution-early-stop'
os.environ['MPLCONFIGDIR']=str(study/'matplotlib');last=-1
while True:
 try:d=json.loads((study/'results.json').read_text());n=len(d['runs'])
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(30);continue
 if n!=last:
  subprocess.run([sys.executable,str(root/'ot-scripts/report.py'),'--root',str(root)],check=True)
  subprocess.run([sys.executable,str(root/'ot-scripts/compare_reference.py')],check=True)
  reportdir=study/'report'
  import shutil
  shutil.copy2(study/'reference-comparison.json',reportdir/'reference-comparison.json')
  archive=root/'ot-current-report.tar.gz';temporary=archive.with_suffix('.tmp')
  with tarfile.open(temporary,'w:gz') as tar:tar.add(reportdir,arcname='report')
  temporary.replace(archive)
  status={'recorded_cases':n,'planned_cases':21,'done':n==21 or bool(d.get('worker_errors')),'worker_errors':d.get('worker_errors',[]),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'updated_unix':time.time()}
  path=root/'ot-report-status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path)
  last=n
 if n==21 or d.get('worker_errors'):break
 time.sleep(30)
print('FINAL_OT_REPORT_READY',flush=True)
