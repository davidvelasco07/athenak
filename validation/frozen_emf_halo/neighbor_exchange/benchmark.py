from physics_suite import *
import statistics
s=(W/'runs/fv8_r2_min/input.athinput').read_text()
s=setparam(s,'mhd','mood_test_dump','false');s=setparam(s,'time','nlim',32);s=setparam(s,'time','tlim',1);s=setparam(s,'time','ndiag',100)
for k in ['nx1','nx2']:s=setparam(s,'meshblock',k,32)
rows=[]
for i in range(3):
 for mode,ng in [('frozen',8),('legacy',18)]:
  deck=setparam(setparam(s,'mesh','nghost',ng),'mhd','mood_frozen_emf','true' if mode=='frozen' else 'false')
  name=f'bench_{mode}_{i}';p=RUNS/name;p.mkdir(parents=True,exist_ok=True);(p/'input.athinput').write_text(deck)
  cmd=[str(EXE),'-i','input.athinput']
  with (p/'run.log').open('w') as out:code=subprocess.run(cmd,cwd=p,stdout=out,stderr=subprocess.STDOUT,timeout=120).returncode
  assert code==0
  log=(p/'run.log').read_text();cpu=float(re.search(r'cpu time used\s*=\s*([\d.e+-]+)',log)[1]);mem=None
  row={'mode':mode,'seconds':cpu,'rss_bytes':mem,'ng':ng};rows.append(row);print(row,flush=True)
(W/'benchmark-results.json').write_text(json.dumps({'runs':rows,'median_seconds':{m:statistics.median(r['seconds'] for r in rows if r['mode']==m) for m in ['frozen','legacy']}},indent=2))
