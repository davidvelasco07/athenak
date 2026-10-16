"""Diagnostic-only forced cases for changing, capped and stable-invalid maps."""
from pathlib import Path
import argparse,collections,difflib,json,re,shlex,subprocess,sys
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--build',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);repo=args.repo.resolve();build=args.build.resolve()
sys.path.insert(0,str(repo/'vis/python'));import bin_convert as reader
source=repo/'src/mhd/mhd_mood.cpp';original=source.read_text()
s=original.replace('#include <algorithm>','#include <algorithm>\n#include <cstdlib>\n#include "globals.hpp"')
needle='  const bool frozen = mood_frozen_emf;'
s=s.replace(needle,needle+'\n  const int audit_mode = std::getenv("MOOD_CASCADE_AUDIT") ? std::atoi(std::getenv("MOOD_CASCADE_AUDIT")) : 0;\n  const bool audit_last = global_variable::my_rank == global_variable::nranks-1;\n  const bool audit_first = global_variable::my_rank == 0;')
needle='      // DC cannot descend further.'
force='''      const bool audit_pad_cell = audit_last && m == 0 && j == js && k == ks;
      if (audit_pad_cell && audit_mode == 1 && i == is && rev <= 2) danger = true;
      if (audit_pad_cell && (audit_mode == 2 || audit_mode == 4) && i == is) danger = true;
      if (audit_pad_cell && audit_mode == 3 && i == is+rev-1) danger = true;
'''
assert s.count(needle)==1;s=s.replace(needle,force+needle)
needle='      // The check after the last allowed correction must not change the levels:'
force='''      if (audit_mode == 4 && audit_first && m == 0 && k == ks && j == js+1 &&
          i == is+rev-1 && rev <= 5 && lv < 1) trouble = true;
'''
s=s.replace(needle,force+needle)
needle='      mood_nad_requests = nnad;'
s=s.replace(needle,'''      printf("AUDIT_CHECK rank=%d stage=%d revision=%d\\n",
             global_variable::my_rank,stage,mood_revision);
'''+needle)
copy=root/'audit_mhd_mood.cpp';copy.write_text(s)
(root/'audit.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='src/mhd/mhd_mood.cpp',tofile='audit/mhd_mood.cpp')))
entries=json.loads((build/'compile_commands.json').read_text());entry=next(e for e in entries if e['file']==str(source))
cmd=shlex.split(entry['command']);cmd.insert(1,'-I'+str(repo/'src/mhd'));cmd[cmd.index('-c')+1]=str(copy);cmd[cmd.index('-o')+1]=str(root/'audit.o')
with (root/'audit-build.log').open('w') as log:subprocess.run(cmd,cwd=entry['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
link=shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text());link[link.index('CMakeFiles/athena.dir/mhd/mhd_mood.cpp.o')]=str(root/'audit.o');exe=root/'athena-audit';link[link.index('-o')+1]=str(exe)
with (root/'audit-link.log').open('w') as log:subprocess.run(link,cwd=build/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
records=[]
base=(Path(__file__).resolve().parent/'uniform.athinput').read_text().replace('mood_nad_v = comps','mood_nad_v = off')
plan=[(mode,mpi,diag,8) for mode in (1,2,3,4) for mpi in (1,4) for diag in (False,True)]+[(2,4,True,2),(3,4,True,1),(4,4,True,3)]
for mode,mpi,diag,cap in plan:
 out=root/f'audit_m{mode}_p{mpi}_d{int(diag)}_r{cap}';out.mkdir(exist_ok=True)
 deck=re.sub(r'^mood_diagnostics = .*$',f'mood_diagnostics = {str(diag).lower()}',base,flags=re.M)
 deck=re.sub(r'^mood_max_revs = .*$',f'mood_max_revs = {cap}',deck,flags=re.M)
 (out/'input.athinput').write_text(deck)
 import os
 env={**os.environ,'MOOD_CASCADE_AUDIT':str(mode)}
 with (out/'stdout.txt').open('w') as log:
  p=subprocess.run((['mpiexec','-n',str(mpi)] if mpi>1 else [])+[str(exe),'-i','input.athinput'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
 text=(out/'stdout.txt').read_text();assert p.returncode==0,(out,text[-2000:])
 expected_rev=min(cap,{1:2,2:2,3:cap,4:5}[mode])
 expected_status='converged' if mode==1 and cap>=2 else 'stalled_unphysical' if (mode==2 and cap>=2) or (mode==4 and cap>=5) else 'revision_limit'
 checks=collections.defaultdict(list)
 for rank,stage,rev in re.findall(r'AUDIT_CHECK rank=(\d+) stage=(\d+) revision=(\d+)',text):checks[(int(rank),int(stage))].append(int(rev))
 assert len(checks)==mpi*3,(out,checks)
 assert all(v==list(range(expected_rev+1)) for v in checks.values()),(out,checks,expected_rev)
 rounds=[]
 for line in text.splitlines():
  if line.startswith('MOOD '):rounds.append(dict(re.findall(r'(\w+)=(\w+)',line)))
 if diag:
  assert len(rounds)==3,(out,rounds)
  for r in rounds:
   assert int(r['revisions'])==expected_rev and int(r['checks'])==expected_rev+1 and r['status']==expected_status,(out,r)
   assert int(r['nad_demotions'])==(expected_rev if mode==4 else 0),(out,r)
   assert int(r['pad_demotions'])==(expected_rev if mode in (1,2,3) else min(expected_rev,2)),(out,r)
   assert int(r['nad_pending'])==(1 if mode==4 and cap<5 else 0),(out,r)
   assert int(r['pad_pending'])==(1 if mode==3 or (mode in (1,2) and cap<2) else 0),(out,r)
   assert int(r['dc_bad'])==(1 if mode in (2,4) and expected_rev>=2 else 0),(out,r)
 else:assert not rounds
 files=sorted((out/'bin').glob('*fb_level*.bin'));d=reader.read_binary(str(files[-1]));vals=np.concatenate([np.asarray(v).ravel() for v in d['mb_data'].values()])
 expected_dc=int(mode in (1,2,4) and expected_rev>=2)
 expected_plm=expected_rev if mode==3 else expected_rev if mode==4 else int(expected_rev==1)
 assert np.count_nonzero(vals==2)==expected_dc and np.count_nonzero(vals==1)==expected_plm,(out,np.unique(vals,return_counts=True))
 records.append({'name':out.name,'mode':mode,'mpi':mpi,'diagnostics':diag,'cap':cap,'revisions':expected_rev,'status':expected_status,'checks_per_rank':{str(k):v for k,v in checks.items()},'rounds':rounds,'final_plm_cells':expected_plm,'final_dc_cells':expected_dc})
 (root/'audit-results.json').write_text(json.dumps(records,indent=2)+'\n')
 print('AUDIT_PASS',out.name,expected_rev,expected_status,flush=True)
print('ALL_CASCADE_AUDITS_PASS',len(records),flush=True)
