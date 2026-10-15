"""Freeze completed sweep records and render figures while later cases run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--runs',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
data=json.loads((a.runs/'results.json').read_text())
(a.output/'results.json').write_text(json.dumps(data,indent=2)+'\n')
active=[]
for proc in Path('/proc').glob('[0-9]*/cwd'):
    try:
        cwd=proc.resolve()
        if cwd.parent==a.runs.resolve():active.append(cwd.name)
    except (OSError,RuntimeError):pass
(a.output/'checkpoint.json').write_text(json.dumps({
    'captured_at_utc':datetime.now(timezone.utc).isoformat(),
    'study_path':str(a.runs.resolve()),'active_cases':sorted(set(active)),
    'completed_cases':len(data['runs']),
    'expected_cases':len(data['config']['orders'].split(','))*len(data['config']['resolutions'].split(','))
},indent=2)+'\n')
for r in data['runs']:
    src=a.runs/r['name']
    target=a.output/'inputs'/r['name']
    target.mkdir(parents=True,exist_ok=True)
    for name in ('input.athinput','result.json','stdout.txt','ot.log','ot.hst'):
        if (src/name).exists():shutil.copy2(src/name,target/name)
    for history in src.glob('*.hst'):shutil.copy2(history,target/history.name)
    # The plotter reads unchanged binaries from completed cases. Archives should
    # exclude these links; the large raw fields remain in the original run tree.
    link=a.output/r['name']
    if not link.exists():link.symlink_to(src.resolve(),target_is_directory=True)
for t,out in [('.5',a.output/'figures'),('1',a.output/'figures/t1')]:
    subprocess.run([sys.executable,str(Path(__file__).with_name('plot_study.py')),
                    '--source',str(a.source),'--runs',str(a.output),
                    '--output',str(out),'--snapshot-time',t],check=True)
print('CHECKPOINT_READY',len(data['runs']),flush=True)
