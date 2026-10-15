from pathlib import Path
import importlib.util,json,subprocess,sys,time
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Check magnetic volume-average detection across blocks and MPI ranks.')
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--exe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=args.repo.resolve();exe=args.exe.resolve();root=args.output.resolve()
root.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('study',source/'validation/frozen_emf_halo/ot_resolution/run_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sys.path.insert(0,str(source/'vis/python'));import bin_convert as reader
runs=[];comparisons=[]
for order in (4,6,8):
    paths={}
    for name,block,ranks in [('one_block',64,1),('four_blocks',32,1),('mpi4',32,4)]:
        run=root/f'fv{order}_{name}';run.mkdir(exist_ok=True)
        (run/'input.athinput').write_text(study.input_deck(order,64,block,.01,revisions=4,rtol=1.e-8))
        cmd=([str(exe)] if ranks==1 else ['mpiexec','-n',str(ranks),str(exe)])+['-i','input.athinput']
        started=time.monotonic()
        with (run/'stdout.txt').open('w') as out:
            proc=subprocess.run(cmd,cwd=run,stdout=out,stderr=subprocess.STDOUT,timeout=180)
        result={'case':run.name,'code':proc.returncode,'wall_seconds':time.monotonic()-started,
                **study.analyze(run,64,.01,reader)}
        result['clean']=result['clean'] and proc.returncode==0
        runs.append(result);paths[name]=run
        study.write_json(root/'results.json',{'runs':runs,'comparisons':comparisons})
        print(run.name,result['clean'],proc.returncode,flush=True)
        assert result['clean'],result
    for name in ('four_blocks','mpi4'):
        files=[]
        for f in sorted((paths['one_block']/'bin').glob('*.bin')):
            a,aa=study.assemble(f,reader);b,bb=study.assemble(paths[name]/'bin'/f.name,reader)
            files.append({'file':f.name,'same_time':a['time']==b['time'],
                'identical':bool(aa.keys()==bb.keys() and all(np.array_equal(aa[v],bb[v]) for v in aa))})
        comparison={'order':order,'against':name,'files':files,
                    'all_identical':bool(files) and all(c['same_time'] and c['identical'] for c in files)}
        comparisons.append(comparison)
        study.write_json(root/'results.json',{'runs':runs,'comparisons':comparisons})
        assert comparison['all_identical'],comparison
print('CPU_MPI_CHECKS_PASS',flush=True)
