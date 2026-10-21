#!/usr/bin/env python3
"""Run real hydro/SC pilots. Timing runs never write visualization frames or full states."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess


def run(exe,deck,folder,options):
    folder.mkdir(parents=True,exist_ok=True)
    command=[str(exe),'-i',str(deck),*options]
    result=subprocess.run(command,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,timeout=600)
    (folder/'run.log').write_text(result.stdout)
    (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(f'Failed {folder}:\n'+result.stdout[-6000:])
    times=re.findall(r'physics_evolution_seconds=([0-9.eE+\-]+)',result.stdout)
    assert len(times)==1,(folder,times)
    report_file=folder/('tracers.json' if 'hydro' in deck.name or 'tracers' in deck.name else 'beam.json')
    report=json.loads(report_file.read_text());report['evolution_seconds']=float(times[0])
    if 'converged' in report and not report['converged']:
        raise RuntimeError(f'Unconverged beam run: {folder}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',choices=['benchmark','movies'],default='benchmark')
    parser.add_argument('--trials',type=int,default=5)
    args=parser.parse_args()
    exe=args.exe.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    hydro=root/'tst/inputs/sparse_tracers.athinput'
    beam=root/'tst/inputs/sc_beam_sparse.athinput'
    if args.suite=='movies':
        for mode in ['native','sparse']:
            options=[f'passive_tracers/storage={mode}','passive_tracers/nfields=4',
                     'hydro/nscalars='+('4' if mode=='native' else '0'),
                     'passive_tracers/compare_dense=false','mesh/nx1=192','mesh/nx2=192',
                     'time/nlim=320','problem/vx=4.0','problem/vy=0.0','problem/radius=0.035',
                     'passive_tracers/capacity=240','passive_tracers/frame_interval=4',
                     'passive_tracers/frames=frames']
            run(exe,hydro,out/('hydro_'+mode),options)
        for mode in ['dense','sparse']:
            options=[f'nr_radiation/intensity_storage={mode}','mesh/nx1=256','mesh/nx2=128',
                     'problem/ihor1=64','problem/ihor2=191','nr_radiation/dump_final=false',
                     'nr_radiation/frame_interval=1','nr_radiation/frames=frames']
            run(exe,beam,out/('beam_'+mode),options)
        return
    cases={
        'hydro_small':(hydro,['time/nlim=32','passive_tracers/nfields=8',
                             'passive_tracers/capacity=384'],8,['native','dense','forced','sparse']),
        'hydro_large':(hydro,['mesh/nx1=512','mesh/nx2=512','meshblock/nx1=32',
                             'meshblock/nx2=32','time/nlim=48','passive_tracers/nfields=32',
                             'problem/radius=0.02','passive_tracers/capacity=2048'],32,
                       ['native','dense','forced','sparse']),
        'beam_small':(beam,[],0,['dense','selected','forced','sparse']),
        'beam_large':(beam,['mesh/nx1=512','mesh/nx2=256','meshblock/nx1=32',
                           'meshblock/nx2=32','problem/ihor1=128','problem/ihor2=383'],0,
                      ['dense','selected','forced','sparse']),
    }
    if (out/'timings.jsonl').exists():
        raise RuntimeError('Choose a fresh output directory; timings already exist')
    rows=[]
    rng=random.Random(7331)
    for case,(deck,base,nf,modes) in cases.items():
        # One untimed/discarded full run per mode; randomized subsequent trial order.
        for trial in range(-1,args.trials):
            order=modes.copy();rng.shuffle(order)
            for mode in order:
                if nf:
                    options=[f'passive_tracers/storage={mode}',
                             f'hydro/nscalars={nf if mode=="native" else 0}',
                             'passive_tracers/compare_dense=false',
                             'passive_tracers/frame_interval=0','passive_tracers/profile=false']
                else:
                    options=[f'nr_radiation/intensity_storage={mode}',
                             'nr_radiation/frame_interval=0','nr_radiation/dump_final=false']
                report=run(exe,deck,out/case/f'{mode}_{trial}',base+options)
                if trial>=0:
                    report.update(case=case,trial=trial)
                    rows.append(report)
                    with (out/'timings.jsonl').open('a') as stream:
                        stream.write(json.dumps(report)+'\n')
                print(case,mode,trial,report['evolution_seconds'],flush=True)
    (out/'binary-sha256.txt').write_text(hashlib.sha256(exe.read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
