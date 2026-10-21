#!/usr/bin/env python3
"""Compare hydro tracer block sizes at fixed resolution, physics and step count."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from run_physics import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[8,16,32,64])
    parser.add_argument('--trials',type=int,default=5)
    args = parser.parse_args()
    exe=args.exe.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'timings.jsonl').exists() or (out/'probes.json').exists():
        raise RuntimeError('Choose a fresh output directory')
    if any(n<4 or 512%n for n in args.sizes) or len(set(args.sizes))!=len(args.sizes):
        raise ValueError('Block sizes must be distinct divisors of 512, at least 4')
    deck=Path(__file__).resolve().parents[2]/'tst/inputs/sparse_tracers.athinput'
    base=['mesh/nx1=512','mesh/nx2=512','time/nlim=48',
          'passive_tracers/nfields=32','problem/radius=0.02',
          'passive_tracers/frame_interval=0','passive_tracers/profile=false']
    probes={};budgets={}
    for size in args.sizes:
        pairs=(512//size)**2*32
        options=base+[f'meshblock/nx1={size}',f'meshblock/nx2={size}',
                      'passive_tracers/storage=sparse',f'passive_tracers/capacity={pairs}',
                      'hydro/nscalars=32','passive_tracers/compare_dense=true']
        record=run(exe,deck,out/f'probe_{size}',options)
        assert record['cycles']==48 and record['compare_native']
        # Explicit offline capacity tuning, not an automatic production predictor.
        # 25% headroom rounded to 64 slots; identical rule on each GPU.
        capacity=min(pairs,64*math.ceil(record['peak_pairs']*1.25/64))
        record.update(block_size=size,possible_pairs=pairs,chosen_capacity=capacity)
        probes[size]=record;budgets[size]=capacity
        print('probe',size,'peak',record['peak_pairs'],'budget',capacity,flush=True)
    (out/'probes.json').write_text(json.dumps(probes,indent=2)+'\n')
    # Warm each (size,mode), then randomize across both block size and storage mode.
    rng=random.Random(2102026)
    cases=[(size,mode) for size in args.sizes for mode in ['native','dense','sparse']]
    for trial in range(-1,args.trials):
        order=cases.copy();rng.shuffle(order)
        for size,mode in order:
            options=base+[f'meshblock/nx1={size}',f'meshblock/nx2={size}',
                          f'passive_tracers/storage={mode}',
                          f'passive_tracers/capacity={budgets[size]}',
                          f'hydro/nscalars={32 if mode=="native" else 0}',
                          'passive_tracers/compare_dense=false']
            record=run(exe,deck,out/f'block_{size}'/f'{mode}_{trial}',options)
            assert record['cycles']==48
            record.update(block_size=size,capacity=budgets[size],trial=trial)
            if trial>=0:
                with (out/'timings.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
            print(size,mode,trial,record['evolution_seconds'],flush=True)
    (out/'binary-sha256.txt').write_text(hashlib.sha256(exe.read_bytes()).hexdigest()+'\n')


if __name__=='__main__':
    main()
