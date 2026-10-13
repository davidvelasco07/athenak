#!/usr/bin/env python3
"""L1 error of the perturbation energy of a 1D pulse run against a fine reference run,
reference cell-averaged onto the coarse grid.  usage: conv1d.py refdir dir [dir...]"""
import sys, os, glob, numpy as np
sys.path.insert(0, os.environ.get('ATHENAK_VIS'))
import bin_convert as bc
def epert(d):
    f = sorted(glob.glob(f'{d}/bin/*_wpert*.bin'))[-1]
    b = bc.read_binary(f); md = b['mb_data']; lg = np.asarray(b['mb_logical'])
    nx = b['nx1_out_mb']; order = np.argsort(lg[:,0])
    return np.concatenate([np.asarray(md['epert'])[m].ravel()[:nx] for m in order]), b['time']
ref, tref = epert(sys.argv[1]); Nr = ref.size
for d in sys.argv[2:]:
    e, t = epert(d); N = e.size; r = Nr//N
    rc = ref.reshape(N, r).mean(axis=1)
    log = f'{d}/hse.log'; nm = 0
    if os.path.exists(log):
        for l in open(log):
            if not l.startswith('#'): nm += int(l.split()[8])
    print(f'{os.path.basename(d):28s} N={N:5d} t={t:.3f} L1={np.mean(np.abs(e-rc)):.3e} Linf={np.max(np.abs(e-rc)):.3e} nmood={nm}')
