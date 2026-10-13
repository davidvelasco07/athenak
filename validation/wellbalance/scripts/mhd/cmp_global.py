#!/usr/bin/env python3
"""Decomposition-aware comparison of two AthenaK run dirs: every bin dump is reassembled on
the global uniform grid from the block logical locations, then compared variable by variable
(max |a-b|, exact 0.0 = bitwise invariant).  Usage: cmp_global.py dirA dirB"""
import sys, os, os, glob, numpy as np
sys.path.insert(0, os.environ.get('ATHENAK_VIS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'vis', 'python')))
import bin_convert as bc
def assemble(path):
    d = bc.read_binary(path)
    nx1, nx2, nx3 = d['nx1_out_mb'], d['nx2_out_mb'], d['nx3_out_mb']
    lg = np.asarray(d['mb_logical'])
    out = {}
    for v in d['var_names']:
        a = np.asarray(d['mb_data'][v])
        G = np.zeros((d['Nx3'], d['Nx2'], d['Nx1']))
        for m in range(d['n_mbs']):
            l1, l2, l3 = lg[m][0], lg[m][1], lg[m][2]
            G[l3*nx3:(l3+1)*nx3, l2*nx2:(l2+1)*nx2, l1*nx1:(l1+1)*nx1] = a[m]
        out[v] = G
    return d['time'], out
a, b = sys.argv[1], sys.argv[2]; ok = True
for f in sorted(glob.glob(f'{a}/bin/*.bin')):
    g = f'{b}/bin/{os.path.basename(f)}'
    ta, A = assemble(f); tb, B = assemble(g)
    worst = max(float(np.max(np.abs(A[v]-B[v]))) for v in A)
    if ta != tb or worst != 0.0:
        ok = False; print(f'DIFF {os.path.basename(f)}: t {ta} vs {tb}, max|diff| = {worst:.3e}')
print('IDENTICAL' if ok else 'DIFFERENT', a, 'vs', b)
