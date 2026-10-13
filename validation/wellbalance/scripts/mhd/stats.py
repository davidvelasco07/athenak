#!/usr/bin/env python3
"""max|var| over all blocks of a bin file, plus max Mach for hydro_w."""
import sys, os, glob, numpy as np
sys.path.insert(0, os.environ.get('ATHENAK_VIS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'vis', 'python')))
import bin_convert as bc
for f in sys.argv[1:]:
    d = bc.read_binary(f)
    md = d['mb_data']
    out = [f"t={d['time']:.4g}"]
    for v in d['var_names']:
        a = np.asarray(md[v])
        out.append(f"{v}:{np.max(np.abs(a)):.3e}")
    if 'velx' in d['var_names']:
        g = 5.0/3.0
        v2 = sum(np.asarray(md[c])**2 for c in ('velx','vely','velz'))
        cs = np.sqrt(g*(g-1)*np.asarray(md['eint'])/np.asarray(md['dens']))
        out.append(f"Mmax:{np.max(np.sqrt(v2)/cs):.3e}")
    print(f.split('/')[-1], ' '.join(out))
