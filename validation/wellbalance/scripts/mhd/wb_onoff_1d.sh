#!/bin/bash
# max|w'_WBon - w'_WBoff| per scheme and binary (bash: zsh would not word-split $C)
set -u; REF=$1; NEW=$2; cd ${RUNS:?set RUNS}; export ATHENAK_VIS=${ATHENAK_VIS:?set ATHENAK_VIS to vis/python}
C="fluid=hydro iprob=1 nx2=1 mbx2=1 mood=true mood_detect=false integrator=rk4 rsolver=hllc amp=1e-3 pulse_x1=0.5 tlim=0.1 bin_dt=0.1"
run() { d=conv1d_off/$1; rm -rf $d; mkdir -p $d; python3 mkdeck.py $d/in.athinput "${@:3}" $C; (cd $d && $2 -i in.athinput > stdout.txt 2>&1 < /dev/null) || echo "rc!=0 $d"; }
for N in 32 64; do c=$(python3 -c "print(0.3*32/$N)")
  run ref_fv6off_N$N $REF wb=false recon=fv6 ng=6 nx1=$N mbx1=$N cfl=$c
  run ref_fv8off_N$N $REF wb=false recon=fv8 ng=7 nx1=$N mbx1=$N cfl=$c
  run new_fv8off_N$N $NEW wb=false recon=fv8 ng=7 nx1=$N mbx1=$N cfl=$c
  run ref_fv4off_N$N $REF wb=false recon=fv4 ng=5 nx1=$N mbx1=$N cfl=$c
  run ref_fv4_N$N $REF wb=true recon=fv4 ng=5 nx1=$N mbx1=$N cfl=$c
  run new_fv4off_N$N $NEW wb=false recon=fv4 ng=5 nx1=$N mbx1=$N cfl=$c
done
python3 - <<'PY'
import sys, os, glob, numpy as np
sys.path.insert(0, os.environ['ATHENAK_VIS']); import bin_convert as bc
def ep(d):
    f = sorted(glob.glob(f'conv1d_off/{d}/bin/*_wpert*.bin'))[-1]; b = bc.read_binary(f); md=b['mb_data']; lg=np.asarray(b['mb_logical'])
    nx=b['nx1_out_mb']; o=np.argsort(lg[:,0]); return np.concatenate([np.asarray(md['epert'])[m].ravel()[:nx] for m in o])
print("max|e'_WBon - e'_WBoff| (same binary, same scheme; pulse amp 1e-3, detection off, max|e'| ~ 8e-4)")
for sch in ('fv4','fv6','fv8'):
    for N in (32,64):
        row=[]
        for v in ('ref','new'):
            try: row.append(f"{v}: {np.max(np.abs(ep(f'{v}_{sch}_N{N}')-ep(f'{v}_{sch}off_N{N}'))):.2e}")
            except Exception as e: row.append(f'{v}: n/a ({type(e).__name__})')
        print(f'  {sch} N={N:3d}  ' + '   '.join(row))
PY
