#!/bin/bash
# eta sweep of a pressure pulse over magnetized hydrostatic atmospheres (iprob 3 uniform oblique B,
# iprob 4 Parker), WB on/off, fv4+MOOD (uct_hlld, rk3) and plm+FOFC (ct_contact).  64^2, t=0.25.
NEW=${ATH:?set ATH to the athena binary}
cd ${RUNS:?set RUNS to a run directory holding mkdeck.py and cmp_global.py}; mkdir -p sweep; : > sweep/results.txt
for prob in "m3|fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25 pulse_x1=0.3 pulse_x2=0.3" "m4|fluid=mhd iprob=4 gdir=2 beta=1.0"; do
  pn=${prob%%|*}; pa=${prob#*|}
  for sch in "fv4mood|recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=6" "plmfofc|recon=plm fofc=true rsolver=hlld emf=ct_contact ng=3"; do
    sn=${sch%%|*}; sa=${sch#*|}
    for eta in 1e-2 1e-4 1e-8 1e-12; do for wb in true false; do
      d=sweep/${pn}_${sn}_eta${eta}_wb$wb; rm -rf $d; mkdir -p $d
      python3 mkdeck.py $d/in.athinput wb=$wb $pa $sa amp=$eta tlim=0.25 bin_dt=0.25 'tail=<output8>\nfile_type = log\ndcycle = 1\n'
      (cd $d && $NEW -i in.athinput > log.txt 2>&1); rc=$?
      last=$(ls $d/bin/*_wpert*.bin 2>/dev/null | tail -1)
      python3 - "$pn" "$sn" "$eta" "$wb" "$rc" "$last" "$d/hse.log" >> sweep/results.txt <<'PY'
import sys, numpy as np, os
sys.path.insert(0, os.environ.get('ATHENAK_VIS', os.path.join(os.getcwd(), '..', '..', '..', '..', 'vis', 'python')))
import bin_convert as bc
pn,sn,eta,wb,rc,f,log=sys.argv[1:]
eta=float(eta); out=dict(prob=pn,scheme=sn,eta=eta,wb=wb,rc=int(rc))
if f and os.path.exists(f):
    d=bc.read_binary(f); md=d['mb_data']
    dp=np.max(np.abs(np.asarray(md['epert'])))*(5.0/3.0-1.0)   # delta p = (gamma-1) delta e
    out.update(t=d['time'], dp_over_eta=dp/eta, dd_over_eta=np.max(np.abs(np.asarray(md['dpert'])))/eta,
               vmax=np.max(np.sqrt(sum(np.asarray(md[c])**2 for c in ('vxpert','vypert','vzpert')))))
nm=0; nf=0
if os.path.exists(log):
    for l in open(log):
        if not l.startswith('#'):
            v=[int(x) for x in l.split()]; nm+=v[8]; nf+=v[1]+v[2]+v[3]
out.update(nmood=nm, nfloor=nf)
print(' '.join(f'{k}={v}' for k,v in out.items()))
PY
    done; done
  done
done
