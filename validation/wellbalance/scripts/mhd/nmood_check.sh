#!/bin/bash
# demotion count at rest: WB on, fv4+MOOD, event log every cycle
NEW=${ATH:?set ATH to the athena binary}
cd ${RUNS:?set RUNS to a run directory holding mkdeck.py and cmp_global.py}; mkdir -p p2/nmood
while IFS='|' read -r n a; do
  d=p2/nmood/$n; rm -rf $d; mkdir -p $d
  python3 mkdeck.py $d/in.athinput wb=true $a 'tail=<output8>\nfile_type = log\ndcycle = 1\n'
  (cd $d && $NEW -i in.athinput > log.txt 2>&1); rc=$?
  printf "%-8s rc=%s cycles=%s mood_demotions=%s floors=%s\n" $n $rc "$(grep -c '^ *[0-9]' $d/hse.log)" "$(awk 'NR>2{s+=$9}END{print s+0}' $d/hse.log)" "$(awk 'NR>2{s+=$2+$3+$4}END{print s+0}' $d/hse.log)"
done <<'LIST'
m3ct|fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=ct_contact ng=5 tlim=0.5
m4hll|fluid=mhd iprob=4 gdir=2 beta=1.0 recon=fv4 mood=true integrator=rk3 rsolver=lhlld emf=uct_hll ng=7 tlim=0.5
m5hlld|fluid=mhd iprob=5 gdir=2 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=7 tlim=0.5 mbx1=16 mbx2=16
m5off|fluid=mhd iprob=5 gdir=2 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=7 tlim=0.5 wb=false
LIST
