#!/bin/bash
# MOOD demotion counts with WB on a 1D hydro pulse, detection ON: REF (perturbation-scaled
# tolerance) vs NEW (full-state tolerance) vs WB off.  Smooth pulse amp 1e-3 (should not
# demote) and a strong pulse amp 1.0 that steepens (MUST demote with WB on and off).
# usage: nad_pulse.sh REF NEW
set -u; REF=$1; NEW=$2; cd ${RUNS:?set RUNS}; mkdir -p nad
T='tail=<output8>\nfile_type = log\ndcycle = 1\n'
run() { d=nad/$1; rm -rf $d; mkdir -p $d; python3 mkdeck.py $d/in.athinput "${@:3}" "$T"; (cd $d && $2 -i in.athinput > stdout.txt 2>&1 < /dev/null) || echo "rc!=0 $d"; }
nm() { awk 'NR>2{s+=$9}END{print s+0}' nad/$1/hse.log; }
C="fluid=hydro iprob=1 nx2=1 mbx2=1 mood=true integrator=rk3 rsolver=hllc pulse_x1=0.5 tlim=0.1 bin_dt=0.1 cfl=0.3"
printf "%-6s %-5s %-5s %10s %10s %10s\n" scheme N amp "nmood_ref" "nmood_new" "nmood_off"
for amp in 1e-3 1.0; do for sch in "fv4 5" "fv6 6" "fv8 7"; do set -- $sch
  for N in 64 128 256; do
    run ref_$1_a${amp}_N$N $REF wb=true recon=$1 ng=$2 nx1=$N mbx1=$N amp=$amp $C
    run new_$1_a${amp}_N$N $NEW wb=true recon=$1 ng=$2 nx1=$N mbx1=$N amp=$amp $C
    run off_$1_a${amp}_N$N $NEW wb=false recon=$1 ng=$2 nx1=$N mbx1=$N amp=$amp $C
    printf "%-6s %-5s %-5s %10s %10s %10s\n" $1 $N $amp "$(nm ref_$1_a${amp}_N$N)" "$(nm new_$1_a${amp}_N$N)" "$(nm off_$1_a${amp}_N$N)"
  done
done; done
echo "--- 2D MHD iprob 3 strong pulse (amp -0.95, the decomposition deck), fv4+MOOD uct, 64^2"
B="fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25 rsolver=hlld tlim=0.3 amp=-0.95 recon=fv4 mood=true integrator=rk3 emf=uct ng=7 bin_dt=0.3"
run ref_m3strong $REF wb=true $B; run new_m3strong $NEW wb=true $B; run off_m3strong $NEW wb=false $B
printf "m3 amp=-0.95: nmood ref=%s new=%s off=%s\n" "$(nm ref_m3strong)" "$(nm new_m3strong)" "$(nm off_m3strong)"
echo "--- decomposition invariance of the new detector (fv4+MOOD uct, WB on, 1 vs 4x4 blocks)"
for nb in 64 16; do run dec_new_$nb $NEW wb=true $B mbx1=$nb mbx2=$nb; done
python3 -I cmp_global.py nad/dec_new_64 nad/dec_new_16 | tail -1
