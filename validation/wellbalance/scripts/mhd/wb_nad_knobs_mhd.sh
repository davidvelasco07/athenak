#!/bin/bash
# One-knob-at-a-time demotion counts of the WB MOOD detector (diagnostic of the amplitude-
# independent fences fixed 2026-10-13).  usage: RUNS=<dir with mkdeck.py> <script> <athena>
set -u; NEW=$1; cd ${RUNS:?set RUNS}; mkdir -p knobs_mhd
C="fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct ng=7 tlim=0.25 bin_dt=0.25 wb=true"
run() { n=$1; shift; d=knobs_mhd/$n; rm -rf $d; mkdir -p $d; python3 mkdeck.py $d/in.athinput $C "$@" "tail=<output8>\nfile_type = log\ndcycle = 1\n<mhd>\n$KN"; (cd $d && $NEW -i in.athinput > stdout.txt 2>&1 < /dev/null) || { echo "rc!=0 $n"; tail -2 $d/stdout.txt; }; printf "%-22s nmood=%-8s floors=%s\n" $n "$(awk 'NR>2{s+=$9}END{print s+0}' $d/hse.log)" "$(awk 'NR>2{s+=$2+$3+$4}END{print s+0}' $d/hse.log)"; }
for eta in 1e-2 1e-12; do
KN=""                       run base_$eta        amp=$eta
KN="mood_rtol = 1e-2"       run rtol1e-2_$eta    amp=$eta
KN="mood_atol = 1e-5"       run atol1e-5_$eta    amp=$eta
KN="mood_nad_energy = false" run noenergy_$eta   amp=$eta
KN="mood_sed = false"       run nosed_$eta       amp=$eta
KN=""                       run nadbmag_$eta     amp=$eta nad_b=mag
KN="mood_nad_scale = relative" run relative_$eta amp=$eta
KN=""                       run ctcontact_$eta   amp=$eta emf=ct_contact ng=5
KN=""                       run off_$eta         amp=$eta wb=false
done
KN="" run rest amp=0
