#!/bin/bash
# One-knob-at-a-time demotion counts of the WB MOOD detector (diagnostic of the amplitude-
# independent fences fixed 2026-10-13).  usage: RUNS=<dir with mkdeck.py> <script> <athena>
set -u; NEW=$1; cd ${RUNS:?set RUNS}; mkdir -p knobs
C="fluid=hydro iprob=1 nx2=1 mbx2=1 mood=true integrator=rk3 rsolver=hllc pulse_x1=0.5 tlim=0.1 bin_dt=0.1 cfl=0.3 recon=fv4 ng=5 nx1=128 mbx1=128 wb=true"
run() { n=$1; shift; d=knobs/$n; rm -rf $d; mkdir -p $d; python3 mkdeck.py $d/in.athinput $C "$@" "tail=<output8>\nfile_type = log\ndcycle = 1\n<hydro>\n$KN"; (cd $d && $NEW -i in.athinput > stdout.txt 2>&1 < /dev/null) || { echo "rc!=0 $n"; tail -2 $d/stdout.txt; }; printf "%-22s nmood=%s\n" $n "$(awk 'NR>2{s+=$9}END{print s+0}' $d/hse.log)"; }
KN=""              run base            amp=1e-3
KN=""              run amp1e-6         amp=1e-6
KN=""              run amp0            amp=0
KN="mood_rtol = 1e-2"      run rtol1e-2  amp=1e-3
KN="mood_atol = 1e-5"      run atol1e-5  amp=1e-3
KN="mood_nad_energy = false" run noenergy amp=1e-3
KN="mood_sed = false"      run nosed     amp=1e-3
KN="mood_nad_scale = relative" run relative amp=1e-3
KN="mood_nad_scale = gcfl" run gcfl      amp=1e-3
KN=""              run off             amp=1e-3 wb=false
KN="mood_rtol = 1e-2"      run off_rtol1e-2 amp=1e-3 wb=false
