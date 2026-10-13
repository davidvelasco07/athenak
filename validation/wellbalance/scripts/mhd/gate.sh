#!/bin/bash
# WB exact gate for one hydrostatic case: runs the deck with well_balanced=true and =false
# (wb_diagnostics) and prints max|W'| / max|B'| of the last perturbation dump (0.0 = exact)
# and the max Mach number.  usage: ATH=<athena> RUNS=<dir with mkdeck.py,stats.py> gate.sh name "mkdeck args" [np]
set -u
ATH=${ATH:?set ATH to the athena binary}; RUNS=${RUNS:?set RUNS}
cd "$RUNS"; name=$1; args=$2; np=${3:-1}
for wb in true false; do
  d=gate/${name}_wb$wb; rm -rf $d; mkdir -p $d
  python3 mkdeck.py $d/in.athinput wb=$wb $args
  if [ "$np" -gt 1 ]; then (cd $d && mpirun -np $np $ATH -i in.athinput > log.txt 2>&1)
  else (cd $d && $ATH -i in.athinput > log.txt 2>&1); fi
  rc=$?
  last_u=$(ls $d/bin/*_wpert*.bin 2>/dev/null | tail -1)
  last_w=$(ls $d/bin/*_w.0*.bin 2>/dev/null | tail -1)
  s=$(python3 -I stats.py $last_u 2>/dev/null | sed 's/.*t=/t=/')
  m=$(python3 -I stats.py $last_w 2>/dev/null | grep -o "Mmax:[^ ]*")
  printf "%-30s wb=%-5s rc=%d %s %s\n" "$name" "$wb" "$rc" "$m" "$s"
  [ $rc -ne 0 ] && tail -3 $d/log.txt
done
