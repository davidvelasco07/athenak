#!/bin/bash
# the rank rows of the WB gate matrix with an MPI binary: exact gate on 4 ranks (rank count
# printed from the log) and 1-vs-4-rank bitwise with the SAME binary.  usage: rank_gates.sh ATH_MPI
set -u; export ATH=$1; export RUNS=${RUNS:?set RUNS}; cd $RUNS; G="$(cd "$(dirname "$0")" && pwd)/gate.sh"
B3="fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25"; PB="$B3 gdir=2 ix1=periodic ox1=periodic ix2=reflect ox2=outflow"
SMR='tail=<mesh_refinement>\nrefinement=static\nnum_levels=2\n<refined_region1>\nlevel=1\nx1min=0.3\nx1max=0.7\nx2min=0.0\nx2max=1.0\n'
AMR='tail=<mesh_refinement>\nrefinement=adaptive\nnum_levels=3\nrefinement_interval=5\nmax_nmb_per_rank=1024\n<amr_criterion0>\nmethod=user\n'
echo "--- exact gates on 4 ranks"
$G r4_m3_fv4_ct      "$B3 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=ct_contact ng=5 tlim=0.5 mbx1=16 mbx2=16" 4
$G r4_m3_fv6_uct     "$B3 recon=fv6 mood=true integrator=rk3 rsolver=hlld emf=uct ng=10 tlim=0.5 mbx1=16 mbx2=16" 4
$G r4_smr_plm_fofc   "$B3 recon=plm fofc=true rsolver=lhlld emf=uct ng=4 tlim=0.5 mbx1=16 mbx2=16 $SMR" 4
$G r4_amr_fv4_uct    "$PB recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct ng=8 tlim=0.5 mbx1=16 mbx2=16 extra=amr_test=band\namr_band_w=0.2\namr_band_v=2.0 $AMR" 4
$G r4_h1_fv8         "fluid=hydro iprob=1 recon=fv8 mood=true integrator=rk3 rsolver=hllc ng=7 tlim=0.5 mbx1=16 mbx2=16" 4
echo "--- 1 vs 4 ranks bitwise (perturbed, amp 1e-2), same binary"
mkdir -p rk
while IFS='|' read -r n a; do
  for np in 1 4; do d=rk/${n}_np$np; rm -rf $d; mkdir -p $d; python3 mkdeck.py $d/in.athinput amp=1e-2 $a
    (cd $d && mpirun -np $np $ATH -i in.athinput > log.txt 2>&1 < /dev/null) || echo "rc!=0 $d"; done
  printf "%-20s ranks=%s/%s " $n "$(grep -m1 'parallel ranks' rk/${n}_np1/log.txt | awk '{print $NF}')" "$(grep -m1 'parallel ranks' rk/${n}_np4/log.txt | awk '{print $NF}')"
  python3 -I cmp_global.py rk/${n}_np1 rk/${n}_np4 | tail -1
done <<'LIST'
m3_fv4_uct_wb|fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25 wb=true recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct ng=7 tlim=0.3 mbx1=16 mbx2=16
m3_fv4_uct_nowb|fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25 wb=false recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct ng=7 tlim=0.3 mbx1=16 mbx2=16
m4_fv6_ct_wb|fluid=mhd iprob=4 gdir=2 beta=1.0 wb=true recon=fv6 mood=true integrator=rk3 rsolver=lhlld emf=ct_contact ng=6 tlim=0.3 mbx1=16 mbx2=16
m5_fv8_uct_wb|fluid=mhd iprob=5 gdir=2 wb=true recon=fv8 mood=true integrator=rk3 rsolver=hlld emf=uct ng=13 tlim=0.3 mbx1=16 mbx2=16
h1_fv6_wb|fluid=hydro iprob=1 wb=true recon=fv6 mood=true integrator=rk3 rsolver=hllc ng=6 tlim=0.3 mbx1=16 mbx2=16
LIST
