#!/bin/bash
# MHD well-balanced exact-gate matrix (the gates of the 2026-10 MHD WB work, phases 0-4):
# uniform-field atmosphere (iprob 3), Parker atmosphere (iprob 4), current-free arcade (iprob 5)
# x {plm, plm+FOFC, fv4+MOOD} x {ct_contact, uct_hll, uct_hlld} x {hlld, lhlld, hlle, llf},
# plus SMR, physical walls, the sweeping AMR band and 4 MPI ranks.  Every wb=true line must show
# 0.000e+00 for all *pert columns and the matching wb=false control must not.
# usage: ATH=<athena> RUNS=<run dir containing mkdeck.py and stats.py> mhd_gates.sh
set -u; H=$(cd "$(dirname "$0")" && pwd); G="$H/gate.sh"
B3="fluid=mhd iprob=3 b1=1.0 b2=0.5 b3=0.25"
B4="fluid=mhd iprob=4 gdir=2 beta=1.0"
B5="fluid=mhd iprob=5 gdir=2"
SMR='tail=<mesh_refinement>\nrefinement=static\nnum_levels=2\n<refined_region1>\nlevel=1\nx1min=0.3\nx1max=0.7\nx2min=0.0\nx2max=1.0\n'
AMR='tail=<mesh_refinement>\nrefinement=adaptive\nnum_levels=3\nrefinement_interval=5\nmax_nmb_per_rank=1024\n<amr_criterion0>\nmethod=user\n'
PB="$B3 gdir=2 ix1=periodic ox1=periodic ix2=reflect ox2=outflow"
echo "--- base scheme (plm)"
$G m3_plm_ct       "$B3 recon=plm rsolver=hlld emf=ct_contact ng=3 tlim=0.5"
$G m3_plm_ucthlld  "$B3 recon=plm rsolver=hlld emf=uct_hlld ng=4 tlim=0.5"
$G m3_plm_ct_lhlld "$B3 recon=plm rsolver=lhlld emf=ct_contact ng=3 tlim=0.5"
$G m4_plm_ucthll   "$B4 recon=plm rsolver=hlle emf=uct_hll ng=4 tlim=0.5"
$G m5_plm_ucthlld_16 "$B5 recon=plm rsolver=hlld emf=uct_hlld ng=4 tlim=0.5 mbx1=16 mbx2=16"
echo "--- fv4 + MOOD"
$G m3_fv4_ct       "$B3 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=ct_contact ng=5 tlim=0.5"
$G m3_fv4_ucthlld  "$B3 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=6 tlim=0.5"
$G m4_fv4_ct_lhlld "$B4 recon=fv4 mood=true integrator=rk3 rsolver=lhlld emf=ct_contact ng=5 tlim=0.5"
$G m4_fv4_ucthll   "$B4 recon=fv4 mood=true integrator=rk3 rsolver=hlle emf=uct_hll ng=6 tlim=0.5"
$G m5_fv4_ucthlld_16 "$B5 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=6 tlim=0.5 mbx1=16 mbx2=16"
$G m5_fv4_ucthlld_lhlld "$B5 recon=fv4 mood=true integrator=rk3 rsolver=lhlld emf=uct_hlld ng=6 tlim=0.5"
$G m5_fv4_ucthll_blend "$B5 recon=fv4 mood=true integrator=rk3 rsolver=llf emf=uct_hll ng=6 tlim=0.5 mood_edge=blend"
$G m5_fv4_ucthll_mag "$B5 recon=fv4 mood=true integrator=rk3 rsolver=hlle emf=uct_hll ng=6 tlim=0.5 nad_b=mag"
$G m3_fv4_ct_4ranks "$B3 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=ct_contact ng=5 tlim=0.5 mbx1=16 mbx2=16" 4
echo "--- plm + FOFC"
$G m3_plm_fofc_ct     "$B3 recon=plm fofc=true rsolver=hlld emf=ct_contact ng=3 tlim=0.5"
$G m4_plm_fofc_ucthll "$B4 recon=plm fofc=true rsolver=hlle emf=uct_hll ng=4 tlim=0.5"
$G m5_plm_fofc_ucthlld_16 "$B5 recon=plm fofc=true rsolver=hlld emf=uct_hlld ng=4 tlim=0.5 mbx1=16 mbx2=16"
echo "--- SMR (level-1 region straddling the stratification)"
$G smr_plm_ct        "$B3 recon=plm rsolver=hlld emf=ct_contact ng=4 tlim=0.5 mbx1=16 mbx2=16 $SMR"
$G smr_fv4mood_uct   "$B3 recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=6 tlim=0.5 mbx1=16 mbx2=16 $SMR"
$G smr_plm_fofc_4rk  "$B3 recon=plm fofc=true rsolver=lhlld emf=uct_hll ng=4 tlim=0.5 mbx1=16 mbx2=16 $SMR" 4
echo "--- physical walls (gravity -x2: reflect bottom, outflow top)"
$G bc_plm_ct         "$PB recon=plm rsolver=hlld emf=ct_contact ng=4 tlim=0.5"
$G bc_fv4mood_uct    "$PB recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=6 tlim=0.5 mbx1=16 mbx2=16"
echo "--- AMR: refinement band sweeping across x1 every 5 steps, reflect walls"
$G amr_plm_ct        "$PB recon=plm rsolver=hlld emf=ct_contact ng=4 tlim=0.5 mbx1=16 mbx2=16 extra=amr_test=band\namr_band_w=0.2\namr_band_v=2.0 $AMR"
$G amr_fv4mood_uct_4rk "$PB recon=fv4 mood=true integrator=rk3 rsolver=hlld emf=uct_hlld ng=6 tlim=0.5 mbx1=16 mbx2=16 extra=amr_test=band\namr_band_w=0.2\namr_band_v=2.0 $AMR" 4
