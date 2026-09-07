#!/bin/bash
#SBATCH -J mood_ring
#SBATCH -o %x-%j.out
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 08:00:00
#SBATCH -p gh
#SBATCH -A AST22008

# Flagship 128^3 ringing/turbulence comparison (requires PROBLEM=fluids/turb build).
# All NR MHD schemes use UCT-HLLD; PPM+MOOD uses the dedicated UCT input.

set -euo pipefail

REPO="${REPO:-$HOME/athenak-fallback-validation}"
ATHENA="${ATHENA:-$REPO/build_turb/src/athena}"
INPUT_CT="$REPO/inputs/mhd/turb_ringing_ppm_fb_128.athinput"
INPUT_UCT="$REPO/inputs/mhd/turb_ringing_ppm_fb_128_uct.athinput"
OUTROOT="${OUTROOT:-$SCRATCH/fallback_validation/ringing_128}"

mkdir -p "$OUTROOT"
schemes=(plm wenoz teno ppm_fb)

for s in "${schemes[@]}"; do
  dir="$OUTROOT/$s"
  mkdir -p "$dir"
  case "$s" in
    plm)
      INPUT="$INPUT_CT"
      ov=(time/integrator=rk2 mhd/reconstruct=plm mhd/mood=false mesh/nghost=3
          mhd/emf=uct_hlld mhd/rsolver=hlld)
      ;;
    wenoz)
      INPUT="$INPUT_CT"
      ov=(time/integrator=rk3 mhd/reconstruct=wenoz mhd/mood=false mesh/nghost=5
          mhd/emf=uct_hlld mhd/rsolver=hlld)
      ;;
    teno)
      INPUT="$INPUT_CT"
      ov=(time/integrator=rk3 mhd/reconstruct=teno mhd/mood=false mesh/nghost=5
          mhd/emf=uct_hlld mhd/rsolver=hlld)
      ;;
    ppm_fb)
      INPUT="$INPUT_UCT"
      ov=(time/integrator=rk3 mhd/reconstruct=ppm mhd/mood=true mesh/nghost=6
          mhd/emf=uct_hlld mhd/rsolver=hlld mhd/mood_edge=auto
          mhd/mood_nad_scale=gcfl mhd/mood_rtol=1.0e-5 mhd/mood_nad_v=off)
      ;;
  esac
  echo "=== $s ==="
  srun -n 8 "$ATHENA" -i "$INPUT" -d "$dir" job/basename="ring_${s}" "${ov[@]}" \
    2>&1 | tee "$dir/run.log"
done
