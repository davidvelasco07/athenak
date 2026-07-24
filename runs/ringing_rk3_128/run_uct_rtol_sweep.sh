#!/bin/bash
# 128^3 PPM+FB UCT (uct_hlld, mood_nad_b=comps, mood_nad_v=comps, revs=2):
# tighten mood_rtol.  rtol=1e-5 (default) already exists as uct_bcomps_vcomps.
set -uo pipefail

source ~/.bash_aliases 2>/dev/null || true
gpus 2>/dev/null || {
  module load anaconda3/2024.06
  module load nvhpc/24.5
  module load cudatoolkit/12.4
  module load openmpi/cuda-12.4/nvhpc-24.5/4.1.6
}

ATHENA=/home/velasco/athenak/fallback/tst/build_gpu/src/athena
INPUT=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_ppm_fb_128_uct.athinput
RUNROOT=/home/velasco/athenak/fallback/runs/ringing_rk3_128

run_one() {
  local name=$1
  local gpu=$2
  local rtol=$3
  local dir="${RUNROOT}/uct_${name}"
  mkdir -p "${dir}"
  echo "=== Starting uct_${name} on GPU ${gpu} (rtol=${rtol}) at $(date) ==="
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${INPUT}" \
    -d "${dir}" \
    job/basename="turb_ringing_ppm_fb" \
    mhd/mood_rtol=${rtol} \
    mhd/mood_nad_b=comps \
    mhd/mood_nad_v=comps \
    2>&1 | tee "${dir}/run.log" | tail -5
  echo "=== Finished uct_${name} at $(date) ==="
}

run_one rtol6 0 1.0e-6 &
run_one rtol7 1 1.0e-7 &
wait

echo "UCT rtol sweep complete at $(date)."
