#!/bin/bash
# 128^3 PPM+FB with UCT (uct_hlld): sweep the 4 NAD combinations of
# mood_nad_b x mood_nad_v in {mag, comps}^2.
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
  shift 2
  local dir="${RUNROOT}/uct_${name}"
  mkdir -p "${dir}"
  echo "=== Starting uct_${name} on GPU ${gpu} at $(date) ==="
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${INPUT}" \
    -d "${dir}" \
    job/basename="turb_ringing_ppm_fb" \
    "$@" \
    2>&1 | tee "${dir}/run.log" | tail -5
  echo "=== Finished uct_${name} at $(date) ==="
}

# Wave 1
run_one bmag_vmag     0 mhd/mood_nad_b=mag   mhd/mood_nad_v=mag &
run_one bmag_vcomps   1 mhd/mood_nad_b=mag   mhd/mood_nad_v=comps &
wait

# Wave 2
run_one bcomps_vmag   0 mhd/mood_nad_b=comps mhd/mood_nad_v=mag &
run_one bcomps_vcomps 1 mhd/mood_nad_b=comps mhd/mood_nad_v=comps &
wait

echo "UCT sweep complete at $(date)."
