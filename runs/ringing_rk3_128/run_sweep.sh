#!/bin/bash
# 128^3 PPM+FB mood_rtol tightening sweep for the RK3 ringing reproducer.
set -uo pipefail

source ~/.bash_aliases 2>/dev/null || true
gpus 2>/dev/null || {
  module load anaconda3/2024.06
  module load nvhpc/24.5
  module load cudatoolkit/12.4
  module load openmpi/cuda-12.4/nvhpc-24.5/4.1.6
}

ATHENA=/home/velasco/athenak/fallback/tst/build_gpu/src/athena
INPUT=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_ppm_fb_128.athinput
RUNROOT=/home/velasco/athenak/fallback/runs/ringing_rk3_128

run_one() {
  local name=$1
  local gpu=$2
  shift 2
  local dir="${RUNROOT}/${name}"
  mkdir -p "${dir}"
  echo "=== Starting ${name} on GPU ${gpu} at $(date) ==="
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${INPUT}" \
    -d "${dir}" \
    job/basename="turb_ringing_ppm_fb" \
    "$@" \
    2>&1 | tee "${dir}/run.log" | tail -5
  echo "=== Finished ${name} at $(date) ==="
}

# mood_rtol tightening: default is 1e-5; smaller = tighter DMP = more demotion
run_one rtol6 0 mhd/mood_rtol=1.0e-6 &
run_one rtol7 1 mhd/mood_rtol=1.0e-7 &
wait

echo "Sweep complete at $(date)."
