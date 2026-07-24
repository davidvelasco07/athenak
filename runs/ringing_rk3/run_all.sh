#!/bin/bash
set -euo pipefail

source ~/.bash_aliases 2>/dev/null || true
gpus 2>/dev/null || {
  module load anaconda3/2024.06
  module load nvhpc/24.5
  module load cudatoolkit/12.4
  module load openmpi/cuda-12.4/nvhpc-24.5/4.1.6
}

ATHENA=/home/velasco/athenak/fallback/tst/build_gpu/src/athena
INPUT=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_reproducer.athinput
RUNROOT=/home/velasco/athenak/fallback/runs/ringing_rk3

run_one() {
  local recon=$1
  local gpu=$2
  local dir="${RUNROOT}/${recon}"
  mkdir -p "${dir}"
  echo "=== Starting turb_ringing_${recon} on GPU ${gpu} at $(date) ==="
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${INPUT}" \
    -d "${dir}" \
    job/basename="turb_ringing_${recon}" \
    mhd/reconstruct="${recon}" \
    2>&1 | tee "${dir}/run.log"
  echo "=== Finished turb_ringing_${recon} at $(date) ==="
}

# PLM and PPMX in parallel on the two Apollo GPUs; WENO-Z afterward on GPU 0.
run_one plm 0 &
pid_plm=$!
run_one ppmx 1 &
pid_ppmx=$!
wait "${pid_plm}"
wait "${pid_ppmx}"
run_one wenoz 0

echo "All three RK3 ringing reproducer runs complete."
