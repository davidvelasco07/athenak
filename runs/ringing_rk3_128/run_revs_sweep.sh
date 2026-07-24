#!/bin/bash
# 128^3 PPM+FB UCT (uct_hlld, mood_nad_b=comps, mood_nad_v=comps):
# sweep mood_max_revs.  nghost must satisfy ng >= revs + 4 (UCT+ppm halo).
# revs=2 already exists as uct_bcomps_vcomps (nghost=6).
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
  local revs=$3
  local ng=$4
  local dir="${RUNROOT}/uct_${name}"
  mkdir -p "${dir}"
  echo "=== Starting uct_${name} on GPU ${gpu} (revs=${revs}, nghost=${ng}) at $(date) ==="
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${INPUT}" \
    -d "${dir}" \
    job/basename="turb_ringing_ppm_fb" \
    mesh/nghost=${ng} \
    mhd/mood_max_revs=${revs} \
    mhd/mood_nad_b=comps \
    mhd/mood_nad_v=comps \
    2>&1 | tee "${dir}/run.log" | tail -5
  echo "=== Finished uct_${name} at $(date) ==="
}

# Wave 1
run_one revs1 0 1 5 &
run_one revs3 1 3 7 &
wait

# Wave 2
run_one revs4 0 4 8 &
wait

echo "Revs sweep complete at $(date)."
