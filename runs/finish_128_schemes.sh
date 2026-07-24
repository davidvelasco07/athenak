#!/bin/bash
# Finish the failed 128^3 non-mood scheme runs (emf was missing from input).
# uct_hll/ppm_fb is already running on GPU 1; use GPU 0 for the rest.
set -uo pipefail
source ~/.bash_aliases 2>/dev/null || true
gpus 2>/dev/null || true

ATHENA=/home/velasco/athenak/fallback/tst/build_gpu/src/athena
INPUT=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_reproducer_128.athinput
RUNBASE=/home/velasco/athenak/fallback/runs/ringing_rk3_128

is_done() {
  local dir=$1
  [ -f "${dir}/run.log" ] || return 1
  rg -q "Terminating on time limit" "${dir}/run.log" || return 1
  ls "${dir}"/bin/rank_*/turb_ringing_*.slice_x1_0_bmag.00006.bin >/dev/null 2>&1
}

run_one() {
  local emf=$1 recon=$2 gpu=$3 nghost=$4
  local dir="${RUNBASE}/${emf}/${recon}"
  mkdir -p "${dir}"
  if is_done "${dir}"; then
    echo "=== SKIP ${emf}/${recon} ==="
    return 0
  fi
  # wipe partial failed logs so is_done can't false-positive later
  rm -rf "${dir}/bin" "${dir}/rst"
  echo "=== Starting ${emf}/${recon} on GPU ${gpu} at $(date) ==="
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${INPUT}" \
    -d "${dir}" \
    job/basename="turb_ringing_${recon}" \
    mesh/nghost=${nghost} \
    mesh/nx1=128 mesh/nx2=128 mesh/nx3=128 \
    mhd/reconstruct="${recon}" \
    mhd/rsolver=hlld \
    mhd/emf="${emf}" \
    mhd/mood=false \
    2>&1 | tee "${dir}/run.log" | tail -3
  echo "=== Finished ${emf}/${recon} at $(date) ==="
}

# Wait until GPU 1 frees (uct_hll/ppm_fb) OR run sequentially on GPU 0 first.
# Wave A on GPU 0 while ppm_fb occupies GPU 1
run_one uct_hlld plm  0 3
run_one uct_hlld ppmx 0 5
run_one uct_hlld wenoz 0 5

# Wave B: both GPUs (ppm_fb may still be going — wait for it)
while pgrep -f "uct_hll/ppm_fb" >/dev/null 2>&1; do
  echo "... waiting for uct_hll/ppm_fb ($(date))"
  sleep 60
done

run_one uct_hll plm  0 3 &
run_one uct_hll ppmx 1 5 &
wait
run_one uct_hll wenoz 0 5

echo "128 finish-up complete at $(date)"
