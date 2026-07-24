#!/bin/bash
# Scheme x EMF mosaics: plm / ppmx / wenoz / ppm_fb  x  uct_hlld / uct_hll
# at 128^3 then 256^3.  Skip any case that already finished to tlim.
set -uo pipefail

source ~/.bash_aliases 2>/dev/null || true
gpus 2>/dev/null || {
  module load anaconda3/2024.06
  module load nvhpc/24.5
  module load cudatoolkit/12.4
  module load openmpi/cuda-12.4/nvhpc-24.5/4.1.6
}

ATHENA=/home/velasco/athenak/fallback/tst/build_gpu/src/athena
RUNBASE=/home/velasco/athenak/fallback/runs

is_done() {
  local dir=$1
  # finished if run.log has "Terminating on time limit" AND slice 00006 exists
  [ -f "${dir}/run.log" ] || return 1
  rg -q "Terminating on time limit" "${dir}/run.log" || return 1
  local bin_dir
  bin_dir=$(find "${dir}/bin" -type d -name 'rank_*' 2>/dev/null | head -1)
  [ -n "${bin_dir}" ] || return 1
  ls "${bin_dir}"/*slice_x1_0_bmag.00006.bin >/dev/null 2>&1
}

run_case() {
  local nx=$1 emf=$2 recon=$3 gpu=$4
  local root="${RUNBASE}/ringing_rk3_${nx}/${emf}"
  local dir="${root}/${recon}"
  mkdir -p "${dir}"

  if is_done "${dir}"; then
    echo "=== SKIP ${nx}/${emf}/${recon} (already complete) ==="
    return 0
  fi

  local input nghost mood_args basename
  if [ "${recon}" = "ppm_fb" ]; then
    recon_arg=ppm
    mood_args="mhd/mood=true mhd/mood_nad_b=comps mhd/mood_nad_v=comps mhd/mood_max_revs=2 mhd/mood_rtol=1.0e-5"
    nghost=6
    basename=turb_ringing_ppm_fb
    if [ "${nx}" = "128" ]; then
      input=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_ppm_fb_128_uct.athinput
    else
      input=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_ppm_fb.athinput
    fi
  else
    recon_arg=${recon}
    mood_args="mhd/mood=false"
    basename=turb_ringing_${recon}
    if [ "${recon}" = "plm" ]; then
      nghost=3
    else
      nghost=5   # ppmx / wenoz + UCT
    fi
    if [ "${nx}" = "128" ]; then
      input=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_reproducer_128.athinput
    else
      input=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_reproducer.athinput
    fi
  fi

  echo "=== Starting ${nx}/${emf}/${recon} on GPU ${gpu} at $(date) ==="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${input}" \
    -d "${dir}" \
    job/basename="${basename}" \
    mesh/nghost=${nghost} \
    mesh/nx1=${nx} mesh/nx2=${nx} mesh/nx3=${nx} \
    mhd/reconstruct="${recon_arg}" \
    mhd/rsolver=hlld \
    mhd/emf="${emf}" \
    ${mood_args} \
    2>&1 | tee "${dir}/run.log" | tail -3
  local rc=${PIPESTATUS[0]}
  echo "=== Finished ${nx}/${emf}/${recon} rc=${rc} at $(date) ==="
  return ${rc}
}

# ---- 128^3 ----
# Wave plan: reuse uct_hlld/ppm_fb; run the other 7.
# Pair across GPUs.
echo "########## 128^3 wave 1: uct_hlld plm + ppmx ##########"
run_case 128 uct_hlld plm  0 &
run_case 128 uct_hlld ppmx 1 &
wait

echo "########## 128^3 wave 2: uct_hlld wenoz + uct_hll plm ##########"
run_case 128 uct_hlld wenoz 0 &
run_case 128 uct_hll  plm   1 &
wait

echo "########## 128^3 wave 3: uct_hll ppmx + wenoz ##########"
run_case 128 uct_hll ppmx  0 &
run_case 128 uct_hll wenoz 1 &
wait

echo "########## 128^3 wave 4: uct_hll ppm_fb (+ uct_hlld ppm_fb skip) ##########"
run_case 128 uct_hlld ppm_fb 0 &   # should skip
run_case 128 uct_hll  ppm_fb 1 &
wait

echo "########## 128^3 COMPLETE at $(date) ##########"
