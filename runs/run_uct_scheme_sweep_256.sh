#!/bin/bash
# 256^3 scheme x EMF sweep.  No prior UCT runs exist at 256 (old runs were ct_contact).
set -uo pipefail
source ~/.bash_aliases 2>/dev/null || true
gpus 2>/dev/null || {
  module load anaconda3/2024.06
  module load nvhpc/24.5
  module load cudatoolkit/12.4
  module load openmpi/cuda-12.4/nvhpc-24.5/4.1.6
}

ATHENA=/home/velasco/athenak/fallback/tst/build_gpu/src/athena
RUNBASE=/home/velasco/athenak/fallback/runs/ringing_rk3_256

is_done() {
  local dir=$1
  [ -f "${dir}/run.log" ] || return 1
  rg -q "Terminating on time limit" "${dir}/run.log" || return 1
  ls "${dir}"/bin/rank_*/*.slice_x1_0_bmag.00006.bin >/dev/null 2>&1
}

run_case() {
  local emf=$1 recon=$2 gpu=$3
  local dir="${RUNBASE}/${emf}/${recon}"
  mkdir -p "${dir}"
  if is_done "${dir}"; then
    echo "=== SKIP 256/${emf}/${recon} (already complete) ==="
    return 0
  fi

  local input nghost mood_args basename recon_arg
  if [ "${recon}" = "ppm_fb" ]; then
    recon_arg=ppm
    mood_args="mhd/mood=true mhd/mood_nad_b=comps mhd/mood_nad_v=comps mhd/mood_max_revs=2 mhd/mood_rtol=1.0e-5"
    nghost=6
    basename=turb_ringing_ppm_fb
    input=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_ppm_fb.athinput
  else
    recon_arg=${recon}
    mood_args="mhd/mood=false"
    basename=turb_ringing_${recon}
    if [ "${recon}" = "plm" ]; then nghost=3; else nghost=5; fi
    input=/home/velasco/athenak/fallback/inputs/mhd/turb_ringing_reproducer.athinput
  fi

  echo "=== Starting 256/${emf}/${recon} on GPU ${gpu} at $(date) ==="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${gpu} "${ATHENA}" \
    -i "${input}" \
    -d "${dir}" \
    job/basename="${basename}" \
    mesh/nghost=${nghost} \
    mesh/nx1=256 mesh/nx2=256 mesh/nx3=256 \
    mhd/reconstruct="${recon_arg}" \
    mhd/rsolver=hlld \
    mhd/emf="${emf}" \
    ${mood_args} \
    2>&1 | tee "${dir}/run.log" | tail -3
  echo "=== Finished 256/${emf}/${recon} at $(date) ==="
}

# 8 cases, 2 GPUs.  Pair light with heavy when possible.
echo "########## 256 wave 1: uct_hlld plm + ppmx ##########"
run_case uct_hlld plm  0 &
run_case uct_hlld ppmx 1 &
wait

echo "########## 256 wave 2: uct_hlld wenoz + ppm_fb ##########"
run_case uct_hlld wenoz  0 &
run_case uct_hlld ppm_fb 1 &
wait

echo "########## 256 wave 3: uct_hll plm + ppmx ##########"
run_case uct_hll plm  0 &
run_case uct_hll ppmx 1 &
wait

echo "########## 256 wave 4: uct_hll wenoz + ppm_fb ##########"
run_case uct_hll wenoz  0 &
run_case uct_hll ppm_fb 1 &
wait

echo "########## 256 COMPLETE at $(date) ##########"
