#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load anaconda3/2024.06 nvhpc/24.5 cudatoolkit/12.4 openmpi/cuda-12.4/nvhpc-24.5/4.1.6
export CUDA_VISIBLE_DEVICES=1
base=/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution
root=$base/cascade-halo-validation/minimum
build_dir=/home/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution/pad-audit/build
cmake --build "$build_dir" -j8 > "$root/build.log" 2>&1
cp "$build_dir/src/athena" "$root/athena-cascade-flags"
sha256sum "$root/athena-cascade-flags" > "$root/executable.sha256"
ready=false
for attempt in $(seq 1 180); do
 if grep -q CASCADE_FLAGS_GPU_CHECKS_FINISHED "$base/cascade-halo-validation/validation.log"; then ready=true; break; fi
 if ! kill -0 488197 2>/dev/null; then echo 'Prior GPU checks failed'; exit 1; fi
 sleep 5
done
if ! "$ready"; then echo 'Prior GPU checks did not finish'; exit 1; fi
python "$root/check_gpu.py" > "$root/checks.log" 2>&1
printf 'MINIMUM_CASCADE_GPU_CHECKS_FINISHED\n'
