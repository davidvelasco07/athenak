#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load anaconda3/2024.06 nvhpc/24.5 cudatoolkit/12.4 openmpi/cuda-12.4/nvhpc-24.5/4.1.6
export CUDA_VISIBLE_DEVICES=1
base=/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution
root=$base/cascade-halo-validation
source_dir=$base/pad-audit/source
build_dir=/home/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution/pad-audit/build
if [ ! -e "$root/prior-source.tar.gz" ]; then
  tar -C "$source_dir" --exclude=kokkos -czf "$root/prior-source.tar.gz" .
  cp "$build_dir/CMakeCache.txt" "$root/prior-CMakeCache.txt"
fi
tar -C "$source_dir" -xzf "$root/source-changes.tar.gz"
cmake -S "$source_dir" -B "$build_dir" -DAthena_ENABLE_MPI=ON > "$root/configure.log" 2>&1
cmake --build "$build_dir" -j8 > "$root/build.log" 2>&1
cp "$build_dir/src/athena" "$root/athena-cascade-flags"
sha256sum "$root/athena-cascade-flags" > "$root/executable.sha256"
python "$root/check_gpu.py" > "$root/checks.log" 2>&1
printf 'CASCADE_FLAGS_GPU_CHECKS_FINISHED\n'
