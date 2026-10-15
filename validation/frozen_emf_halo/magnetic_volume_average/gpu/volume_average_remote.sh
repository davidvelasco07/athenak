#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load anaconda3/2024.06 nvhpc/24.5 cudatoolkit/12.4 openmpi/cuda-12.4/nvhpc-24.5/4.1.6
export CUDA_VISIBLE_DEVICES=1
base=/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution
root=$base/volume-average-validation
source_dir=$base/pad-audit/source
build_dir=/home/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution/pad-audit/build
if [ ! -e "$root/prior-source.tar.gz" ]; then
  tar -C "$source_dir" --exclude=kokkos -czf "$root/prior-source.tar.gz" .
  cp "$build_dir/src/athena" "$root/athena-before"
fi
tar -C "$source_dir" -xzf "$root/changed-source.tar.gz"
cmake --build "$build_dir" -j8 > "$root/build.log" 2>&1
cp "$build_dir/src/athena" "$root/athena-volume-average"
sha256sum "$root/athena-volume-average" > "$root/executable.sha256"
cmake -S "$source_dir/validation/frozen_emf_halo/magnetic_volume_average" \
  -B "$root/operator-build" -DKokkos_DIR="$build_dir/kokkos" \
  -DATHENA_BUILD="$build_dir" -DCMAKE_CXX_COMPILER="$source_dir/kokkos/bin/nvcc_wrapper" \
  > "$root/operator-configure.log" 2>&1
cmake --build "$root/operator-build" -j4 > "$root/operator-build.log" 2>&1
"$root/operator-build/check_volume_average" > "$root/operator-checks.log" 2>&1
python "$root/check_gpu.py" > "$root/checks.log" 2>&1
printf 'VOLUME_AVERAGE_GPU_CHECKS_FINISHED\n'
