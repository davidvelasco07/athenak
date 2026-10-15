#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load anaconda3/2024.06 nvhpc/24.5 cudatoolkit/12.4 openmpi/cuda-12.4/nvhpc-24.5/4.1.6
cd /scratch/lustre/velasco/codex-validation/early-stop-20261015
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake -S source -B build-prod -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$PWD/source/kokkos/bin/nvcc_wrapper" -DAthena_ENABLE_MPI=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF -DPROBLEM=fluids/blast
cmake --build build-prod -j8
printf 'GPU_EARLY_STOP_BUILD_FINISHED\n'
