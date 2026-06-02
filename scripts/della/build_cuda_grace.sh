#!/bin/bash
# Build AthenaK with Kokkos CUDA on Della Grace Hopper (della-gh, aarch64).
#
# Run ONLY on della-gh.princeton.edu after updating the Kokkos submodule:
#   cd ~/athenak/kokkos && git fetch --tags origin && git checkout 4.7.02
#
# Usage:
#   ./scripts/della/build_cuda_grace.sh
#   ./scripts/della/build_cuda_grace.sh --clean

set -euo pipefail

ATHENA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ATHENA_ROOT}/build"
KOKKOS_DIR="${ATHENA_ROOT}/kokkos"
NVCC_WRAPPER="${KOKKOS_DIR}/bin/nvcc_wrapper"

CLEAN=0
if [[ "${1:-}" == "--clean" ]]; then
  CLEAN=1
fi

# Kokkos source must be tag 4.7.02+ (CUDA 13 API compatibility), not 4.4 with edited CMakeLists.
if ! grep -q 'CUDART_VERSION >= 13000' "${KOKKOS_DIR}/core/src/Cuda/Kokkos_Cuda_Instance.hpp" 2>/dev/null; then
  echo "ERROR: Kokkos at ${KOKKOS_DIR} lacks CUDA 13 fixes." >&2
  echo "       Only CMakeLists may have been bumped to 4.7.x while source is still 4.4." >&2
  echo "       Fix:" >&2
  echo "         cd ${KOKKOS_DIR} && git fetch --tags origin && git checkout -f 4.7.02" >&2
  exit 1
fi

echo "=== Modules ==="
module purge
module load cudatoolkit/13.1

# Use Princeton cudatoolkit for nvcc + headers (avoid mixing NVHPC 12.9 headers with CUDA 13 APIs).
if [[ -d /usr/local/cuda-13.1 ]]; then
  export CUDA_ROOT=/usr/local/cuda-13.1
elif [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_ROOT="${CUDA_HOME}"
fi
export PATH="${CUDA_ROOT}/bin:${PATH}"

# Host compiler for nvcc_wrapper (gcc from system or gcc-toolset on della-gh)
if command -v g++ >/dev/null 2>&1; then
  export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
fi

echo "ATHENA_ROOT:  ${ATHENA_ROOT}"
echo "CUDA_ROOT:    ${CUDA_ROOT:-<unset>}"
echo "nvcc:         $(command -v nvcc)"
echo "nvcc version: $(nvcc --version | grep release | head -1)"
echo "Kokkos:       $(cd "${KOKKOS_DIR}" && git describe --tags --always)"
echo "CXX wrapper:  ${NVCC_WRAPPER}"
echo "Host CXX:     ${NVCC_WRAPPER_DEFAULT_COMPILER:-<nvcc_wrapper default>}"

if [[ ! -x "${NVCC_WRAPPER}" ]]; then
  echo "ERROR: nvcc_wrapper not found at ${NVCC_WRAPPER}" >&2
  exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: build must run on della-gh (aarch64). Current host: $(hostname) ($(uname -m))" >&2
  echo "       ssh della-gh.princeton.edu" >&2
  echo "       Or: sbatch scripts/della/build_cuda_grace.slurm" >&2
  exit 1
fi

if [[ "${CLEAN}" -eq 1 ]]; then
  echo "=== Removing ${BUILD_DIR} ==="
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cmake -B "${BUILD_DIR}" -S "${ATHENA_ROOT}" \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DKokkos_ENABLE_ATOMICS_BYPASS=OFF \
  -DKokkos_ENABLE_DEBUG=ON \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo ""
echo "=== Build OK ==="
echo "Binary: $(ls -la "${BUILD_DIR}/src/athena")"
file "${BUILD_DIR}/src/athena"
grep -E 'Kokkos_ENABLE_CUDA:BOOL|Kokkos_ARCH_HOPPER90:BOOL|Kokkos_ENABLE_ATOMICS_BYPASS:BOOL|Kokkos_ENABLE_DEBUG:BOOL|Kokkos_ENABLE_DEBUG_BOUNDS_CHECK:BOOL|CMAKE_BUILD_TYPE:' "${BUILD_DIR}/CMakeCache.txt" || true
