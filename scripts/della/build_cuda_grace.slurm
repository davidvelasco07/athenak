#!/bin/bash
#SBATCH --job-name=athenak-build
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/TEYSSIER/%u/athenak_profiles/logs/build-%j.out
#SBATCH --error=/scratch/gpfs/TEYSSIER/%u/athenak_profiles/logs/build-%j.err

set -euo pipefail

ATHENA_ROOT="${ATHENA_ROOT:-$HOME/athenak}"
NETID="${NETID:-$USER}"
LOG_DIR="/scratch/gpfs/TEYSSIER/${NETID}/athenak_profiles/logs"
mkdir -p "${LOG_DIR}"

echo "=== Build AthenaK CUDA on $(hostname) ($(uname -m)) ==="
echo "ATHENA_ROOT=${ATHENA_ROOT}"
cd "${ATHENA_ROOT}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: grace build must run on aarch64 (got $(uname -m))" >&2
  exit 1
fi

"${ATHENA_ROOT}/scripts/della/build_cuda_grace.sh" --clean 2>&1 | tee "${LOG_DIR}/build-${SLURM_JOB_ID}.log"

echo "=== Done ==="
file "${ATHENA_ROOT}/build/src/athena"
grep -E 'Kokkos_ENABLE_CUDA|Kokkos_ENABLE_DEBUG|Kokkos_ENABLE_ATOMICS_BYPASS|Kokkos_ENABLE_DEBUG_BOUNDS_CHECK|CMAKE_BUILD_TYPE' \
  "${ATHENA_ROOT}/build/CMakeCache.txt" || true
