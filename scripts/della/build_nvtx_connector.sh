#!/bin/bash
# Build Kokkos Tools nvtx-connector for AthenaK profiling on Della (aarch64 / della-gh).
#
# Usage (on della-gh.princeton.edu):
#   ./scripts/della/build_nvtx_connector.sh
#
# Then in profile_config.sh:
#   export KOKKOS_TOOLS_LIBS=$HOME/kokkos-tools/profiling/nvtx-connector/kp_nvtx_connector.so

set -euo pipefail

KOKKOS_TOOLS_ROOT="${KOKKOS_TOOLS_ROOT:-${HOME}/kokkos-tools}"
KOKKOS_ROOT="${KOKKOS_ROOT:-${ATHENA_ROOT:-$HOME/athenak}/kokkos}"
BUILD_DIR="${KOKKOS_TOOLS_ROOT}/build"
CONNECTOR_SO="${KOKKOS_TOOLS_ROOT}/profiling/nvtx-connector/kp_nvtx_connector.so"

echo "=== Kokkos Tools nvtx-connector ==="
echo "KOKKOS_TOOLS_ROOT: ${KOKKOS_TOOLS_ROOT}"
echo "KOKKOS_ROOT:       ${KOKKOS_ROOT}"
echo "Host:              $(uname -m)"

if [[ ! -d "${KOKKOS_TOOLS_ROOT}/.git" && ! -f "${KOKKOS_TOOLS_ROOT}/CMakeLists.txt" ]]; then
  echo "ERROR: clone kokkos-tools first:" >&2
  echo "  git clone --depth 1 https://github.com/kokkos/kokkos-tools.git ${KOKKOS_TOOLS_ROOT}" >&2
  exit 1
fi

module purge 2>/dev/null || true
module load cudatoolkit/13.1 2>/dev/null || true

cmake -S "${KOKKOS_TOOLS_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DKokkos_ROOT="${KOKKOS_ROOT}/build" \
  -DKokkos_DIR="${KOKKOS_ROOT}/build"

cmake --build "${BUILD_DIR}" --target kp_nvtx_connector -j "$(nproc)"

if [[ ! -f "${CONNECTOR_SO}" ]]; then
  echo "ERROR: expected ${CONNECTOR_SO} after build" >&2
  find "${KOKKOS_TOOLS_ROOT}" -name 'kp_nvtx_connector.so' 2>/dev/null || true
  exit 1
fi

echo ""
echo "=== Build OK ==="
echo "export KOKKOS_TOOLS_LIBS=${CONNECTOR_SO}"
file "${CONNECTOR_SO}"
