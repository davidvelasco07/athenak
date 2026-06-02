#!/bin/bash
# Preflight checks before submit_profile.sh (also sourced by submit_profile.sh).
# Exits non-zero with actionable errors if the grace job would fail immediately.

validate_profile_setup() {
  local err=0

  if [[ -z "${ATHENA_BIN:-}" ]]; then
    echo "ERROR: ATHENA_BIN is not set in profile_config.sh" >&2
    err=1
  elif [[ ! -e "${ATHENA_BIN}" ]]; then
    echo "ERROR: ATHENA_BIN does not exist: ${ATHENA_BIN}" >&2
    err=1
  elif [[ ! -f "${ATHENA_BIN}" ]]; then
    echo "ERROR: ATHENA_BIN is not a regular file: ${ATHENA_BIN}" >&2
    err=1
  elif [[ ! -x "${ATHENA_BIN}" ]]; then
    echo "ERROR: ATHENA_BIN is not executable: ${ATHENA_BIN}" >&2
    echo "       Run: chmod +x ${ATHENA_BIN}  or rebuild on della-gh" >&2
    err=1
  else
    local bin_info host_arch
    bin_info="$(file -b "${ATHENA_BIN}" 2>/dev/null || true)"
    host_arch="$(uname -m 2>/dev/null || true)"
    echo "OK: ATHENA_BIN exists and is executable"
    echo "    ${ATHENA_BIN}"
    echo "    ${bin_info}"
    if [[ "${host_arch}" == "aarch64" && "${bin_info}" != *"aarch64"* && "${bin_info}" != *"ARM"* ]]; then
      echo "WARNING: login node is aarch64 but binary does not look like an ARM build." >&2
      echo "         Grace requires a build from della-gh." >&2
      err=1
    fi
    if [[ "${host_arch}" == "x86_64" && "${bin_info}" == *"aarch64"* ]]; then
      echo "WARNING: binary is aarch64 (della-gh build) but you are on ${host_arch}." >&2
      echo "         Submitting to grace is still OK; this check is informational." >&2
    fi
    local cache="${ATHENA_ROOT:-}/build/CMakeCache.txt"
    if [[ -f "${cache}" ]]; then
      if grep -q 'Kokkos_ENABLE_CUDA:BOOL=ON' "${cache}" 2>/dev/null; then
        echo "OK: CMakeCache Kokkos_ENABLE_CUDA=ON"
      else
        echo "ERROR: build is not CUDA-enabled (Kokkos_ENABLE_CUDA is OFF in ${cache})." >&2
        echo "       nsys/ncu will show no GPU kernels. Rebuild on della-gh:" >&2
        echo "         ./scripts/della/build_cuda_grace.sh --clean" >&2
        err=1
      fi
    else
      echo "WARNING: no ${cache}; cannot verify Kokkos CUDA is enabled." >&2
    fi
  fi

  if [[ -z "${INPUT_FILE:-}" ]]; then
    echo "ERROR: INPUT_FILE is not set in profile_config.sh" >&2
    err=1
  elif [[ ! -f "${INPUT_FILE}" ]]; then
    echo "ERROR: INPUT_FILE does not exist: ${INPUT_FILE}" >&2
    err=1
  else
    echo "OK: INPUT_FILE ${INPUT_FILE}"
  fi

  if [[ -z "${PROFILE_DIR:-}" ]]; then
    echo "ERROR: PROFILE_DIR is not set" >&2
    err=1
  else
    if ! mkdir -p "${PROFILE_DIR}/logs" 2>/dev/null; then
      echo "ERROR: cannot create PROFILE_DIR/logs: ${PROFILE_DIR}" >&2
      err=1
    elif [[ ! -w "${PROFILE_DIR}" ]]; then
      echo "ERROR: PROFILE_DIR is not writable: ${PROFILE_DIR}" >&2
      err=1
    else
      echo "OK: PROFILE_DIR ${PROFILE_DIR}"
    fi
  fi

  if [[ "${SLURM_PARTITION:-grace}" != "grace" ]]; then
    echo "ERROR: SLURM_PARTITION must be 'grace' (got '${SLURM_PARTITION:-}')" >&2
    err=1
  else
    echo "OK: SLURM_PARTITION=grace"
  fi

  if [[ "${USE_NVTX_CONNECTOR:-1}" == "1" ]]; then
    local connector_lib="" candidate
    for candidate in \
        "${NVTX_CONNECTOR_LIB:-}" \
        "${KOKKOS_TOOLS_LIBS:-}" \
        "${KOKKOS_TOOLS_ROOT:-}/profiling/nvtx-connector/kp_nvtx_connector.so" \
        "${ATHENA_ROOT:-}/../kokkos-tools/profiling/nvtx-connector/kp_nvtx_connector.so" \
        "${HOME}/kokkos-tools/profiling/nvtx-connector/kp_nvtx_connector.so"; do
      if [[ -n "${candidate}" && -f "${candidate}" ]]; then
        connector_lib="${candidate}"
        break
      fi
    done
    if [[ -n "${connector_lib}" ]]; then
      echo "OK: nvtx-connector ${connector_lib}"
      echo "    $(file -b "${connector_lib}" 2>/dev/null || true)"
    else
      echo "ERROR: USE_NVTX_CONNECTOR=1 but kp_nvtx_connector.so not found." >&2
      echo "       Build: scripts/della/build_nvtx_connector.sh (on della-gh)" >&2
      echo "       Then set KOKKOS_TOOLS_LIBS in profile_config.sh" >&2
      err=1
    fi
  fi

  if [[ "${err}" -ne 0 && -n "${ATHENA_ROOT:-}" && -d "${ATHENA_ROOT}/build" ]]; then
    echo "" >&2
    echo "Searching for 'athena' under ${ATHENA_ROOT}/build ..." >&2
    find "${ATHENA_ROOT}/build" \( -name athena -o -name athena.exe \) -type f 2>/dev/null | while read -r p; do
      echo "  found: ${p}" >&2
      file -b "${p}" 2>/dev/null | sed 's/^/          /' >&2
    done
    if ! find "${ATHENA_ROOT}/build" -name athena -type f 2>/dev/null | grep -q .; then
      echo "  (none — rebuild on della-gh, then set ATHENA_BIN to the path above)" >&2
    fi
  fi

  return "${err}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CONFIG="${SCRIPT_DIR}/profile_config.sh"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONFIG}"
  validate_profile_setup
fi
