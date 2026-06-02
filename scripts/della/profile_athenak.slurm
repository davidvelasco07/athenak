#!/bin/bash
#SBATCH --job-name=athenak-profile
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Slurm stdout/stderr paths are set by submit_profile.sh (--output/--error).
# Batch job: profile AthenaK on Della Grace Hopper (partition grace, ARM binary).

set -uo pipefail

log() { echo "[$(date -Is)] $*" | tee -a "${JOB_LOG:-/dev/stderr}"; }
die() { log "ERROR: $*"; exit 1; }

init_modules() {
  # Slurm sets MODULES_RUN_QUARANTINE; still source Modules so module load works in batch.
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck source=/dev/null
    source /usr/share/Modules/init/bash
  else
    log "WARNING: could not source Modules init; using explicit tool paths"
  fi
}

resolve_nvtx_connector_lib() {
  local candidate
  if [[ -n "${NVTX_CONNECTOR_LIB:-}" && -f "${NVTX_CONNECTOR_LIB}" ]]; then
    echo "${NVTX_CONNECTOR_LIB}"
    return 0
  fi
  if [[ -n "${KOKKOS_TOOLS_LIBS:-}" && -f "${KOKKOS_TOOLS_LIBS}" ]]; then
    echo "${KOKKOS_TOOLS_LIBS}"
    return 0
  fi
  local -a search_dirs=()
  if [[ -n "${KOKKOS_TOOLS_ROOT:-}" ]]; then
    search_dirs+=("${KOKKOS_TOOLS_ROOT}/profiling/nvtx-connector")
    search_dirs+=("${KOKKOS_TOOLS_ROOT}/build/profiling/nvtx-connector")
  fi
  if [[ -n "${ATHENA_ROOT:-}" ]]; then
    search_dirs+=("${ATHENA_ROOT}/../kokkos-tools/profiling/nvtx-connector")
    search_dirs+=("${HOME}/kokkos-tools/profiling/nvtx-connector")
  fi
  search_dirs+=("${HOME}/kokkos-tools/profiling/nvtx-connector")
  for candidate in \
      "${search_dirs[@]}/kp_nvtx_connector.so" \
      "${search_dirs[@]}/libkp_nvtx_connector.so"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

setup_kokkos_nvtx_connector() {
  local connector_lib require_connector=0
  if [[ "${USE_NVTX_CONNECTOR:-1}" == "1" && "${PROFILER:-}" != "run" ]]; then
    require_connector=1
  fi

  if connector_lib="$(resolve_nvtx_connector_lib)"; then
    export KOKKOS_TOOLS_LIBS="${connector_lib}"
    log "KOKKOS_TOOLS_LIBS=${KOKKOS_TOOLS_LIBS}"
    export NVTX_CONNECTOR_ACTIVE=1
  elif [[ "${require_connector}" -eq 1 ]]; then
    die "nvtx-connector not found. Build on della-gh, then set in profile_config.sh:" \
      "  export KOKKOS_TOOLS_LIBS=\${HOME}/kokkos-tools/profiling/nvtx-connector/kp_nvtx_connector.so" \
      "  (or export NVTX_CONNECTOR_LIB=...)"
  elif [[ -n "${KOKKOS_TOOLS_LIBS:-}" ]]; then
    log "WARNING: KOKKOS_TOOLS_LIBS not found: ${KOKKOS_TOOLS_LIBS}"
  fi
}

# Map legacy/alias rename modes to values accepted by ncu 13.x (none|kernel).
resolve_ncu_nvtx_rename() {
  local mode="${NCU_PRINT_NVTX_RENAME:-kernel}"
  case "${mode}" in
    most-recent|per-nvtx|kernel) echo "kernel" ;;
    none|off|"") echo "none" ;;
    *) echo "${mode}" ;;
  esac
}

# nsys: -t cuda,nvtx,...  ncu: NCU_NVTX_ARGS=-t nvtx (Kokkos wiki) or --nvtx on CUDA 13+ ncu.
append_profiler_nvtx_args() {
  local -n _out=$1
  local ncu_bin="${2:-}"
  local nvtx_rename

  if [[ "${PROFILER_ENABLE_NVTX:-1}" != "1" ]]; then
    return 0
  fi

  if [[ -z "${ncu_bin}" ]]; then
    return 0
  fi

  nvtx_rename="$(resolve_ncu_nvtx_rename)"

  if [[ -n "${NCU_NVTX_ARGS:-}" ]]; then
    if [[ "${NCU_NVTX_ARGS}" == *"-t"* ]] \
        && ! "${ncu_bin}" --help 2>/dev/null | grep -qE '(^|[[:space:]])-t,'; then
      log "WARNING: ${ncu_bin} has no -t option; using --nvtx instead of ${NCU_NVTX_ARGS}"
      _out+=(--nvtx)
      if [[ "${nvtx_rename}" != "none" ]]; then
        _out+=(--print-nvtx-rename "${nvtx_rename}")
      fi
    else
      local -a _nvtx=()
      # shellcheck disable=SC2206
      _nvtx=(${NCU_NVTX_ARGS})
      _out+=("${_nvtx[@]}")
    fi
  else
    _out+=(--nvtx)
    if [[ "${nvtx_rename}" != "none" ]]; then
      _out+=(--print-nvtx-rename "${nvtx_rename}")
    fi
  fi
  if [[ -n "${NCU_NVTX_INCLUDE:-}" ]]; then
    _out+=(--nvtx-include "${NCU_NVTX_INCLUDE}")
  fi
  if [[ -n "${NCU_NVTX_EXCLUDE:-}" ]]; then
    _out+=(--nvtx-exclude "${NCU_NVTX_EXCLUDE}")
  fi
}

ensure_nsys_trace_includes_nvtx() {
  local trace="${1}"
  if [[ "${PROFILER_ENABLE_NVTX:-1}" == "1" && "${trace}" != *nvtx* ]]; then
    trace="nvtx,${trace}"
  fi
  echo "${trace}"
}

append_ncu_args() {
  local -n _out=$1
  # Default filter: Kokkos constant-memory launches only (excludes cuda_parallel_launch_local_memory).
  local kernel_name="${NCU_KERNEL_NAME:-regex:cuda_parallel_launch_constant_memory}"
  if [[ -n "${kernel_name}" ]]; then
    _out+=(--kernel-name "${kernel_name}")
    log "ncu kernel filter: ${kernel_name}"
  fi
  if [[ -n "${NCU_KERNEL_NAME_BASE:-}" ]]; then
    _out+=(--kernel-name-base "${NCU_KERNEL_NAME_BASE}")
  fi
  if [[ -n "${NCU_LAUNCH_SKIP:-}" && "${NCU_LAUNCH_SKIP}" != "0" ]]; then
    _out+=(--launch-skip "${NCU_LAUNCH_SKIP}")
  fi
  if [[ "${NCU_PROFILE_FROM_START:-1}" == "0" ]]; then
    _out+=(--profile-from-start off)
  fi
  local import_source="${NCU_IMPORT_SOURCE:-yes}"
  if [[ "${import_source}" != "0" && "${import_source}" != "no" \
      && "${import_source}" != "off" ]]; then
    _out+=(--import-source yes)
  fi
  if [[ -n "${NCU_EXTRA_ARGS:-}" ]]; then
    local -a _extra=()
    # shellcheck disable=SC2206
    _extra=(${NCU_EXTRA_ARGS})
    _out+=("${_extra[@]}")
  fi
}

check_run_success() {
  local rc=$1
  if ! grep -q 'Setup complete, executing task list' "${JOB_LOG}"; then
    die "Simulation did not reach setup complete (see ${JOB_LOG})"
  fi
  if ! grep -qE 'cycle=10 ' "${JOB_LOG}"; then
    die "Simulation did not finish cycle=10 (exit ${rc}, see ${JOB_LOG})"
  fi
  if [[ ${rc} -ne 0 ]]; then
    die "Simulation exit code ${rc} (see ${JOB_LOG})"
  fi
  log "OK: simulation completed through cycle=10"
}

resolve_ncu_bin() {
  if [[ -n "${NCU_BIN:-}" && -x "${NCU_BIN}" ]]; then
    echo "${NCU_BIN}"
    return 0
  fi
  if command -v ncu >/dev/null 2>&1; then
    command -v ncu
    return 0
  fi
  local ver path
  ver="${CUDA_MODULE##*/}"
  ver="${ver#cudatoolkit/}"
  for path in "/usr/local/cuda-${ver}/bin/ncu" /usr/local/cuda/bin/ncu; do
    if [[ -x "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

on_err() {
  local rc=$?
  log "FAILED: command exit ${rc} at line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
  exit "${rc}"
}
trap on_err ERR

PROFILER="${PROFILER:-nsys}"
if [[ "${PROFILER}" != "nsys" && "${PROFILER}" != "ncu" && "${PROFILER}" != "run" ]]; then
  die "PROFILER must be nsys, ncu, or run (got '${PROFILER}')"
fi

if [[ -n "${PROFILE_CONFIG:-}" && -f "${PROFILE_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PROFILE_CONFIG}"
fi

: "${ATHENA_BIN:?Set ATHENA_BIN in profile_config.sh}"
: "${INPUT_FILE:?Set INPUT_FILE in profile_config.sh}"
: "${PROFILE_DIR:?Set PROFILE_DIR in profile_config.sh}"

# Require grace partition (GH200); sbatch line + #SBATCH above should both request it.
if [[ -n "${SLURM_JOB_PARTITION:-}" && "${SLURM_JOB_PARTITION}" != "grace" ]]; then
  die "Refusing to run: SLURM_JOB_PARTITION='${SLURM_JOB_PARTITION}' (expected grace)"
fi

RUN_DIR="${PROFILE_DIR}/${PROFILER}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
JOB_LOG="${RUN_DIR}/job.log"
touch "${JOB_LOG}"

log "=== AthenaK profiling on Della (Grace Hopper) ==="
log "Profiler:   ${PROFILER}"
log "Run dir:    ${RUN_DIR}"
log "Host:       $(hostname)"
log "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
log "Job ID:     ${SLURM_JOB_ID}"

init_modules
module purge 2>&1 | tee -a "${JOB_LOG}" || true
if [[ -n "${CUDA_MODULE:-}" ]]; then
  log "module load ${CUDA_MODULE}"
  if ! module load "${CUDA_MODULE}" 2>&1 | tee -a "${JOB_LOG}"; then
    log "WARNING: module load ${CUDA_MODULE} failed"
  fi
fi
if [[ -n "${EXTRA_MODULES:-}" ]]; then
  log "module load ${EXTRA_MODULES}"
  # shellcheck disable=SC2086
  if ! module load ${EXTRA_MODULES} 2>&1 | tee -a "${JOB_LOG}"; then
    log "WARNING: some EXTRA_MODULES failed to load (see above)"
  fi
fi

if [[ ! -x "${ATHENA_BIN}" ]]; then
  die "AthenaK binary not found or not executable: ${ATHENA_BIN}"
fi
if [[ ! -f "${INPUT_FILE}" ]]; then
  die "Input file not found: ${INPUT_FILE}"
fi

# della-gh builds must be aarch64; grace compute nodes are ARM.
BIN_ARCH="$(file -b "${ATHENA_BIN}")"
HOST_ARCH="$(uname -m)"
log "Binary:     ${ATHENA_BIN}"
log "file:       ${BIN_ARCH}"
log "uname -m:   ${HOST_ARCH}"
if [[ "${HOST_ARCH}" == "aarch64" ]] && [[ "${BIN_ARCH}" != *"aarch64"* ]]; then
  die "Binary is not aarch64 but the grace node is ARM. Rebuild on della-gh: ${ATHENA_BIN}"
fi
if [[ "${HOST_ARCH}" == "x86_64" ]] && [[ "${BIN_ARCH}" == *"aarch64"* ]]; then
  die "Binary is aarch64 (della-gh build) but this node is x86_64. Check #SBATCH --partition=grace"
fi

INPUT_BASE="$(basename "${INPUT_FILE}")"
cp -f "${INPUT_FILE}" "${RUN_DIR}/${INPUT_BASE}"
cd "${RUN_DIR}" || die "cannot cd to ${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

setup_kokkos_nvtx_connector

ATHENA_ARGS=(-i "${INPUT_BASE}")
if [[ -n "${RUN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ATHENA_ARGS+=(${RUN_ARGS})
fi

log "=== GPU ==="
nvidia-smi 2>&1 | tee -a "${JOB_LOG}" || log "WARNING: nvidia-smi failed"
log "=== Modules ==="
module list 2>&1 | tee -a "${JOB_LOG}" || true
log "=== Run command ==="
log "${ATHENA_BIN} ${ATHENA_ARGS[*]}"

run_athena() {
  if [[ "${USE_MPI:-0}" == "1" ]]; then
    srun --wait=0 --ntasks="${SLURM_NTASKS:-1}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
      "${ATHENA_BIN}" "${ATHENA_ARGS[@]}" 2>&1 | tee -a "${JOB_LOG}"
  else
    "${ATHENA_BIN}" "${ATHENA_ARGS[@]}" 2>&1 | tee -a "${JOB_LOG}"
  fi
  return "${PIPESTATUS[0]}"
}

case "${PROFILER}" in
  run)
    log "=== bare simulation (no nsys/ncu) ==="
    set +e
    trap - ERR
    RUN_RC=0
    run_athena || RUN_RC=$?
    trap on_err ERR
    set -e
    check_run_success "${RUN_RC}"
    ;;

  nsys)
    NSYS_BIN="${NSYS_BIN:-/usr/local/bin/nsys}"
    if [[ ! -x "${NSYS_BIN}" ]]; then
      if command -v nsys >/dev/null 2>&1; then
        NSYS_BIN="$(command -v nsys)"
      else
        die "nsys not found (tried ${NSYS_BIN} and PATH)"
      fi
    fi
    TRACE="$(ensure_nsys_trace_includes_nvtx "${NSYS_TRACE:-cuda,nvtx,osrt}")"
    if [[ "${USE_MPI:-0}" == "1" ]]; then
      TRACE="${TRACE},mpi"
    fi
    REPORT="${RUN_DIR}/nsys_${SLURM_JOB_ID}"
    NSYS_ARGS=(
      profile
      -t "${TRACE}"
      --sample=none
      --cpuctxsw=none
      -f true
      -o "${REPORT}"
    )
    log "=== nsys profile ==="
    log "${NSYS_BIN} ${NSYS_ARGS[*]} -- ${ATHENA_BIN} ..."
    set +e
    trap - ERR
    if [[ "${USE_MPI:-0}" == "1" ]]; then
      srun --wait=0 --ntasks="${SLURM_NTASKS:-1}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
        "${NSYS_BIN}" "${NSYS_ARGS[@]}" "${ATHENA_BIN}" "${ATHENA_ARGS[@]}" \
        2>&1 | tee -a "${JOB_LOG}"
    else
      "${NSYS_BIN}" "${NSYS_ARGS[@]}" "${ATHENA_BIN}" "${ATHENA_ARGS[@]}" \
        2>&1 | tee -a "${JOB_LOG}"
    fi
    NSYS_RC=${PIPESTATUS[0]}
    trap on_err ERR
    set -e
    if [[ ! -f "${REPORT}.nsys-rep" ]]; then
      die "nsys did not create ${REPORT}.nsys-rep (exit ${NSYS_RC})"
    fi
    if [[ ${NSYS_RC} -ne 0 ]]; then
      log "WARNING: nsys/athena exit code ${NSYS_RC}; report was still written"
    fi
    log "=== nsys stats ==="
    "${NSYS_BIN}" stats "${REPORT}.nsys-rep" 2>&1 | tee "${RUN_DIR}/nsys_stats.txt" || true
    log "Report: ${REPORT}.nsys-rep"
    ;;

  ncu)
    NCU_BIN_RESOLVED="$(resolve_ncu_bin)" || \
      die "ncu not found (set NCU_BIN in profile_config.sh, e.g. /usr/local/cuda-13.1/bin/ncu)"
    log "ncu:        ${NCU_BIN_RESOLVED}"
    REPORT="${RUN_DIR}/ncu_${SLURM_JOB_ID}"
    NCU_ARGS=(
      --set "${NCU_SET:-full}"
      --launch-count "${NCU_LAUNCH_COUNT:-20}"
      -o "${REPORT}"
    )
    if [[ "${NCU_FORCE_OVERWRITE:-1}" != "0" && "${NCU_FORCE_OVERWRITE:-1}" != "no" \
        && "${NCU_FORCE_OVERWRITE:-1}" != "off" ]]; then
      NCU_ARGS+=(-f)
    fi
    append_profiler_nvtx_args NCU_ARGS "${NCU_BIN_RESOLVED}"
    append_ncu_args NCU_ARGS
    log "=== ncu (slow) ==="
    log "${NCU_BIN_RESOLVED} ${NCU_ARGS[*]} -- ${ATHENA_BIN} ..."
    set +e
    trap - ERR
    if [[ "${USE_MPI:-0}" == "1" ]]; then
      "${NCU_BIN_RESOLVED}" "${NCU_ARGS[@]}" -- srun --ntasks="${SLURM_NTASKS:-1}" \
        "${ATHENA_BIN}" "${ATHENA_ARGS[@]}" 2>&1 | tee -a "${JOB_LOG}"
    else
      "${NCU_BIN_RESOLVED}" "${NCU_ARGS[@]}" -- "${ATHENA_BIN}" "${ATHENA_ARGS[@]}" \
        2>&1 | tee -a "${JOB_LOG}"
    fi
    NCU_RC=${PIPESTATUS[0]}
    trap on_err ERR
    set -e
    if [[ ! -f "${REPORT}.ncu-rep" ]]; then
      die "ncu did not create ${REPORT}.ncu-rep (exit ${NCU_RC})"
    fi
    if [[ ${NCU_RC} -ne 0 ]]; then
      log "WARNING: ncu/athena exit code ${NCU_RC}; report was still written"
    fi
    if grep -q 'Setup complete, executing task list' "${JOB_LOG}" \
        && grep -qE 'cycle=10 ' "${JOB_LOG}"; then
      log "OK: simulation completed through cycle=10 under ncu"
    elif grep -q 'could not find a valid execution configuration' "${JOB_LOG}"; then
      log "HINT: Kokkos TeamPolicy failed under ncu. Use NCU_KERNEL_NAME=regex:Hydro in profile_config.sh"
      log "      to profile only labeled hydro kernels and skip Kokkos setup launches."
    fi
    log "Report: ${REPORT}.ncu-rep"
    log "View:   ncu-ui ${REPORT}.ncu-rep  (Nsight Compute, not nsys-ui)"
    ;;
esac

log "=== Done ==="
ls -la "${RUN_DIR}" 2>&1 | tee -a "${JOB_LOG}"
