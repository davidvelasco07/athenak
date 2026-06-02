#!/bin/bash
# Submit AthenaK GPU profiling jobs on Princeton Della (Grace Hopper / partition grace).
#
# Usage:
#   cp scripts/della/profile_config.example.sh scripts/della/profile_config.sh
#   # edit profile_config.sh
#   ./scripts/della/validate_profile_setup.sh   # check binary, input, paths first
#   ./scripts/della/submit_profile.sh run    # bare GPU simulation (sanity check)
#   ./scripts/della/submit_profile.sh nsys
#   ./scripts/della/submit_profile.sh ncu
#
# Submit from della9, della-gpu, or della-gh (after module purge). Binary must be
# built on della-gh (aarch64). Output is under PROFILE_DIR on /scratch/gpfs/...

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 run|nsys|ncu" >&2
  exit 1
fi

PROFILER="$1"
if [[ "${PROFILER}" != "nsys" && "${PROFILER}" != "ncu" && "${PROFILER}" != "run" ]]; then
  echo "First argument must be 'run', 'nsys', or 'ncu'" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/profile_config.sh"
SLURM_SCRIPT="${SCRIPT_DIR}/profile_athenak.slurm"
VALIDATE="${SCRIPT_DIR}/validate_profile_setup.sh"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing ${CONFIG}" >&2
  echo "Copy profile_config.example.sh to profile_config.sh and edit it." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG}"

if [[ ! -f "${VALIDATE}" ]]; then
  echo "Missing ${VALIDATE}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${VALIDATE}"
echo "Preflight checks:"
if ! validate_profile_setup; then
  echo "" >&2
  echo "Fix the errors above before submitting. Run on della-gh after building:" >&2
  echo "  find \"\${ATHENA_ROOT}/build\" -name athena -type f" >&2
  echo "  export ATHENA_BIN=<that path>  # in profile_config.sh" >&2
  exit 1
fi
echo ""

# Always target Grace Hopper (user build on della-gh).
SLURM_PARTITION="${SLURM_PARTITION:-grace}"
if [[ "${SLURM_PARTITION}" != "grace" ]]; then
  echo "WARNING: SLURM_PARTITION was '${SLURM_PARTITION}'; forcing grace" >&2
  SLURM_PARTITION="grace"
fi

mkdir -p "${PROFILE_DIR}/logs"

module purge 2>/dev/null || true

if [[ -x /usr/bin/sbatch ]]; then
  SBATCH_BIN="/usr/bin/sbatch"
else
  SBATCH_BIN="$(command -v sbatch)"
fi
if [[ -z "${SBATCH_BIN}" || ! -x "${SBATCH_BIN}" ]]; then
  echo "sbatch not found after module purge" >&2
  exit 1
fi

# Export everything the batch script needs (sourcing PROFILE_CONFIG again is backup).
EXPORT_VARS="ALL,PROFILER=${PROFILER},PROFILE_CONFIG=${CONFIG},SLURM_PARTITION=${SLURM_PARTITION}"

SBATCH_ARGS=(
  --export="${EXPORT_VARS}"
  --job-name="${SLURM_JOB_NAME:-athenak-${PROFILER}}"
  --partition="${SLURM_PARTITION}"
  --output="${PROFILE_DIR}/logs/${PROFILER}-%j.out"
  --error="${PROFILE_DIR}/logs/${PROFILER}-%j.err"
)

if [[ -n "${SLURM_TIME:-}" ]]; then
  SBATCH_ARGS+=(--time="${SLURM_TIME}")
fi
if [[ -n "${SLURM_MEM:-}" ]]; then
  SBATCH_ARGS+=(--mem="${SLURM_MEM}")
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  SBATCH_ARGS+=(--cpus-per-task="${SLURM_CPUS_PER_TASK}")
fi
if [[ -n "${SLURM_GPUS:-}" ]]; then
  SBATCH_ARGS+=(--gres="gpu:${SLURM_GPUS}")
fi
if [[ "${USE_MPI:-0}" == "1" ]]; then
  SBATCH_ARGS+=(--ntasks="${SLURM_NTASKS:-1}")
fi
if [[ -n "${SLURM_CONSTRAINT:-}" ]]; then
  SBATCH_ARGS+=(--constraint="${SLURM_CONSTRAINT}")
fi
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="${SLURM_ACCOUNT}")
fi
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
  SBATCH_ARGS+=(--mail-type=END,FAIL --mail-user="${SLURM_MAIL_USER}")
fi

if [[ "${PROFILER}" == "run" ]]; then
  echo "Submitting bare simulation job (partition=${SLURM_PARTITION})..."
else
  echo "Submitting ${PROFILER} profiling job (partition=${SLURM_PARTITION})..."
fi
echo "  Config:     ${CONFIG}"
echo "  sbatch:     ${SBATCH_BIN} ($(${SBATCH_BIN} --version 2>&1 | head -1))"
echo "  Host:       $(hostname)"
echo "  Binary:     ${ATHENA_BIN}"
echo "  PROFILE_DIR: ${PROFILE_DIR}"

SUBMIT_OUT="$("${SBATCH_BIN}" "${SBATCH_ARGS[@]}" "${SLURM_SCRIPT}")"
echo "${SUBMIT_OUT}"

JOB_ID="$(echo "${SUBMIT_OUT}" | awk '{print $NF}')"
if [[ -z "${JOB_ID}" || ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse job ID from sbatch output" >&2
  exit 1
fi

RESULT_DIR="${PROFILE_DIR}/${PROFILER}/${JOB_ID}"
LOG_OUT="${PROFILE_DIR}/logs/${PROFILER}-${JOB_ID}.out"
LOG_ERR="${PROFILE_DIR}/logs/${PROFILER}-${JOB_ID}.err"

echo ""
echo "Job ${JOB_ID} submitted to partition '${SLURM_PARTITION}'."
echo "  Slurm log (read this if the result folder is empty):"
echo "    ${LOG_OUT}"
echo "    ${LOG_ERR}"
echo "  Profile output (created when the job runs on grace):"
echo "    ${RESULT_DIR}/"
echo "    ${RESULT_DIR}/job.log"
echo ""
echo "Monitor:  squeue -j ${JOB_ID}"
echo "          sacct -j ${JOB_ID} --format=JobID,Partition,State,ExitCode,Elapsed,NodeList"
echo ""
if [[ -d "${HOME}/scratch/athenak_profiles" && "${HOME}/scratch/athenak_profiles" != "${PROFILE_DIR}"* ]]; then
  echo "NOTE: ~/scratch/athenak_profiles is NOT the configured PROFILE_DIR."
  echo "      Use ${PROFILE_DIR} (see profile_config.sh)."
fi
