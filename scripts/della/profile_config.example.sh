# Copy to profile_config.sh and edit for your account:
#   cp scripts/della/profile_config.example.sh scripts/della/profile_config.sh
#
# profile_config.sh is sourced by submit_profile.sh (git-ignored).

# --- AthenaK paths (required) ---
export ATHENA_ROOT="${ATHENA_ROOT:-$HOME/athenak}"
export ATHENA_BIN="${ATHENA_BIN:-${ATHENA_ROOT}/build/src/athena}"
export INPUT_FILE="${INPUT_FILE:-${ATHENA_ROOT}/tst/inputs/lwave_hydro_uniform_3d.athinput}"
# Uniform 3D linear wave, no AMR, nlim=10 in file; add overrides here if needed:
export RUN_ARGS="${RUN_ARGS:-}"

# --- Profiling output on GPFS scratch (NOT ~/scratch unless that is your GPFS path) ---
export NETID="${NETID:-$USER}"
export PROFILE_DIR="${PROFILE_DIR:-/scratch/gpfs/TEYSSIER/${NETID}/athenak_profiles}"

# --- Modules (loaded on the grace compute node inside the batch job) ---
export CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/13.1}"
# Match modules used when you built on della-gh:
export EXTRA_MODULES="${EXTRA_MODULES:-nvhpc/25.5 openmpi/nvhpc-25.5/4.1.8}"

# --- Kokkos nvtx-connector (build: scripts/della/build_nvtx_connector.sh on della-gh) ---
export KOKKOS_TOOLS_ROOT="${KOKKOS_TOOLS_ROOT:-${HOME}/kokkos-tools}"
export KOKKOS_TOOLS_LIBS="${KOKKOS_TOOLS_LIBS:-${KOKKOS_TOOLS_ROOT}/profiling/nvtx-connector/kp_nvtx_connector.so}"
export USE_NVTX_CONNECTOR="${USE_NVTX_CONNECTOR:-1}"
export PROFILER_ENABLE_NVTX="${PROFILER_ENABLE_NVTX:-1}"

# --- Slurm: Grace Hopper only ---
# submit_profile.sh always uses partition=grace. Build athena on della-gh (aarch64).
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-athenak-profile}"
export SLURM_PARTITION="${SLURM_PARTITION:-grace}"
export SLURM_TIME="${SLURM_TIME:-00:30:00}"
export SLURM_MEM="${SLURM_MEM:-32G}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
export SLURM_GPUS="${SLURM_GPUS:-1}"

# --- MPI (set USE_MPI=1 if built with -DAthena_ENABLE_MPI=ON) ---
export USE_MPI="${USE_MPI:-0}"
export SLURM_NTASKS="${SLURM_NTASKS:-1}"

# --- nsys (timeline on compute nodes; uses -t cuda,nvtx,... when PROFILER_ENABLE_NVTX=1) ---
export NSYS_BIN="${NSYS_BIN:-/usr/local/bin/nsys}"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"

# --- ncu (kernel analysis; very slow) — open .ncu-rep with ncu-ui, not nsys-ui ---
export NCU_BIN="${NCU_BIN:-/usr/local/cuda-13.1/bin/ncu}"
export NCU_SET="${NCU_SET:-full}"
export NCU_FORCE_OVERWRITE="${NCU_FORCE_OVERWRITE:-1}"
export NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-20}"
# Only profile Kokkos constant-memory launches (excludes cuda_parallel_launch_local_memory).
export NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:cuda_parallel_launch_constant_memory}"
export NCU_KERNEL_NAME_BASE="${NCU_KERNEL_NAME_BASE:-demangled}"
export NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
export NCU_PROFILE_FROM_START="${NCU_PROFILE_FROM_START:-1}"
export NCU_NVTX_ARGS="${NCU_NVTX_ARGS:--t nvtx}"
export NCU_PRINT_NVTX_RENAME="${NCU_PRINT_NVTX_RENAME:-kernel}"
export NCU_IMPORT_SOURCE="${NCU_IMPORT_SOURCE:-yes}"
# Optional extra NVTX range filter: export NCU_NVTX_INCLUDE="regex:Hydro"
# Extra ncu flags (space-separated). Reduces instrumentation overhead on GH200.
export NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS:---cache-control none}"
