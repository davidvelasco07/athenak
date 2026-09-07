#!/usr/bin/env bash
# Run paper-resolution 2D stress suite on Apollo login-node A100s.
# Usage (on apollo): bash validation/fallback/scripts/run_stress_apollo_gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

source ~/.bash_aliases 2>/dev/null || true
if declare -F gpus >/dev/null; then gpus; fi
if declare -F build_athenak_gpu >/dev/null; then
  :
else
  echo "build_athenak_gpu not found in ~/.bash_aliases; loading modules manually"
  module load anaconda3/2024.06 nvhpc/24.5 cudatoolkit/12.4 2>/dev/null || true
fi

NJOBS="${ATHENAK_NJOBS:-8}"
NVCC="$ROOT/kokkos/bin/nvcc_wrapper"
if [[ ! -x "$NVCC" ]]; then
  echo "Missing $NVCC" >&2
  exit 1
fi

build_one() {
  local bdir="$1" problem="$2"
  echo "=== Building $bdir (-DPROBLEM=$problem, CUDA) ==="
  cmake -B "$bdir" -DCMAKE_BUILD_TYPE=Release \
    -DAthena_ENABLE_MPI=OFF \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DCMAKE_CXX_COMPILER="$NVCC" \
    -DPROBLEM="$problem"
  cmake --build "$bdir" --target athena -j"$NJOBS"
  ls -la "$bdir/src/athena"
}

# Built-in (implode, OT) + PROBLEM binaries
build_one build_gpu_builtin built_in_pgens
build_one build_gpu_slotted fluids/slotted_cyl
build_one build_gpu_kh fluids/kh
build_one build_gpu_cs fluids/current_sheet

# Point manifest binaries at GPU trees via symlinks expected by the suite
ln -sfn build_gpu_slotted build_slotted
ln -sfn build_gpu_kh build_kh
ln -sfn build_gpu_cs build_current_sheet
# default athena binary for implode/OT
mkdir -p build/src
ln -sfn "$ROOT/build_gpu_builtin/src/athena" build/src/athena

VAL="$ROOT/validation/fallback"
rm -rf "$VAL/results/stress"
mkdir -p "$VAL/results/_logs"

echo "=== Running stress suite on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  python3 "$VAL/scripts/run_suite.py" \
    --athena "$ROOT/build/src/athena" \
    --suite stress \
    --schemes ppm_fb,plm,wenoz,teno \
  2>&1 | tee "$VAL/results/_logs/stress_apollo_gpu.log"

python3 "$VAL/scripts/analyze.py" --write
python3 "$VAL/scripts/plot_2d_stress.py"
echo "Done. Sync results/figures back to laptop."
