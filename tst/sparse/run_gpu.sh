#!/usr/bin/env bash
# Run inside an allocation with one visible GPU, after building the sparse targets.
# Usage: bash tst/sparse/run_gpu.sh build-sparse-cuda results-sparse
set -euo pipefail
build=$(cd "${1:?build directory required}" && pwd)
mkdir -p "${2:?results directory required}"
out=$(cd "$2" && pwd)
bin="$build/tst/sparse"

date -u > "$out/environment.txt"
hostname >> "$out/environment.txt"
nvidia-smi >> "$out/environment.txt"
sha256sum src/sparse/*.hpp src/sparse/*.cpp tst/sparse/*.hpp tst/sparse/*.cpp \
  > "$out/source-sha256.txt"
sha256sum "$bin/sparse_storage_test" "$bin/sparse_benchmark" \
  "$bin/sparse_advection_test" > "$out/binary-sha256.txt"
"$bin/sparse_storage_test" --kokkos-print-configuration >> "$out/environment.txt"
ctest --test-dir "$build" --output-on-failure | tee "$out/ctest.txt"

# Large streaming working sets and a small launch/metadata-sensitive case.
# 256 blocks * 128 fields * 32*8*8 doubles = 512 MiB in dense mode.
# Five trials per mode; each is 100 launches with a fence around the full interval.
for shape in large small; do
  if [[ "$shape" == large ]]; then
    blocks=256; fields=128; nx=32; ny=8; nz=8
  else
    blocks=64; fields=32; nx=32; ny=1; nz=1
  fi
  for fraction in 0.01 0.1 0.5 1.0; do
    "$bin/sparse_benchmark" --blocks "$blocks" --fields "$fields" \
      --cells "$nx" --ny "$ny" --nz "$nz" --fraction "$fraction" \
      --iterations 100 --repeats 5 > "$out/${shape}_${fraction}.jsonl"
  done
done

# End-to-end comparison: legacy host allocator, GPU pool, and a fused dense control.
# The GPU pool includes its session setup and final host reconciliation in timing.
for trial in 1 2 3 4 5; do
  "$bin/sparse_advection_test" --blocks 64 --cells 64 --fields 64 --steps 256 \
    > "$out/advection_${trial}.jsonl"
done
# Larger cell work and expanding support exercise different cost/occupancy regimes.
for trial in 1 2 3; do
  "$bin/sparse_advection_test" --blocks 128 --cells 512 --fields 64 --steps 128 \
    > "$out/advection_large_${trial}.jsonl"
  "$bin/sparse_advection_test" --blocks 32 --cells 64 --fields 32 --steps 128 \
    --courant 0.4 > "$out/advection_diffusive_${trial}.jsonl"
done
sanitizer=$(command -v compute-sanitizer || true)
if [[ -z "$sanitizer" && -x /usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer ]]; then
  sanitizer=/usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer
fi
if [[ -n "$sanitizer" ]]; then
  "$sanitizer" --tool memcheck --error-exitcode 3 \
    "$bin/sparse_storage_test" > "$out/memcheck-storage.txt" 2>&1
  "$sanitizer" --tool memcheck --error-exitcode 3 \
    "$bin/sparse_advection_test" --blocks 4 --cells 16 --fields 5 --steps 64 \
    > "$out/memcheck-advection.txt" 2>&1
fi
