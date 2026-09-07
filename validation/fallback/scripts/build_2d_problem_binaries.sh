#!/usr/bin/env bash
# Build PROBLEM-specific Athena binaries needed for 2D stress tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
pairs=(
  "build_slotted:fluids/slotted_cyl"
  "build_kh:fluids/kh"
  "build_current_sheet:fluids/current_sheet"
  "build_jet:fluids/jet"
  "build_rotor:fluids/rotor"
  "build_blast:fluids/blast"
  "build_mjet:fluids/mhd_jet"
)
for pair in "${pairs[@]}"; do
  dir="${pair%%:*}"
  prob="${pair##*:}"
  echo "=== $dir ($prob) ==="
  cmake -B "$dir" -DCMAKE_BUILD_TYPE=Release -DAthena_ENABLE_MPI=OFF -DPROBLEM="$prob"
  cmake --build "$dir" --target athena -j"${NPROC:-8}"
  ls -la "$dir/src/athena"
done
echo "All PROBLEM binaries ready."
