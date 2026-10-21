#!/usr/bin/env bash
# Run on an allocated GPU. All frame output is in a separate invocation from timing.
set -euo pipefail
source_root=$PWD
build=$(cd "${1:?build directory}" && pwd)
mkdir -p "${2:?results directory}"
out=$(cd "$2" && pwd)
date -u > "$out/environment.txt"
hostname >> "$out/environment.txt"
nvidia-smi >> "$out/environment.txt"
python3 - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
paths=[p for top in ['src','tst'] for p in Path(top).rglob('*')
       if p.is_file() and (p.suffix in ['.cpp','.hpp','.py','.athinput'] or p.name=='CMakeLists.txt')]
paths.append(Path('CMakeLists.txt'))
(Path(sys.argv[1])/'source-sha256.json').write_text(json.dumps(
 {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
PY
ctest --test-dir "$build" --output-on-failure | tee "$out/ctest.txt"
python3 tst/sparse/run_physics.py --exe "$build/src/athena" --output "$out/benchmark" --trials 5
# Optional fenced component profile: explicitly separate from end-to-end trials.
mkdir -p "$out/component"
(
 cd "$out/component"
 "$build/src/athena" -i "$source_root/tst/inputs/sparse_tracers.athinput" \
   hydro/nscalars=0 passive_tracers/compare_dense=false passive_tracers/profile=true \
   passive_tracers/capacity=384 > run.log 2>&1
)
san=$(command -v compute-sanitizer || true)
if [[ -z "$san" && -x /usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer ]]; then
 san=/usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer
fi
if [[ -n "$san" ]]; then
 for tool in memcheck synccheck; do
  mkdir -p "$out/$tool-tracers" "$out/$tool-beam"
  (
   cd "$out/$tool-tracers"
   "$san" --tool "$tool" --error-exitcode 3 "$build/src/athena" \
     -i "$source_root/tst/inputs/sparse_tracers.athinput" mesh/nx1=32 mesh/nx2=32 \
     problem/radius=0.1 time/nlim=3 passive_tracers/capacity=32 > check.log 2>&1
  )
  (
   cd "$out/$tool-beam"
   "$san" --tool "$tool" --error-exitcode 3 "$build/src/athena" \
     -i "$source_root/tst/inputs/sc_beam_sparse.athinput" mesh/nx1=32 mesh/nx2=16 \
     problem/ihor1=8 problem/ihor2=23 nr_radiation/intensity_storage=sparse > check.log 2>&1
  )
 done
fi
