# Frozen fallback on Apollo: hydro and MHD in NR/SR/GR

The hydro staging copies described by this initial benchmark have since been removed.
See [direct hydro exchange validation and timings](direct_exchange/README.md) for the update.

Validated on 2026-10-14, Apollo GPU 1, NVIDIA A100-PCIE-40GB, CUDA 12.4.
All six configurations explicitly enable the new immutable-candidate path.
The detector uses cell-local `rcfl`, `rtol=1e-4`, SED and two revisions.
The common solver is HLLE, with UCT-HLL for MHD; MHD velocity NAD is disabled.

## Implementation and halo

- Hydro exchanges the initial owner face fluxes before local PLM/DC revisions
  (`mood_frozen_flux=true`). High-order reconstruction runs once per stage.
  NR/SR and Minkowski GR reuse the primitive ghosts from the existing conservative
  exchange, since their C2P conversion is local. Curved GR additionally exchanges
  owner primitives, although curved-metric runs are not certified by this matrix.
- MHD uses the same immutable owner flux/EMF exchange and cached PLM/DC EMFs in
  NR, SR and GR (`mood_frozen_emf=true`). The relativistic Riemann solver is retained
  for both the initial candidate and each fallback tier.
- With two revisions, **hydro needs 5 ghosts for FV4/FV6/FV8**:
  `max(reconstruction_radius + 1, revisions + 3)`.
  **MHD needs 5/6/8**: `max(FV_order, revisions + 3)`.
- These paths are the default for supported uniform periodic ideal-gas FV cases.
  Explicit `false` selects the legacy path used for the comparisons below.
  AMR/SMR, physical boundaries, excision, well balance, diffusion, shear and passive
  scalars retain the existing path. Excision is explicitly rejected for the new
  path because its separate FOFC correction requires additional integration.
- `relative` remains the detector default; `rcfl` is restored as a local option.
  No domain-wide speed, spacing or range reduction is used by either tolerance.
  The stored velocity-component convention and well-balanced signal contribution
  of rcfl are retained, but evaluated per cell. Normal timestep selection remains.

## Correctness evidence

The CPU build enabled Kokkos bounds checking. Both builds use double arithmetic;
normal binary field output is single precision, and histories retain double precision.

| Check | Runs | Result |
|---|---:|---|
| CPU, 2D, FV4/6/8, six configurations | 36 | Wide-halo single block vs minimum-halo four blocks: identical saved fields/flags |
| GPU, same 2D matrix | 36 | Identical saved fields/flags across decompositions |
| CPU and GPU, 3D FV8, six configurations | 12 each | One block vs eight blocks: identical saved fields/flags |
| CPU MPI, FV8, six configurations | 12 | One rank/block vs four ranks/blocks: identical saved fields/flags |
| CUDA memcheck, two steps of FV8, six configurations | 6 | Zero errors |

The 2D cases use 64² cells and 16 RK3 steps; the 3D cases use 32³ and four steps.
MPI checks use eight RK3 steps. Every run activates fallback, with zero density or
energy floors and zero primitive-conversion failures. GPU conservation drift is
at most 1.78e-15 in the reported normalized history sums; max |div B| is 1.89e-15.

CPU/GPU physical fields agree to within 7.11e-15 after normalization by
`max(1, max(abs(reference variable)))` in these saved outputs. Detector
flags are **not fully portable bitwise**: the 3D NR-MHD case has 1,205 differing
final flags between CPU and GPU despite round-off-level state differences. Within
each architecture, decomposition comparisons are exact. This distinction is
recorded in `results.json`; flag agreement was not silently treated as numerical
state agreement.

The blast runs mainly exercise PLM. A separate validation-only `force-dc.patch`
forces a cell adjacent to a block interface through both revisions to DC. Its
12 two-dimensional and 12 three-dimensional runs reached DC. Conserved/primitive
fields and flags match exactly across decompositions; the largest divB difference
is 1.93e-34. All six forced-DC CUDA memory checks report zero errors. Results are
recorded in `forced-dc-results.json` and `forced-dc-3d-results.json`; the probe is absent from the
production executable and source.

## GPU performance with fallback active

Both paths have `mood=true`, detection enabled and nonzero demotion counts.
These are finite-duration blast workloads, not a general scaling claim. One warmup
per path is discarded, followed by three alternating measured repeats. Times are
the solver's elapsed-time measurement, including lightweight history/event output;
intermediate binary field output is disabled for timings. No GPU MPI is involved.

### 2D: 256² cells, 64² blocks, 32 RK3 steps

Entries are **new time / legacy time**; below 1 is faster.

| Configuration | FV4 | FV8 |
|---|---:|---:|
| NR HYDRO | 1.010 | 1.210 |
| NR MHD | 1.243 | 1.238 |
| SR HYDRO | 1.167 | 1.181 |
| SR MHD | 1.236 | 1.156 |
| GR HYDRO | 1.161 | 1.084 |
| GR MHD | 1.281 | 1.193 |

### 3D: 64³ cells, 32³ blocks, FV8, 32 RK3 steps

| Configuration | Legacy (s) | New (s) | Change |
|---|---:|---:|---|
| NR HYDRO | 0.2196 | 0.2820 | 28.4% slower |
| NR MHD | 1.9792 | 1.1713 | 40.8% faster |
| SR HYDRO | 0.2617 | 0.3218 | 23.0% slower |
| SR MHD | 2.1236 | 1.2399 | 41.6% faster |
| GR HYDRO | 0.2968 | 0.3466 | 16.8% slower |
| GR MHD | 2.3254 | 1.3245 | 43.0% faster |

The new MHD path is 41–43% faster in this 3D workload, but slower in the small 2D
workloads. Hydro remains 17–28% slower in 3D after removing redundant primitive
exchange and reducing its ghost count to five. The halo reduction is validated;
it is not a universal speed improvement. Kernel profiling would be needed to
attribute the remaining runtime costs quantitatively.

## Scope

GR tests exercise the fixed-metric GR solver on **Minkowski spacetime**. They do
not certify curved-metric source terms, black-hole excision, nonperiodic boundaries,
AMR, dynamic spacetime, other EOS models or GPU MPI. This matrix tests HLLE/UCT-HLL;
it does not certify every Riemann solver. Earlier CPU studies cover additional
Newtonian solvers and smooth-wave accuracy. These blast tests do not establish
formal FV convergence order.

## Reproduce

On Apollo, load the same modules and build this checkout in a separate directory:

```bash
source /etc/profile.d/modules.sh
module load anaconda3/2024.06 nvhpc/24.5 cudatoolkit/12.4 openmpi/cuda-12.4/nvhpc-24.5/4.1.6
export CUDA_VISIBLE_DEVICES=1
cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DAthena_ENABLE_MPI=OFF \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DPROBLEM=fluids/blast
cmake --build build_gpu -j8
python3 validation/frozen_emf_halo/gpu/run_matrix.py --source . \
  --exe build_gpu/src/athena --output gpu-correctness --steps 16
python3 validation/frozen_emf_halo/gpu/run_matrix.py --source . \
  --exe build_gpu/src/athena --output gpu-3d --orders 8 --dim 3 --nx 32 --steps 4
python3 validation/frozen_emf_halo/gpu/run_matrix.py --source . \
  --exe build_gpu/src/athena --output gpu-benchmark --phase benchmark \
  --orders 4,8 --nx 256 --steps 32 --repeats 3
python3 validation/frozen_emf_halo/gpu/run_matrix.py --source . \
  --exe build_gpu/src/athena --output gpu-benchmark-3d --phase benchmark \
  --orders 8 --dim 3 --nx 64 --block 32 --steps 32 --repeats 3
```

For CPU MPI, build with MPI and bounds checking enabled, then use `--mpi 4
--orders 8 --steps 8`; the reference runs on one rank and the minimum-halo case on
four. The script writes every input deck, stdout, event counts, health metrics,
comparison and timing sample. It requires NumPy and the repository's binary reader.

`results.json` contains the final combined evidence. MHD and final hydro use two
recorded executable hashes because hydro was trimmed after the initial matrix;
the MHD implementation did not change. `recorded_runs/` preserves one measured
repeat's input and stdout per timing variant. Full raw outputs remain in the
Apollo validation directory `/home/velasco/codex-validation/local-rcfl-gpu-20261014`.

To reproduce the DC audit, apply `force-dc.patch` in a separate checkout, rebuild,
and run the correctness matrix with `ATHENA_MOOD_FORCE_DC=1`, `--orders 8`,
`--steps 2`, and `--require-dc` (add `--dim 3 --nx 32` for the 3D check). The probe
forces a single cell in 2D or a column in 3D adjacent to a block interface through
both tiers. Never use that audit executable for performance or production runs.
