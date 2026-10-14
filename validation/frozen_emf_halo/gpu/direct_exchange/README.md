# Direct hydro owner-flux exchange

## Exchange schedule and halo

The immutable high-order flux halo is exchanged **once per RK stage**, after the
initial reconstruction and before `ApplyFallback`. Every MOOD revision then runs
locally. There is no communication inside the hydro revision loop.

The current requirement is
`max(reconstruction_radius + 1, mood_max_revs + 3)`: five ghosts for two revisions
and six for three revisions at FV4/6/8. The `+3` covers the smooth-extrema
detector (SED), which reads three cells beyond each tested cell. The exchange supplies high-order fluxes
that cannot be reconstructed throughout the revision halo using that smaller
primitive halo. This optimization preserves that schedule and bound; it does not
establish that six is a mathematically minimal bound for three revisions.

## Implementation

`PackAndSendOwnedFluxCC` reads the direction/variable directly from the face-flux
arrays using the existing uniform CC owner indices. It writes local-neighbor
receive buffers or the existing rank-packed MPI send buffer.
`RecvAndUnpackOwnedFluxCC` writes only imported ghost/shared faces directly into
the face-flux arrays. Interior owner faces stay in place.

This removes the full-volume `mood_candidate` allocation and both volume-copy
kernels, plus the generic CC exchange's unused Z4c passes. Packing and unpacking
remain separate kernels so self-periodic neighbors cannot overwrite data before
all outgoing values have been read. The existing primitive exchange for curved
GR is unchanged. MHD uses its existing path.

## Validation and timings

See `results.json` for individual runs, input parameters and executable hashes.
CPU builds enable MPI and Kokkos bounds checking. GR cases use Minkowski spacetime.

CPU validation completed 42 runs across NR/SR/GR hydro:

- 2D FV4/6/8: one wide-halo block versus four minimum-halo blocks (18 runs).
- 3D FV8: one block versus eight blocks (6 runs).
- FV8 MPI: one rank versus four ranks in both 2D and 3D (12 runs).
- Three revisions, FV8: one rank/wide halo versus four ranks/six ghosts (6 runs).

Saved physical fields and flags are identical across each decomposition comparison.
The 12 minimum-halo 2D/3D cases also exactly match the previous staged-exchange
outputs on the same architecture. All runs activate fallback, with zero density
or energy floors and zero primitive-conversion failures.

GPU validation on Apollo A100 GPU 1 covers the same 18 2D and 6 3D runs,
plus 6 runs with three revisions/six ghosts. All pass; saved fields and flags
match across decompositions and the 12 minimum-halo cases match the prior staged
implementation exactly. CUDA memcheck reports zero errors for all three hydro
regimes in 3D. GPU MPI and curved-spacetime GR were not tested.

### Longer GR stress run: pre-existing failure

An attempted 128-step 3D FV8 GR timing run reached 192 primitive-conversion
failures, so it was rejected as a valid benchmark. Repeating that input with
staged exchange and legacy hydro produced exactly the same 192 failures,
demotion count, final time, and conservation drift (mass 5.05e-6; energy 1.85e-5).
Saved conserved fields are identical between direct and staged exchange. This
is an existing limitation of this GR blast configuration, not a regression from
the direct exchange. No long-duration GR robustness claim is made.

### Matched 32-step timings

The reported comparison uses the original 32-step workload, with one discarded
warmup and seven alternating measured runs for each of direct exchange, staged
exchange, and legacy hydro. All three paths have fallback active. Timings are
solver elapsed times; kernel-level profiling has not been performed. Individual
samples are retained so timing variation is visible.

#### 3D: 64³ cells, 32³ blocks, FV8, 32 RK3 steps

| Regime | Direct (s) | Staged (s) | Legacy (s) | Direct vs staged | Direct vs legacy |
|---|---:|---:|---:|---:|---:|
| NR | 0.256943 | 0.277028 | 0.225344 | -7.2% | +14.0% |
| SR | 0.300406 | 0.317757 | 0.261454 | -5.5% | +14.9% |
| GR | 0.325280 | 0.347975 | 0.288679 | -6.5% | +12.7% |

Negative changes mean lower runtime. Removing staging reduces 3D hydro runtime
by 5–7%, but this version still takes 13–15% longer than legacy hydro. The first
32-step 2D timing window was noisy and gave mixed signs; those samples are retained
in `results.json`, with no consistent 2D speedup inferred from that short window.

#### 2D: 256² cells, 64² blocks, 128 RK3 steps

To reduce the short-window variation, this follow-up used one discarded warmup
and five measured repeats per path. All runs pass the same health checks.

| Regime | Order | Direct (s) | Staged (s) | Legacy (s) | Direct vs staged | Direct vs legacy |
|---|---:|---:|---:|---:|---:|---:|
| NR | FV4 | 0.372935 | 0.422419 | 0.364124 | -11.7% | +2.4% |
| NR | FV8 | 0.389645 | 0.387825 | 0.358167 | +0.5% | +8.8% |
| SR | FV4 | 0.449484 | 0.471404 | 0.398901 | -4.7% | +12.7% |
| SR | FV8 | 0.437133 | 0.440397 | 0.387150 | -0.7% | +12.9% |
| GR | FV4 | 0.448676 | 0.493208 | 0.419490 | -9.0% | +7.0% |
| GR | FV8 | 0.469423 | 0.471776 | 0.446533 | -0.5% | +5.1% |

FV4 median runtimes improve by 5–12%. FV8 changes are below 1% and are not
resolved against run-to-run variation. In this longer 2D workload, direct exchange
still takes 2–13% longer than legacy hydro. These measurements support the 3D
improvement and a smaller allocation, without claiming a universal speedup.
All measured direct/staged pairs have identical demotion counts.

## Reproduce

Build the parent implementation (`4aa93691e`) separately using the same compiler,
Kokkos checkout and CMake options as the current build. Its executable supplies
`--baseline-exe`; this does not switch production code at runtime. Use the build
instructions in the parent GPU report, then run from this repository:

```sh
python3 validation/frozen_emf_halo/gpu/run_matrix.py --source . \
  --exe build_gpu/src/athena --baseline-exe /path/to/baseline/src/athena \
  --output direct-3d --cases nr_hydro,sr_hydro,gr_hydro --phase benchmark \
  --orders 8 --dim 3 --nx 64 --block 32 --steps 32 --repeats 7
python3 validation/frozen_emf_halo/gpu/run_matrix.py --source . \
  --exe build_gpu/src/athena --baseline-exe /path/to/baseline/src/athena \
  --output direct-2d --cases nr_hydro,sr_hydro,gr_hydro --phase benchmark \
  --orders 4,8 --nx 256 --steps 128 --repeats 5
```

For correctness, omit `--phase benchmark` and `--baseline-exe`; use `--steps 16`
for 2D, or `--orders 8 --dim 3 --nx 32 --steps 4` for 3D. Add `--mpi 4` with an
MPI build. The three-revision test uses `--orders 8 --revisions 3 --steps 8`.
As before, arithmetic is double precision, saved binary fields are single
precision, and history sums are double precision. GR uses Minkowski spacetime.

Full raw data and both GPU executables remain under
`/home/velasco/codex-validation/local-rcfl-gpu-20261014/direct-exchange` on Apollo.
The committed stress-run decks/stdout preserve the unsuccessful longer GR case.
