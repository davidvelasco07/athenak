# Frozen-EMF MOOD halo experiment

Enable `<mhd>/mood_frozen_emf=true` to retain the initial high-order corner EMFs
and replace only edges adjacent to demoted cells using cached PLM/DC EMFs.
The default recomposition path is unchanged. This is an experimental reference
implementation: whole fallback tiers are cached each stage; memory and runtime
have not been optimized.

The source branch enforces `nghost >= FV_order + mood_max_revs`: for two revisions,
FV4/FV6/FV8 require 6/8/10 ghost cells per side. It rejects unsupported combinations
including AMR/SMR, relativity, well balance, diffusion, shear, passive scalars,
nonideal EOS, non-FV reconstruction, and `mood_touched_only=true`. Validation used
periodic boundaries and CPU execution only; GPU and other boundary conditions
remain unverified.

## Evidence and reproduction

`halo-test-report.txt` summarizes 72 completed CPU runs and 48 comparisons,
including 2D, 3D, four MPI ranks, eight-step runs, and negative controls.
`halo-test-results.json` records comparisons and source/executable hashes.
`recorded_runs/` contains the original decks, parameters, and logs.

The scripts and patches reproduce the original audit from baseline commit
`2d251bb09d67fe76ac97d419041c33dc906db2ca`, rather than patching this branch again:

```sh
python3 validation/frozen_emf_halo/reproduce.py \
  --repo /path/to/athenak --kokkos /path/to/kokkos \
  --destination /path/to/new-experiment --jobs 8
```

Requires Python 3.12+, NumPy, CMake, a C++17 compiler, and OpenMPI. The test build
enables Kokkos bounds checks. Large raw snapshots are regenerated, not committed.
The audit-only patch permits intentionally undersized halos, adds a synthetic
cascade detector, and writes double-precision binary output. These changes are
not enabled in the normal source build. The source implementation differs from
the original audit by stricter configuration/ghost guards and by requiring MOOD
when the frozen option is set; the numerical kernels are identical.

This experiment establishes exact agreement with a large-block reference for
its tested configurations. One fewer input ghost changes outer-halo candidates,
even when the final interior fields happen to agree. The revision halo itself
must remain one cell per revision in the adversarial cascade tests.

The follow-up [maximum-bound audit](maximum_bound/README.md) tests
`max(FV_order, revisions + 3)` using exchanges of immutable owner data before the
revision loop. Its two-revision counts are 5/6/8. The audit patch and 50-run
record are included; the normal source retains the conservative local N+revisions
bound until scalable neighbor exchanges are implemented.
