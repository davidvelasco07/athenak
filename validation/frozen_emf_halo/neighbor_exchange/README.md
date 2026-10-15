# Frozen MOOD with neighbor exchange and local NAD

## Branch behavior

The frozen-EMF method is the default for dynamic ideal NR/SR/GR MHD with
FV4/FV6/FV8, UCT, a uniform periodic multidimensional grid, `mood_edge=auto|flag`,
`mood_nad_bcc=avg`, and `mood_touched_only=false`. Diffusion, shear, well balance,
passive scalars and refinement are excluded. Unsupported configurations keep the
legacy method; explicitly requesting frozen EMFs for them produces an error.
Set `mood_frozen_emf=false` to compare against legacy recomposition.

The current implementation exchanges integer cascade levels after every owned-cell
detection pass. FV4/FV6/FV8 require **3/4/5 ghost cells** for HLLD/HLLE/LLF, and
**4/5/6** for L-HLLD's extra sensor cell, independently of revision count.
Each active block dimension must be at least `nghost`. See the
[cascade exchange validation](../cascade_exchange/README.md) for the derivation,
forced interface controls, GPU/MPI results and current reproduction commands.
The older local-loop measurements below are historical.

Both hydro and MHD support local `relative` and `rcfl` NAD tolerances:

```
mood_nad_scale = relative   # default; rcfl also supported
mood_rtol = 1.e-5           # default
mood_atol = 0.0             # default
```

For `mood_nad_scale=rcfl`, `mood_rtol` defaults to `1.e-4`. The relative band is
multiplied by `min(1, dt * speed(cell) / dx(cell))`, using the immutable stage-input
state, full-step timestep, maximum absolute stored velocity component, and minimum
active-direction spacing of that cell's block. Well-balanced runs retain the
local sound-speed (hydro) or fast-speed upper-bound (MHD) contribution. The
absolute and round-off floors are unchanged. The factor applies to every NAD
variable, including scalar concentrations. It adds no stencil or reduction.
Normal timestep selection still determines `dt`.

`rcfl` previously used global velocity/spacing extrema; the restored option is
cell-local, so old `rcfl` results are not expected to match. `relative` remains
bitwise unchanged. See [local rcfl validation](local-rcfl-results.json).

`gcfl`, `grange`, `gdu`, and `mood_nad_theta` remain removed. Old settings
produce an explicit error and must be migrated. This includes saved restart
parameters; the restart test below concerns checkpoints written by the new code.
No global field range, update amplitude, speed, or mesh-size reduction is used
for the detector tolerance. The existing per-pack demotion count for local loop
termination remains, as does normal timestep selection.

## Implementation

Two phases precede the revision loop:

1. Exchange the stage primitives and cell-centered B.
2. Compute the initial high-order face fluxes and edge EMFs, then exchange their
   owner values into the candidate halo.

Each face/edge is assigned to the cell with the same integer indices. Packing
these values into a CC payload lets the existing neighbor routines fill face,
edge and corner ghosts, including periodic self-neighbors. Dedicated boundary
objects and task dependencies isolate these messages from normal U/B exchanges.
The existing MPI aggregation by destination rank is reused. Within revisions,
one scalar integer cascade level per halo cell is exchanged with neighbors.
Ghost levels determine local face and cached edge replacement; detection and SED
run on owned cells. There are no global-sized arrays or new detector collectives.
All payload packing is in device kernels.

PLM/DC EMFs are cached in persistent device arrays. Revisions replace adjacent
face fluxes and adjacent edge EMFs; other edges retain the initial high-order
value. Final CT uses these selected EMFs. No high-order reconstruction or corner
composition is repeated during revisions. The candidate payload also restores
initial fluxes/EMFs after fallback-cache construction, removing the prototype's
per-stage high-order and face-EMF copies.

## Cell-local rcfl validation

`local_rcfl.py` checks actual first-stage tolerance multipliers against each
cell's velocity and spacing. Variable-speed Orszag–Tang cells must differ from
the former global multiplier; stationary hydro cells have zero advective factor.
The executed checks matched every candidate/flag snapshot and final field bitwise
for FV4/6/8 at
5/6/8 ghosts (eight steps) and FV8 on four MPI ranks. FV8 in 3D after two steps
had identical flags and maximum field difference 1.39e-17 (divB 4.17e-16); a
`relative` control also showed field differences of 1.39e-17. Hydro and MHD
physics runs compare one block against four blocks, and the remaining global
options must reject. Before/after `relative` checks additionally require the
previous executable saved as `athena-before-rcfl`.

## Earlier relative-mode validation results

The following results were recorded before restoring local `rcfl` and used
`mood_nad_scale=relative` throughout.

All builds used CPU execution, double-precision arithmetic and Kokkos bounds
checking. Audit output is double precision; normal binary output is unchanged.

- **52 local-detector halo runs:** 2D FV4/FV6/FV8 with 1–4 revisions, 3D with two
  revisions, eight-step runs, four MPI ranks, and 8-cell blocks (FV8 nghost=8).
  All 26 positive comparisons matched every candidate, flag, final field and
  div(B) bitwise. All four shortened-halo negative controls changed the final
  fields as expected.
- Before removing the global-scale options, the neighbor implementation also
  matched the global-exchange prototype bitwise in nine 2D/3D/multistep checks.
  Those checks isolated the communication implementation from the detector change.
- **Orszag–Tang, 64², t=0.25:** all FV orders, single block versus four blocks.
  All fields, flags and div(B) match bitwise. FV8 on four MPI ranks also matches.
  Minimum density >=0.0640 and minimum internal energy >=0.0251. Maximum
  normalized conserved-mean drift <=3.22e-15; max|div(B)| <=1.82e-13.
- **Other solvers:** LLF, HLLE and L-HLLD, eight steps, bitwise agreement across
  block decompositions. HLLD is covered throughout the main tests.
- **Hydro:** a periodic Sod-type initial condition through t=0.05, bitwise
  agreement across decompositions. Before/after checks for hydro and MHD confirm
  the previous `relative` arithmetic is unchanged by removing the other scales.
- **Restart:** resume at cycle 4 and continue to cycle 8; final binary fields
  match the uninterrupted run bitwise.
- **Input guards at the original validation commit:** all four then-removed modes
  rejected for hydro and MHD (`rcfl` is restored and tested below); insufficient
  ghosts and explicit frozen mode with a nonperiodic boundary reject.
- **Normal build:** default selection at 5/6/8 ghosts matches the audited results
  at the normal output format's float precision.

### Smooth-wave accuracy

One fast-wave period at 32² and 64², amplitude 1e-6, default velocity NAD:

| Method | Order | RMS-L1 at 32² | RMS-L1 at 64² |
|---|---:|---:|---:|
| Frozen | FV4 | 1.798973e-8 | 3.298506e-9 |
| Legacy | FV4 | 1.793331e-8 | 3.299673e-9 |
| Frozen | FV6 | 1.748129e-8 | 2.910328e-9 |
| Legacy | FV6 | 1.724340e-8 | 2.874603e-9 |
| Frozen | FV8 | 1.732070e-8 | 3.939018e-9 |
| Legacy | FV8 | 1.777476e-8 | 3.895997e-9 |

These errors differ by less than 3%. With `mood_nad_v=off`, as in the existing
linear-wave validation manifest, FV8 frozen/legacy RMS-L1 errors are
6.592683e-10 / 6.592682e-10 at 32² and 1.623232e-10 / 1.623228e-10 at 64².
This confirms similar accuracy in these tests; it does **not** establish formal
fourth/sixth/eighth-order convergence. The measured two-resolution behavior is
roughly second order and the velocity detector substantially increases this
small-amplitude wave's error in both methods.

### Runtime

A 64² field loop, four 32² blocks, FV8, 32 RK3 steps, three serial CPU repeats:
median solver time 1.905 s frozen (nghost=8), 2.690 s legacy (nghost=18), about
29% less time. The legacy count covers the full initial candidate dependency.
This is one small CPU case, not a scaling or GPU benchmark. No peak-memory
measurement is claimed.

## Limits

The original study below did not certify GPU execution or relativity. The
[new GPU matrix](../gpu/README.md) covers the subsequent NR/SR/GR extension.
SMR/AMR, nonperiodic boundaries, excision, diffusion, shear,
passive scalars, well balance and touched-only revisions are not certified for
the frozen method. The unsupported physical configurations stay on the legacy path by default.
The original CPU results on this page remain unchanged; GPU evidence is separate.
The local NAD change applies to hydro/MHD generally; the previous studies using
global scales are historical results and must be rerun to assess those regimes.

## Reproduction

The script below reproduces the historical local-loop experiment against its
recorded source version. Use the [current controls](../cascade_exchange/README.md)
for the present integer-exchange implementation.

```
python3 validation/frozen_emf_halo/neighbor_exchange/reproduce.py \
  --repo /path/to/athenak --kokkos /path/to/kokkos \
  --destination /path/to/new-neighbor-validation --jobs 8
```

Use a checkout containing this implementation. Requires Python 3.12+, NumPy,
CMake, C++17 and OpenMPI. The script builds a separate audited source tree and
runs halo, physics, accuracy, restart/guard and runtime checks. The audit patch
adds snapshots, forced detectors, double output and a problem dispatcher; it
is not part of the normal source build. Historical before/after equivalence
checks additionally need the earlier neighbor binary as `athena-pre-local`.

JSON files contain detailed results; `recorded_runs/` holds decks/logs rather
than large snapshots/checkpoints. `provenance.json` records source and executable
hashes. Historical prototype scripts in the parent directory still reproduce
their pinned old baseline and intentionally retain their recorded `rcfl` inputs.
