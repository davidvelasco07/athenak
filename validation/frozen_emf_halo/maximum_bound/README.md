# Testing the maximum bound

The tested sufficient bound is `nghost >= max(FV_order, revisions + 3)` with the
smooth-extrema detector enabled. For two revisions the counts are **5/6/8** for
FV4/FV6/FV8, down from the frozen local-construction counts **6/8/10**.
This is a dependency experiment, not a proof that every smaller count fails.

| Revisions | FV4 | FV6 | FV8 |
|---|---:|---:|---:|
| 1 | 4 | 6 | 8 |
| 2 | 5 | 6 | 8 |
| 3 | 6 | 6 | 8 |
| 4 | 7 | 7 | 8 |

## What permits the maximum

The high-order candidate is needed in the revision halo, even though its
high-order reconstruction must not be repeated during revision. Constructing
that halo independently on every block gives an additive input footprint.
Instead, this audit exchanges immutable stage primitives and cell-centered B,
computes the high-order fluxes and EMFs for owned cells/edges, and copies those
owner values into neighboring halos before the revision loop. The PLM/DC caches
and revisions then use the smaller local stencil. There are no new halo
exchanges inside the revision loop.

PLM reconstruction reaches two cells. The existing smooth-extrema detector
reaches **three**, because it evaluates a +/-2 stencil centered at offsets
-1, 0, and +1. Thus the tested low-order-side bound is revisions+3. The audit
also corrects the plain face average at the outermost ghost cell to use its
existing upper face; the previous code clamped that average to its lower face.

## Evidence

50 completed runs, Kokkos bounds checking, double-precision audit output,
RK3, HLLD/UCT-HLLD, ideal MHD, periodic field loops:

- 2D FV4/FV6/FV8 with 1–4 revisions, one block versus four blocks.
- 3D FV4/FV6/FV8 with two revisions, one block versus eight blocks.
- Eight steps for all three orders at the two-revision bound.
- Four MPI ranks for FV8 in both 2D and 3D at nghost=8.
- Forced FV4 cascades for 1–4 revisions at the maximum bound.
- Four negative controls shorten the detection halo by one; all change the
  final fields, confirming that the revision halo itself remains necessary.

All 24 positive comparisons are **bitwise equal** for fields, flags, div(B),
and every saved candidate/flag in the shrinking detector halo. All four
negative controls fail as expected. `max-halo-results.json` includes these
comparisons, parameters, source hashes, and the executable hash.

Compared separately with the previous no-exchange large-block references,
2D one-step and eight-step solutions are bitwise equal for all orders. The
3D periodic owner normalization changes values at roundoff level: final U/B
maximum differences <=2.78e-17 and div(B) differences <=4.45e-16; final flags
remain identical. The new 3D one-block and many-block runs are bitwise equal.

## Scope and reproduction

`owner-exchange-audit.patch` applies **after both original audit patches** in the
parent directory, against baseline 2d251bb09. It is not applied to the production
source branch. The branch retains its validated local-construction N+revisions
guard. This patch uses dense global owner-value arrays and MPI collectives,
requires a uniform periodic grid with one pack per rank, and is unsuitable for
production scaling. A production version needs task dependencies and neighbor
exchanges for the initial data. SMR/AMR, GPU, nonperiodic boundaries, relativity,
and diffusion have not been validated.

```sh
python3 validation/frozen_emf_halo/maximum_bound/reproduce.py \
  --repo /path/to/athenak --kokkos /path/to/kokkos \
  --destination /path/to/new-maximum-audit --jobs 8
```

Requires Python 3.12+, NumPy, CMake, C++17 and OpenMPI. The script reruns the
50-case maximum-bound suite. The nine historical comparisons in the saved
results additionally use the original suite's raw snapshots. Raw snapshots
are regenerated rather than committed; input decks and logs are recorded.
