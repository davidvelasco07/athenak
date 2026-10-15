# Integer cascade exchange and reduced MHD halos

The supported frozen-EMF path now detects **owned cells only** and exchanges one
integer cascade level per halo cell after every revision. Each block rebuilds
the affected face fluxes and selects cached fallback EMFs using these owner
levels. Original high-order fluxes and EMFs remain frozen.

The revision count drops out of the ghost bound. Tested bounds are:

| Reconstruction | HLLD / HLLE / LLF | L-HLLD |
|---|---:|---:|
| FV4 | 3 | 4 |
| FV6 | 4 | 5 |
| FV8 | 5 | 6 |

These bounds apply to the existing supported uniform periodic configurations.
SR/GR use their supported LLF/HLLE solvers; L-HLLD is Newtonian. Unsupported
configurations retain the legacy method and its existing bounds.

## Data needed on each side of an interface

Let r be the cell-polynomial radius: 2/3/4 for FV4/FV6/FV8.

1. The face solve needs r+1 input cells across a boundary. The corner composition
   samples faces transversely through the same r+1 reach. Those supports act on
   different axes, so their coordinate-wise maximum is r+1. The earlier frozen
   guard of 2r was unnecessarily conservative.
2. Newtonian magnetic volume averaging needs candidate CT faces through r-1
   beyond the owned cells. Their EMFs depend on cascade levels through r cells.
   PLM EMF caches over that region need r+1 stage-input cells.
3. SED reads immutable stage inputs through +/-3. It is evaluated only for owned
   cells, so it adds no revision-dependent ghost region.
4. L-HLLD's transverse compression sensor needs one additional input cell.

Thus the enforced frozen bound is

```text
nghost >= max(3, r + 1 + [solver == L-HLLD])
```

The integer exchange has width r for Newtonian volume-average detection and
width one for SR/GR's existing two-face magnetic average. Initial primitive and
high-order flux/EMF exchanges still occur once per RK stage. Inside revisions,
the new payload is **MPI_INT: one four-byte integer per communicated cell**.
Face, edge and corner neighbors are included. The existing aggregation by peer
rank and one-time neighbor layout exchange are reused. The original fixed-count
validation below used no convergence collective; the current early-exit loop adds
one nonblocking integer consensus per detector check. See
[early termination](../early_stop/README.md).

## Revision sequence

For each RK stage:

1. Exchange immutable primitives and magnetic volume averages; compute and
   exchange the original high-order face fluxes and edge EMFs.
2. Build PLM and donor-cell EMF caches over the fixed averaging support.
3. Construct the candidate conserved state and CT faces from the fixed RK stage
   inputs and current mixed fluxes/EMFs; form its magnetic volume averages.
4. Detect owned cells. Exchange their cascade levels and identify which imported
   ghost levels increased.
5. Reduce one integer indicating whether any owned detector violations remain.
   Stop when all ranks are clear. Otherwise, exchange levels and revise adjacent
   physical faces and cached fallback EMFs, up to the configured correction cap.
6. Check the candidate again after the final allowed correction, without changing
   accepted cascade levels. Report unresolved limits separately from convergence.

`ApplyFallback` yields while the convergence collective or integer receive is
pending. Ranks with no local violations keep participating while another rank
needs repairs. The cap is a maximum; a clear initial candidate uses zero corrections. `mood_halo0=0` for this path; no ghost cell performs NAD or
SED. No candidate U/B or floating-point flux/EMF payload is sent inside revisions.
The existing EOS repair policy is retained when the correction cap is exhausted.
The CPU/GPU numbers below describe the earlier fixed-count implementation; the
early-exit equivalence controls are recorded separately.

## CPU validation

All CPU builds use double precision, MPI and Kokkos bounds checking.

- Initial bound controls: **44 runs**, revisions 1–4, 2D/3D, and four MPI ranks.
  All 29 comparisons / 232 saved datasets match the preceding larger-halo
  executable bitwise. This includes the proposed four-revision 6/6/8 settings.
- Final 3/4/5 bound: **45 runs**, 36 comparisons / 288 saved datasets, all bitwise
  identical to the larger-halo executable. Includes revisions 1–4, 3D, and
  LLF/HLLE/L-HLLD controls; L-HLLD uses 4/5/6 ghosts.
- L-HLLD with UCT-HLLD: four further comparisons, including 3D, all bitwise
  identical at the extra-cell bound.
- SR/GR flat-spacetime shock tubes: six comparisons at 3/4/5 ghosts, all bitwise
  identical across four ranks. Velocity NAD is disabled in these controls because
  of the separately documented relativistic velocity-representation issue.
- Six insufficient-ghost inputs reject with the expected bound.

### Forced interface and corner repairs

The separate `audit.patch` forces a PLM repair followed by donor cell at cells
r away from a block's last owned cell, then moves the repair to the next block.
It also forces the periodic corner. Some ranks have no local repairs during
individual revisions. Every owned candidate U/B value and cascade level is
recorded in double precision for all four revisions and three RK stages.

FV4/FV6/FV8 in 2D and 3D compare one block with four ranks and 8-cell blocks:
**all 72 candidate/level snapshots are bitwise identical at 3/4/5 ghosts**.
Negative controls deliberately exchange only one layer of cascade levels for
FV8. Both 2D and 3D then differ by up to `3.6365861599274574e-6` in the candidate.
This demonstrates that the tests exercise the distant magnetic-average dependency.
The audit modifications are not present in production source.

## GPU validation

Apollo A100, double precision, RK3, CFL=0.25, four revisions, local rcfl
tolerance 1e-8, SED, HLLD/UCT-HLLD. Reference executable: the volume-average
implementation at `bcc400b3d` with its former 11/15/20 halos.

| Case | Block size | Ghosts | Outcome |
|---|---:|---:|---|
| FV4 128² | 32² | 3 | t=1, zero EOS floors/conversion failures |
| FV6 128² | 32² | 4 | t=1, zero EOS floors/conversion failures |
| FV8 128² | 32² | 5 | t=1, zero EOS floors/conversion failures |
| FV4 512² | 64² | 3 | t=1, zero EOS floors/conversion failures |

All eighty saved datasets match the reference bitwise, including fallback levels
and staggered div B. Fallback is active. A separate eight-step FV8 comparison at
five ghosts matches bitwise between one and four CUDA-aware MPI ranks.

The preceding 4/6/8-ghost GPU phase also passed all five full t=1 cases, including
FV8 128² on four CUDA-aware MPI ranks: all one hundred saved datasets match the
reference. MPI ranks share one physical GPU for correctness testing; these are
not MPI performance-scaling measurements.

Four short FV8 CUDA memory checks, HLLD at five ghosts and L-HLLD at six ghosts,
each in 2D and extruded 3D, report **zero errors**. They use one MPI rank and
`--target-processes application-only`; MPI CUDA hooks are disabled for these
single-rank memory diagnostics. Actual CUDA-aware integer transfers are tested
by the separate multi-rank field comparisons. The default sanitizer process
tracking failed to attach during MPI startup; those diagnostic logs are preserved.

The final GPU executable SHA256 is
`4f781dec3ade2c1a05d00cbac90319e0dd4dedccca92eb3e4354dd078118e1cf`.
All eight changed C++/CMake files match `gpu/minimum/source-sha256.json` exactly.

These checks establish the stated halo equivalence for the tested cases. The
full corrected 2048²/4096² study and a final-revision failure policy remain
separate work. Refinement, physical boundaries, well balance, diffusion, shear,
passive scalars and touched-only detection remain outside this frozen path.

## Evidence and reproduction

`cpu/` and `gpu/` contain result/comparison JSON, selected exact decks and logs,
histories, build logs, source/executable hashes and launch diagnostics. Raw fields
remain in the validation work directories; forced double snapshots are retained
in the local `work/cascade-halo-validation/forced-minimum` directory.

Build a normal CPU executable with `Athena_ENABLE_MPI=ON`,
`Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON` and `PROBLEM=built_in_pgens`, then run:

```sh
python validation/frozen_emf_halo/cascade_exchange/reproduce.py \
  --repo /path/to/athenak --exe /path/to/new-cpu-athena \
  --reference /path/to/cpu-athena-built-from-bcc400b3d \
  --output /path/to/fresh-controls
```

Use the same compiler/backend for the reference. For the forced controls, apply
`audit.patch` to a separate source copy, build it with MPI/bounds checking, and
run `verify_forced.py --repo /path/to/audited-source --exe /path/to/audited-athena
--output /path/to/fresh-forced-controls`. Its short-halo environment switch exists
only in the audit patch.

The OT runner now defaults to 3/4/5 ghosts for its HLLD/HLLE frozen runs.
`--pre-cascade-exchange-halo` reproduces the preceding 11/15/20 policy for
historical executables; `--pre-pad-fix-halo` retains the older policy. These
historical flags are mutually exclusive. Recorded GPU controllers/decks are
under `gpu/`; `gpu/minimum/check_memcheck.py` includes the working sanitizer setup.
