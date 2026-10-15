# Production consistent magnetic PAD correction

**The corrected FV4 512² GPU run reaches t = 1 without EOS floors.** The old
executable, using the same eleven-ghost input, first requires energy floors at
cycle 5263 near t = 0.884. Both runs use four revisions and local rcfl tolerance
1e-8. This isolates the correction from the halo-size change.

## Implementation

- PAD and production primitive conversion share `eos/mhd_bcc.hpp`, preserving
  the existing unlimited FV face-to-cell magnetic interpolant and its boundary
  reductions. PAD no longer substitutes the simple face average for magnetic energy.
- Candidate CT faces extend through the interpolation stencil. Internal-energy
  NAD also uses the production magnetic energy; magnetic-component NAD retains
  its face-average reference and candidate representation.
- The revision halo accounts for this additional dependency. Base fluxes and
  EMFs remain frozen; no base corner composition is repeated during revisions.
- After final CT and U/B boundary synchronization, owned cells receive another
  PAD and finiteness check using the actual evolved fields. It checks all fallback
  tiers, including donor cell. If a revision ceiling leaves an inadmissible state,
  the run stops explicitly before primitive conversion instead of accepting an
  EOS floor repair. This guard does not implement timestep retry.

The wider candidate field disables the touched-only optimization for Newtonian
ideal UCT. The unchanged average-field SR/GR paths keep their previous halo rule.
The separate relativistic velocity-NAD issue in the earlier audit is not addressed
by this magnetic-energy correction.

## Halo requirements

For Newtonian ideal frozen UCT, let r = FV order / 2 and R = maximum revisions.
The conservative requirement is

```text
halo0 = r * R
nghost >= max(FV_order, halo0 + max(3, r))
```

| Reconstruction | Two revisions | Four revisions |
|---|---:|---:|
| FV4 | 7 | 11 |
| FV6 | 9 | 15 |
| FV8 | 12 | 20 |

A revised edge affects CT faces, and their production cell-centred interpolant
can change PAD up to r cells away. Freezing base EMFs removes their recomposition
dependency but does not remove this magnetic-energy dependency. The earlier
5/6/8 and 7/7/8 bounds omitted it. This is a conservative correctness bound, not
an empirical claim that every smaller halo fails.

SR/GR MHD retains `max(FV_order, R+3)`. Hydro is unchanged. Retaining smaller
Newtonian halos requires a different communication or magnetic-field treatment.

## GPU results

Apollo A100 GPU 0, double precision, stock Newtonian OT, HLLD/UCT-HLLD, RK3,
CFL 0.25, four revisions, rcfl tolerance 1e-8, SED enabled. Clean means t = 1,
finite positive snapshots and zero EOS floors or conversion failures.

| Case | Block size | Ghosts | Outcome | PAD rejection count |
|---|---:|---:|---|---:|
| Old FV4 512² | 64² | 11 | Energy floors, first cycle 5263 | 0 |
| Corrected FV4 512² | 64² | 11 | Clean to t = 1 | 4 |
| Corrected FV4 128² | 64² | 11 | Clean to t = 1 | 8 |
| Corrected FV4 128² | 32² | 11 | Clean to t = 1 | 16 |
| Corrected FV6 128² | 64² | 15 | Clean to t = 1 | 2 |
| Corrected FV8 128² | 64² | 20 | Clean to t = 1 | 168 |

PAD counts include duplicated halo cells; they are not counts of unique physical
cells. All corrected runs have zero density/energy/temperature floors, velocity
ceilings and conversion failures. Corrected FV4 512² maximum relative total-energy
drift is 3.656e-13.

All twenty saved field datasets in the 32²/64² block comparison are bitwise
identical, including fallback levels and div B. All four initial field datasets
in the old/corrected FV4 512² comparison are bitwise identical, confirming that
the production interpolation was preserved. See `fixed-validation/comparisons.json`.

CUDA compute-sanitizer memcheck reports **zero errors** in short 2D and 3D FV8
32-cell-per-direction runs at twenty ghosts. These exercise the expanded candidate
stencils and final-state guard; the 3D initial data is an extruded OT state and
the end time is 0.002. They are memory-access checks, not a full 3D robustness study.

## Evidence and reproduction

- `fixed-validation/checks.json`: focused case outcomes.
- Per-case directories: exact input decks, stdout, logs, histories and result JSON.
- `fixed-validation/memcheck_2d/` and `memcheck_3d/`: sanitizer input and output.
- `fixed-validation/executable.sha256`: corrected executable provenance.
- `fixed-validation/source.patch`, `source-sha256.json`: exact tested numerical
  changes from source base `903b6b9606a440ab2476aa49c17775e74e9ba7d9`.
- `build-fixed.log`: successful GPU build log.
- `reproduce_apollo.py`: commands used for the focused checks on Apollo.

The published source also has explanatory comment cleanup after the build.
No additional numerical changes were made after these tests.

The [extended resolution sweep](../four_revision_rtol1e8/README.md) continues with
its original uncorrected executable. Its results must not be presented as results
of this correction.

A separate corrected sweep is now running on Apollo GPU 0 at 64², 128², 256²,
512², 1024², 2048² and 4096² for all three orders, with four revisions and rcfl
tolerance 1e-8. It uses 256² blocks (or one smaller block at the smallest grids)
to reduce ghost-memory overhead for 4096². The corrected FV4 512² run also passes
with 256² blocks, and all twenty saved field datasets match the 64²-block run
bitwise; see `fixed-validation/blocks64_vs256.json`.

The new sweep uses an immutable copy of the validated executable and is stored at
`/scratch/lustre/velasco/codex-validation/local-rcfl-gpu-20261014/ot-resolution/pad-corrected-resolution`.
The full corrected 2048²/4096² matrix remains in progress.
