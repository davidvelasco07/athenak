# Convergent termination of the frozen MHD revision loop

`mood_max_revs` is now a maximum number of flux/EMF corrections per RK stage.
The stage exits as soon as all ranks report clear owned-cell detector tests.
An initially clear candidate performs zero corrections. High-order fluxes and
EMFs remain frozen, and the 3/4/5 ghost bounds for FV4/FV6/FV8 are unchanged.

## Algorithm

1. Build and test the candidate using the current mixed fluxes and cached EMFs.
2. On multiple MPI ranks, post a nonblocking `MPI_Iallreduce` of one integer:
   whether this rank has any owned detector violations. Yield until it completes.
   A dedicated communicator separates this consensus from other collectives.
   One-rank and non-MPI executions use the local result directly.
3. If every rank is clear, finish immediately. A locally quiet rank continues
   participating while another rank needs a repair.
4. If corrections remain available, exchange integer cascade levels, revise
   affected faces/edges, increment the correction count, and return to step 1.
5. After the final permitted correction, test the resulting candidate once more.
   This last check is read-only with respect to accepted cascade levels. A
   remaining violation is reported as a revision limit, rather than convergence.

PAD/nonfinite violations in cells already at DC still count as violations even
though those cells cannot descend further. The existing EOS floor policy at the
cap is retained. This change adds no timestep rejection or post-CT abort.
NAD tolerances, smooth-extrema exemptions, and the existing NAD floor at PLM are
unchanged. The known SR/GR velocity-representation mismatch is separate work.
The termination predicate covers owned candidates; ghost EOS conversions are
outside that predicate.

Enable `<mhd>/mood_diagnostics=true` to record the actual count for every stage:

```text
MOOD cycle=0 stage=1 revisions=0 checks=1 status=converged
```

`revisions` counts applied corrections; `checks` is `revisions + 1`. At the cap,
`status=revision_limit` indicates unresolved detector tests. Failed-trial/FOFC
counters can change because the final candidate is now checked explicitly.

## CPU verification

Release, double precision, Kokkos bounds checking, MPI and Serial backend.
The independent fixed-count reference was built from the preserved source
archive of `abea4ac2fdf6e0b62ea655734c9c85fd51bb0a93`.

- 17 normal/reference runs: uniform states, FV4/FV6/FV8 OT, extruded 3D OT,
  and SR/flat-GR shock tubes. One-rank and four-rank results agree bitwise.
  All ten comparisons, totaling 300 saved variable arrays, pass; three compare
  directly against the original fixed-count implementation.
- Uniform states stop with zero corrections on one and four ranks. Tested OT
  stages use between one and eight corrections; SR/GR shocks use one or two.
- Two MPI-disabled runs match their one-rank MPI counterparts in all 60 arrays.
- A legacy FV4 UCT control remains bitwise identical in all 30 saved arrays.
- Separate detector-only audit: only the final MPI rank receives forced PAD
  flags. All ranks follow its two repairs and converge together. A persistent
  forced DC violation reaches eight corrections and remains unresolved.
- A moving forced violation verifies the terminal check leaves the accepted
  level map unchanged: eight PLM cells at the eight-correction cap, and one at
  the one-correction cap. These controls pass on one/four ranks (R1 on four).
  The audit disables velocity NAD in its uniform inputs to isolate the forced
  PAD cells from zero-velocity roundoff flags. Audit hooks are absent from
  production source.

`cpu/` contains normal/audit results, build/source hashes, and selected logs.
`detector-audit.patch` is diagnostic-only and applies to a separate source copy.
Raw states and audit inputs remain in `work/early-stop-validation/`.

To repeat the production CPU checks:

```sh
python validation/frozen_emf_halo/early_stop/check_cpu.py \
  --repo /path/to/repo --exe /path/to/new-athena \
  --reference /path/to/abea4ac2-athena --output /path/to/fresh-results
```

Both executables use `PROBLEM=built_in_pgens`, the same compiler/backend, MPI,
and Kokkos bounds checking. The GPU controls use the blast generator and the
preserved stress-study outputs through `check_gpu.py --help`.

## GPU verification

Apollo A100, double-precision Release, MPI-enabled production configuration
(`Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF`). Eight runs pass. Six comparisons,
totaling **240 saved float32 variable arrays**, are bitwise identical. This
includes the prior NR/SR/flat-GR FV8 R4 blast outputs, the GR R8 control, and
uniform/active one-rank versus four-rank runs. Physical EOS counters are also
identical to the earlier R4 production runs in all three regimes.

| Control | Cap | RK stages | Mean rounds | Converged stages | Unresolved at cap |
|---|---:|---:|---:|---:|---:|
| Uniform | 8 | 3 | 0.000 | 3 | 0 |
| NR low-beta blast | 4 | 8127 | 3.976 | 488 | 7639 |
| SR UCT blast | 4 | 333 | 2.459 | 279 | 54 |
| GR UCT blast | 4 | 855 | 2.646 | 661 | 194 |
| GR UCT blast | 8 | 855 | 3.420 | 766 | 89 |
| Active four-rank NR control | 8 | 9 | 2.333 | 9 | 0 |

One- and four-rank stage traces agree exactly for both the uniform and active
GPU controls. The four ranks share one GPU for correctness, not scaling
measurements. Remaining DC violations can still exhaust the cap, as the table
shows; early termination does not establish floor-free admissibility.

A separate bounds-enabled CUDA build was stopped after its assembler spent
more than 18 minutes optimizing individual translation units. The production
configuration completed and supplied all GPU controls. CPU bounds checking
passed as recorded above. This compiler diagnostic did not change numerical
source or the selected GPU configuration.

See `gpu/results.json`, `gpu/gpu-build-provenance.json`, selected inputs/logs and
`gpu/build-prod.sh`. Raw GPU fields remain on Apollo under
`/scratch/lustre/velasco/codex-validation/early-stop-20261015/results/`. The earlier
109-entry stress report remains pinned to its original fixed-count executable;
these are targeted equivalence and termination controls for the correction.
