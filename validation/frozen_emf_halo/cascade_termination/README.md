# Stop frozen MHD revisions when the cascade map is stable

Frozen MHD now requests another correction only when an owned cell needs a higher
cascade level somewhere in the MPI domain. NAD remains limited to high order →
PLM. Only PAD/nonfinite failures can continue through PLM → donor cell (DC).
This replaces the [previous any-violation stopping rule](../early_stop/README.md).

## Decision and communication

The existing detector reduction counts three disjoint owned-cell categories:

- `nad_requests`: a high-order cell requests PLM after NAD/SED testing.
- `pad_requests`: an inadmissible/nonfinite cell below DC requests the next tier.
- `dc_bad`: an inadmissible/nonfinite cell is already at DC.

Requests are evaluated even after the correction budget is exhausted. That final
check remains read-only: its proposed levels never replace the accepted level map.
The continuation flag is `nad_requests || pad_requests`; physical failure is
`pad_requests || dc_bad`. A stable DC failure cannot request another correction.

Without diagnostics, a nonblocking `MPI_Iallreduce` combines the two flags in one
integer with `MPI_BOR`. A quiet rank still joins corrections requested elsewhere.
With diagnostics enabled, the same single collective instead sums three unsigned
64-bit counts, and the global flags are derived from those totals. One-rank and
non-MPI runs use the local result directly. No second collective, array comparison,
additional detection pass, or wider halo is introduced.

| Final check | Status |
|---|---|
| No requests and no physical failure | `converged` |
| No requests, but DC remains unphysical | `stalled_unphysical` |
| Requests remain at the correction cap | `revision_limit` |

A stable unphysical map exits even when the configured maximum is larger. Its
status does not claim admissibility. The existing EOS repair policy after the
stage is retained; this change adds no timestep rejection or post-CT abort.
High-order F/E stay frozen, the integer level exchange is unchanged, and minimum
HLLD/HLLE/LLF ghosts remain 3/4/5 for FV4/6/8. Hydro and legacy fallback retain
their preceding behaviour.

## Diagnostics

With `<mhd>/mood_diagnostics=true`, each terminal stage line includes:

```text
MOOD cycle=0 stage=1 revisions=2 checks=3 status=stalled_unphysical nad_demotions=0 pad_demotions=2 nad_pending=0 pad_pending=0 dc_bad=1
```

All counts are global sums over owned cells. `nad_demotions` and `pad_demotions`
count accepted level increases during this RK stage. A PAD cell can contribute
once for high order → PLM and again for PLM → DC. The pending counts and `dc_bad`
describe the final candidate check; pending requests at the cap are not applied.
These are detector counts, not actual EOS repairs. Trial/PAD rejection counters
can decrease because redundant checks are skipped; EOS counters retain their
original meaning.

The OT study parser accepts these optional fields and records accepted demotions
by reason, terminal requests, and terminal DC failures. Older pinned logs remain
readable without manufacturing missing reason counts.

## Verification

- CPU: 17 normal/reference runs, 10 comparisons and 300 saved variable arrays
  agree bitwise with the preceding implementation. Covers uniform data,
  FV4/6/8 OT, extruded 3D OT, SR/flat-GR shocks, and one/four MPI ranks.
- All seven one/four-rank global diagnostic traces match exactly. Their accepted
  NAD+PAD demotion totals agree with the existing event logs.
- 19 synthetic detector controls cover diagnostics on/off, one/four ranks, a
  quiet rank joining remote repairs, physical failure already at DC, moving
  demotions, and read-only checks at caps 1, 2 and 3.
- The mixed-rank case keeps one rank unphysical at DC while a different rank
  creates five successive NAD demotions. Both consensus paths continue for five
  corrections, then report `stalled_unphysical`. This catches the incorrect use
  of `MPI_MAX` on packed flags, which would lose the remote pending bit.
- Persistent DC failure stops after two corrections with cap eight. A moving
  failure still performs eight genuine corrections and reports `revision_limit`;
  its ninth check does not add a ninth demotion.

The non-MPI uniform/OT controls and one legacy control add 90 bitwise-equal
arrays. All CPU builds use Release, double precision and Kokkos bounds checking.
The new parser was checked against new global reason counts and historical logs
without those optional fields.

## Reproduction

Use the normal CPU executable with `PROBLEM=built_in_pgens`; keep the preceding
`b3f756c38` executable as the field reference:

```sh
python validation/frozen_emf_halo/cascade_termination/check_cpu.py \
  --repo /path/to/repo --exe /path/to/new-athena \
  --reference /path/to/preceding-athena --output /path/to/results/cpu
python validation/frozen_emf_halo/cascade_termination/check_audit.py \
  --repo /path/to/repo --build /path/to/cpu-build --output /path/to/results/audit
```

The audit helper compiles only a diagnostic replacement for `mhd_mood.cpp` and
relinks the unchanged CPU core. Forced PAD/NAD flags are absent from production.
The captured uniform input, diagnostic patch, exact source hashes and logs are
preserved with the results. The Apollo scripts preserve their deployed directory
layout and immutable reference executable paths.

The running OT resolution batch retains its pinned `bdef110c3` numerical source
and executable. Its files are not relabelled as this version. The current change
is validated separately; the 2048²/4096² runs continue independently.

## GPU results

A fresh normal CUDA Release build (double precision, MPI, A100, bounds checking
off) uses the same 347 verified numerical source files as the CPU build.
Thirteen short new/reference runs compare 210 saved arrays bitwise: uniform
one/four-rank runs, FV4/6/8 OT against the preceding GPU executable, active FV8
one/four-rank integer exchange and exact global traces, and SR/flat-GR shocks.

A separate GR blast at Rmax=8 compares 50 additional arrays bitwise against the
preceding production executable. All 855 stages retain their correction counts:
mean 3.419883. The previous 89 `revision_limit` stages become 87 `revision_limit`
and two `stalled_unphysical` stages. Those two maps become stable only at the
cap, so this particular case saves no rounds. The other 87 still request genuine PAD/nonfinite level increases, with zero
terminal NAD requests in those stages. Raw energy-floor applications remain 1379 in both runs; density,
temperature, velocity and conversion-failure counters remain zero. Raw EOS counts
include ghost cells; these are not owned-cell floor totals.

The blast selector recompiles only its generator, dispatcher and configuration
display, and relinks the freshly built numerical core. Its macro-use scan, core
hashes, commands and binary hashes are preserved. GPU checks briefly share GPU 0
with the pinned resolution batch and establish correctness only; no wall-time
performance comparison is claimed. The original study executable and source are
unchanged.

Across the CPU and GPU field comparisons, **650 saved variable arrays are
bitwise identical**. The forced audit adds 19 behavioural controls. Raw binary
states remain in `work/cascade-stop-validation/` and the recorded Apollo scratch
directory; portable inputs, compressed logs, histories, summaries and hashes are
preserved here.

These comparisons establish stopping and field equivalence. The previously
identified SR/GR velocity-NAD representation issue remains separate work; this
change does not revise detector variables, NAD tolerances or SED exemptions.

The short GPU OT controls (64², 16² blocks, Rmax=8, t=0.01) hit the cap in
2/1/2 stages for FV4/FV6/FV8 respectively. All five terminal cases request new
NAD demotions, with no PAD requests or failed DC cells. This confirms productive
NAD cascades in those GPU controls; it does not attribute every capped stage in
the longer resolution study.
