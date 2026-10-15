# FV4 magnetic energy consistency audit

**The FV4 512² energy-floor event is a physical-cell failure that the current
PAD magnetic-field representation can miss.** A diagnostic-only GPU build
reproduces the original first-floor cycle and records positive internal energy
with the PAD face average but negative internal energy with production FV4
interpolation. The numerical update and sweep executable were not changed.

## Detection variables

In this Newtonian run, NAD checks eight quantities: density, internal energy
density, three velocity components and three magnetic components. Each uses the
stage-input 3 by 3 neighborhood. SED may exempt smooth extrema from NAD;
it never exempts PAD or nonfinite states. NAD demotes the base tier to PLM.
PAD and nonfinite checks can demote PLM to donor cell.

The rcfl relative tolerance is local:

```text
C_local = min(1, dt * max(abs(vx), abs(vy), abs(vz)) / min(dx, dy))
rtol_local = 1e-8 * C_local
epsilon_lower = max(rtol_local * abs(q_min), atol, eps0 * abs(q_min))
epsilon_upper = max(rtol_local * abs(q_max), atol, eps0 * abs(q_max))
```

Here `atol=0`, `eps0=1e-12`, and `dt` is the full-step timestep.
PAD asks the EOS whether the candidate would require density, energy or
temperature floors. Candidate internal energy is total energy less kinetic
and magnetic energies; pressure differs by the fixed factor gamma minus one.

## Confirmed mismatch

The candidate CT update is followed by a plain average of adjacent faces in
`src/mhd/mhd_mood.cpp:737`. That averaged field is passed to floor-only
`ConsToPrim` at line 752. Floor-only conversion consumes that supplied field in
`src/eos/ideal_mhd.cpp:159`.

Production conversion follows a different branch at
`src/eos/ideal_mhd.cpp:168`: it applies the unlimited FV4/FV6/FV8 central
face-to-cell interpolant. Thus PAD does not test the magnetic energy used to
accept the evolved state. Changing `mood_nad_bcc` alone does not repair this:
that option changes the NAD reference, not the candidate PAD field.

The base fluxes and corner EMFs are frozen as intended. `SendMOODCandidate`
composes the base edges once, and the loop caches PLM/DC edge values and replaces
selected edges. The discrepancy occurs when the revised face field is converted
to cell-centred magnetic energy, after those operations.

## GPU reproduction

Apollo A100 GPU 1; stock OT, FV4, 512², 64² blocks, HLLD/UCT-HLLD, RK3,
CFL 0.25, four revisions, seven ghosts and rcfl tolerance 1e-8. The input deck
is in `reproduce/ot_fv4_n512/input.athinput`. Instrumentation adds diagnostics
only, before the EOS applies any floor; see `instrument_pad.py`.

At the first event, mesh time is **0.8836558531863674**:

| Quantity | First physical cell | Second physical cell |
|---|---:|---:|
| MeshBlock index | 47 | 16 |
| Array indices j, i | 25, 18 | 52, 59 |
| Current fallback tier | 0, base | 0, base |
| Internal energy with face average | +3.2027923956e-3 | +3.2045387636e-3 |
| Internal energy with FV4 field | −1.0961832710e-5 | −6.1531053039e-6 |

All **35** recorded energy-floor events in the observed prefix are in physical
cells, all remain at the base tier, and all have positive internal energy when
evaluated with the face average. There are **zero PAD rejections** in the event
log. The first event is logged at completed cycle **5263**, matching the original
sweep. Diagnostics print the in-progress cycle 5262 before the driver increments
the completed-step counter. The EOS energy-density floor is 1.5e-12.

`events.json` contains the parsed trace. `stdout.txt`, `ot.log`, `ot.mhd.hst`
and full result metrics are under `reproduce/ot_fv4_n512/`.
`executable.sha256` and `configure.log` identify the diagnostic build.
The main sweep was briefly suspended during this reproduction and automatically
resumed; both transitions are recorded in the accompanying JSON files.

EOS counters elsewhere still include ghosts. This reproduction establishes the
physical location of the FV4 512² events only; other floor cases need their own
classification.

## Separating halo size from the regression

A control with **two revisions, rtol 1e-4 and seven ghosts** reaches t = 1 with
zero floors. All values in its twenty saved field files are bitwise identical
to the earlier five-ghost run: conserved fields, primitive fields, fallback levels
and div B at all five saved times. This rules out the change from five to seven
ghosts as the source of this FV4 512² regression at these settings.

The input, result and field comparison are under `controls/`.
Separate two-revision 1e-8 and four-revision 1e-4 controls remain in progress on
Apollo under `ot-resolution/factor-controls`; they are not needed to establish
the magnetic-energy mismatch, but separate the two parameter changes.

## Additional relativistic velocity check

A separate source-level issue occurs in the velocity NAD pass:
`mhd_mood.cpp:833` forms candidate components as momentum divided by conserved
density for every regime. Its reference is `w0`, which stores spatial components
of four-velocity in SR, as documented in `eos/ideal_srmhd.cpp:47`.
Those quantities differ by enthalpy and magnetic terms; the Newtonian division
does not recover the relativistic primitive velocity. There is no Newtonian-only
guard on this pass. This issue is independent of the Newtonian OT reproduction
and has not been tested here with a dedicated SR/GR run.

## Remaining acceptance checks

Two further properties of the loop matter when designing the correction:

- A cell already at the donor-cell tier is skipped by the detector
  (`mhd_mood.cpp:773`), even if PAD just flagged it.
- After the last allowed face/edge revision, the loop exits without another
  candidate test (`mhd_mood.cpp:1062`). The final revised state therefore has no
  explicit acceptance check at that point.

These properties do not explain away the reproduced base-tier failure. They
mean that matching the PAD and production magnetic fields is necessary but does
not by itself establish positivity of every final revised state. A correction
must validate the magnetic field and energy actually accepted after revisions
and shared-edge synchronization, with an explicit response if donor-cell
fallback remains inadmissible. Its stencil and halo requirements must then be
rechecked. No numerical correction is included in this audit.
