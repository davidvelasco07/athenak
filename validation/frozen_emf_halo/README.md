# Frozen-EMF MOOD validation

## Current method

The [neighbor-exchange implementation](neighbor_exchange/README.md) is the default
for supported uniform periodic FV4/FV6/FV8 MHD configurations, including NR/SR/GR.
Hydro now has the analogous immutable-flux exchange (`mood_frozen_flux`).
See the [GPU matrix](gpu/README.md) for the six-configuration validation. It exchanges the
immutable high-order data before the revision loop, retains untouched EMFs and
uses cached PLM/DC values for revised edges. Newtonian magnetic volume averaging
uses the [normal polynomial integrals](magnetic_volume_average/README.md).
The current [integer cascade exchange](cascade_exchange/README.md) communicates
one level per halo cell after each owned-cell detection pass. MHD ghost counts
are **3/4/5** for FV4/FV6/FV8 with HLLD/HLLE/LLF, and **4/5/6** for L-HLLD's
extra sensor cell, independently of revision count. SR/GR use their supported
LLF/HLLE solvers with **3/4/5**. The preceding wider-halo results are historical.
The [early-termination correction](early_stop/README.md) makes `mood_max_revs`
a maximum for frozen MHD: all ranks stop as soon as their owned detector checks
are clear. It uses one nonblocking integer consensus per check, retains DC PAD
failures as unresolved, and checks the final corrected candidate without changing
its accepted levels. Enable `mood_diagnostics` for actual per-stage counts.

Hydro is unchanged: it needs **5** for all three orders at two revisions,
`max(reconstruction_radius + 1, revisions + 3)`.

Hydro/MHD detection supports `relative` (default) and cell-local `rcfl` tolerances.
`gcfl`, `grange`, `gdu` and the associated `mood_nad_theta` option remain removed. The current
[report and reproduction scripts](neighbor_exchange/README.md) describe scope,
input migration, conservation, accuracy, MPI, restart and runtime checks.

## Orszag–Tang robustness

The [current OT resolution study](ot_resolution/early_exit/README.md) reruns FV4/FV6/FV8
from 64² through 4096² on Apollo with four revisions maximum, `rcfl` tolerance
`1e-8`, magnetic volume averages, 3/4/5 ghosts, and early termination. It records
actual revision counts, unresolved final checks, EOS repairs, conservation, and
t = 1 maps. Larger resolutions remain pending until their recorded runs complete.

The [earlier two-revision](ot_resolution/README.md) and
[four-revision](ot_resolution/four_revision_rtol1e8/README.md) sweeps retain their
original executables and are historical. Their magnetic-energy inconsistency was
identified by the [PAD audit](ot_resolution/four_revision_rtol1e8/pad_audit/README.md),
then addressed by the [correction and focused GPU checks](ot_resolution/pad_correction/README.md).
Those old positivity failures do not describe the corrected implementation.

## Historical experiments

These records remain reproducible against their pinned baseline commit
`2d251bb09d67fe76ac97d419041c33dc906db2ca`:

- `halo-test-report.txt`, `halo-test-results.json`, and the scripts in this
  directory: 72 runs of the first local-construction frozen-EMF prototype,
  establishing the additive N+revisions bound (6/8/10 for two revisions).
- [maximum_bound/](maximum_bound/README.md): 50 runs demonstrating 5/6/8 using
  dense global owner exchanges. This was a dependency audit, not a scalable
  implementation.

Their saved inputs and patches intentionally preserve the old detector settings.
Use `neighbor_exchange/reproduce.py` to validate the current source.
