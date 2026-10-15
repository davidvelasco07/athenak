# Frozen-EMF MOOD validation

## Current method

The [neighbor-exchange implementation](neighbor_exchange/README.md) is the default
for supported uniform periodic FV4/FV6/FV8 MHD configurations, including NR/SR/GR.
Hydro now has the analogous immutable-flux exchange (`mood_frozen_flux`).
See the [GPU matrix](gpu/README.md) for the six-configuration validation. It exchanges the
immutable high-order data before the revision loop, retains untouched EMFs and
uses cached PLM/DC values for revised edges. The two-revision MHD ghost counts are
**5/6/8**, with `nghost >= max(FV_order, revisions + 3)`. Hydro needs **5** for all three
orders at two revisions: `max(reconstruction_radius + 1, revisions + 3)`.

Hydro/MHD detection supports `relative` (default) and cell-local `rcfl` tolerances.
`gcfl`, `grange`, `gdu` and the associated `mood_nad_theta` option remain removed. The current
[report and reproduction scripts](neighbor_exchange/README.md) describe scope,
input migration, conservation, accuracy, MPI, restart and runtime checks.

## Orszag–Tang robustness

The [OT resolution study](ot_resolution/README.md) tests FV4/6/8 from 64² to 1024²
on Apollo with two revisions and the new frozen path. Only six of fifteen cases
remain floor-free to t = 1; all three orders require energy floors at 1024².
A wider halo does not remove the FV8 128² failure. A third revision fixes that
case but still requires floors at 256². These are current positivity limitations
under the documented HLLD/UCT-HLLD and cell-local rcfl settings.

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
