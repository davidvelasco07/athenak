# Magnetic volume averages for the MHD candidate

**Averaging validation at `bcc400b3d`:** the averaging weights remain current.
The larger halos and local revision protocol described below are superseded by
[integer cascade exchange](../cascade_exchange/README.md), now validated at
3/4/5 ghosts for FV4/FV6/FV8 with HLLD. The measurements on this page retain their
original test settings.

This implements the averaging part of the agreed candidate/revision sequence for
Newtonian ideal MHD with FV4/FV6/FV8 and UCT. Face fields remain the fields evolved
by CT. `utest` and `bcctest` together hold the eight-component cell-average
candidate; magnetic components are refreshed from the candidate CT faces before
each existing detection pass.

## Averaging operator

A staggered component is an area average over its normal face. Reconstruct that
area-averaged function in the normal direction, then integrate the polynomial
over the central cell. This produces a volume average without further
interpolation in the tangential directions. On a uniform grid, the weights in
increasing face-index order are:

| Method | Numerators | Denominator |
|---|---|---:|
| FV4 | -1, 13, 13, -1 | 24 |
| FV6 | 11, -93, 802, 802, -93, 11 | 1440 |
| FV8 | -191, 1879, -9531, 68323, 68323, -9531, 1879, -191 | 120960 |

These weights integrate monomials through degrees 3, 5 and 7 respectively.
They differ from `FV4Face/FV6Face/FV8Face`, which reconstruct an interface value
from cell averages. Reusing those interface weights on face-area averages was
incorrect for this operation.

`eos/mhd_bcc.hpp` supplies the operator used by production primitive conversion
and the candidate. It loads only the stencil needed by the selected order and
reduces through FV8/FV6/FV4/two-face averaging at outer array boundaries. Collapsed
dimensions use their two bounding planes.

Magnetic NAD now checks the same candidate volume averages used by PAD and
internal-energy NAD. Its reference is the stage-input volume average; on the
frozen path, these reference values have already been exchanged from their cell
owners. No separate two-face magnetic candidate is substituted during FV
detection. The other NAD variable choices and local `rcfl` scaling are unchanged.

## Revision behavior and remaining decisions

High-order fluxes and EMFs remain cached. Revisions select fallback face fluxes
and cached fallback edge EMFs, rebuild the candidate from fixed RK stage inputs,
CT-update the candidate faces, and apply the averaging operator before detection.
The averaging weights remain those of the selected FV order during revisions.

The previously added automatic post-CT PAD abort has been removed. Revision-limit
handling, rechecking the candidate after the final permitted correction, and
handling an inadmissible donor-cell candidate still need an agreed policy.
Ordinary EOS floor/conversion counters remain available to validation, which
classifies any such event as an unsuccessful robustness run.

This change does **not** redesign revision communication or reduce the halo.
The averaging stencil still reaches 2/3/4 faces for FV4/FV6/FV8. A changed edge
changes nearby CT faces; their volume averages can change up to that reach away.
Thus fixed high-order averaging still introduces a revision dependency even
though the original high-order fluxes and EMFs are frozen. The current shrinking
halo is retained for isolated validation; its bound is not asserted to be the
minimum possible. Neighbor exchanges or a deliberately adaptive averaging
operator require a subsequent design decision.

SR/GR conversion continues to use its existing two-face average. Non-FV methods
retain their existing interpolation and magnetic NAD behavior. This result does
not certify those configurations, refinement or well balance.

## Operator validation

`check_volume_average.cpp` executes the actual helper through Kokkos kernels and
compares it with analytic integrals, including all three component directions,
small arrays and boundary reductions, and products with nonconstant tangential
face averages. CPU bounds checking is enabled in the recorded build.

CPU and CUDA: **1,548 integral checks pass on each**, maximum absolute error
`2.2204460492503131e-16`. Sine-average convergence rates are 3.96368, 5.93283 and
7.89999 for FV4/FV6/FV8. These are convergence rates of the averaging operator,
not a claim about the order of the full MHD solver.

Reproduce against an existing double-precision AthenaK/Kokkos build:

```sh
cmake -S validation/frozen_emf_halo/magnetic_volume_average -B /tmp/bavg-check \
  -DKokkos_DIR=/path/to/athena-build/kokkos \
  -DATHENA_BUILD=/path/to/athena-build
cmake --build /tmp/bavg-check -j4
ctest --test-dir /tmp/bavg-check --output-on-failure
```

For CUDA, use the same `nvcc_wrapper` compiler as the AthenaK build.

## Focused solver validation

All runs use four revisions, local `rcfl` tolerance `1e-8`, SED, and the retained
conservative halo. The tests isolate averaging from a communication or halo change.

- CPU: nine 64² OT runs to t=0.01, FV4/FV6/FV8 each with one block, four blocks,
  and four MPI ranks. All finish without EOS floors/conversion failures. All 48
  compared binary datasets match bitwise across the six decomposition comparisons,
  including fallback levels and div B. Fallback is active in every run. Log
  counters are rank-local and can also count duplicate halo tests.
- CUDA: Apollo A100, double precision, RK3, HLLD/UCT-HLLD, CFL=0.25. All five
  focused cases below reach t=1 with finite positive saved states, zero
  density/energy/temperature floors, and zero conversion failures or velocity
  ceilings. PAD rejection counts include duplicated halo cells.

| Case | Block size | Test ghosts | PAD rejections |
|---|---:|---:|---:|
| FV4 128² | 64² | 11 | 2 |
| FV4 128² | 32² | 11 | 4 |
| FV6 128² | 64² | 15 | 0 |
| FV8 128² | 64² | 20 | 12 |
| FV4 512² | 64² | 11 | 4798 |

All twenty saved GPU datasets in the FV4 32²/64² block comparison are bitwise
identical. FV4 512² maximum relative total-energy drift is `3.6525e-13` and
maximum saved staggered `|div B|` is `1.5021e-11`.

Short CUDA compute-sanitizer tests in 2D and extruded 3D, FV8 at 32 cells per
active direction through t=0.002, each report **zero memory errors**. These test
memory access, not long-time 3D robustness.

Evidence is under `cpu/` and `gpu/`: exact decks, logs, histories, JSON results,
comparison records, build/operator logs and executable/source hashes. All seven
numerical source files hashed in `gpu/source-sha256.json` match this checkout
exactly. The tested GPU executable SHA256 is
`f396af501beb1d1bccbcf71b594a8f57f1d06e49c3ce74656201cfbd3cb976eb`.

Reproduce the OT settings with the existing runner against a build containing
these source changes:

```sh
python validation/frozen_emf_halo/ot_resolution/run_study.py \
  --source /path/to/athenak --exe /path/to/athena --output /path/to/new-runs \
  --resolutions 128 --orders 4,6,8 --block 64 --revisions 4 --rtol 1e-8 \
  --pre-cascade-exchange-halo
```

The runner's default halos remain conservative. The counts in the table are
test settings, not a measurement of the minimum halo. These focused results do
not establish full-solver formal order or high-resolution robustness.

The superseded OT sweeps were intentionally stopped, with all completed and
partial files preserved. The old PAD patch's report describes a historical
executable and should not be treated as evidence for this averaging change.
