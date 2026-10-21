# Sparse cell fields

`athena_sparse` is a Kokkos library independent of any physics module. It allocates a
scalar field on a whole block or leaves it absent. The owning module supplies the block
count and storage extents, including ghosts. It also chooses activation and deallocation
rules. This version is not connected to hydro, chemistry, or RT state arrays.

## Interface and lifetime

```cpp
sparse::Registry fields({{7, "tracer_a", 0.0}, {42, "tracer_b", 0.0}});
sparse::Variables vars("tracers", fields, nblocks, n3, n2, n1, 2);
vars.Allocate({{0, 42}, {1, 7}});  // pairs use global field IDs
auto view = vars.GetDeviceView();
```

Global IDs and names are unique and immutable. IDs can be signed and non-contiguous.
`fields.Index(id)` returns the ordinal used by device accessors. A slot owns every state
register for one `(block, field)` pair. Registers have the same shape and share allocation
status, but their numerical meanings belong to the caller. Spatial extents are uniform
across this store; different shapes need different stores in this initial version.

`view.Read(reg, block, ordinal, k, j, i)` returns the registered default if absent.
`view.At(reg, slot, k, j, i)` is a writable reference for an allocated slot. Iterate
`work = 0 .. view.active_count-1`, get `slot = view.ActiveSlot(work)`, and recover the block
and field ordinal with `Block(slot)` and `FieldIndex(slot)`. Work is sorted by block and
registry ordinal. Writes never implicitly allocate and must not target an absent slot.

`SetAllocation` replaces the requested set; `Allocate` and `Deallocate` update it. Requests
are validated before mutation. New and reused slots are filled with their default across
all cells and registers. Retained pairs preserve slot numbers and values through growth.
The pool grows geometrically, capped at the total number of possible pairs. Freed capacity
is retained for reuse. There is no automatic shrink or threshold-based removal.

Allocation operations fence outstanding device work. Each change increments an allocation
generation. **Acquire a new DeviceView after every change.** `IsCurrent(view)` checks a
snapshot on the host; checking generation inside every device access is intentionally
avoided. Old Kokkos views may retain memory, so discard stale snapshots promptly. This is
a single host owner API, not a thread-safe concurrent allocator. Kernel indices/ordinals
must be valid, as with ordinary Kokkos views.

Passing `force_dense=true` allocates all pairs and disables removal. This keeps the same
access path for debugging and measuring sparse-interface overhead.

`Memory()` accounts for occupied payload, reserved pool payload, equivalent dense payload,
and persistent device metadata. It excludes host metadata, temporary allocation/copy
buffers, stale snapshots, and allocator overhead. Metadata includes an O(blocks*fields)
slot table. Neither memory accounting nor active counts imply a runtime speedup.

## GPU-resident allocation sessions

The synchronous host API remains useful for setup and occasional changes. Frequent
changes can instead use a fixed-budget device session:

```cpp
vars.BeginDeviceAllocation(slot_budget);  // reserve; synchronize once
const auto pool = vars.GetDevicePoolView();
// Allocate requests(nblocks, fields.Size()) and initialize it before the loop.
// In each epoch, a device producer marks every desired pair with that epoch.
vars.ApplyDeviceMask(requests, epoch);
// Consumers iterate 0 .. pool.capacity-1 and skip !pool.IsAllocated(slot).
const auto stats = vars.EndDeviceAllocation();  // synchronize; restore host queries
```

`ApplyDeviceMask` enqueues two kernels on the default execution-space stream. The first
releases unwanted pairs and queues new pairs; the second consumes a device free stack
and fills all new slot cells/registers with their defaults. Neither operation allocates
memory, copies metadata to/from the host, or fences. Existing pairs retain their slots
and values. The second kernel uses one team with vectorized cell initialization; large
simultaneous activation bursts may need a more scalable initializer later.

Every desired pair must have `requests(block,ordinal)==epoch`. Use positive, distinct
epochs, or clear the mask before reusing an epoch. Publish marks atomically if multiple
producers can mark the same pair. Requests, producers, allocation, and consumers must
use the same default ordered stream. Do not mutate the mask concurrently, use cached
slots after their pair is removed, or read/write removed slots. Keep request arrays alive
until `EndDeviceAllocation`. Deallocate only at a module-defined safe lifetime boundary.

The budget is bounded by the number of possible pairs and must cover the peak live set.
Previously reserved capacity is retained, so `pool.capacity` may exceed the requested
budget. Exceeding actual capacity **aborts explicitly on the device** rather than dropping
fields. There is no implicit asynchronous growth or retry. Choose a conservative budget
or end the session and reserve more at a known synchronization point. Transitions to an
empty set and zero-block stores are supported. Forced dense sessions preserve all pairs.

Begin and End invalidate ordinary snapshots. Within a session, `pool` pointers stay
valid across transitions; `pool.fields.active_count` is -1 and its compact work list must
not be used. Iterate capacity and test `IsAllocated`. Host count/map/memory queries and
host allocation operations throw during a session. End copies the map/counters once and
restores the ordered compact work list. For small CUDA kernels, keep captures small and consider Kokkos's `HintLightWeight`
policy property: the pinned Kokkos otherwise stages sufficiently large functors in
constant memory and synchronizes reuse on each launch. The device advection example
uses this hint and captures its field view only once.

Device slot assignment is not deterministic;
logical `(block, field)` membership and values are the reproducibility contract. Persistent
metadata also includes a free stack, a possible-pair-sized request queue, and counters.

## Build and verify

```sh
git submodule update --init kokkos
cmake -S . -B build-sparse -DAthena_BUILD_SPARSE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sparse --target sparse_storage_test sparse_advection_test \
  sparse_benchmark -j4
ctest --test-dir build-sparse --output-on-failure
```

Configure Release separately for performance. For CUDA, use the unmodified pinned
`kokkos/bin/nvcc_wrapper` with a supported host compiler, `Kokkos_ENABLE_CUDA=ON`, and the
appropriate architecture: `AMPERE80` (Apollo A100), `HOPPER90` (Vista GH200), or
`BLACKWELL100` (Horizon GB200). Example:

```sh
cmake -S . -B build-sparse-cuda -DAthena_BUILD_SPARSE_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON
cmake --build build-sparse-cuda --target sparse_storage_test sparse_advection_test \
  sparse_benchmark -j4
# Inside a single-GPU allocation:
bash tst/sparse/run_gpu.sh build-sparse-cuda results-sparse
python3 tst/sparse/summarize.py results-sparse --csv results-sparse/summary.csv
```

The site job files under `tst/sparse/jobs` are starting points using the user's current
partitions/account. Check local policy and compiler environment before submitting.

## What the executables test

* `sparse_storage_test`: absent/default reads, non-contiguous IDs, three registers,
  growth preserving values, GPU session transitions/default initialization, slot reuse,
  explicit device capacity failure, invalid requests without partial commit,
  ordered work lists, zero-block stores, forced dense mode, and memory accounting.
* `sparse_benchmark`: controlled occupancy with a manufactured decay update and analytic
  verification. Runs ordinary dense, forced dense through the sparse interface, and
  genuinely sparse storage. `--cells`, `--ny`, and `--nz` set the block's storage shape.
  Results are JSONL, including all timing samples. Five warm-up kernels precede timed
  trials. Each trial resets data, fences before timing, runs the requested number of
  launches, then fences. Allocation time is separate. Actual occupancy is quantized to
  an integer number of blocks per field and is reported alongside requested occupancy.
* `sparse_advection_test`: a standalone 1D uniform periodic mesh of logical blocks,
  unrelated to the main AthenaK Mesh object. A prescribed constant velocity transports
  compact top-hat patches by donor-cell finite volumes. All five modes are checked
  against an independent host reference, per-field conservation, positivity, and (at
  Courant number one) the exact translated discrete profile. Both flow directions and
  periodic wraparound are tested. There is one ghost cell per end and two state
  registers. An outgoing nonzero boundary value activates its receiver before the update.
  Optional exact-zero pruning occurs only after the completed update, when the old
  register and old ghosts are dead. This is not a generic RK deallocation policy.

The advection test reports end-to-end evolution time. `--mode sparse` retains the original
host allocation path (device/host flags, scans, metadata publication, fences, ghosts,
pruning). `--mode device` uses GPU allocation and combines direct periodic neighbor reads,
transport, liveness, and next-step halo requests in one kernel. Only mask epochs change;
there is no per-step host copy or global mask clear. Its interval includes Begin/End and
all allocation kernels. `--mode dense_fused` combines neighbor reads and transport too,
providing a control for fusion independent of sparse storage. `dense` and `forced` retain
the original separate ghost/transport kernels. The default `--mode all` checks all five.
The reported `allocation_seconds` measures synchronous host allocation calls only in the
legacy path; zero in device mode does not mean its allocation kernels are free. Use a
GPU profiler for that attribution.

Initial field setup, the independent reference, and final verification are excluded from
all timings. The device advection budget follows a conservative bound for this particular
patch transport problem; it is not an automatic general-purpose occupancy predictor.
These are direct local kernels on uniform logical blocks, with no MPI or dependency on
AthenaK boundary classes. MPI, AMR, restart, arbitrary topology, vector fields, and threshold pruning remain
future work. Limited physics integrations are described below.

## Next steps

Use these tests and the physics pilots below to choose capacity and metadata policies.
Then add sparse MPI manifests/payloads,
AMR transfer and flux correction, restart masks, and module-specific lifetime rules.
Keep exact-zero omission separate from approximate positive thresholds and validate the
conservation and physics error of the latter explicitly. RT can later use global ray IDs;
the generic store does not prescribe angular quadrature or decide which rays to retain.

## Physics pilots: passive tracers, then SC beam

Two opt-in integrations now exercise the store with real AthenaK physics. These are
single-rank, single-pack uniform-grid pilots, not general sparse MPI/AMR support.
The generic allocation library remains independent of either module.

### Passive tracers

Run `athena -i tst/inputs/sparse_tracers.athinput` for an in-process comparison with
AthenaK's existing appended passive scalars. Each field stores conserved `rho*C`;
the transport uses the actual hydro mass flux, DC/PLM mass-fraction reconstruction,
and SSP RK1/RK2/RK3 coefficients. `Hydro::Fluxes` advances the tracers after computing
gas fluxes and before updating gas. RK4, FOFC, nonperiodic boundaries, diffusion,
source terms, relativistic coordinates, and coupled modules are rejected explicitly.

At each stage, the GPU requests current nonzero pairs, live saved RK registers, and
face-neighbor receivers. The allocation mask is applied before any receiver update.
Each slot groups the current state, saved RK state, and one flux array per active
dimension. At the completed step only exact-zero current states may be dropped.
There is no threshold truncation: tiny advective tails can retain storage indefinitely.
The pool is a user-specified fixed capacity, and overflow fails rather than dropping data.

`passive_tracers/storage` selects these controls:

| Mode | Storage and transport |
| --- | --- |
| `native` | Original `hydro/nscalars` kernels and buffers; requires `nscalars=nfields` |
| `dense` | Ordinary dense array using the pilot's transport kernels |
| `forced` | Same pilot kernels through an all-allocated sparse slot map |
| `sparse` | GPU allocation requests, bounded slots, and exact-zero pruning |

For independent timing use `compare_dense=false` and `hydro/nscalars=0` outside native
mode. For correctness use `compare_dense=true` and `nscalars=nfields`; the problem
generator checks every conserved tracer cell against native scalars, each label's total
mass, finiteness, and positivity within precision-appropriate roundoff. `tracers.json`
reports these errors and reserved payload/metadata. Native payload attribution includes
eight existing scalar arrays; the 2D pilot uses four. Thus native-to-sparse savings
include both fewer buffers and spatial sparsity; compare `dense` to isolate the latter.
Common gas arrays, boundary buffers, allocator overhead and transient peaks are excluded.

### Non-emitting SC beam

Run `athena -i tst/inputs/sc_beam_sparse.athinput
nr_radiation/intensity_storage=sparse`. This is a fixed-support 2D `sc_beam` pilot with
user/vacuum boundaries, tiled sweeps, integral coupling, no fluid, no scattering, and
no local emission. It must not be used as an angular-selection policy for emitting LTE
gas. Original global ordinate IDs and quadrature weights are retained unchanged.

The host computes conservative reachability from injected boundary cells using both
actual short-characteristics interpolation donors. A block-direction pair is stored
if support touches its interior or required ghost footprint. This includes numerical
beam broadening, not just a geometric ray. The compact work list is reused for all
sweeps. Absent intensities read as zero, and a nonzero write outside support aborts.
Ghost exchange uses explicit local mesh neighbors; moments use the original weights.

`nr_radiation/intensity_storage` controls:

| Mode | Stored intensities |
| --- | --- |
| `dense` | Original full-angle storage, sweeps, moments and boundary exchange |
| `selected` | Dense injected directions on every block, using the pilot kernels |
| `forced` | All original directions through the sparse map, using pilot kernels |
| `sparse` | Only supported block-direction pairs |

The selected-direction dense control is essential: removing globally dark directions
can explain most of a full-angle speedup without spatial sparsity helping runtime.
`beam_diagnostics=true` writes `beam.json`; `dump_final=true` also writes every original
intensity and all ten moments for comparison. Reported memory is intensity storage
and its metadata, excluding common moments/material arrays, boundary buffers, and
transient allocations. Existing native SC configurations retain their original boundary
kernel and do not write pilot diagnostics by default.

### Validation, timing and movies

With sparse tests enabled, `ctest --test-dir BUILD --output-on-failure` covers generic
storage, transport, native tracer comparisons in 1D/2D/3D, and full-intensity/moment
beam comparisons for multiple tile/block shapes and injection faces. The existing
dense SC beam SMR regression also exercises the unchanged native paths. Neither pilot
supports restart; both read and write paths fail explicitly.

On an allocated GPU, from the repository root:

```sh
bash tst/sparse/run_physics_gpu.sh BUILD FRESH_RESULTS
python3 tst/sparse/run_physics.py --exe BUILD/src/athena \
  --suite movies --output FRESH_FRAMES
MPLCONFIGDIR=SCRATCH/mpl-cache python3 tst/sparse/render_physics.py \
  --frames FRESH_FRAMES --output VIDEOS
```

The timing runner discards one warm-up per mode and randomizes five measured trials.
`time/profile_evolution=true` fences the complete evolution task loop; setup and final
verification/output are outside this measurement. Per-stage fenced tracer profiling
and sanitizer runs are separate. Movies enable additional synchronized frame I/O and
must not be used as timing evidence. Tracer movies show physical time; the quasi-static
beam movie shows convergence iterations, not photon propagation in physical time.
Movie allocation maps count live pairs, while memory figures include the reserved pool
capacity. Default full-size capacity does not save reserved payload memory.
