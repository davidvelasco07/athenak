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
AthenaK boundary classes. MPI, AMR, restart, arbitrary topology, vector fields, threshold
pruning, and physics-module integration remain future work.

## Next steps

Use these tests to choose capacity and metadata policies before connecting the library
to MeshBlockPack and real mesh neighbor lists. Then add sparse MPI manifests/payloads,
AMR transfer and flux correction, restart masks, and module-specific lifetime rules.
Keep exact-zero omission separate from approximate positive thresholds and validate the
conservation and physics error of the latter explicitly. RT can later use global ray IDs;
the generic store does not prescribe angular quadrature or decide which rays to retain.
