# Sparse cell fields: initial implementation

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
  growth preserving values, slot reuse, invalid requests without partial commit,
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
  compact top-hat patches by donor-cell finite volumes. All three modes are checked
  against an independent host reference, per-field conservation, positivity, and (at
  Courant number one) the exact translated discrete profile. Both flow directions and
  periodic wraparound are tested. There is one ghost cell per end and two state
  registers. An outgoing nonzero boundary value activates its receiver before the update.
  Optional exact-zero pruning occurs only after the completed update, when the old
  register and old ghosts are dead. This is not a generic RK deallocation policy.

The advection test reports end-to-end evolution time including device/host activation
flags, host scans, metadata publication, allocation fences, ghost filling, and pruning.
Its host control path prioritizes correctness and is expected to be costly for small
GPU workloads. Initial setup, the independent reference, and final verification are
outside that timed interval. It uses direct local ghost kernels, not AthenaK's existing
boundary communication classes. It does not implement MPI, AMR, restart, arbitrary
topology, vector fields, threshold pruning, or integration with a physics module.

## Next steps

Use these tests to choose capacity and metadata policies before connecting the library
to MeshBlockPack and real mesh neighbor lists. Then add sparse MPI manifests/payloads,
AMR transfer and flux correction, restart masks, and module-specific lifetime rules.
Keep exact-zero omission separate from approximate positive thresholds and validate the
conservation and physics error of the latter explicitly. RT can later use global ray IDs;
the generic store does not prescribe angular quadrature or decide which rays to retain.
