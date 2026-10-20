#ifndef SPARSE_SPARSE_VARIABLES_HPP_
#define SPARSE_SPARSE_VARIABLES_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sparse_variables.hpp
//! \brief Host-managed, block-sparse cell fields with a compact device work list.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "athena.hpp"

namespace sparse {

struct Field {
  int id;
  std::string name;
  Real default_value = 0.0;
};

//! Global field IDs need not be contiguous. Device indexing uses registry ordinals.
class Registry {
 public:
  explicit Registry(std::vector<Field> fields);
  int Index(int id) const;
  int Size() const { return static_cast<int>(fields_.size()); }
  const Field &At(int index) const { return fields_.at(index); }

 private:
  std::vector<Field> fields_;
  std::unordered_map<int, int> index_;
};

struct BlockField {
  int block;
  int field_id;
};

//! A snapshot for capture by a Kokkos kernel. Acquire again after any allocation change.
//! Reads of absent fields return their default; writes require an allocated slot.
struct DeviceView {
  DvceArray5D<Real> data;                // register, slot, k, j, i
  DvceArray2D<const int> slot_of;        // block, registry ordinal
  DvceArray1D<const int> active_slots;   // sorted by block then registry ordinal
  DvceArray1D<const int> block_of;
  DvceArray1D<const int> field_of;
  DvceArray1D<const int> field_ids;
  DvceArray1D<const Real> defaults;
  std::uintptr_t owner_token;  // host identity only; never dereferenced on device
  std::uint64_t generation;
  int active_count;
  int nregisters;
  int nblocks;
  int nfields;

  KOKKOS_INLINE_FUNCTION
  int Slot(int block, int field) const { return slot_of(block, field); }
  KOKKOS_INLINE_FUNCTION
  bool Contains(int block, int field) const { return Slot(block, field) >= 0; }
  KOKKOS_INLINE_FUNCTION
  int ActiveSlot(int work) const { return active_slots(work); }
  KOKKOS_INLINE_FUNCTION
  int Block(int slot) const { return block_of(slot); }
  KOKKOS_INLINE_FUNCTION
  int FieldIndex(int slot) const { return field_of(slot); }
  KOKKOS_INLINE_FUNCTION
  int FieldID(int field) const { return field_ids(field); }
  KOKKOS_INLINE_FUNCTION
  Real Read(int reg, int block, int field, int k, int j, int i) const {
    const int slot = Slot(block, field);
    return slot < 0 ? defaults(field) : data(reg, slot, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  Real &At(int reg, int slot, int k, int j, int i) const {
    return data(reg, slot, k, j, i);
  }
};

struct MemoryUsage {
  std::size_t occupied_payload;
  std::size_t reserved_payload;
  std::size_t dense_payload;
  std::size_t device_metadata;
};

//! One slot owns all registers (including ghosts) for one block/field pair.
//! n1/n2/n3 are storage extents; the owner defines interior/ghost index ranges.
//! Mutations are synchronous host operations that fence outstanding device work.
//! A snapshot is valid only until the next mutation; IsCurrent checks its generation.
//! Pool capacity is retained after freeing slots. No cached snapshot may be used after
//! mutation, even if its Kokkos Views keep the old allocation alive.
class Variables {
 public:
  Variables(std::string label, Registry registry, int nblocks,
            int n3, int n2, int n1, int nregisters = 1, bool force_dense = false);
  Variables(const Variables &) = delete;
  Variables &operator=(const Variables &) = delete;

  //! Replace the allocated set. Validate every request before mutating any state.
  //! Retained pairs preserve their slots and all register values. New/reused slots
  //! are initialized to the registered default over every cell and register.
  void SetAllocation(const std::vector<BlockField> &pairs);
  void Allocate(const std::vector<BlockField> &pairs);
  void Deallocate(const std::vector<BlockField> &pairs);
  DeviceView GetDeviceView() const;
  bool IsCurrent(const DeviceView &view) const;
  int Slot(int block, int field_id) const;
  int ActiveCount() const { return active_count_; }
  int Capacity() const { return capacity_; }
  const Registry &Fields() const { return registry_; }
  MemoryUsage Memory() const;

 private:
  std::size_t Key(const BlockField &pair) const;
  void ApplyMask(const std::vector<unsigned char> &wanted);
  std::string label_;
  Registry registry_;
  int nblocks_, n3_, n2_, n1_, nregisters_;
  bool force_dense_;
  int capacity_ = 0, active_count_ = 0;
  std::uint64_t generation_ = 0;
  std::vector<int> slots_;
  DvceArray5D<Real> data_;
  DvceArray2D<int> slot_of_;
  DvceArray1D<int> active_slots_, block_of_, field_of_, field_ids_;
  DvceArray1D<Real> defaults_;
};

}  // namespace sparse
#endif  // SPARSE_SPARSE_VARIABLES_HPP_
