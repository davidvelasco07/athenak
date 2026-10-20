//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sparse_variables.cpp
//! \brief Slot lifetimes, allocation generations, and device metadata publication.

#include "sparse/sparse_variables.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace sparse {
namespace {
// CUDA extended lambdas must be enclosed by a free or public function. Keep the
// allocation mutation private and put only its initialization kernel here.
void InitializeSlots(DvceArray5D<Real> data, DvceArray1D<int> fresh,
                     DvceArray1D<int> field_of, DvceArray1D<Real> defaults,
                     int nregisters, int n3, int n2, int n1) {
  const std::int64_t cells = static_cast<std::int64_t>(n1)*n2*n3;
  const std::int64_t per_slot = cells*nregisters;
  using Policy = Kokkos::RangePolicy<DevExeSpace, Kokkos::IndexType<std::int64_t>>;
  Kokkos::parallel_for("sparse_initialize_slots", Policy(0,per_slot*fresh.size()),
      KOKKOS_LAMBDA(std::int64_t idx) {
    const int slot = fresh(idx/per_slot);
    const int reg = (idx%per_slot)/cells;
    const int cell = idx%cells;
    data(reg,slot,cell/(n2*n1),(cell/n1)%n2,cell%n1) = defaults(field_of(slot));
  });
}
}  // namespace

Registry::Registry(std::vector<Field> fields) : fields_(std::move(fields)) {
  if (fields_.empty() || fields_.size() > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Sparse registry must have 1..INT_MAX fields");
  }
  std::unordered_set<std::string> names;
  for (int f = 0; f < Size(); ++f) {
    const auto &field = fields_[f];
    if (field.name.empty() || !std::isfinite(field.default_value) ||
        !index_.emplace(field.id, f).second || !names.insert(field.name).second) {
      throw std::invalid_argument("Sparse fields require unique IDs/names and finite "
                                  "defaults");
    }
  }
}

int Registry::Index(int id) const {
  const auto it = index_.find(id);
  if (it == index_.end()) throw std::out_of_range("Unknown sparse field ID");
  return it->second;
}

Variables::Variables(std::string label, Registry registry, int nblocks,
                     int n3, int n2, int n1, int nregisters, bool force_dense)
    : label_(std::move(label)), registry_(std::move(registry)), nblocks_(nblocks),
      n3_(n3), n2_(n2), n1_(n1), nregisters_(nregisters), force_dense_(force_dense) {
  const auto max_int = std::numeric_limits<int>::max();
  if (nblocks < 0 || n1 <= 0 || n2 <= 0 || n3 <= 0 || nregisters <= 0 ||
      static_cast<std::int64_t>(nblocks) * registry_.Size() > max_int ||
      static_cast<std::int64_t>(n1) * n2 > max_int ||
      static_cast<std::int64_t>(n1) * n2 * n3 > max_int) {
    throw std::invalid_argument("Invalid sparse storage extents");
  }
  const int nf = registry_.Size();
  slots_.assign(static_cast<std::size_t>(nblocks) * nf, -1);
  slot_of_ = DvceArray2D<int>(label_ + ".slot_of", nblocks, nf);
  field_ids_ = DvceArray1D<int>(label_ + ".field_ids", nf);
  defaults_ = DvceArray1D<Real>(label_ + ".defaults", nf);
  auto ids = Kokkos::create_mirror_view(field_ids_);
  auto defaults = Kokkos::create_mirror_view(defaults_);
  for (int f = 0; f < nf; ++f) {
    ids(f) = registry_.At(f).id;
    defaults(f) = registry_.At(f).default_value;
  }
  Kokkos::deep_copy(field_ids_, ids);
  Kokkos::deep_copy(defaults_, defaults);
  Kokkos::deep_copy(slot_of_, -1);
  if (force_dense_) ApplyMask(std::vector<unsigned char>(slots_.size(), 1));
}

std::size_t Variables::Key(const BlockField &pair) const {
  if (pair.block < 0 || pair.block >= nblocks_) {
    throw std::out_of_range("Invalid sparse block index");
  }
  return static_cast<std::size_t>(pair.block) * registry_.Size()
         + registry_.Index(pair.field_id);
}

int Variables::Slot(int block, int field_id) const {
  return slots_[Key({block, field_id})];
}

void Variables::SetAllocation(const std::vector<BlockField> &pairs) {
  std::vector<unsigned char> mask(slots_.size(), force_dense_ ? 1 : 0);
  for (const auto &pair : pairs) mask[Key(pair)] = 1;
  ApplyMask(mask);
}

void Variables::Allocate(const std::vector<BlockField> &pairs) {
  if (pairs.empty()) return;
  std::vector<unsigned char> mask(slots_.size());
  for (std::size_t key = 0; key < slots_.size(); ++key) mask[key] = slots_[key] >= 0;
  for (const auto &pair : pairs) mask[Key(pair)] = 1;
  ApplyMask(mask);
}

void Variables::Deallocate(const std::vector<BlockField> &pairs) {
  if (pairs.empty()) return;
  std::vector<unsigned char> mask(slots_.size());
  for (std::size_t key = 0; key < slots_.size(); ++key) mask[key] = slots_[key] >= 0;
  for (const auto &pair : pairs) mask[Key(pair)] = force_dense_ ? 1 : 0;
  ApplyMask(mask);
}

void Variables::ApplyMask(const std::vector<unsigned char> &wanted) {
  int count = 0;
  bool changed = false;
  for (std::size_t key = 0; key < slots_.size(); ++key) {
    count += wanted[key] != 0;
    changed = changed || ((slots_[key] >= 0) != (wanted[key] != 0));
  }
  if (!changed) return;
  // A slot can be reused, so even mutations without a realloc must wait for kernels.
  Kokkos::fence();
  if (count > capacity_) {
    const auto max_pairs = static_cast<std::int64_t>(slots_.size());
    capacity_ = static_cast<int>(std::min(max_pairs,
        std::max<std::int64_t>(count, 2LL * capacity_)));
    Kokkos::resize(data_, nregisters_, capacity_, n3_, n2_, n1_);
    Kokkos::realloc(active_slots_, capacity_);
    Kokkos::realloc(block_of_, capacity_);
    Kokkos::realloc(field_of_, capacity_);
  }
  std::vector<unsigned char> used(capacity_, 0);
  for (std::size_t key = 0; key < slots_.size(); ++key) {
    if (!wanted[key]) slots_[key] = -1;
    if (slots_[key] >= 0) used[slots_[key]] = 1;
  }
  std::vector<int> new_slots;
  int next = 0;
  for (std::size_t key = 0; key < slots_.size(); ++key) {
    if (wanted[key] && slots_[key] < 0) {
      while (used[next]) ++next;
      slots_[key] = next;
      used[next] = 1;
      new_slots.push_back(next);
    }
  }
  auto map = Kokkos::create_mirror_view(slot_of_);
  auto active = Kokkos::create_mirror_view(active_slots_);
  auto blocks = Kokkos::create_mirror_view(block_of_);
  auto fields = Kokkos::create_mirror_view(field_of_);
  for (int s = 0; s < capacity_; ++s) {
    active(s) = blocks(s) = fields(s) = -1;
  }
  int work = 0;
  const int nf = registry_.Size();
  for (int b = 0; b < nblocks_; ++b) {
    for (int f = 0; f < nf; ++f) {
      const int slot = slots_[static_cast<std::size_t>(b) * nf + f];
      map(b, f) = slot;
      if (slot >= 0) {
        active(work++) = slot;
        blocks(slot) = b;
        fields(slot) = f;
      }
    }
  }
  Kokkos::deep_copy(slot_of_, map);
  Kokkos::deep_copy(active_slots_, active);
  Kokkos::deep_copy(block_of_, blocks);
  Kokkos::deep_copy(field_of_, fields);
  if (!new_slots.empty()) {
    DvceArray1D<int> fresh(label_ + ".fresh", new_slots.size());
    auto fresh_h = Kokkos::create_mirror_view(fresh);
    for (std::size_t s = 0; s < new_slots.size(); ++s) fresh_h(s) = new_slots[s];
    Kokkos::deep_copy(fresh, fresh_h);
    InitializeSlots(data_,fresh,field_of_,defaults_,nregisters_,n3_,n2_,n1_);
  }
  Kokkos::fence();
  active_count_ = count;
  ++generation_;
}

DeviceView Variables::GetDeviceView() const {
  return {data_, slot_of_, active_slots_, block_of_, field_of_, field_ids_, defaults_,
          reinterpret_cast<std::uintptr_t>(this), generation_, active_count_,
          nregisters_, nblocks_, registry_.Size()};
}

bool Variables::IsCurrent(const DeviceView &view) const {
  return view.generation == generation_ &&
         view.owner_token == reinterpret_cast<std::uintptr_t>(this);
}

MemoryUsage Variables::Memory() const {
  const std::size_t per_slot = static_cast<std::size_t>(nregisters_) * n3_ * n2_ * n1_
                             * sizeof(Real);
  const auto metadata = (slot_of_.size() + active_slots_.size() + block_of_.size()
                         + field_of_.size() + field_ids_.size()) * sizeof(int)
                        + defaults_.size() * sizeof(Real);
  return {per_slot * active_count_, per_slot * capacity_, per_slot * slots_.size(),
          metadata};
}

}  // namespace sparse
