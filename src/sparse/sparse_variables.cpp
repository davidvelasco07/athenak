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
enum { Active, Free, Pending, Added, Removed, Peak, StateSize };

void UpdateDeviceMask(DvceArray2D<const int> requests, int epoch, bool forced,
                      DvceArray2D<int> map, DvceArray1D<int> blocks,
                      DvceArray1D<int> fields, DvceArray1D<int> free_slots,
                      DvceArray1D<int> pending, DvceArray1D<std::int64_t> state,
                      DvceArray5D<Real> data, DvceArray1D<Real> defaults,
                      int nregisters, int n3, int n2, int n1) {
  const int nf = map.extent_int(1);
  Kokkos::parallel_for("sparse_device_requests",
      Kokkos::RangePolicy<DevExeSpace>(0,map.size()), KOKKOS_LAMBDA(int key) {
    const int b = key/nf, f = key%nf, slot = map(b,f);
    const bool wanted = forced || requests(b,f) == epoch;
    if (slot >= 0 && !wanted) {
      map(b,f) = -1;
      blocks(slot) = fields(slot) = -1;
      const int index = Kokkos::atomic_fetch_add(&state(Free),std::int64_t(1));
      free_slots(index) = slot;
      Kokkos::atomic_dec(&state(Active));
      Kokkos::atomic_inc(&state(Removed));
    } else if (slot < 0 && wanted) {
      const int index = Kokkos::atomic_fetch_add(&state(Pending),std::int64_t(1));
      pending(index) = key;
    }
  });
  // One team consumes the queue so its final barrier can reset the counter without
  // another kernel or host synchronization. Vector lanes initialize contiguous cells.
  using Team = Kokkos::TeamPolicy<DevExeSpace>;
  const int cells = n1*n2*n3;
  int vector_length = 1;
#ifdef KOKKOS_ENABLE_CUDA
  vector_length = 32;
#endif
  Kokkos::parallel_for("sparse_device_allocate",Team(1,Kokkos::AUTO,vector_length),
      KOKKOS_LAMBDA(const Team::member_type &team) {
    const int count = state(Pending);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count),[&](int index) {
      int slot = -1;
      Kokkos::single(Kokkos::PerThread(team),[&](int &result) {
        const int free_index = Kokkos::atomic_fetch_sub(&state(Free),std::int64_t(1))-1;
        if (free_index < 0) Kokkos::abort("Sparse device allocation capacity exceeded");
        result = free_slots(free_index);
        const int key = pending(index), b = key/nf, f = key%nf;
        map(b,f) = result;
        blocks(result) = b;
        fields(result) = f;
        Kokkos::atomic_inc(&state(Active));
        Kokkos::atomic_inc(&state(Added));
      },slot);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,
          static_cast<std::int64_t>(nregisters)*cells),[&](std::int64_t c) {
        const int cell = c%cells;
        data(c/cells,slot,cell/(n2*n1),(cell/n1)%n2,cell%n1) = defaults(fields(slot));
      });
    });
    team.team_barrier();
    Kokkos::single(Kokkos::PerTeam(team),[&]() {
      state(Peak) = state(Active) > state(Peak) ? state(Active) : state(Peak);
      state(Pending) = 0;
    });
  });
}

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
  RequireHost();
  return slots_[Key({block, field_id})];
}

void Variables::SetAllocation(const std::vector<BlockField> &pairs) {
  RequireHost();
  std::vector<unsigned char> mask(slots_.size(), force_dense_ ? 1 : 0);
  for (const auto &pair : pairs) mask[Key(pair)] = 1;
  ApplyMask(mask);
}

void Variables::Allocate(const std::vector<BlockField> &pairs) {
  RequireHost();
  if (pairs.empty()) return;
  std::vector<unsigned char> mask(slots_.size());
  for (std::size_t key = 0; key < slots_.size(); ++key) mask[key] = slots_[key] >= 0;
  for (const auto &pair : pairs) mask[Key(pair)] = 1;
  ApplyMask(mask);
}

void Variables::Deallocate(const std::vector<BlockField> &pairs) {
  RequireHost();
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

DeviceView Variables::View() const {
  return {data_, slot_of_, active_slots_, block_of_, field_of_, field_ids_, defaults_,
          reinterpret_cast<std::uintptr_t>(this), generation_, active_count_,
          nregisters_, nblocks_, registry_.Size()};
}

void Variables::RequireHost() const {
  if (device_mode_) throw std::logic_error("End sparse device allocation session first");
}

int Variables::ActiveCount() const { RequireHost(); return active_count_; }

DeviceView Variables::GetDeviceView() const { RequireHost(); return View(); }

void Variables::BeginDeviceAllocation(int capacity) {
  RequireHost();
  if (capacity < active_count_ || capacity < 0 ||
      static_cast<std::size_t>(capacity) > slots_.size()) {
    throw std::invalid_argument("Invalid sparse device capacity");
  }
  Kokkos::fence();
  if (capacity > capacity_) {
    Kokkos::resize(data_,nregisters_,capacity,n3_,n2_,n1_);
    Kokkos::resize(active_slots_,capacity);
    Kokkos::resize(block_of_,capacity);
    Kokkos::resize(field_of_,capacity);
    capacity_ = capacity;
  }
  Kokkos::resize(free_slots_,capacity_);
  Kokkos::resize(pending_,slots_.size());
  Kokkos::resize(state_,StateSize);
  auto blocks = Kokkos::create_mirror_view(block_of_);
  auto fields = Kokkos::create_mirror_view(field_of_);
  auto free = Kokkos::create_mirror_view(free_slots_);
  auto state = Kokkos::create_mirror_view(state_);
  for (int s = 0; s < capacity_; ++s) blocks(s) = fields(s) = -1;
  const int nf = registry_.Size();
  for (std::size_t key = 0; key < slots_.size(); ++key) {
    const int slot = slots_[key];
    if (slot >= 0) { blocks(slot) = key/nf; fields(slot) = key%nf; }
  }
  int available = 0;
  for (int s = 0; s < capacity_; ++s) if (blocks(s) < 0) free(available++) = s;
  for (int i = 0; i < StateSize; ++i) state(i) = 0;
  state(Active) = state(Peak) = active_count_;
  state(Free) = available;
  Kokkos::deep_copy(block_of_,blocks);
  Kokkos::deep_copy(field_of_,fields);
  Kokkos::deep_copy(free_slots_,free);
  Kokkos::deep_copy(state_,state);
  Kokkos::fence();
  device_mode_ = true;
  ++generation_;
}

DevicePoolView Variables::GetDevicePoolView() const {
  if (!device_mode_) throw std::logic_error("Begin sparse device allocation first");
  auto view = View();
  view.active_count = -1;  // Never expose a stale host count as a device work bound.
  return {view,capacity_};
}

void Variables::ApplyDeviceMask(DvceArray2D<const int> requests, int epoch) {
  if (!device_mode_) throw std::logic_error("Begin sparse device allocation first");
  if (requests.extent_int(0) != nblocks_ ||
      requests.extent_int(1) != registry_.Size() || epoch <= 0) {
    throw std::invalid_argument("Invalid sparse device request mask/epoch");
  }
  UpdateDeviceMask(requests,epoch,force_dense_,slot_of_,block_of_,field_of_,
                   free_slots_,pending_,state_,data_,defaults_,nregisters_,n3_,n2_,n1_);
}

AllocationStats Variables::EndDeviceAllocation() {
  if (!device_mode_) throw std::logic_error("Begin sparse device allocation first");
  Kokkos::fence();
  auto map = Kokkos::create_mirror_view_and_copy(HostMemSpace(),slot_of_);
  auto state = Kokkos::create_mirror_view_and_copy(HostMemSpace(),state_);
  auto active = Kokkos::create_mirror_view(active_slots_);
  int count = 0;
  for (int b = 0; b < nblocks_; ++b) {
    for (int f = 0; f < registry_.Size(); ++f) {
      const int slot = map(b,f);
      slots_[static_cast<std::size_t>(b)*registry_.Size()+f] = slot;
      if (slot >= 0) active(count++) = slot;
    }
  }
  for (int s = count; s < capacity_; ++s) active(s) = -1;
  Kokkos::deep_copy(active_slots_,active);
  Kokkos::fence();
  active_count_ = count;
  device_mode_ = false;
  ++generation_;
  return {state(Added),state(Removed),static_cast<int>(state(Peak))};
}

bool Variables::IsCurrent(const DeviceView &view) const {
  return !device_mode_ && view.generation == generation_ &&
         view.owner_token == reinterpret_cast<std::uintptr_t>(this);
}

MemoryUsage Variables::Memory() const {
  RequireHost();
  const std::size_t per_slot = static_cast<std::size_t>(nregisters_) * n3_ * n2_ * n1_
                             * sizeof(Real);
  const auto metadata = (slot_of_.size() + active_slots_.size() + block_of_.size()
                         + field_of_.size() + field_ids_.size() + free_slots_.size()
                         + pending_.size()) * sizeof(int)
                        + state_.size() * sizeof(std::int64_t)
                        + defaults_.size() * sizeof(Real);
  return {per_slot * active_count_, per_slot * capacity_, per_slot * slots_.size(),
          metadata};
}

}  // namespace sparse
