// Allocation lifecycle tests, including growth, slot reuse, and nonzero defaults.
#include "common.hpp"

#include <functional>

namespace {
using sparse_test::Require;

void Throws(const std::function<void()> &operation) {
  bool caught = false;
  try { operation(); } catch (const std::exception &) { caught = true; }
  Require(caught, "Expected validation failure");
}

void CheckDefault(const sparse::Variables &vars, int reg, int block, int field,
                  Real expected) {
  const auto view = vars.GetDeviceView();
  const Real invalid = std::numeric_limits<Real>::max();
  Real error = 0;
  Kokkos::parallel_reduce("check_default", sparse_test::Range(0, 24),
      KOKKOS_LAMBDA(std::int64_t c, Real &err) {
    const Real value = view.Read(reg, block, field, c/12, (c/4)%3, c%4);
    const Real difference = fabs(value-expected);
    if (!Kokkos::isfinite(value)) err = invalid;
    else if (difference > err) err = difference;
  }, Kokkos::Max<Real>(error));
  Require(error == 0, "Default or retained register value is wrong");
}

void Run() {
  using sparse::Field;
  Throws([]() { sparse::Registry registry({}); });
  Throws([]() { sparse::Registry registry({{1,"a"}, {1,"b"}}); });
  Throws([]() { sparse::Registry registry({{1,"a"}, {2,"a"}}); });
  Throws([]() { sparse::Registry registry({{1,""}}); });
  Throws([]() { sparse::Registry registry(
      {{1,"a",std::numeric_limits<Real>::quiet_NaN()}}); });
  sparse::Registry registry({{-17,"negative",2.5}, {42,"zero",0}, {9,"third",-3}});
  Require(registry.Index(42) == 1 && registry.Index(-17) == 0,
          "IDs confused with ordinals");
  Throws([&]() { registry.Index(0); });
  Throws([&]() { sparse::Variables bad("bad", registry, -1, 2, 3, 4); });
  Throws([&]() { sparse::Variables bad("bad", registry, 2, 0, 3, 4); });
  sparse::Variables empty_rank("empty_rank", registry, 0, 2, 3, 4, 2);
  Require(empty_rank.ActiveCount() == 0 && empty_rank.Memory().reserved_payload == 0,
          "Empty rank has payload");
  sparse::Variables other_empty("other_empty", registry, 0, 2, 3, 4, 2);
  Require(!empty_rank.IsCurrent(other_empty.GetDeviceView()), "Foreign empty view accepted");

  sparse::Variables vars("lifecycle", registry, 4, 2, 3, 4, 3);
  Require(vars.Capacity() == 0 && vars.ActiveCount() == 0, "Initially allocated");
  CheckDefault(vars, 0, 3, 0, 2.5);
  CheckDefault(vars, 2, 1, 2, -3);
  auto snapshot = vars.GetDeviceView();
  vars.SetAllocation({{2,42}, {0,-17}, {2,42}});  // duplicates are idempotent
  Require(!vars.IsCurrent(snapshot), "Allocation did not invalidate snapshot");
  Require(vars.ActiveCount() == 2 && vars.Capacity() == 2, "Wrong active count");
  CheckDefault(vars, 0, 0, 0, 2.5);
  CheckDefault(vars, 1, 0, 0, 2.5);
  CheckDefault(vars, 2, 0, 0, 2.5);
  snapshot = vars.GetDeviceView();
  auto view = vars.GetDeviceView();
  const int keep = vars.Slot(0,-17);
  Kokkos::parallel_for("seed_registers", sparse_test::Range(0, 72),
      KOKKOS_LAMBDA(std::int64_t idx) {
    const int reg = idx/24, cell = idx%24;
    view.At(reg, keep, cell/12, (cell/4)%3, cell%4) = 10+reg;
  });
  // No explicit fence: allocation must protect outstanding work and preserve values.
  vars.Allocate({{1,9}, {3,42}, {2,-17}, {0,9}, {3,9}});
  Require(vars.Capacity() >= 7 && vars.ActiveCount() == 7, "Growth failed");
  Require(!vars.IsCurrent(snapshot), "Growth did not invalidate snapshot");
  Require(vars.Slot(0,-17) == keep, "Growth moved retained slot");
  for (int r = 0; r < 3; ++r) CheckDefault(vars, r, 0, 0, 10+r);
  for (int r = 0; r < 3; ++r) CheckDefault(vars, r, 1, 2, -3);

  const int released = vars.Slot(2,42);
  vars.Deallocate({{2,42}});
  CheckDefault(vars, 1, 2, 1, 0);
  snapshot = vars.GetDeviceView();
  vars.Allocate({{1,-17}});
  Require(vars.Slot(1,-17) == released, "Freed slot was not reused");
  Require(!vars.IsCurrent(snapshot), "Reuse did not invalidate snapshot");
  for (int r = 0; r < 3; ++r) CheckDefault(vars, r, 1, 0, 2.5);

  snapshot = vars.GetDeviceView();
  vars.Allocate({{1,-17}});
  Require(vars.IsCurrent(snapshot), "No-op invalidated a snapshot");
  Throws([&]() { vars.Allocate({{0,42}, {4,42}}); });
  Throws([&]() { vars.SetAllocation({{0,42}, {0,123456}}); });
  Require(vars.IsCurrent(snapshot) && vars.Slot(0,42) < 0,
          "Invalid request partially committed");
  auto order = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                 snapshot.active_slots);
  auto blocks = Kokkos::create_mirror_view_and_copy(HostMemSpace(), snapshot.block_of);
  auto fields = Kokkos::create_mirror_view_and_copy(HostMemSpace(), snapshot.field_of);
  int last = -1;
  for (int w = 0; w < snapshot.active_count; ++w) {
    const int slot = order(w), key = 3*blocks(slot)+fields(slot);
    Require(key > last, "Work list is not compact and ordered");
    last = key;
  }
  const auto before = vars.Memory();
  Require(before.occupied_payload == 7*3*24*sizeof(Real), "Wrong memory accounting");
  vars.SetAllocation({});
  Require(vars.ActiveCount() == 0 && vars.Memory().occupied_payload == 0 &&
          vars.Memory().reserved_payload == before.reserved_payload,
          "Freed storage accounting is incorrect");
  vars.Allocate({{3,9}});
  for (int r = 0; r < 3; ++r) CheckDefault(vars, r, 3, 2, -3);

  sparse::Variables dense("forced", registry, 4, 2, 3, 4, 2, true);
  Require(dense.ActiveCount() == 12, "Forced dense failed");
  auto dense_snapshot = dense.GetDeviceView();
  dense.SetAllocation({});
  dense.Deallocate({{1,42}});
  Require(dense.IsCurrent(dense_snapshot) && dense.ActiveCount() == 12,
          "Forced dense deallocated a field");
  Require(!dense.IsCurrent(vars.GetDeviceView()), "Foreign snapshot accepted");
  CheckDefault(dense, 1, 2, 0, 2.5);
  CheckDefault(dense, 1, 2, 2, -3);
  std::cout << "{\"type\":\"storage\",\"passed\":true}\n";
}
}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  try { sparse_test::PrintEnvironment(); Run(); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; result = 1; }
  Kokkos::finalize();
  return result;
}
