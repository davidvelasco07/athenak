// Controlled occupancy benchmark: identical values and updates in all three modes.
#include "common.hpp"

#include <algorithm>
#include <memory>

namespace {
using sparse_test::Range;
using sparse_test::Require;

struct Options {
  int blocks, fields, nx, ny, nz, iterations, repeats;
  double fraction;
};

KOKKOS_INLINE_FUNCTION
Real Initial(int block, int field, int selected, int nblocks) {
  return ((block+17*field)%nblocks < selected)
      ? Real(1) + Real((block+3*field)%17)/Real(32) : Real(0);
}

void Run(const Options &o, const std::string &mode) {
  const int selected = static_cast<int>(std::round(o.fraction*o.blocks));
  const int nx = o.nx, ny = o.ny, nz = o.nz, nf = o.fields, nb = o.blocks;
  const std::int64_t cells = static_cast<std::int64_t>(nx)*ny*nz;
  const std::int64_t total = cells*nb*nf;
  const Real alpha = Real(1)-Real(1)/Real(1024);
  DvceArray5D<Real> dense;
  std::unique_ptr<sparse::Variables> sparse;
  sparse::DeviceView view{};
  Kokkos::fence();
  Kokkos::Timer allocation_timer;
  if (mode == "dense") {
    dense = DvceArray5D<Real>("benchmark_dense", nb, nf, nz, ny, nx);
  } else {
    sparse = std::make_unique<sparse::Variables>("benchmark_sparse",
        sparse_test::MakeRegistry(nf), nb, nz, ny, nx, 1, mode == "forced");
    std::vector<sparse::BlockField> pairs;
    for (int b = 0; b < nb; ++b) {
      for (int f = 0; f < nf; ++f) {
        if ((b+17*f)%nb < selected) pairs.push_back({b, 17*f-5});
      }
    }
    sparse->SetAllocation(pairs);
    view = sparse->GetDeviceView();
  }
  Kokkos::fence();
  const double allocation_seconds = allocation_timer.seconds();
  const bool plain = mode == "dense";
  auto initialize = [&]() {
    if (plain) {
      Kokkos::parallel_for("dense_initialize", Range(0, total),
          KOKKOS_LAMBDA(std::int64_t idx) {
        const int pair = idx/cells, c = idx%cells;
        const int b = pair/nf, f = pair%nf;
        dense(b,f,c/(ny*nx),(c/nx)%ny,c%nx) = Initial(b,f,selected,nb);
      });
    } else {
      Kokkos::parallel_for("sparse_initialize", Range(0, cells*view.active_count),
          KOKKOS_LAMBDA(std::int64_t idx) {
        const int slot = view.ActiveSlot(idx/cells), c = idx%cells;
        view.At(0,slot,c/(ny*nx),(c/nx)%ny,c%nx)
            = Initial(view.Block(slot),view.FieldIndex(slot),selected,nb);
      });
    }
  };
  auto update = [&]() {
    if (plain) {
      Kokkos::parallel_for("dense_scale", Range(0, total),
          KOKKOS_LAMBDA(std::int64_t idx) {
        const int pair = idx/cells, c = idx%cells;
        dense(pair/nf,pair%nf,c/(ny*nx),(c/nx)%ny,c%nx) *= alpha;
      });
    } else {
      Kokkos::parallel_for("sparse_scale", Range(0, cells*view.active_count),
          KOKKOS_LAMBDA(std::int64_t idx) {
        const int slot = view.ActiveSlot(idx/cells), c = idx%cells;
        view.At(0,slot,c/(ny*nx),(c/nx)%ny,c%nx) *= alpha;
      });
    }
  };
  initialize();
  for (int i = 0; i < 5; ++i) update();
  Kokkos::fence();
  std::vector<double> times;
  for (int repeat = 0; repeat < o.repeats; ++repeat) {
    initialize();
    Kokkos::fence();
    Kokkos::Timer timer;
    for (int it = 0; it < o.iterations; ++it) update();
    Kokkos::fence();
    times.push_back(timer.seconds());
  }
  const Real factor = static_cast<Real>(std::pow(static_cast<long double>(alpha),
                                                o.iterations));
  Real error = 0;
  const Real invalid = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("benchmark_verify", Range(0, total),
      KOKKOS_LAMBDA(std::int64_t idx, Real &maximum) {
    const int pair = idx/cells, c = idx%cells;
    const int b = pair/nf, f = pair%nf;
    const Real value = plain ? dense(b,f,c/(ny*nx),(c/nx)%ny,c%nx)
        : view.Read(0,b,f,c/(ny*nx),(c/nx)%ny,c%nx);
    const Real difference = fabs(value - Initial(b,f,selected,nb)*factor);
    if (!Kokkos::isfinite(value)) maximum = invalid;
    else if (difference > maximum) maximum = difference;
  }, Kokkos::Max<Real>(error));
  Require(error <= 16*std::numeric_limits<Real>::epsilon()*o.iterations,
          "Benchmark differs from analytic decay");
  auto sorted = times;
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted.size()%2 ? sorted[sorted.size()/2]
      : 0.5*(sorted[sorted.size()/2-1]+sorted[sorted.size()/2]);
  const std::size_t dense_bytes = static_cast<std::size_t>(total)*sizeof(Real);
  const auto memory = plain ? sparse::MemoryUsage{dense_bytes,dense_bytes,dense_bytes,0}
                            : sparse->Memory();
  std::cout << "{\"type\":\"benchmark\",\"mode\":\"" << mode
      << "\",\"blocks\":" << nb << ",\"fields\":" << nf
      << ",\"nx\":" << nx << ",\"ny\":" << ny << ",\"nz\":" << nz
      << ",\"requested_fraction\":" << o.fraction
      << ",\"nonzero_fraction\":" << static_cast<double>(selected)/nb
      << ",\"active_pairs\":" << (plain ? nb*nf : sparse->ActiveCount())
      << ",\"occupied_payload_bytes\":" << memory.occupied_payload
      << ",\"reserved_payload_bytes\":" << memory.reserved_payload
      << ",\"device_metadata_bytes\":" << memory.device_metadata
      << ",\"dense_payload_bytes\":" << memory.dense_payload
      << ",\"allocation_seconds\":" << allocation_seconds
      << ",\"iterations\":" << o.iterations << ",\"seconds_median\":" << median
      << ",\"max_error\":" << error << ",\"samples_seconds\":[";
  for (std::size_t i = 0; i < times.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << times[i];
  }
  std::cout << "]}\n";
}

void Main(int argc, char **argv) {
  sparse_test::Arguments args(argc,argv);
  Options o{args.Integer("blocks",128), args.Integer("fields",64),
      args.Integer("cells",64), args.Integer("ny",1), args.Integer("nz",1),
      args.Integer("iterations",100), args.Integer("repeats",5),
      args.Number("fraction",0.1)};
  const auto mode = args.String("mode","all");
  args.Finish();
  Require(o.blocks > 0 && o.fields > 0 && o.nx > 0 && o.ny > 0 && o.nz > 0 &&
          o.iterations > 0 && o.repeats > 0 && o.fraction >= 0 && o.fraction <= 1,
          "Invalid benchmark dimensions, fraction, or repetitions");
  Require(static_cast<std::int64_t>(o.blocks)*o.fields <=
          std::numeric_limits<int>::max() &&
          static_cast<std::int64_t>(o.nx)*o.ny*o.nz <=
          std::numeric_limits<int>::max(), "Benchmark dimensions too large");
  Require(mode == "all" || mode == "dense" || mode == "forced" || mode == "sparse",
          "Mode must be all, dense, forced, or sparse");
  sparse_test::PrintEnvironment();
  for (const std::string &candidate : {"dense","forced","sparse"}) {
    if (mode == "all" || mode == candidate) Run(o,candidate);
  }
}
}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc,argv);
  int result = 0;
  try { Main(argc,argv); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; result = 1; }
  Kokkos::finalize();
  return result;
}
