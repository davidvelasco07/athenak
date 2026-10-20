// Standalone single-rank periodic finite-volume transport on uniform logical blocks.
// This deliberately has no dependency on a fluid solver or its state arrays.
#include "common.hpp"

#include <algorithm>
#include <memory>

namespace {
using sparse_test::Range;
using sparse_test::Require;

struct Options {
  int blocks, cells, fields, steps, direction, width, prune;
  Real courant;
};

struct Result {
  std::vector<Real> values;
  double seconds = 0, allocation_seconds = 0;
  std::size_t reserved = 0, metadata = 0;
  int initial_pairs = 0, final_pairs = 0, peak_pairs = 0;
  int activations = 0, deallocations = 0;
};

KOKKOS_INLINE_FUNCTION
Real Initial(int field, int cell, int ncell, int width) {
  const int start = (37LL*field)%ncell;
  return ((cell-start+ncell)%ncell < width) ? Real(1)+Real(field%17)/Real(16) : Real(0);
}

std::vector<Real> Reference(const Options &o) {
  const int n = o.blocks*o.cells;
  std::vector<Real> q(static_cast<std::size_t>(o.fields)*n), next(q.size());
  for (int f = 0; f < o.fields; ++f) {
    for (int x = 0; x < n; ++x) q[f*n+x] = Initial(f,x,n,o.width);
  }
  for (int step = 0; step < o.steps; ++step) {
    for (int f = 0; f < o.fields; ++f) {
      for (int x = 0; x < n; ++x) {
        const int upwind = (x-o.direction+n)%n;
        next[f*n+x] = (Real(1)-o.courant)*q[f*n+x] + o.courant*q[f*n+upwind];
      }
    }
    q.swap(next);
  }
  return q;
}

Result Dense(const Options &o) {
  const int nb = o.blocks, nf = o.fields, nx = o.cells, n = nb*nx;
  const int width = o.width, direction = o.direction;
  const Real courant = o.courant;
  DvceArray4D<Real> q("advection_dense",2,nb,nf,nx+2);
  Kokkos::deep_copy(q,0.0);
  Kokkos::parallel_for("dense_initial", Range(0,static_cast<std::int64_t>(nb)*nf*nx),
      KOKKOS_LAMBDA(std::int64_t idx) {
    const int pair = idx/nx, i = idx%nx, b = pair/nf, f = pair%nf;
    q(0,b,f,i+1) = Initial(f,b*nx+i,n,width);
  });
  Kokkos::fence();
  Kokkos::Timer timer;
  int current = 0;
  for (int step = 0; step < o.steps; ++step) {
    const int next = 1-current;
    Kokkos::parallel_for("dense_ghost_exchange", Range(0,nb*nf),
        KOKKOS_LAMBDA(std::int64_t idx) {
      const int b = idx/nf, f = idx%nf;
      q(current,b,f,0) = q(current,(b+nb-1)%nb,f,nx);
      q(current,b,f,nx+1) = q(current,(b+1)%nb,f,1);
    });
    Kokkos::parallel_for("dense_advect", Range(0,static_cast<std::int64_t>(nb)*nf*nx),
        KOKKOS_LAMBDA(std::int64_t idx) {
      const int pair = idx/nx, i = idx%nx+1, b = pair/nf, f = pair%nf;
      q(next,b,f,i) = (Real(1)-courant)*q(current,b,f,i)
                     + courant*q(current,b,f,i-direction);
    });
    current = next;
  }
  Kokkos::fence();
  Result result;
  result.seconds = timer.seconds();
  result.initial_pairs = result.final_pairs = result.peak_pairs = nb*nf;
  result.reserved = q.size()*sizeof(Real);
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),q);
  result.values.resize(static_cast<std::size_t>(nf)*n);
  for (int f = 0; f < nf; ++f) {
    for (int x = 0; x < n; ++x) result.values[f*n+x] = host(current,x/nx,f,x%nx+1);
  }
  return result;
}

Result Sparse(const Options &o, bool forced) {
  const int nb = o.blocks, nf = o.fields, nx = o.cells, n = nb*nx;
  const int width = o.width, direction = o.direction;
  const Real courant = o.courant;
  sparse::Variables store("advection_sparse", sparse_test::MakeRegistry(nf),
                           nb,1,1,nx+2,2,forced);
  std::vector<sparse::BlockField> initial;
  for (int b = 0; b < nb; ++b) {
    for (int f = 0; f < nf; ++f) {
      for (int i = 0; i < nx; ++i) {
        if (Initial(f,b*nx+i,n,width) != 0) {
          initial.push_back({b,17*f-5});
          break;
        }
      }
    }
  }
  store.SetAllocation(initial);
  auto view = store.GetDeviceView();
  Kokkos::parallel_for("sparse_initial", Range(0,view.active_count*nx),
      KOKKOS_LAMBDA(std::int64_t idx) {
    const int slot = view.ActiveSlot(idx/nx), i = idx%nx;
    view.At(0,slot,0,0,i+1) = Initial(view.FieldIndex(slot),view.Block(slot)*nx+i,n,width);
  });
  DvceArray2D<int> flags("sparse_advection_flags",nb,nf);
  auto flags_h = Kokkos::create_mirror_view(flags);
  Result result;
  result.initial_pairs = result.peak_pairs = store.ActiveCount();
  int current = 0;
  Kokkos::fence();
  Kokkos::Timer total;
  for (int step = 0; step < o.steps; ++step) {
    view = store.GetDeviceView();
    // Detect outgoing flux BEFORE the update, so a receiver participates in that update.
    // There is one upwind sender per (destination block, field); no atomics are needed.
    if (!forced) {
      Kokkos::deep_copy(flags,0);
      Kokkos::parallel_for("sparse_activation_requests", Range(0,view.active_count),
        KOKKOS_LAMBDA(std::int64_t w) {
      const int slot = view.ActiveSlot(w), b = view.Block(slot), f = view.FieldIndex(slot);
      const int edge = direction > 0 ? nx : 1;
      if (view.At(current,slot,0,0,edge) != 0) flags((b+direction+nb)%nb,f) = 1;
      });
      Kokkos::deep_copy(flags_h,flags);
      std::vector<sparse::BlockField> additions;
      for (int b = 0; b < nb; ++b) {
        for (int f = 0; f < nf; ++f) {
          if (flags_h(b,f) && store.Slot(b,17*f-5) < 0) additions.push_back({b,17*f-5});
        }
      }
      Kokkos::Timer allocation_timer;
      store.Allocate(additions);
      result.allocation_seconds += allocation_timer.seconds();
      result.activations += additions.size();
    }
    result.peak_pairs = std::max(result.peak_pairs,store.ActiveCount());
    view = store.GetDeviceView();
    Kokkos::parallel_for("sparse_ghost_exchange", Range(0,view.active_count),
        KOKKOS_LAMBDA(std::int64_t w) {
      const int slot = view.ActiveSlot(w), b = view.Block(slot), f = view.FieldIndex(slot);
      view.At(current,slot,0,0,0) = view.Read(current,(b+nb-1)%nb,f,0,0,nx);
      view.At(current,slot,0,0,nx+1) = view.Read(current,(b+1)%nb,f,0,0,1);
    });
    const int next = 1-current;
    Kokkos::parallel_for("sparse_advect", Range(0,view.active_count*nx),
        KOKKOS_LAMBDA(std::int64_t idx) {
      const int slot = view.ActiveSlot(idx/nx), i = idx%nx+1;
      view.At(next,slot,0,0,i) = (Real(1)-courant)*view.At(current,slot,0,0,i)
                               + courant*view.At(current,slot,0,0,i-direction);
    });
    current = next;
    if (o.prune && !forced) {
      // Exact-zero removal AFTER all updates. Old-register/ghost values are dead here.
      // A general RK module must choose its own safe lifetime boundary instead.
      Kokkos::deep_copy(flags,0);
      Kokkos::parallel_for("sparse_live_pairs",
          Kokkos::TeamPolicy<DevExeSpace>(view.active_count,Kokkos::AUTO),
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<DevExeSpace>::member_type &team) {
        const int slot = view.ActiveSlot(team.league_rank());
        int nonzero = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,nx),
            [&](const int i, int &count) {
          if (view.At(current,slot,0,0,i+1) != 0) ++count;
        },nonzero);
        Kokkos::single(Kokkos::PerTeam(team),[&]() {
          flags(view.Block(slot),view.FieldIndex(slot)) = nonzero > 0;
        });
      });
      Kokkos::deep_copy(flags_h,flags);
      std::vector<sparse::BlockField> keep;
      for (int b = 0; b < nb; ++b) {
        for (int f = 0; f < nf; ++f) if (flags_h(b,f)) keep.push_back({b,17*f-5});
      }
      const int removed = store.ActiveCount()-static_cast<int>(keep.size());
      Kokkos::Timer allocation_timer;
      store.SetAllocation(keep);
      result.allocation_seconds += allocation_timer.seconds();
      result.deallocations += removed;
    }
  }
  Kokkos::fence();
  result.seconds = total.seconds();
  const auto memory = store.Memory();
  result.reserved = memory.reserved_payload;
  result.metadata = memory.device_metadata;
  result.final_pairs = store.ActiveCount();
  view = store.GetDeviceView();
  DvceArray2D<Real> output("sparse_final",nf,n);
  Kokkos::parallel_for("sparse_extract", Range(0,static_cast<std::int64_t>(nf)*n),
      KOKKOS_LAMBDA(std::int64_t idx) {
    const int f = idx/n, x = idx%n;
    output(f,x) = view.Read(current,x/nx,f,0,0,x%nx+1);
  });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),output);
  result.values.resize(static_cast<std::size_t>(nf)*n);
  for (int f = 0; f < nf; ++f) {
    for (int x = 0; x < n; ++x) result.values[f*n+x] = host(f,x);
  }
  return result;
}

void Check(const Options &o, const Result &r, const std::vector<Real> &reference,
           const std::string &mode) {
  const int n = o.blocks*o.cells;
  double error = 0, mass_error = 0, translated_error = 0;
  const double tolerance = 128*std::numeric_limits<Real>::epsilon()*std::max(1,o.steps);
  for (int f = 0; f < o.fields; ++f) {
    double initial_mass = 0, mass = 0;
    for (int x = 0; x < n; ++x) {
      const Real value = r.values[f*n+x];
      Require(std::isfinite(value) && value >= 0, "Non-finite/negative advected value");
      error = std::max(error,static_cast<double>(fabs(value-reference[f*n+x])));
      initial_mass += Initial(f,x,n,o.width);
      mass += value;
      if (o.courant == 1) {
        const int origin = (x-o.direction*(o.steps%n)+n)%n;
        translated_error = std::max(translated_error,
            static_cast<double>(fabs(value-Initial(f,origin,n,o.width))));
      }
    }
    mass_error = std::max(mass_error,fabs(mass-initial_mass)/initial_mass);
  }
  Require(error < tolerance && mass_error < tolerance && translated_error < tolerance,
          "Sparse transport failed dense reference, conservation, or translation");
  if (mode == "sparse" && o.courant == 1 && o.blocks > 2 && o.width < o.cells &&
      o.steps >= n && o.prune) {
    Require(r.activations > 0 && r.deallocations > 0,
            "Full-period transport did not exercise activation and slot reuse");
  }
  std::cout << "{\"type\":\"advection\",\"mode\":\"" << mode
      << "\",\"blocks\":" << o.blocks << ",\"fields\":" << o.fields
      << ",\"cells_per_block\":" << o.cells << ",\"steps\":" << o.steps
      << ",\"courant\":" << o.courant << ",\"direction\":" << o.direction
      << ",\"prune\":" << o.prune << ",\"initial_pairs\":" << r.initial_pairs
      << ",\"final_pairs\":" << r.final_pairs << ",\"peak_pairs\":" << r.peak_pairs
      << ",\"activations\":" << r.activations << ",\"deallocations\":" << r.deallocations
      << ",\"reserved_payload_bytes\":" << r.reserved
      << ",\"device_metadata_bytes\":" << r.metadata
      << ",\"seconds\":" << r.seconds
      << ",\"allocation_seconds\":" << r.allocation_seconds
      << ",\"max_reference_error\":" << error
      << ",\"max_relative_mass_error\":" << mass_error
      << ",\"max_translation_error\":" << translated_error << "}\n";
}

void Main(int argc, char **argv) {
  sparse_test::Arguments args(argc,argv);
  Options o{args.Integer("blocks",16),args.Integer("cells",32),
      args.Integer("fields",32),args.Integer("steps",512),args.Integer("direction",1),
      args.Integer("width",4),args.Integer("prune",1),
      static_cast<Real>(args.Number("courant",1))};
  args.Finish();
  Require(o.blocks > 0 && o.cells > 0 && o.fields > 0 && o.steps > 0 &&
          (o.direction == 1 || o.direction == -1) && o.courant > 0 && o.courant <= 1 &&
          (o.prune == 0 || o.prune == 1) && o.width > 0 &&
          static_cast<std::int64_t>(o.blocks)*o.cells <= std::numeric_limits<int>::max() &&
          static_cast<std::int64_t>(o.blocks)*o.fields*o.cells <=
          std::numeric_limits<int>::max() && o.width <= o.blocks*o.cells,
          "Invalid advection parameters");
  const auto reference = Reference(o);
  sparse_test::PrintEnvironment();
  Check(o,Dense(o),reference,"dense");
  Check(o,Sparse(o,true),reference,"forced");
  Check(o,Sparse(o,false),reference,"sparse");
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
