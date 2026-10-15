// Static/dynamic radial-profile performance benchmark with synthetic centers.
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "utils/radial_profile.hpp"
#include "bvals/bvals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif


DvceArray1D<RadialProfileCenter> DomainCenter(Mesh *pm, int num_centers) {
  const auto &size = pm->mesh_size;
  const auto &indcs = pm->mesh_indcs;
  const std::uint64_t nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const std::uint64_t num_cells = nx1*nx2*nx3;
  if (num_centers <= 0 || static_cast<std::uint64_t>(num_centers) > num_cells) {
    std::cerr << "### FATAL ERROR in radial_profile_benchmark: "
              << "problem/num_centers must be positive and no larger than "
              << "the global active-cell count (" << num_cells << "), got "
              << num_centers << std::endl;
    #if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    #endif
    std::exit(EXIT_FAILURE);
  }

  // Reproduce the same synthetic, uniformly distributed centers on every rank.
  std::mt19937_64 generator(0);
  std::uniform_int_distribution<std::uint64_t> cell_id(0, num_cells - 1);
  std::vector<std::uint64_t> center_ids;
  while (center_ids.size() < num_centers) {
    const auto id = cell_id(generator);
    if (std::find(center_ids.begin(), center_ids.end(), id) == center_ids.end()) {
      center_ids.push_back(id);
    }
  }
  std::sort(center_ids.begin(), center_ids.end());

  DvceArray1D<RadialProfileCenter> centers("domain_centers", num_centers);
  auto host_centers = Kokkos::create_mirror_view(centers);
  for (int n = 0; n < num_centers; ++n) {
    const auto id = center_ids[n];
    const std::uint64_t i = id%nx1, j = (id/nx1)%nx2, k = id/(nx1*nx2);
    host_centers(n) = {
      id,
      CellCenterX(i, indcs.nx1, size.x1min, size.x1max),
      CellCenterX(j, indcs.nx2, size.x2min, size.x2max),
      CellCenterX(k, indcs.nx3, size.x3min, size.x3max),
      0.0, 0.0, 0.0  // stationary uniform benchmark
    };
  }
  Kokkos::deep_copy(centers, host_centers);
  return centers;
}

void BenchmarkRadialProfile(ParameterInput *pin, Mesh *pm) {
  const int num_centers = pin->GetOrAddInteger("problem", "num_centers", 100);
  const int num_repeats = pin->GetOrAddInteger("problem", "num_repeats", 100);
  if (num_repeats <= 0) {
    std::cerr << "### FATAL ERROR in radial_profile_benchmark: "
              << "problem/num_repeats must be positive, got " << num_repeats
              << std::endl;
    #if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    #endif
    std::exit(EXIT_FAILURE);
  }
  Real rmax = pin->GetReal("problem", "rmax");
  int nbins_subcell_corrected = pin->GetInteger("problem", "nbins_subcell_corrected");
  int nsub = pin->GetInteger("problem", "nsub");

  RadialProfile radial_profile(pm, rmax, nbins_subcell_corrected, nsub);
  const auto centers = DomainCenter(pm, num_centers);

  // Warm up outside timing and obtain the actual radial-bin count from the result.
  radial_profile.measure_time = false;
  int num_radial_bins;
  {
    const auto profiles = radial_profile.Compute(centers);
    num_radial_bins = profiles.extent(2);
  }
  Kokkos::fence();

  // Time whole calls, including allocation, MPI, flux, and result destruction.
  // Phase instrumentation and center generation are excluded.
  #if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  Kokkos::Timer clock;
  for (int repeat = 0; repeat < num_repeats; ++repeat) {
    radial_profile.Compute(centers);
  }
  Kokkos::fence();
  const double elapsed = clock.seconds();
  double max_elapsed = elapsed;
  #if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);
  #endif
  const auto &indcs = pm->mesh_indcs;
  const std::uint64_t num_cells = static_cast<std::uint64_t>(indcs.nx1) *
                                  indcs.nx2 * indcs.nx3;
  const double seconds_per_batch = max_elapsed/num_repeats;
  const double seconds_per_center = seconds_per_batch/num_centers;
  const double batches_per_second = 1.0/seconds_per_batch;
  const double cell_center_pairs_per_second =
      static_cast<double>(num_cells)*num_centers/seconds_per_batch;

  // One separate, instrumented batch; its phase fences do not affect primary timing.
  #if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  radial_profile.measure_time = true;
  radial_profile.Compute(centers);
  Kokkos::fence();
  const auto &t = radial_profile.timings;
  const char *phase_names[] = {
    "allocation", "accumulation", "shell_reduction", "normalization",
    "magnetic_flux", "total"
  };
  const double phase_seconds[] = {
    t.allocation, t.accumulation, t.reduction, t.normalization, t.flux, t.total
  };
  double phase_sums[6], phase_maxima[6];
  #if MPI_PARALLEL_ENABLED
  MPI_Reduce(phase_seconds, phase_sums, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(phase_seconds, phase_maxima, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  #else
  std::copy(phase_seconds, phase_seconds + 6, phase_sums);
  std::copy(phase_seconds, phase_seconds + 6, phase_maxima);
  #endif

  if (global_variable::my_rank == 0) {
    const auto previous_precision = std::cout.precision(12);
    const auto &mb_indcs = pm->mb_indcs;
    std::cout << "RadialProfile performance... " << std::endl;
    std::cout << "rprof.global_nx1 = " << indcs.nx1 << std::endl;
    std::cout << "rprof.global_nx2 = " << indcs.nx2 << std::endl;
    std::cout << "rprof.global_nx3 = " << indcs.nx3 << std::endl;
    std::cout << "rprof.meshblock_nx1 = " << mb_indcs.nx1 << std::endl;
    std::cout << "rprof.meshblock_nx2 = " << mb_indcs.nx2 << std::endl;
    std::cout << "rprof.meshblock_nx3 = " << mb_indcs.nx3 << std::endl;
    std::cout << "rprof.global_active_cells = " << num_cells << std::endl;
    std::cout << "rprof.mpi_ranks = " << global_variable::nranks << std::endl;
    std::cout << "rprof.num_centers = " << num_centers << std::endl;
    std::cout << "rprof.num_repeats = " << num_repeats << std::endl;
    std::cout << "rprof.rmax = " << rmax << std::endl;
    std::cout << "rprof.num_radial_bins = " << num_radial_bins << std::endl;
    std::cout << "rprof.nbins_subcell_corrected = " << nbins_subcell_corrected
              << std::endl;
    std::cout << "rprof.nsub = " << nsub << std::endl;
    std::cout << "rprof.elapsed_seconds = " << max_elapsed << std::endl;
    std::cout << "rprof.seconds_per_batch = " << seconds_per_batch << std::endl;
    std::cout << "rprof.seconds_per_center = " << seconds_per_center << std::endl;
    std::cout << "rprof.batches_per_second = " << batches_per_second << std::endl;
    std::cout << "rprof.cell_center_pairs_per_second = "
              << cell_center_pairs_per_second << std::endl;
    std::cout << "Cell-center throughput measures nominal workload, not kernel "
              << "operations or MHD zone-cycles." << std::endl;
    std::cout << "Phase timings: one separately instrumented batch, rank means "
              << "and maxima in seconds. Independent phase maxima need not sum "
              << "to maximum total time." << std::endl;
    for (int phase = 0; phase < 6; ++phase) {
      std::cout << "rprof.phase." << phase_names[phase] << ".mean_seconds = "
                << phase_sums[phase]/global_variable::nranks << std::endl;
      std::cout << "rprof.phase." << phase_names[phase] << ".max_seconds = "
                << phase_maxima[phase] << std::endl;
    }
    std::cout.precision(previous_precision);
  }
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  pgen_final_func = BenchmarkRadialProfile;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  auto u0 = pack->phydro != nullptr ? pack->phydro->u0 : pack->pmhd->u0;
  Kokkos::deep_copy(u0, 0.0);
  Kokkos::deep_copy(Kokkos::subview(u0, Kokkos::ALL(), static_cast<int>(IDN),
                    Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), 1.0);
  if (pack->pmhd != nullptr) {
    Kokkos::deep_copy(pack->pmhd->b0.x1f, 0.0);
    Kokkos::deep_copy(pack->pmhd->b0.x2f, 0.0);
    Kokkos::deep_copy(pack->pmhd->b0.x3f, 1.0);
  }
}
