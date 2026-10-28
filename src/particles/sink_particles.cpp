//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles.cpp
//! \brief SinkParticles constructor, bookkeeping, timers, restart import and sanity check

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"
#include "sink.hpp"
#include "sink_particles.hpp"
#include "sink_names.hpp"
#include "particle_mesh.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor: reads the sink parameters of the population's input block and allocates
// the particle-mesh coupling, the potential exchange and the accretion staging buffers

SinkParticles::SinkParticles(ParticlePopulation *pop, MeshBlockPack *ppack,
                             ParameterInput *pin, const std::string &input_block) :
    ppop(pop),
    pmy_pack(ppack) {
  const std::string &blk = input_block;
  // host/MPI phase timers, off unless SINK_TIMERS is set (see sink_particles.hpp)
  if (const char *e = std::getenv("SINK_TIMERS")) { st_mode_ = std::atoi(e); }

  // control-volume gas accretion onto sinks: opt-in (accretion tests set it true);
  // orbit/gravity tests with inert or no gas leave it off so the kernel never runs
  accretion = pin->GetOrAddBoolean(blk,"accretion",false);
  // sink creation (LP threshold + potential minimum; see sink_creation.cpp)
  creation = pin->GetOrAddBoolean(blk,"creation",false);
  creation_finest_only = pin->GetOrAddBoolean(blk, "creation_finest_only", false);
  // sink-sink merging on overlapping control volumes (see sink_merger.cpp)
  const std::string setup = pin->GetOrAddString(blk, "sink_setup", "tigris");
  if (setup != "tigris" && setup != "legacy") {
    std::cerr << "Unknown " << blk << "/sink_setup: " << setup << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const bool reference = (setup == "tigris");
  merging = pin->GetOrAddBoolean(blk, "merging", reference && creation);
  merge_bound = pin->GetOrAddBoolean(blk, "merge_bound", !reference);
  merge_iterative = pin->GetOrAddBoolean(blk, "merge_iterative", reference);
  merge_cell_centers = pin->GetOrAddBoolean(blk, "merge_cell_centers", reference);
  merge_face_contact = pin->GetOrAddBoolean(blk, "merge_face_contact", false);
  creation_exclusion = pin->GetOrAddBoolean(blk, "creation_exclusion", !reference);
  creation_zero_velocity = pin->GetOrAddBoolean(blk, "creation_zero_velocity",
                                                reference);
  accrete_old_position = pin->GetOrAddBoolean(blk, "accrete_old_position", !reference);
  reject_negative_accretion = pin->GetOrAddBoolean(blk, "reject_negative_accretion",
                                                   reference);
  sanity_vmax = pin->GetOrAddReal(blk, "sanity_vmax", 0.0);
  sanity_flag_ = DvceArray1D<int>("sanity_flag", 1);
  sanity_rec_ = DvceArray1D<Real>("sanity_rec", kSanityRec);
  const std::string convergence = pin->GetOrAddString(
      blk, "sink_convergence", reference ? "mass_flux" : "off");
  if (convergence != "off" && convergence != "mass_flux" && convergence != "velocity") {
    std::cerr << "Unknown " << blk << "/sink_convergence: " << convergence << std::endl;
    std::exit(EXIT_FAILURE);
  }
  sink_convergence = convergence == "off" ? 0 : (convergence == "mass_flux" ? 1 : 2);
  if (!merge_iterative && (merge_cell_centers || merge_face_contact)) {
    std::cerr << "Cell/face merger geometry requires merge_iterative=true." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (creation && !creation_exclusion && (!merging || !merge_iterative)) {
    std::cerr << "Creation without exclusion requires iterative merging." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // particle CFL number (see sink_particles.hpp); 0.5 guarantees <= 1 cell per step
  cfl_par = pin->GetOrAddReal(blk,"cfl_par",0.5);

  // restarts of runs whose restart files hold no particles (see sink_particles.hpp)
  restart_trk = pin->GetOrAddString(blk, "restart_trk", "");
  restart_without_sinks = pin->GetOrAddBoolean(blk, "restart_without_sinks", false);

  // TODO(SMOON) This is temporary treatment of source term on particle.
  // Later, we need to implement a more general way to include source terms
  point_mass_gm = pin->GetOrAddReal(blk,"point_mass_gm",0.0);

  // 1 slot for rho_particles; extend (to 4 with momentum) when needed.
  ppm = new ParticleMesh(ppack, pin, 1);
  // Own boundary-values object for the potential halo exchange (1 variable). Separate
  // from ppm->pmbval so phi's exchange state cannot interact with the deposit's.
  pbval_phi = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_phi->InitializeBuffers(1);

  // staging buffer + communicator for the cross-rank control-volume reset (MPI). Sized
  // to hold every reset cell of every sink twice over (new + old CV = 54 cells), which
  // bounds the off-rank-destined subset; overflow is guarded + warned in the kernel.
  if (accretion) {
    cvemit_max_ = std::max(1, ppop->nprtcl_thispack)*64;
    Kokkos::realloc(cvemit_, cvemit_max_, NCVEMIT);
    Kokkos::realloc(cvemit_cnt_, 1);
    Kokkos::realloc(accskip_, 1);
#if MPI_PARALLEL_ENABLED
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_cvscat_);
#endif
  }
}

//----------------------------------------------------------------------------------------
//! \fn ParticlePopulation* FindSinkPopulation(MeshBlockPack *pmbp)
//! \brief the sink population of the pack, or nullptr if the run has none

ParticlePopulation* FindSinkPopulation(MeshBlockPack *pmbp) {
  return (pmbp->ppart == nullptr) ? nullptr :
      pmbp->ppart->FindPopulationOfType(ParticleType::sink);
}

//----------------------------------------------------------------------------------------
//! \fn ParticlePopulation* SinkPopulation(MeshBlockPack *pmbp)
//! \brief the sink population of the pack; problem generators use it to seed sinks

ParticlePopulation* SinkPopulation(MeshBlockPack *pmbp) {
  ParticlePopulation *pop = FindSinkPopulation(pmbp);
  if (pop == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "This problem needs a particle population of type 'sink'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return pop;
}

//----------------------------------------------------------------------------------------
// destructor

SinkParticles::~SinkParticles() {
  delete ppm;
  delete pbval_phi;
}

//----------------------------------------------------------------------------------------
//! \fn void SinkParticles::RefreshMeshParticleCounts()
//! \brief Re-derive the Mesh's global particle bookkeeping (nprtcl_thisrank,
//! nprtcl_eachrank, nprtcl_total) after the sink arrays changed size. Collective.

void SinkParticles::RefreshMeshParticleCounts() {
  SinkTimerScope _st(this, ST_REFRESH);
  pmy_pack->pmesh->UpdateParticleCounts();
}

//----------------------------------------------------------------------------------------
//! \fn double SinkParticles::StBegin(int id) / void SinkParticles::StEnd(int, double)
//! \brief start/stop one host-MPI phase timer. See the SINK_TIMERS block in
//! sink_particles.hpp for what the two modes mean and why mode 2 perturbs on purpose.

double SinkParticles::StBegin(int id) {
  if (st_mode_ <= 0) return 0.0;
  if (st_mode_ >= 2) {
    // Isolate the region from work already in flight: fence the device, then barrier so
    // every rank starts together. What the barrier absorbs is load imbalance -- charged
    // to "wait", not to the region, because a collective exposes imbalance rather than
    // creating it, and conflating the two is how a cheap collective gets blamed.
    const double b = st_clock_.seconds();
    Kokkos::fence();
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) { MPI_Barrier(MPI_COMM_WORLD); }
#endif
    st_wait_[id] += st_clock_.seconds() - b;
  }
  return st_clock_.seconds();
}

void SinkParticles::StEnd(int id, double t0) {
  if (st_mode_ <= 0) return;
  st_time_[id] += st_clock_.seconds() - t0;
  st_calls_[id] += 1;
}

//----------------------------------------------------------------------------------------
//! \fn void SinkParticles::ReportSinkTimers(double wall_seconds)
//! \brief print the accumulated host/MPI phase times. Rank 0 prints; the max over ranks
//! is what matters for a synchronous step, so both max and mean are shown.

void SinkParticles::ReportSinkTimers(double wall_seconds) {
  if (st_mode_ <= 0) return;
  const int n = ST_NUM;
  const char *name[ST_NUM] = {"colour(host+mirror)", "cv_reset(total)",
                              "  MPI_Alltoall", "create_sinks", "merge_sinks",
                              "refresh_counts"};
  double tmax[ST_NUM], tsum[ST_NUM], wmax[ST_NUM];
  std::int64_t csum[ST_NUM];
  for (int i = 0; i < n; ++i) {
    tmax[i] = st_time_[i]; tsum[i] = st_time_[i];
    wmax[i] = st_wait_[i]; csum[i] = st_calls_[i];
  }
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, tmax, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, tsum, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, wmax, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, csum, n, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank != 0) return;
  const int nr = global_variable::nranks;
  std::printf("\n### SINK TIMERS  mode=%d  ranks=%d  wall=%.4f s"
              "  (host/MPI regions only -- device kernels are kokkos-tools' job)\n",
              st_mode_, nr, wall_seconds);
  std::printf("###   ST_ALLTOALL is nested in ST_CVRESET, and refresh_counts in "
              "create/merge: totals are INCLUSIVE, do not sum the column.\n");
  std::printf("### %-20s %10s %12s %12s %12s %8s\n",
              "region", "calls", "max_s", "mean_s", "wait_max_s", "%wall");
  for (int i = 0; i < n; ++i) {
    if (csum[i] == 0) continue;
    std::printf("### %-20s %10" PRId64 " %12.5f %12.5f %12.5f %7.2f%%\n",
                name[i], csum[i]/nr, tmax[i], tsum[i]/nr, wmax[i],
                (wall_seconds > 0.0) ? 100.0*tmax[i]/wall_seconds : 0.0);
  }
  std::printf("\n");
}

//----------------------------------------------------------------------------------------
//! \fn void SinkParticles::ResizeForSeededParticles(int npart_new)
//! \brief Resize the particle arrays to npart_new, grow the buffers that were sized from
//! the ppc-derived count, and refresh the Mesh's global particle bookkeeping. New slots
//! start PACTIVE; the caller fills every other column.
//!
//! The constructor sizes cvemit_ (the cross-rank control-volume reset staging buffer) as
//! max(1, nprtcl_thispack)*64 records, from the count ppc implied BEFORE the pgen runs. A
//! benchmark or IC that seeds sinks explicitly typically runs with ppc = 0, so that
//! buffer would hold 64 records total no matter how many sinks are seeded. Grow it here,
//! so a pgen only has to state how many particles it wants.

void SinkParticles::ResizeForSeededParticles(int npart_new) {
  const int npart_old = ppop->nprtcl_thispack;
  Kokkos::resize(ppop->prtcl_rdata, ppop->nrdata, npart_new);
  Kokkos::resize(ppop->prtcl_idata, ppop->nidata, npart_new);
  ppop->nprtcl_thispack = npart_new;
  if (npart_new > npart_old) {
    auto status = Kokkos::subview(ppop->prtcl_idata, static_cast<int>(PSTATUS),
                                  std::make_pair(npart_old, npart_new));
    Kokkos::deep_copy(status, static_cast<int>(PACTIVE));
  }

  // grow-only: never shrink below what a previous sizing already allocated
  if (accretion) {
    const int need = std::max(1, npart_new)*64;
    if (need > cvemit_max_) {
      cvemit_max_ = need;
      Kokkos::realloc(cvemit_, cvemit_max_, NCVEMIT);
    }
  }

  RefreshMeshParticleCounts();
}

//----------------------------------------------------------------------------------------
//! \fn void SinkParticles::RestoreSinksFromTrk(const std::string &path)
//! \brief Rebuild the sinks of a restarted run from the trk output of
//! feature/sink-particles-only (see sink_particles.hpp). Columns are matched by name, so
//! the interleaved layout of those files (x vx y vy ...) maps onto this one. Exact
//! gates: the dump's cycle must equal the restart cycle and its time the restart time,
//! every stored column must be present, and every sink must be claimed by exactly one
//! MeshBlock globally.

void SinkParticles::RestoreSinksFromTrk(const std::string &path) {
  auto *pm = pmy_pack->pmesh;
  const int want_cycle = pm->ncycle;
  const int nrdata = ppop->nrdata;
  int nsink = 0, nrec = 0;
  double trk_time = 0.0;
  std::vector<Real> recs;
  // file column of every real column of this layout, then of the tag
  std::vector<int> col(nrdata + 1, -1);
  if (global_variable::my_rank == 0) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "restart_trk '" << path << "' could not be opened"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    char line[4096];
    int found = 0, last_cycle = -1;
    std::vector<std::string> names;
    // each dump: "# AthenaK particle data", "# time=...", "# columns: ...", binary block
    while (std::fgets(line, sizeof(line), f) != nullptr) {
      double t; int cyc, nr, np, ni, nd, nc;
      char cols[4096];
      if (std::fgets(line, sizeof(line), f) == nullptr ||
          std::sscanf(line, "# time=%lf cycle=%d nranks=%d nparticles=%d nidata=%d "
                      "nrdata=%d nrec=%d", &t, &cyc, &nr, &np, &ni, &nd, &nc) != 7 ||
          std::fgets(cols, sizeof(cols), f) == nullptr) {       // "# columns: ..."
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "malformed dump header in '" << path << "' after cycle "
                  << last_cycle << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::vector<Real> block(static_cast<std::size_t>(np)*nc);
      if (np > 0 &&
          std::fread(block.data(), sizeof(Real), block.size(), f) != block.size()) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "truncated dump at cycle " << cyc << " in '" << path
                  << "'" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      last_cycle = cyc;
      if (cyc != want_cycle) continue;
      ++found;          // keep the LAST dump of this cycle (a rerun appends after it)
      nsink = np;
      nrec = nc;
      trk_time = t;
      recs.swap(block);
      std::istringstream ss(std::string(cols).substr(std::string("# columns:").size()));
      names.clear();
      for (std::string name; ss >> name;) names.push_back(name);
    }
    std::fclose(f);
    if (found == 0 || std::fabs(trk_time - pm->time) > 1.0e-12*std::max(1.0, pm->time)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "no trk dump in '" << path
                << "' matches the restart (cycle " << want_cycle << ", time " << pm->time
                << "); "
                << (found ? "the cycle matches but the time differs" : "last dump cycle ")
                << (found ? std::string() : std::to_string(last_cycle)) << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int v = 0; v <= nrdata; ++v) {
      const std::string want = (v < nrdata) ? sink::real_names[v] : "tag";
      for (std::size_t c = 0; c < names.size(); ++c) {
        if (names[c] == want) col[v] = static_cast<int>(c);
      }
      if (col[v] < 0 || static_cast<int>(names.size()) != nrec) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "trk dump in '" << path << "' has no column '" << want
                  << "' (or a column list that does not match its record width)"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&nsink, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nrec, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&trk_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(col.data(), nrdata + 1, MPI_INT, 0, MPI_COMM_WORLD);
  recs.resize(static_cast<std::size_t>(nsink)*nrec);
  if (nsink > 0) {
    MPI_Bcast(recs.data(), nsink*nrec, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
#endif
  auto rec = [&](int q, int v) {
    return recs[static_cast<std::size_t>(q)*nrec + col[v]];
  };

  // ownership: lower-inclusive containment, as ParticlePopulation::ReownByPosition
  const int nmb = pmy_pack->nmb_thispack;
  auto &mbsz = pmy_pack->pmb->mb_size;
  std::vector<int> own(nsink, -1);
  int nloc = 0, maxtag = -1;
  for (int q = 0; q < nsink; ++q) {
    const Real x = rec(q, IPX), y = rec(q, IPY), z = rec(q, IPZ);
    maxtag = std::max(maxtag, static_cast<int>(rec(q, nrdata)));
    for (int m = 0; m < nmb; ++m) {
      if (x >= mbsz.h_view(m).x1min && x < mbsz.h_view(m).x1max &&
          y >= mbsz.h_view(m).x2min && y < mbsz.h_view(m).x2max &&
          z >= mbsz.h_view(m).x3min && z < mbsz.h_view(m).x3max) {
        own[q] = m;
        ++nloc;
        break;
      }
    }
  }
  int nglob = nloc;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (nglob != nsink) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "restored sink ownership is inconsistent: " << nglob << " of " << nsink
              << " sinks were claimed by exactly one MeshBlock." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  ResizeForSeededParticles(nloc);   // collective
  if (nloc > 0) {
    auto pr_h = Kokkos::create_mirror_view(ppop->prtcl_rdata);
    auto pi_h = Kokkos::create_mirror_view(ppop->prtcl_idata);
    int slot = 0;
    for (int q = 0; q < nsink; ++q) {
      if (own[q] < 0) continue;
      for (int v = 0; v < nrdata; ++v) pr_h(v, slot) = rec(q, v);
      pi_h(PGID, slot) = pmy_pack->gids + own[q];
      pi_h(PTAG, slot) = static_cast<int>(rec(q, nrdata));
      pi_h(PSTATUS, slot) = PACTIVE;
      ++slot;
    }
    Kokkos::deep_copy(ppop->prtcl_rdata, pr_h);
    Kokkos::deep_copy(ppop->prtcl_idata, pi_h);
  }
  // created sinks are tagged CREATED_TAG_BASE + created_total_ + (rank offset)
  created_total_ = (maxtag >= sink::CREATED_TAG_BASE) ?
                   maxtag - sink::CREATED_TAG_BASE + 1 : 0;
  if (global_variable::my_rank == 0) {
    Real msum = 0.0;
    for (int q = 0; q < nsink; ++q) msum += rec(q, IPM);
    std::printf("### RESTART: restored %d sink(s), total mass %.17e, from '%s' "
                "(cycle %d, time %.17e); next created tag %d\n", nsink, msum,
                path.c_str(), want_cycle, trk_time,
                sink::CREATED_TAG_BASE + created_total_);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SinkParticles::CheckSinkSanity
//! \brief Abort with a full report on the first non-finite or absurd sink (see the
//! sanity_vmax comment in sink_particles.hpp). WHERE names the step that just ran;
//! together with the clean check after the previous step it brackets the update that
//! broke it.

void SinkParticles::CheckSinkSanity(const char *where, int stage) {
  if (sanity_vmax <= 0.0) return;
  auto &pr = ppop->prtcl_rdata;
  const int npart = ppop->nprtcl_thispack;
  const Real vmax2 = sanity_vmax*sanity_vmax;
  int nbad = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("sink_sanity", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &n) {
      const Real v2 = pr(IPVX, p)*pr(IPVX, p) + pr(IPVY, p)*pr(IPVY, p)
                    + pr(IPVZ, p)*pr(IPVZ, p);
      const Real s = pr(IPX, p) + pr(IPY, p) + pr(IPZ, p) + pr(IPM, p);
      // written so that a NaN fails every comparison
      if (!(v2 <= vmax2) || !(Kokkos::fabs(s) < 1.0e300) || !(pr(IPM, p) >= 0.0)) n += 1;
    }, Kokkos::Sum<int>(nbad));
  }
  auto flag_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sanity_flag_);
  if (nbad == 0 && flag_h(0) == 0) return;

  auto hr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppop->prtcl_rdata);
  auto hi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppop->prtcl_idata);
  auto rec = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sanity_rec_);
  auto *pm = pmy_pack->pmesh;
  std::printf("### SINK SANITY: %d bad sink(s) after %s (stage %d) on rank %d, cycle=%d "
              "time=%.17e dt=%.17e, %d local sink(s), sanity_vmax=%g\n", nbad, where,
              stage, global_variable::my_rank, pm->ncycle, pm->time, pm->dt,
              npart, sanity_vmax);
  if (flag_h(0) == 1) {
    std::printf("### SINK SANITY: push produced it: tag=%.0f v_old=(%.17e, %.17e, %.17e) "
                "g=(%.17e, %.17e, %.17e) half_dt=%.17e v_new=(%.17e, %.17e, %.17e) "
                "x=(%.17e, %.17e, %.17e) m=%.17e\n", rec(0), rec(1), rec(2), rec(3),
                rec(4), rec(5), rec(6), rec(7), rec(8), rec(9), rec(10), rec(11), rec(12),
                rec(13), rec(14));
  } else if (flag_h(0) == 2) {
    std::printf("### SINK SANITY: accretion produced it: tag=%.0f m_old=%.17e dm=%.17e "
                "dp=(%.17e, %.17e, %.17e) v_old=(%.17e, %.17e, %.17e) "
                "v_new=(%.17e, %.17e, %.17e) x=(%.17e, %.17e, %.17e)\n", rec(0), rec(1),
                rec(2), rec(3), rec(4), rec(5), rec(6), rec(7), rec(8), rec(9), rec(10),
                rec(11), rec(12), rec(13), rec(14));
  }
  for (int p = 0; p < npart; ++p) {
    std::printf("### SINK SANITY:   p=%d tag=%d gid=%d", p, hi(PTAG, p), hi(PGID, p));
    for (int r = 0; r < ppop->nrdata; ++r) {
      std::printf(" %s=%.17e", sink::real_names[r], hr(r, p));
    }
    std::printf("\n");
  }
  std::fflush(stdout);
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, 86);
#endif
  std::exit(EXIT_FAILURE);
}

} // namespace particles
