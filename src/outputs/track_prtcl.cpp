//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file track_prtcl.cpp
//! \brief writes the FULL per-particle state to an unformatted binary file.
//!
//! Every column the species carries is written: all `nidata` integer properties (gid,
//! tag) followed by all `nrdata` real ones. For sinks that is position, velocity, MASS,
//! the gathered gravitational acceleration and the start-of-step position -- the mass in
//! particular being the reason this output exists at all, since neither the deposited
//! `prtcl_d` field (a per-cell COUNT, no mass) nor the history file (fixed 20 columns)
//! can give a per-sink mass time series.
//!
//! FILE LAYOUT. The file is appended to once per output time. Each dump is a text header
//! followed by a contiguous binary block of `npout_total` records, every record being
//! `nrec = nidata + nrdata` values of type Real, integer properties first (cast to Real,
//! which is exact for gid/tag and keeps full precision for mass). The header states the
//! record width and the column names, so a reader never needs to know the particle type:
//!
//!   # AthenaK particle data
//!   # time=... cycle=... nranks=... nparticles=... nidata=... nrdata=... nrec=...
//!   # columns: gid tag x vx y vy z vz m gx gy gz x0 y0 z0
//!   <npout_total * nrec Reals, little-endian>
//!
//! Records are laid out contiguously in RANK ORDER, each rank writing its own block at an
//! offset derived from the prefix sum of the per-rank counts. This deliberately replaces
//! the previous scheme, which placed each record at an offset derived from its tag and so
//! silently assumed tags ran 0..(nparticles-1) with no gaps. Created sinks are numbered
//! from a global base of 1000000 (see Particles::CreateSinks), so under that assumption
//! they were either written megabytes into the file or, because selection also tested
//! `tag < nparticles`, skipped entirely -- which is why this output was unusable for any
//! run whose sinks were created rather than seeded.
//!
//! SELECTION. `<outputN>/nparticles` is now optional. Omitted or <= 0 writes EVERY
//! particle, which is what a sink run wants; a positive value keeps the legacy behaviour
//! of writing only tags below it, which is how tracer runs pick a fixed tracked subset.

#include <sys/stat.h>  // mkdir
#include <vector>

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

TrackedParticleOutput::TrackedParticleOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("trk",0775);
  // allocate arrays
  npout_eachrank.resize(global_variable::nranks);
  // Optional: omitted or <= 0 means "write every particle". A positive value restricts
  // the output to tags below it (the legacy tracer behaviour).
  ntrack = pin->GetOrAddInteger(op.block_name,"nparticles",0);
  ntrack_thisrank = ntrack;
  nrec = 0;
  npout = 0;
}

//----------------------------------------------------------------------------------------
// TrackedParticleOutput::LoadOutputData()
// Gathers the full state of every selected particle on this rank into outpart

void TrackedParticleOutput::LoadOutputData(Mesh *pm) {
  particles::Particles *pp = pm->pmb_pack->ppart;
  if (pp == nullptr) {
    npout = 0;
    return;
  }
  const int nid = pp->nidata;
  const int nrd = pp->nrdata;
  nrec = nid + nrd;
  const int npart = pp->nprtcl_thispack;
  auto &pr = pp->prtcl_rdata;
  auto &pi = pp->prtcl_idata;

  // Gather into a device record array, compacting the selected particles.
  // The compaction counter must live in device memory. A host stack int captured into a
  // device kernel and atomically updated there is undefined on GPUs (the device writes are
  // not coherently visible to the subsequent host read, so npout comes back stale -- e.g. 0
  // -- which later zero-sizes outpart and segfaults the write loop). It only "works"
  // on CPU because host==device. Use a device View and copy it back; the deep_copy also
  // fences, guaranteeing the count is valid before it is read on the host.
  DualArray2D<Real> rec("trk_rec", std::max(1,npart), nrec);
  Kokkos::View<int, DevExeSpace> counter_d("trk_counter");
  Kokkos::deep_copy(counter_d, 0);
  const int ntrk = ntrack;  // localize: capturing the member would capture `this` (host ptr)
  par_for("trk_gather",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    // ntrk <= 0 selects everything; otherwise keep the legacy "tags below ntrk" subset
    if (ntrk > 0 && pi(PTAG,p) >= ntrk) { return; }
    const int idx = Kokkos::atomic_fetch_add(&counter_d(),1);
    for (int v=0; v<nid; ++v) { rec.d_view(idx, v) = static_cast<Real>(pi(v,p)); }
    for (int v=0; v<nrd; ++v) { rec.d_view(idx, nid+v) = pr(v,p); }
  });
  auto counter_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), counter_d);
  npout = counter_h();

  // share number of particles to be output across all ranks
  npout_eachrank[global_variable::my_rank] = npout;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npout, 1, MPI_INT, npout_eachrank.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif

  rec.template modify<DevExeSpace>();
  rec.template sync<HostMemSpace>();

  // copy the compacted records into the host output array
  Kokkos::realloc(outpart, std::max(1,npout), nrec);
  for (int p=0; p<npout; ++p) {
    for (int v=0; v<nrec; ++v) { outpart(p,v) = rec.h_view(p,v); }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TrackedParticleOutput:::WriteOutputFile(Mesh *pm)
//! \brief Appends one self-describing dump: a text header, then this rank's records at
//! its slot in the contiguous rank-ordered block.

void TrackedParticleOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  particles::Particles *pp = pm->pmb_pack->ppart;
  if (pp == nullptr) { return; }

  // total records across all ranks, and this rank's offset into the block
  std::vector<int> rank_offset(global_variable::nranks, 0);
  int npout_total = npout_eachrank[0];
  for (int n=1; n<global_variable::nranks; ++n) {
    rank_offset[n] = rank_offset[n-1] + npout_eachrank[n-1];
    npout_total += npout_eachrank[n];
  }

  // create filename: "trk/file_basename".trk
  std::string fname;
  fname.assign("trk/");
  fname.append(out_params.file_basename);
  fname.append(".trk");

  // Root process opens/creates file and appends the header for this dump
  if (global_variable::my_rank == 0) {
    // Column names, so a reader never has to know the particle type. nidata is (gid,tag)
    // for every species; the real columns follow the ParticlesIndex layout in athena.hpp.
    std::string cols;
    if (pp->nidata == 2) {
      cols = "gid tag";
    } else {
      for (int v=0; v<pp->nidata; ++v) { cols += (v?" i":"i") + std::to_string(v); }
    }
    if (pp->nrdata == NRDATA_SINK) {
      cols += " x vx y vy z vz m gx gy gz x0 y0 z0";
    } else {
      for (int v=0; v<pp->nrdata; ++v) { cols += " r" + std::to_string(v); }
    }
    std::stringstream msg;
    msg << "# AthenaK particle data" << std::endl
        << "# time=" << pm->time
        << " cycle=" << pm->ncycle
        << " nranks=" << global_variable::nranks
        << " nparticles=" << npout_total
        << " nidata=" << pp->nidata
        << " nrdata=" << pp->nrdata
        << " nrec=" << nrec
        << " dtype=Real" << std::endl
        << "# columns: " << cols << std::endl;
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
    }
    std::fprintf(pfile,"%s",msg.str().c_str());
    std::fclose(pfile);
  }
#if MPI_PARALLEL_ENABLED
  // every rank must see rank 0's header before it queries the file length below, or the
  // ranks disagree about where the binary block starts
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Now all ranks open the file and append their own block of records
  IOWrapper partfile;
  partfile.Open(fname.c_str(), IOWrapper::FileMode::append);
  std::size_t header_offset = partfile.GetPosition();

  if (npout > 0) {
    // flatten this rank's records
    std::vector<Real> data(static_cast<std::size_t>(npout)*nrec);
    for (int p=0; p<npout; ++p) {
      for (int v=0; v<nrec; ++v) { data[static_cast<std::size_t>(p)*nrec + v] = outpart(p,v); }
    }
    // contiguous, rank-ordered placement -- no assumption about tag values
    std::size_t myoffset = header_offset
                         + static_cast<std::size_t>(rank_offset[global_variable::my_rank])
                           *nrec*sizeof(Real);
    std::size_t cnt = static_cast<std::size_t>(npout)*nrec;
    if (partfile.Write_any_type_at(data.data(),cnt,myoffset,"Real") != cnt) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // close the output file and clean up
  partfile.Close();

  // increment counters
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
