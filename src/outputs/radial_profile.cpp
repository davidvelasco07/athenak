//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radial_profile.cpp
//! \brief writes normalized radial profiles with a text header and native binary payload

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "outputs.hpp"

namespace {
const char *field_names[RadialProfile::nfields] = {
  "shell_volume",
  "shell_mass",
  "density",
  "density_sq",
  "velocity_x",
  "velocity_y",
  "velocity_z",
  "velocity_xy",
  "velocity_xz",
  "velocity_yz",
  "velocity_x_sq",
  "velocity_y_sq",
  "velocity_z_sq",
  "velocity_mass_weighted_x",
  "velocity_mass_weighted_y",
  "velocity_mass_weighted_z",
  "velocity_mass_weighted_xy",
  "velocity_mass_weighted_xz",
  "velocity_mass_weighted_yz",
  "velocity_mass_weighted_x_sq",
  "velocity_mass_weighted_y_sq",
  "velocity_mass_weighted_z_sq",
  "angular_momentum_density_x",
  "angular_momentum_density_y",
  "angular_momentum_density_z",
  "velocity_1",
  "velocity_2",
  "velocity_3",
  "velocity_mass_weighted_1",
  "velocity_mass_weighted_2",
  "velocity_mass_weighted_3",
  "velocity_1_sq",
  "velocity_2_sq",
  "velocity_3_sq",
  "velocity_mass_weighted_1_sq",
  "velocity_mass_weighted_2_sq",
  "velocity_mass_weighted_3_sq",
  "mass_flux_in",
  "mass_flux_out",
  "bfield_x",
  "bfield_y",
  "bfield_z",
  "bfield_x_sq",
  "bfield_y_sq",
  "bfield_z_sq",
  "bfield_1",
  "bfield_2",
  "bfield_3",
  "bfield_1_sq",
  "bfield_2_sq",
  "bfield_3_sq",
  "potential_mass_weighted",
  "gravity_1",
  "gravity_mass_weighted_1",
  "fraction_negative_gravity_1",
  "rhoxgx",
  "rhoygy",
  "rhozgz",
  "enclosed_field_x",
  "enclosed_field_y",
  "enclosed_field_z",
  "magnetic_flux_upper",
  "magnetic_flux_lower",
  "mass_flux_xx",
  "mass_flux_yy",
  "mass_flux_zz",
  "mass_flux_xy",
  "mass_flux_xz",
  "mass_flux_yz",
  "mass_flux_yx",
  "mass_flux_zx",
  "mass_flux_zy",
};
const char *origin_names[] = {
  "velocity_x_origin", "velocity_y_origin", "velocity_z_origin"
};
}  // namespace

//----------------------------------------------------------------------------------------
RadialProfileOutput::RadialProfileOutput(ParameterInput *pin, Mesh *pm,
                                       OutputParameters op) : BaseTypeOutput(pin, pm, op) {
  const auto &size = pm->mesh_size;
  const Real rmax_default = 0.5*std::min({size.x1max - size.x1min,
                                          size.x2max - size.x2min,
                                          size.x3max - size.x3min});
  const Real rmax = pin->GetOrAddReal(op.block_name, "rmax", rmax_default);
  const int nsc = pin->GetOrAddInteger(op.block_name, "nbins_subcell_corrected", 4);
  const int nsub = pin->GetOrAddInteger(op.block_name, "nsub", 4);

  prp = new RadialProfile(pm, rmax, nsc, nsub);
  mkdir("rprof",0775);
}

RadialProfileOutput::~RadialProfileOutput() {
  delete prp;
}

void RadialProfileOutput::LoadOutputData(Mesh *pm) {
  const auto centers_device = pm->pgen->rprof_center_func(pm);
  const auto rprof_device = prp->Compute(centers_device);
  if (global_variable::my_rank == 0) {
    centers = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), centers_device);
    rprof = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rprof_device);
  }
}

void RadialProfileOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  char number[7];
  std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
  std::string fname = "rprof/" + out_params.file_basename + number + ".rprof";
  IOWrapper binfile;
  binfile.Open(fname.c_str(), IOWrapper::FileMode::write);

  if (global_variable::my_rank == 0) {
    const bool mhd = pm->pmb_pack->pmhd != nullptr;
    const bool gravity = pm->pmb_pack->pgrav != nullptr;
    std::vector<int> fields;
    for (int field=0; field<RadialProfile::nfields; ++field) {
      if (!mhd && ((field >= RadialProfile::bfield_x &&
                    field <= RadialProfile::bfield_3_sq) ||
                   (field >= RadialProfile::enclosed_field_x &&
                    field <= RadialProfile::magnetic_flux_lower))) continue;
      if (!gravity && field >= RadialProfile::potential_mass_weighted &&
                      field <= RadialProfile::rhozgz) continue;
      fields.push_back(field);
    }
    const std::size_t ncenter = rprof.extent(0);
    const std::size_t nr = rprof.extent(2);
    const std::size_t nvariable = 3 + fields.size();
    const int feature_flags = (mhd ? 1 : 0) | (gravity ? 2 : 0);
    const int time_precision = std::numeric_limits<Real>::max_digits10 - 1;
    std::stringstream msg;
    msg << "Athena radial profile output version=1.0" << std::endl
        // Includes this line and the metadata preceding "number of variables".
        << "  size of preheader=9" << std::endl
        << std::scientific << std::setprecision(time_precision)
        << "  time=" << pm->time << std::endl
        << "  cycle=" << pm->ncycle << std::endl
        << "  output number=" << out_params.file_number << std::endl
        << "  feature flags=" << feature_flags << std::endl
        << "  nsub=" << prp->nsub_ << std::endl
        << "  nbins_subcell_corrected=" << prp->nbins_subcell_corrected_ << std::endl
        << "  number of centers=" << ncenter << std::endl
        << "  number of radial bins=" << nr << std::endl
        << "  number of variables=" << nvariable << std::endl
        << "  variables:  ";
    for (std::size_t n=0; n<nvariable; ++n) {
      msg << (n < 3 ? origin_names[n] : field_names[fields[n-3]]) << "  ";
    }
    msg << std::endl;
    binfile.Write_any_type(msg.str().c_str(), msg.str().size(), "byte");

    // Native uint64 IDs and double coordinates, without struct padding.
    std::vector<char> data(ncenter*sizeof(std::uint64_t) + (3*ncenter + nr)*sizeof(double));
    char *pdata = data.data();
    for (std::size_t c=0; c<ncenter; ++c) {
      const std::uint64_t id = centers(c).id;
      std::memcpy(pdata, &id, sizeof(id));
      pdata += sizeof(id);
    }
    for (std::size_t c=0; c<ncenter; ++c) {
      const double position[] = {centers(c).x1, centers(c).x2, centers(c).x3};
      std::memcpy(pdata, position, sizeof(position));
      pdata += sizeof(position);
    }
    for (std::size_t k=0; k<nr; ++k) {
      const double radius = k*double(pm->mesh_size.dx1);
      std::memcpy(pdata, &radius, sizeof(radius));
      pdata += sizeof(radius);
    }
    binfile.Write_any_type(data.data(), data.size(), "byte");

    // The first three variables are center-only; radial rows are center-major.
    for (std::size_t n=0; n<nvariable; ++n) {
      const std::size_t data_size = ncenter*(n < 3 ? 1 : nr)*sizeof(double);
      data.resize(data_size);
      pdata = data.data();
      for (std::size_t c=0; c<ncenter; ++c) {
        if (n < 3) {
          const double velocity[] = {centers(c).vx, centers(c).vy, centers(c).vz};
          std::memcpy(pdata, &velocity[n], sizeof(double));
          pdata += sizeof(double);
        } else {
          for (std::size_t k=0; k<nr; ++k) {
            const double value = rprof(c,fields[n-3],k);
            std::memcpy(pdata, &value, sizeof(value));
            pdata += sizeof(value);
          }
        }
      }
      binfile.Write_any_type(data.data(), data_size, "byte");
    }
  }
  binfile.Close();

  // Match MeshBinaryOutput's native counter and schedule bookkeeping.
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
