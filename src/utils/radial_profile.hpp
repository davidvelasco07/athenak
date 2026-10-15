#ifndef UTILS_RADIAL_PROFILE_HPP_
#define UTILS_RADIAL_PROFILE_HPP_

#include <cstdint>

#include "athena.hpp"

class Mesh;

struct RadialProfileCenter {
  std::uint64_t id;
  Real x1, x2, x3;
  Real vx, vy, vz;
};


// Radial shell averages on uniform, cubic-cell, periodic Cartesian meshes.
// The caller supplies identical ordered centers, including current center-cell
// velocities, on every MPI rank.
// Profile centers must be mesh cell centers. Setting nsub even number avoids coordinate
// singularity at the origin. Outer polar-axis cells use the same subdivision to average
// spherical projections and their moments within each cell, retaining the parent radial
// bin and weight. The caller must ensure rmax <= (half the shortest box length).
// Optional gravity/MHD slots are zero when absent; the writer omits them.

class RadialProfile {
  friend class RadialProfileOutput;

 public:
  enum Field {
    shell_volume = 0,
    shell_mass = 1,
    density = 2,
    density_sq = 3,
    velocity_x = 4,
    velocity_y = 5,
    velocity_z = 6,
    velocity_xy = 7,
    velocity_xz = 8,
    velocity_yz = 9,
    velocity_x_sq = 10,
    velocity_y_sq = 11,
    velocity_z_sq = 12,
    velocity_mass_weighted_x = 13,
    velocity_mass_weighted_y = 14,
    velocity_mass_weighted_z = 15,
    velocity_mass_weighted_xy = 16,
    velocity_mass_weighted_xz = 17,
    velocity_mass_weighted_yz = 18,
    velocity_mass_weighted_x_sq = 19,
    velocity_mass_weighted_y_sq = 20,
    velocity_mass_weighted_z_sq = 21,
    angular_momentum_density_x = 22,
    angular_momentum_density_y = 23,
    angular_momentum_density_z = 24,
    velocity_1 = 25,
    velocity_2 = 26,
    velocity_3 = 27,
    velocity_mass_weighted_1 = 28,
    velocity_mass_weighted_2 = 29,
    velocity_mass_weighted_3 = 30,
    velocity_1_sq = 31,
    velocity_2_sq = 32,
    velocity_3_sq = 33,
    velocity_mass_weighted_1_sq = 34,
    velocity_mass_weighted_2_sq = 35,
    velocity_mass_weighted_3_sq = 36,
    mass_flux_in = 37,
    mass_flux_out = 38,
    bfield_x = 39,
    bfield_y = 40,
    bfield_z = 41,
    bfield_x_sq = 42,
    bfield_y_sq = 43,
    bfield_z_sq = 44,
    bfield_1 = 45,
    bfield_2 = 46,
    bfield_3 = 47,
    bfield_1_sq = 48,
    bfield_2_sq = 49,
    bfield_3_sq = 50,
    potential_mass_weighted = 51,
    gravity_1 = 52,
    gravity_mass_weighted_1 = 53,
    fraction_negative_gravity_1 = 54,
    rhoxgx = 55,
    rhoygy = 56,
    rhozgz = 57,
    enclosed_field_x = 58,
    enclosed_field_y = 59,
    enclosed_field_z = 60,
    magnetic_flux_upper = 61,  // upper outward flux (B*length^2)
    magnetic_flux_lower = 62,  // minus lower outward flux (B*length^2)
    // Volume-weighted rho*v_i*rhat_j moments, with center-relative velocities.
    // First index: velocity; second: radial direction. No symmetrization or area factor.
    mass_flux_xx = 63,
    mass_flux_yy = 64,
    mass_flux_zz = 65,
    mass_flux_xy = 66,
    mass_flux_xz = 67,
    mass_flux_yz = 68,
    mass_flux_yx = 69,
    mass_flux_zx = 70,
    mass_flux_zy = 71,
    nfields = 72
  };
  // Optional fenced phase measurements. Disabled for ordinary/correctness calls.
  bool measure_time = false;
  struct Timings {
    double allocation = 0, accumulation = 0, normalization = 0;
    double reduction = 0, flux = 0, total = 0;
  } timings;
  struct SubcellParent {int c, m, k, j, i;};

  RadialProfile(Mesh *mesh, Real rmax, int nbins_subcell = 4, int nsub = 4);
  DvceArray3D<Real> Compute(const DvceArray1D<const RadialProfileCenter>& centers);
  void AccumulateShells(
      DvceArray3D<Real> &rprof,
      DvceArray1D<SubcellParent> &subcell_parents,
      Kokkos::DualView<int> &parent_count,
      const DvceArray1D<const RadialProfileCenter> &centers);
  void NormalizeProfiles(DvceArray3D<Real> &rprof);
  void CalculateMagneticFlux(
      DvceArray3D<Real> &rprof,
      const DvceArray1D<SubcellParent> &subcell_parents,
      const Kokkos::DualView<int> &parent_count,
      const DvceArray1D<const RadialProfileCenter> &centers);

 private:
  Mesh *mesh_;
  const Real dr_;
  const int nbins_;
  const int nbins_subcell_corrected_;
  const int nsub_;
};
#endif  // UTILS_RADIAL_PROFILE_HPP_
