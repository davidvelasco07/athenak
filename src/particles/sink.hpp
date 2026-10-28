#ifndef PARTICLES_SINK_HPP_
#define PARTICLES_SINK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink.hpp
//! \brief storage layout of self-gravitating sink particles

namespace particles {
namespace sink {

// Positions first, as the common particle machinery requires (IPX/IPY/IPZ = 0/1/2):
//   IPX..IPZ    position
//   IPVX..IPVZ  velocity
//   IPM         mass
//   IPGX..IPGZ  gravitational acceleration gathered from the potential each stage
//   IPX0..IPZ0  start-of-step position (leapfrog: before the drift, read by the
//               accretion's cell-crossing test; rk3: the SSPRK u^n register). Contiguous,
//               so ReownByPosition can shift it with a periodic wrap (prev_pos_index).
//   IPVX0..     start-of-step velocity, rk3 only
// The leapfrog layout stores NREAL columns, the rk3 layout NREAL_RK3.
enum RealIndex {
  IPX=0, IPY=1, IPZ=2, IPVX=3, IPVY=4, IPVZ=5, IPM=6, IPGX=7, IPGY=8, IPGZ=9,
  IPX0=10, IPY0=11, IPZ0=12, NREAL=13,
  IPVX0=13, IPVY0=14, IPVZ0=15, NREAL_RK3=16
};
enum IntIndex {PGID=0, PTAG=1, PSTATUS=2, NINT=3};
inline constexpr const char *real_names[NREAL_RK3] = {
  "x", "y", "z", "vx", "vy", "vz", "m", "gx", "gy", "gz", "x0", "y0", "z0",
  "vx0", "vy0", "vz0"};
inline constexpr const char *int_names[NINT] = {"gid", "ptag", "status"};
inline constexpr bool real_output[NREAL_RK3] = {
  true, true, true, true, true, true, true, true, true, true, true, true, true,
  true, true, true};
inline constexpr bool int_output[NINT] = {false, true, true};
constexpr int RESTART_LAYOUT_VERSION = 1;

// Tags of sinks created during a run start here, above any seeded tag.
constexpr int CREATED_TAG_BASE = 1000000;

} // namespace sink
} // namespace particles

#endif // PARTICLES_SINK_HPP_
