#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.hpp
//! \brief defines MGGravity class

// C headers

// C++ headers

// C++ headers
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../parameter_input.hpp"
#include "../multigrid/multigrid.hpp"
#include "../coordinates/coordinates.hpp"
#include "mg_gravity.hpp"

class MeshBlockPack;
class ParameterInput;
class Coordinates;
class Multigrid;
namespace gravity {

//----------------------------------------------------------------------------------------
//! \struct GravSourceDensity
//! \brief one registered contribution to the gravitating mass density. `parr` points at
//! a source array member (e.g. hydro u0, particle-mesh dmesh) and is dereferenced at
//! assembly time so it survives reallocation (AMR); `comp` selects the component; the
//! contribution is `fac * (*parr)(m, comp, k, j, i)`.

struct GravSourceDensity {
  DvceArray5D<Real> *parr;
  int comp;
  Real fac;
};

class Gravity {
 public:
  Gravity(MeshBlockPack *pmbp, ParameterInput *pin);
  ~Gravity();

  MeshBlockPack* pmy_pack;
  DvceArray5D<Real> phi, coarse_phi;
  DvceArray5D<Real> def;
  // Assembled total gravitating mass density (the Poisson source). Filled by
  // AssembleSource() from the registered contributions; consumed by whichever
  // solver (multigrid now, FFT/FMM in future) computes phi. Keeping the source
  // here -- rather than letting the solver reach into hydro/particles -- keeps
  // the solver agnostic about what gravitates.
  DvceArray5D<Real> rho;
  Real four_pi_G;
  bool output_defect;
  bool fill_ghost;
  MGGravityDriver *pmgd;
  MGGravity *pmg;
  void SaveFaceBoundaries();
  void RestoreFaceBoundaries();

  // Push model: a gravitating module registers its density contribution once at
  // setup; AssembleSource() then sums all contributions into `rho` before each solve.
  void RegisterSourceDensity(DvceArray5D<Real> *parr, int comp, Real fac = 1.0);
  void AssembleSource();

  friend class MGGravityDriver;

 private:
  DvceArray5D<Real> fbuf_[6];
  std::vector<GravSourceDensity> source_terms_;
};
}  // namespace gravity
#endif // GRAVITY_GRAVITY_HPP_
