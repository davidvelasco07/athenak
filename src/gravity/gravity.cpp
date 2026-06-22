//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravity.cpp
//! \brief implementation of functions in class Gravity

// C headers

// C++ headers
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"
#include "../multigrid/multigrid.hpp"

namespace gravity { // NOLINT (build/namespace)
//! constructor, initializes data structures and parameters
//-------------------------------------------------------------------------------------
//! \fn Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Gravity constructor
Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin):
    pmy_pack(pmbp),
    phi("phi",1,1,1,1,1),
    coarse_phi("coarse",1,1,1,1,1),
    def("defect",1,1,1,1,1),
    rho("grav_rho",1,1,1,1,1),
    four_pi_G(-1.0),
    output_defect(false),
    fill_ghost(false) {
    four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G",-1.0);
    output_defect = pin->GetOrAddBoolean("gravity", "output_defect", false);
    fill_ghost = pin->GetOrAddBoolean("gravity", "fill_ghost", true);

    if (four_pi_G == 0.0) {
        std::cout << "### FATAL ERROR in Gravity::Gravity" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
        exit(EXIT_FAILURE);
    }

    // create multigrid driver/solver
    // The driver allocates multigrid instances for root level and meshblock levels
    pmgd = new MGGravityDriver(pmbp, pin);

    // Enroll CellCenteredBoundaryVariable object
    //gbvar.bvar_index = pmb->pbval->bvars.size();
    //pmb->pbval->bvars.push_back(&gbvar);
    //pmb->pbval->pgbvar = &gbvar;
    int nmb = pmy_pack->nmb_thispack;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(phi, nmb, 1, ncells3, ncells2, ncells1);
    Kokkos::realloc(rho, nmb, 1, ncells3, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------
//! \fn void Gravity::RegisterSourceDensity(DvceArray5D<Real> *parr, int comp, Real fac)
//! \brief Register a mass-density contribution to the gravitational source. Called once
//! at setup by each gravitating module (gas, particle-mesh, ...). The solver never needs
//! to know who contributes -- it only consumes the assembled `rho`.

void Gravity::RegisterSourceDensity(DvceArray5D<Real> *parr, int comp, Real fac) {
  source_terms_.push_back(GravSourceDensity{parr, comp, fac});
}

//----------------------------------------------------------------------------------------
//! \fn void Gravity::AssembleSource()
//! \brief Sum all registered density contributions into `rho`, the total gravitating mass
//! density. Run once per solve (ahead of the solver), after the contributing fields are
//! up to date (e.g. after the particle-mesh deposit). Solver-agnostic: multigrid, FFT, or
//! FMM all consume `rho`.

void Gravity::AssembleSource() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  // reallocate if AMR has changed the number of MeshBlocks in this pack
  if (static_cast<int>(rho.extent_int(0)) != nmb) {
    Kokkos::realloc(rho, nmb, 1, ncells3, ncells2, ncells1);
  }

  Kokkos::deep_copy(rho, 0.0);

  auto r = rho;
  for (auto &st : source_terms_) {
    auto src = *(st.parr);
    const int comp = st.comp;
    const Real fac = st.fac;
    par_for("GravAssembleSource", DevExeSpace(),
            0, nmb-1, 0, ncells3-1, 0, ncells2-1, 0, ncells1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      r(m, 0, k, j, i) += fac * src(m, comp, k, j, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn Gravity::~Gravity()
//! \brief Gravity destructor
Gravity::~Gravity() {
    delete pmg;
}
} // namespace gravity
