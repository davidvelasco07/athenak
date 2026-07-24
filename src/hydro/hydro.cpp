//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of Hydro class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    u1("cons1",1,1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    wl_split("wl_split",1,1,1,1,1),
    wr_split("wr_split",1,1,1,1,1),
    utest("utest",1,1,1,1,1),
    fofc("fofc",1,1,1,1),
    fb_level("fb_level",1,1,1,1) {
  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("hydro","eos");
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRHydro(ppack, pin);
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRHydro(ppack, pin);
    } else {
      peos = new IdealHydro(ppack, pin);
    }
    nhydro = 5;
  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "<hydro>/eos = isothermal cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalHydro(ppack, pin);
      nhydro = 4;
    }
  // EOS string not recognized
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro>/eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("hydro","nscalars",0);

  // Viscosity (if requested in input file)
  if (pin->DoesParameterExist("hydro","viscosity")) {
    pvisc = new Viscosity("hydro", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Thermal conduction (if requested in input file)
  if (pin->DoesParameterExist("hydro","conductivity") ||
      pin->DoesParameterExist("hydro","tdep_conductivity")) {
    pcond = new Conduction("hydro", ppack, pin);
  } else {
    pcond = nullptr;
  }

  // Source terms (if needed)
  if (pin->DoesBlockExist("hydro_srcterms")) {
    psrc = new SourceTerms("hydro_srcterms", ppack, pin);
  }

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for conserved and primitive variables
  // With AMR, maximum size of Views are limited by total device memory through an input
  // parameter, which in turn limits max number of MBs that can be created.
  {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(w0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nhydro+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb, (nhydro+nscalars), n_ccells3, n_ccells2, n_ccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers((nhydro+nscalars));

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, (nhydro+nscalars));
    psbox_u = new ShearingBoxCC(ppack, pin, (nhydro+nscalars));
  } else {
    porb_u = nullptr;
    psbox_u = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // The split-kernel flux path supports every reconstruction method (DC/PLM/PPM4/
    // PPMX/WENOZ) and all Newtonian (Advect/LLF/HLLE/HLLC/Roe) and relativistic
    // (LLF/HLLE/HLLC SR; LLF/HLLE GR) Riemann solvers.

    // FOFC: the split-kernel main flux kernels extend their face-normal range by one cell
    // when FOFC is enabled, so the self-contained first-order flux correction
    // (hydro_fofc.cpp) has the fluxes it needs over the [is-1,ie+2] etc. range.
    use_fofc = pin->GetOrAddBoolean("hydro","fofc",false);

    // MOOD a-posteriori fallback (hydro_mood.cpp).  Requires a flux-range extension of
    // mood_max_revs+1 cells (see hydro_fluxes.cpp).  Mutually exclusive with FOFC.
    use_mood = pin->GetOrAddBoolean("hydro","mood",false);
    // NAD tolerance: the DMP bounds are relaxed by eps = max(rtol*SCALE, atol,
    // eps0*|bound|), where SCALE is selected by <hydro>/mood_nad_scale:
    //   "relative" — |bound| itself (the spd modular-gpu form; Mach-blind)
    //   "grange"   — GLOBAL dynamic range of the NAD variable: Mach/resolution-aware
    //                (collapses ~M^2 at low Mach) and uniform across cells
    //   "gdu"      — GLOBAL max per-stage change |u*-u| of the NAD variable, frozen
    //                from the first (pre-revision) candidate each stage: additionally
    //                timestep-aware and stratification-immune (hydrostatic backgrounds
    //                cancel in the per-step difference)
    //   "gcfl"     — grange scaled by the global advective Courant number of the
    //                step (see below): Mach-invariant detection counts.  DEFAULT.
    // All global scales are uniform across cells (single scalar), so detection stays
    // decomposition-invariant.  eps0 is a round-off floor; atol an absolute floor.
    std::string nadsc = pin->GetOrAddString("hydro","mood_nad_scale","gcfl");
    if (nadsc.compare("relative") == 0) {
      mood_nad_scale = 0;
    } else if (nadsc.compare("grange") == 0) {
      mood_nad_scale = 1;
    } else if (nadsc.compare("gdu") == 0) {
      mood_nad_scale = 2;
    } else if (nadsc.compare("gcfl") == 0) {
      // grange scaled by the global advective Courant number: eps = rtol*G*min(1,
      // dt*|v|max/dx).  The CFL_adv factor converts the amplitude scale G into a
      // per-stage dynamics scale, keeping detection counts approximately Mach-
      // invariant (at low Mach the acoustic dt shrinks the per-stage dynamics while
      // grid noise does not; a pure-G tolerance then over- or under-counts).
      mood_nad_scale = 3;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<hydro> mood_nad_scale = '" << nadsc
        << "' not implemented (relative|grange|gdu|gcfl)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Default rtol (calibrated for gcfl): 1e-5 is robust (Sod TV 0.91, implosion
    // dissipation at or below WENOZ, clean RTI at Mach 0.04-0.4 across resolutions)
    // while keeping detection unambiguous.  Strict DMP (rtol=0) gives the identical
    // solution but breaks decomposition invariance and symmetry: at eps->0 marginal
    // detections sit on a round-off fence, so the tiny finite tolerance acts purely
    // as a margin regularizer.  Looser (1e-4..1e-3) admits sub-tolerance ringing.
    mood_eps0  = pin->GetOrAddReal("hydro","mood_eps0",1.0e-12);
    mood_rtol = pin->GetOrAddReal("hydro","mood_rtol",1.0e-5);
    mood_atol = pin->GetOrAddReal("hydro","mood_atol",0.0);
    // Mach-response softening of the grange scale: eps = rtol * G^theta * |bound|^(1-theta).
    // theta=1 is pure grange; theta=0 is the Mach-blind relative form; intermediate
    // values tighten sublinearly at low Mach.  Only used by mood_nad_scale=grange
    // (gcfl handles the Mach response through the Courant factor instead).
    mood_nad_theta = pin->GetOrAddReal("hydro","mood_nad_theta",1.0);
    // include the energy variable in NAD (density is always checked).  The energy
    // channel's noise amplitude scales with the sound speed (acoustic), so it can
    // break the Mach-invariance of detection counts that the G scaling provides.
    mood_nad_energy = pin->GetOrAddBoolean("hydro","mood_nad_energy",true);
    if (mood_nad_theta < 0.0 || mood_nad_theta > 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<hydro> mood_nad_theta must be in [0,1]" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    mood_sed = pin->GetOrAddBoolean("hydro","mood_sed",true);
    if (use_mood && use_fofc) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<hydro> mood=true and fofc=true cannot be used together"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (use_mood && pmy_pack->pcoord->is_general_relativistic &&
        pmy_pack->pcoord->coord_data.bh_excise) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<hydro> mood=true does not support BH excision (use fofc)"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
      if (use_mood) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<hydro> mood=true cannot be used with dc reconstruction "
          << "(no fallback tier below first order)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
      // check that nghost > 2 with PLM+FOFC (FOFC extends recon by one cell)
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (use_fofc && indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 3 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0 ||
               xorder.compare("ppm") == 0) {
      // check that nghost > 2 (the +/-2 stencil requires at least 3 ghost zones)
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // check that nghost > 3 with PPM4(or PPMX or WENOZ)+FOFC
      if (use_fofc && indcs.ng < 4) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 4 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      } else if (xorder.compare("ppm") == 0) {
        recon_method = ReconstructionMethod::ppm;
        // unlimited interpolation is unstable without the a-posteriori fallback
        if (!use_mood) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "<hydro> reconstruct=ppm (unlimited) requires "
            << "mood=true" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> reconstruct = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // MOOD cascade: base -> plm -> dc, or plm -> dc if the base scheme is already PLM.
    // Ghost-zone requirement: detection in the first revision iteration covers a halo
    // of mood_max_revs ghost cells (the halo shrinks by one per iteration so that all
    // decisions near MeshBlock boundaries are made from data both blocks share — this
    // keeps the scheme conservative and decomposition-invariant).  The base fluxes are
    // therefore computed mood_max_revs cells beyond the active domain, which needs
    // stencil_width + mood_max_revs ghost zones (the SED +/-3 stencil needs
    // 3 + mood_max_revs, which is never larger).
    if (use_mood) {
      n_fb_tiers = (recon_method == ReconstructionMethod::plm) ? 1 : 2;
      mood_max_revs = pin->GetOrAddInteger("hydro","mood_max_revs",n_fb_tiers);
      if (mood_max_revs < 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<hydro> mood_max_revs must be >= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      int ng_need = ((recon_method == ReconstructionMethod::plm) ? 2 : 3)
                    + mood_max_revs;
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < ng_need) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "MOOD with " << xorder << " reconstruction and mood_max_revs="
          << mood_max_revs << " requires at least " << ng_need
          << " ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("hydro","rsolver");
    // Special relativistic dynamic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_sr;
        } else if (rsolver.compare("hllc") == 0) {
          rsolver_method = Hydro_RSolver::hllc_sr;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic dynamic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_gr;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for GR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for GR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic dynamic solvers
    } else if (evolution_t.compare("dynamic") == 0) {
      if (rsolver.compare("llf") == 0) {
        rsolver_method = Hydro_RSolver::llf;
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = Hydro_RSolver::hlle;
      } else if (rsolver.compare("hllc") == 0) {
        if (peos->eos_data.is_ideal) {
          rsolver_method = Hydro_RSolver::hllc;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = hllc cannot be used with "
                    << "isothermal EOS" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else if (rsolver.compare("roe") == 0) {
        rsolver_method = Hydro_RSolver::roe;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solvers
    } else {
      if (rsolver.compare("advect") == 0) {
        rsolver_method = Hydro_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // Final memory allocations
    {
      // allocate second registers, fluxes
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(u1,       nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x1f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x2f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x3f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);

      // allocate global per-face L/R buffers for the split-kernel flux path.
      // Indexed by the GLOBAL cell/face index (m,n,k,j,i), so sized to the full
      // cell range (including ghost zones) in every dimension.
      Kokkos::realloc(wl_split, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(wr_split, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);

      // allocate arrays of flags/candidate state used with FOFC and MOOD
      if (use_fofc || use_mood) {
        Kokkos::realloc(fofc,  nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest, nmb, nhydro, ncells3, ncells2, ncells1);
        Kokkos::deep_copy(fofc, false);
      }
      if (use_mood) {
        Kokkos::realloc(fb_level, nmb, ncells3, ncells2, ncells1);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

Hydro::~Hydro() {
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_u;
  if (psrc != nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (pvisc != nullptr) {delete pvisc;}
  delete peos;
}

} // namespace hydro
