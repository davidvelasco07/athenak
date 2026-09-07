//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//! \brief implementation of MHD class constructor and assorted functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    b0("B_fc",1,1,1,1),
    bcc0("B_cc",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    coarse_b0("cB_fc",1,1,1,1),
    u1("cons1",1,1,1,1,1),
    u_sts0("u_sts0",1,1,1,1,1),
    u_sts1("u_sts1",1,1,1,1,1),
    u_sts2("u_sts2",1,1,1,1,1),
    u_sts_rhs("u_sts_rhs",1,1,1,1,1),
    b1("B_fc1",1,1,1,1),
    b_sts0("b_sts0",1,1,1,1),
    b_sts1("b_sts1",1,1,1,1),
    b_sts2("b_sts2",1,1,1,1),
    b_sts_rhs("b_sts_rhs",1,1,1,1),
    b0_test("B_fc_test",1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    efld("efld",1,1,1,1),
    e3x1("e3x1",1,1,1,1),
    e2x1("e2x1",1,1,1,1),
    e1x2("e1x2",1,1,1,1),
    e3x2("e3x2",1,1,1,1),
    e2x3("e2x3",1,1,1,1),
    e1x3("e1x3",1,1,1,1),
    aL_x1f("aL_x1f",1,1,1,1),
    dL_x1f("dL_x1f",1,1,1,1),
    dR_x1f("dR_x1f",1,1,1,1),
    vy_x1f("vy_x1f",1,1,1,1),
    vz_x1f("vz_x1f",1,1,1,1),
    aL_x2f("aL_x2f",1,1,1,1),
    dL_x2f("dL_x2f",1,1,1,1),
    dR_x2f("dR_x2f",1,1,1,1),
    vx_x2f("vx_x2f",1,1,1,1),
    vz_x2f("vz_x2f",1,1,1,1),
    aL_x3f("aL_x3f",1,1,1,1),
    dL_x3f("dL_x3f",1,1,1,1),
    dR_x3f("dR_x3f",1,1,1,1),
    vx_x3f("vx_x3f",1,1,1,1),
    vy_x3f("vy_x3f",1,1,1,1),
    wl3d("wl3d",1,1,1,1,1),
    wr3d("wr3d",1,1,1,1,1),
    bl3d("bl3d",1,1,1,1,1),
    br3d("br3d",1,1,1,1,1),
    wsaved("wsaved",1,1,1,1,1),
    bccsaved("bccsaved",1,1,1,1,1),
    fofc("fofc",1,1,1,1),
    fofc_scal("fofc_scal",1,1,1,1,1),
    utest("utest",1,1,1,1,1),
    bcctest("bcctest",1,1,1,1,1),
    fb_level("fb_level",1,1,1,1),
    bmag_ref("bmag_ref",1,1,1,1),
    pmy_pack(ppack),
    e1_cc("e1_cc",1,1,1,1),
    e2_cc("e2_cc",1,1,1,1),
    e3_cc("e3_cc",1,1,1,1) {
  // defaults for members not always assigned in stationary path
  emf_method = MHD_EMF::ct_contact;
  uct_diag = false;
  mood_edge_flag = 0;

  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("mhd","eos");
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRMHD(ppack, pin);
    } else if (pmy_pack->pcoord->is_dynamical_relativistic) {
      // DynGRMHD uses PrimitiveSolver instead, so use a no-op here.
      peos = new NoOpDynGRMHD(ppack, pin);
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRMHD(ppack, pin);
    } else {
      peos = new IdealMHD(ppack, pin);
    }
    nmhd = 5;

  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
                <<"<mhd> eos = isothermal cannot be used with SR/GR"<< std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalMHD(ppack, pin);
      nmhd = 4;
    }

  // EOS string not recognized
  } else {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
              <<"<mhd> eos = '"<< eqn_of_state <<"' not implemented"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);

  // Viscosity (only constructed if needed)
  if (pin->DoesParameterExist("mhd","nu_iso") ||
      pin->DoesParameterExist("mhd","nu_aniso")) {
    pvisc = new Viscosity("mhd", ppack, pin);
    const bool active = (pvisc->nu_iso != 0.0 || pvisc->nu_aniso != 0.0);
    has_sts_viscosity = active &&
        (pvisc->mode == parabolic::DiffusionSelection::sts_only);
    has_explicit_viscosity = active &&
        (pvisc->mode == parabolic::DiffusionSelection::explicit_only);
    if (active) {
      ppack->RegisterParabolicProcess({"mhd/viscosity",
                                       parabolic::ParabolicProcessOwner::mhd,
                                       pvisc->mode, &(pvisc->dtnew)});
    }
  } else {
    pvisc = nullptr;
  }

  // Resistivity / non-ideal MHD (only constructed if needed). The Resistivity class now
  // also implements ambipolar diffusion, so it is constructed if either the Ohmic
  // coefficient (eta_ohm) or the ambipolar coefficient (eta_ad) is supplied in <mhd>.
  if (pin->DoesParameterExist("mhd","eta_ohm") ||
      pin->DoesParameterExist("mhd","eta_ad")) {
    presist = new Resistivity(ppack, pin);
    const bool active = (presist->eta_ohm != 0.0 || presist->eta_ad != 0.0);
    has_sts_resistivity = active &&
        (presist->mode == parabolic::DiffusionSelection::sts_only);
    has_explicit_resistivity = active &&
        (presist->mode == parabolic::DiffusionSelection::explicit_only);
    if (active) {
      ppack->RegisterParabolicProcess({"mhd/resistivity",
                                       parabolic::ParabolicProcessOwner::mhd,
                                       presist->mode, &(presist->dtnew)});
    }
  } else {
    presist = nullptr;
  }

  // Thermal conduction (only constructed if needed)
  if (pin->DoesParameterExist("mhd","alpha_iso") ||
      pin->DoesParameterExist("mhd","alpha_aniso") ||
      pin->DoesParameterExist("mhd","alpha_spitzer")) {
    if (peos->eos_data.is_ideal) {
      pcond = new Conduction("mhd", ppack, pin);
      const bool active = (pcond->alpha_iso != 0.0 || pcond->alpha_aniso != 0.0 ||
                           pcond->alpha_spitzer);
      has_sts_conduction = active &&
          (pcond->mode == parabolic::DiffusionSelection::sts_only);
      has_explicit_conduction = active &&
          (pcond->mode == parabolic::DiffusionSelection::explicit_only);
      if (active) {
        ppack->RegisterParabolicProcess({"mhd/conduction",
                                         parabolic::ParabolicProcessOwner::mhd,
                                         pcond->mode, &(pcond->dtnew)});
      }
    } else {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Thermal conduction in MHD requires ideal gas EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    pcond = nullptr;
  }

  has_any_sts_cell_update = (has_sts_viscosity || has_sts_conduction ||
                             (has_sts_resistivity && peos->eos_data.is_ideal));
  has_any_sts_field_update = has_sts_resistivity;
  has_any_sts_diffusion = (has_any_sts_cell_update || has_any_sts_field_update);

  // Source terms (if needed)
  if (pin->DoesBlockExist("mhd_srcterms")) {
    psrc = new SourceTerms("mhd_srcterms", ppack, pin);
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
    Kokkos::realloc(u0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(w0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);

    // allocate memory for face-centered and cell-centered magnetic fields
    Kokkos::realloc(bcc0,   nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nmhd+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb, (nmhd+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_b0.x1f, nmb, n_ccells3, n_ccells2, n_ccells1+1);
    Kokkos::realloc(coarse_b0.x2f, nmb, n_ccells3, n_ccells2+1, n_ccells1);
    Kokkos::realloc(coarse_b0.x3f, nmb, n_ccells3+1, n_ccells2, n_ccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) and face-centered variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers((nmhd+nscalars));
  pbval_b = new MeshBoundaryValuesFC(ppack, pin);
  pbval_b->InitializeBuffers(3);

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, (nmhd+nscalars));
    porb_b = new OrbitalAdvectionFC(ppack, pin);
    psbox_u = new ShearingBoxCC(ppack, pin, (nmhd+nscalars));
    psbox_b = new ShearingBoxFC(ppack, pin);
  } else {
    porb_u = nullptr;
    porb_b = nullptr;
    psbox_u = nullptr;
    psbox_b = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // determine if FOFC is enabled.  On the split-recon-rsolver branch the main flux
    // kernels extend their face-normal range by one cell when FOFC is on, so the
    // self-contained first-order flux correction (mhd_fofc.cpp) has the fluxes/EMFs it
    // needs over [is-1,ie+2] etc.
    use_fofc = pin->GetOrAddBoolean("mhd","fofc",false);

    // MOOD a-posteriori fallback (mhd_mood.cpp).
    use_mood = pin->GetOrAddBoolean("mhd","mood",false);
    std::string nadsc = pin->GetOrAddString("mhd","mood_nad_scale","gcfl");
    if (nadsc.compare("relative") == 0) {
      mood_nad_scale = 0;
    } else if (nadsc.compare("grange") == 0) {
      mood_nad_scale = 1;
    } else if (nadsc.compare("gdu") == 0) {
      mood_nad_scale = 2;
    } else if (nadsc.compare("gcfl") == 0) {
      mood_nad_scale = 3;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> mood_nad_scale = '" << nadsc
        << "' not implemented (relative|grange|gdu|gcfl)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    mood_eps0  = pin->GetOrAddReal("mhd","mood_eps0",1.0e-12);
    mood_rtol = pin->GetOrAddReal("mhd","mood_rtol",1.0e-5);
    mood_atol = pin->GetOrAddReal("mhd","mood_atol",0.0);
    mood_nad_theta = pin->GetOrAddReal("mhd","mood_nad_theta",1.0);
    mood_nad_energy = pin->GetOrAddBoolean("mhd","mood_nad_energy",true);
    mood_nad_scalars = pin->GetOrAddBoolean("mhd","mood_nad_scalars",true);
    if (mood_nad_theta < 0.0 || mood_nad_theta > 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> mood_nad_theta must be in [0,1]" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::string nadb = pin->GetOrAddString("mhd","mood_nad_b","comps");
    if (nadb.compare("mag") == 0) {
      mood_nad_b = 0;
    } else if (nadb.compare("comps") == 0) {
      mood_nad_b = 1;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> mood_nad_b = '" << nadb << "' not implemented (mag|comps)"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::string nadv = pin->GetOrAddString("mhd","mood_nad_v","comps");
    if (nadv.compare("off") == 0) {
      mood_nad_v = 0;
    } else if (nadv.compare("mag") == 0) {
      mood_nad_v = 1;
    } else if (nadv.compare("comps") == 0) {
      mood_nad_v = 2;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> mood_nad_v = '" << nadv
        << "' not implemented (off|mag|comps)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    uct_diag = pin->GetOrAddBoolean("mhd","uct_diag",false);
    // "auto": flag for UCT, blend for ct_contact (resolved after emf is known)
    std::string medge = pin->GetOrAddString("mhd","mood_edge","auto");
    if (medge.compare("auto") == 0) {
      mood_edge_flag = -1;
    } else if (medge.compare("blend") == 0) {
      mood_edge_flag = 0;
    } else if (medge.compare("flag") == 0) {
      mood_edge_flag = 1;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> mood_edge = '" << medge
        << "' not implemented (auto|blend|flag)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    mood_sed = pin->GetOrAddBoolean("mhd","mood_sed",true);
    if (use_mood && use_fofc) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> mood=true and fofc=true are mutually exclusive"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
      if (use_mood) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "MOOD with dc reconstruction has no fallback tier"
          << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm") == 0) {
      recon_method = ReconstructionMethod::ppm;
      if (!use_mood) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "reconstruct=ppm has no built-in limiting and "
          << "requires <mhd>/mood=true" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
      // check that nghost > 2 with PLM+FOFC
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (use_fofc && indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 3 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("teno") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // check that nghost > 3 with PPM4(or PPMX or WENOZ or TENO)+FOFC
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
      } else if (xorder.compare("teno") == 0) {
        recon_method = ReconstructionMethod::teno;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("mhd","rsolver");
    // Special relativistic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_sr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_gr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for GR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for GR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic dynamic solvers
    } else if (evolution_t.compare("dynamic") == 0) {
      // LLF solver
      if (rsolver.compare("llf") == 0) {
        rsolver_method = MHD_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = MHD_RSolver::hlle;
      // HLLD solver
      } else if (rsolver.compare("hlld") == 0) {
        rsolver_method = MHD_RSolver::hlld;
      // Roe solver
      // } else if (rsolver.compare("roe") == 0) {
      //   rsolver_method = MHD_RSolver::roe;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solver
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        rsolver_method = MHD_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // select EMF averaging method (default ct_contact).  uct_hll works for Newtonian,
    // SR and GR; uct_hlld is Newtonian-only because it needs rsolver=hlld.
    std::string emf_str = pin->GetOrAddString("mhd","emf","ct_contact");
    if (emf_str.compare("ct_contact") == 0) {
      emf_method = MHD_EMF::ct_contact;
    } else if (emf_str.compare("uct_hll") == 0) {
      emf_method = MHD_EMF::uct_hll;
    } else if (emf_str.compare("uct_hlld") == 0) {
      if (rsolver_method != MHD_RSolver::hlld) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/emf = 'uct_hlld' requires rsolver = 'hlld'"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      emf_method = MHD_EMF::uct_hlld;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/emf = '" << emf_str << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // uct_hll now works for SR and GR.  It previously drove the ENTIRE domain to the
    // density floor (measured: 27.2M floor hits on the 400^2 Minkowski magnetized
    // blast) because the relativistic Riemann solvers fed the UCT composition the
    // spatial FOUR-velocity where the induction equation needs the coordinate
    // transport velocity v^i = u^i/u^0 -- too large by the Lorentz factor and, in GR,
    // missing the lapse and shift.  See the note in each of the four relativistic
    // solvers' uct blocks.  The composition itself (Mignone & Del Zanna 2021 Eq. 33,
    // the form PLUTO uses for RMHD) needs no relativistic modification: the CT
    // machinery, the staggered B and the characteristic speeds are all already in the
    // coordinate frame, exactly as the ct_contact path assumes.
    //
    // uct_hlld remains Newtonian-only, and not by choice: it requires rsolver=hlld,
    // which is rejected outright for SR/GR dynamics.
    if (emf_method == MHD_EMF::uct_hlld &&
        (pmy_pack->pcoord->is_special_relativistic ||
         pmy_pack->pcoord->is_general_relativistic)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/emf = 'uct_hlld' is Newtonian-only (it needs"
                << " rsolver = 'hlld'); use 'uct_hll' or 'ct_contact' with SR/GR"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Ghost-zone requirements for UCT
    if (emf_method != MHD_EMF::ct_contact) {
      bool ho = !(recon_method == ReconstructionMethod::dc ||
                  recon_method == ReconstructionMethod::plm);
      int ng_need = ho ? 5 : 3;
      if (pmy_pack->pmesh->mb_indcs.ng < ng_need) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "UCT EMF with " << (ho ? "high-order" : "dc/plm")
                  << " reconstruction requires at least " << ng_need
                  << " ghost zones, but <mesh>/nghost="
                  << pmy_pack->pmesh->mb_indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // MOOD cascade / ghost requirements (ct_contact: h_f=1; UCT: wider)
    if (use_mood) {
      const bool is_uct = (emf_method != MHD_EMF::ct_contact);
      if (mood_edge_flag == -1) {
        mood_edge_flag = is_uct ? 1 : 0;
      } else if (mood_edge_flag == 1 && !is_uct) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<mhd> mood_edge=flag requires a UCT emf method (ct_contact "
          << "has no edge reconstruction to demote; use mood_edge=blend or auto)"
          << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (pmy_pack->pcoord->is_general_relativistic &&
          pmy_pack->pcoord->coord_data.bh_excise) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "MOOD with BH excision is not supported" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      n_fb_tiers = (recon_method == ReconstructionMethod::plm) ? 1 : 2;
      mood_max_revs = pin->GetOrAddInteger("mhd","mood_max_revs",n_fb_tiers);
      if (mood_max_revs < 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<mhd> mood_max_revs must be >= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      int h_f = 1;
      if (emf_method != MHD_EMF::ct_contact) {
        if (recon_method == ReconstructionMethod::dc) {
          h_f = 1;
        } else if (recon_method == ReconstructionMethod::plm ||
                   recon_method == ReconstructionMethod::ppm) {
          h_f = 2;
        } else {
          h_f = 3;  // wenoz/ppm4/ppmx/teno
        }
      }
      mood_halo0 = h_f + mood_max_revs - 1;
      int stencil = 2;
      if (recon_method == ReconstructionMethod::plm && !mood_sed) { stencil = 1; }
      int ng_need = mood_halo0 + 1 + stencil;
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < ng_need) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "MHD MOOD with " << xorder << " reconstruction, emf="
          << emf_str << " and mood_max_revs=" << mood_max_revs << " requires at least "
          << ng_need << " ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      mood_halo0 = 0;
      if (mood_edge_flag == -1) { mood_edge_flag = 0; }
    }

    // Final memory allocations
    {
      // allocate second registers
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(u1,     nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
      if (has_any_sts_cell_update) {
        Kokkos::realloc(u_sts0,    nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(u_sts1,    nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(u_sts2,    nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(u_sts_rhs, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
      }
      Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);
      if (has_any_sts_field_update) {
        Kokkos::realloc(b_sts0.x1f,    nmb, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(b_sts0.x2f,    nmb, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(b_sts0.x3f,    nmb, ncells3+1, ncells2, ncells1);
        Kokkos::realloc(b_sts1.x1f,    nmb, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(b_sts1.x2f,    nmb, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(b_sts1.x3f,    nmb, ncells3+1, ncells2, ncells1);
        Kokkos::realloc(b_sts2.x1f,    nmb, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(b_sts2.x2f,    nmb, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(b_sts2.x3f,    nmb, ncells3+1, ncells2, ncells1);
        Kokkos::realloc(b_sts_rhs.x1f, nmb, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(b_sts_rhs.x2f, nmb, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(b_sts_rhs.x3f, nmb, ncells3+1, ncells2, ncells1);
      }

      // allocate fluxes, electric fields
      Kokkos::realloc(uflx.x1f, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1+1);
      Kokkos::realloc(uflx.x2f, nmb, (nmhd+nscalars), ncells3, ncells2+1, ncells1);
      Kokkos::realloc(uflx.x3f, nmb, (nmhd+nscalars), ncells3+1, ncells2, ncells1);
      Kokkos::realloc(efld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
      Kokkos::realloc(efld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
      Kokkos::realloc(efld.x3e, nmb, ncells3, ncells2+1, ncells1+1);

      // allocate scratch arrays for face- and cell-centered E used in CornerE
      Kokkos::realloc(e3x1, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2x1, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1x2, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e3x2, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2x3, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1x3, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1_cc, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2_cc, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e3_cc, nmb, ncells3, ncells2, ncells1);

      // allocate global per-face L/R buffers for the split-kernel flux path.
      // Indexed by the GLOBAL cell/face index (m,n,k,j,i), so sized to the full
      // cell range (including ghost zones) in every dimension.  bl/br hold the
      // reconstructed cell-centered magnetic field (3 components).
      Kokkos::realloc(wl3d, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(wr3d, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(bl3d, nmb, 3, ncells3, ncells2, ncells1);
      Kokkos::realloc(br3d, nmb, 3, ncells3, ncells2, ncells1);

      // allocate UCT arrays if UCT method is selected
      if (emf_method == MHD_EMF::uct_hll || emf_method == MHD_EMF::uct_hlld) {
        Kokkos::realloc(aL_x1f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(dL_x1f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(dR_x1f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(vy_x1f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(vz_x1f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(aL_x2f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(dL_x2f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(dR_x2f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(vx_x2f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(vz_x2f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(aL_x3f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(dL_x3f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(dR_x3f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(vx_x3f, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(vy_x3f, nmb, ncells3, ncells2, ncells1);
      }

      // allocate arrays of flags/candidate state used with FOFC and MOOD
      if (use_fofc || use_mood) {
        // MOOD tests passive scalars too, so the candidate must carry them
        bool with_scalars = pmy_pack->pcoord->is_dynamical_relativistic ||
                            (use_mood && mood_nad_scalars);
        int nvars = (with_scalars) ? nmhd+nscalars : nmhd;
        Kokkos::realloc(fofc,    nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest,   nmb, nvars, ncells3, ncells2, ncells1);
        Kokkos::realloc(bcctest, nmb, 3,    ncells3, ncells2, ncells1);
        Kokkos::deep_copy(fofc, false);
        if (nscalars > 0) {
          Kokkos::realloc(fofc_scal,    nmb, nscalars, ncells3, ncells2, ncells1);
          Kokkos::deep_copy(fofc_scal, false);
        }
      }
      if (use_mood) {
        Kokkos::realloc(fb_level, nmb, ncells3, ncells2, ncells1);
        if (mood_nad_b == 0) {
          Kokkos::realloc(bmag_ref, nmb, ncells3, ncells2, ncells1);
        }
        if (emf_method == MHD_EMF::uct_hll || emf_method == MHD_EMF::uct_hlld) {
          Kokkos::realloc(b0_test.x1f, nmb, ncells3, ncells2, ncells1+1);
          Kokkos::realloc(b0_test.x2f, nmb, ncells3, ncells2+1, ncells1);
          Kokkos::realloc(b0_test.x3f, nmb, ncells3+1, ncells2, ncells1);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

MHD::~MHD() {
  if (psbox_b != nullptr) {delete psbox_b;}
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_b != nullptr) {delete porb_b;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_b;
  delete pbval_u;
  if (psrc!= nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (presist!= nullptr) {delete presist;}
  if (pvisc != nullptr) {delete pvisc;}
  delete peos;
}

//----------------------------------------------------------------------------------------
// SetSaveWBcc:  set flag to save primitives and cell-centered B field, e.g., for jcon

void MHD::SetSaveWBcc() {
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  // allocated saved arrays for time derivatives
  Kokkos::realloc(wsaved,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(bccsaved, nmb, 3,               ncells3, ncells2, ncells1);

  wbcc_saved = true;
}

} // namespace mhd
