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
#include "eos/mhd_bcc.hpp"
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
    b_eq("b_eq",1,1,1,1),
    w_eq_face("w_eq_face",1,1,1,1,1),
    bcc_eq_face("bcc_eq_face",1,1,1,1,1),
    f_eq("f_eq",1,1,1,1,1),
    pmy_pack(ppack),
    e1_cc("e1_cc",1,1,1,1),
    e2_cc("e2_cc",1,1,1,1),
    e3_cc("e3_cc",1,1,1,1) {
  // well-balanced arrays: default-constructed at size 1, reallocated only when used
  u_eq = DvceArray5D<Real>("u_eq",1,1,1,1,1);
  w_eq = DvceArray5D<Real>("w_eq",1,1,1,1,1);
  bcc_eq = DvceArray5D<Real>("bcc_eq",1,1,1,1,1);
  w_pert = DvceArray5D<Real>("w_pert",1,1,1,1,1);
  bcc_pert = DvceArray5D<Real>("bcc_pert",1,1,1,1,1);
  e3x1_eq = DvceArray4D<Real>("e3x1_eq",1,1,1,1);
  e2x1_eq = DvceArray4D<Real>("e2x1_eq",1,1,1,1);
  e1x2_eq = DvceArray4D<Real>("e1x2_eq",1,1,1,1);
  e3x2_eq = DvceArray4D<Real>("e3x2_eq",1,1,1,1);
  e2x3_eq = DvceArray4D<Real>("e2x3_eq",1,1,1,1);
  e1x3_eq = DvceArray4D<Real>("e1x3_eq",1,1,1,1);
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
    // NAD uses only each cell's neighboring bounds; no domain-wide scales.
    const std::string nadsc = pin->GetOrAddString("mhd","mood_nad_scale","relative");
    if (nadsc != "relative" && nadsc != "rcfl") {
      std::cerr << "<mhd> mood_nad_scale='" << nadsc
                << "' is unsupported; use relative or rcfl with mood_rtol/mood_atol\n";
      std::exit(EXIT_FAILURE);
    }
    mood_nad_rcfl = (nadsc == "rcfl");
    if (pin->DoesParameterExist("mhd", "mood_nad_theta")) {
      std::cerr << "<mhd> mood_nad_theta was removed with the global NAD scales\n";
      std::exit(EXIT_FAILURE);
    }
    mood_eps0  = pin->GetOrAddReal("mhd","mood_eps0",1.0e-12);
    // well-balanced round-off floor relative to the equilibrium (see hydro.cpp)
    mood_eps0_eq = pin->GetOrAddReal("mhd","mood_eps0_eq",0.0);
    mood_rtol = pin->GetOrAddReal("mhd","mood_rtol",
                                    mood_nad_rcfl ? 1.0e-4 : 1.0e-5);
    mood_atol = pin->GetOrAddReal("mhd","mood_atol",0.0);
    mood_nad_energy = pin->GetOrAddBoolean("mhd","mood_nad_energy",true);
    mood_nad_scalars = pin->GetOrAddBoolean("mhd","mood_nad_scalars",true);

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
    // fvN compares magnetic volume averages on both sides of NAD, consistently with
    // PAD and production conversion. For the other methods, "avg" retains the
    // two-face NAD average and "ho" uses bcc0 as the reference. Both agree under
    // ct_contact. The input names are retained for existing decks.
    mood_nad_bcc_avg =
        (pin->GetOrAddString("mhd","mood_nad_bcc","avg").compare("avg") == 0);
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
    // Trial state the detector judges at every RK stage.  "stage" (default): the RK stage
    // combination gam0*u + gam1*u_n + beta*dt*L(u), the state the update actually writes.
    // "euler": a full-step forward-Euler predictor from the stage-input state,
    // u + dt*L(u), so every stage is tested on what a whole step of its fluxes would do;
    // stricter (13-48% more demotions on Orszag-Tang and Shu-Osher, and less accurate on
    // those and on a smooth CPAW, 2026-10-28).  Only the trial state changes; the
    // accepted RK update is the same either way.
    {
      const std::string pred = pin->GetOrAddString("mhd","mood_predictor","stage");
      if (pred != "euler" && pred != "stage") {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/mood_predictor = '" << pred
                  << "' not implemented (euler|stage)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      mood_euler_predictor = (pred == "euler");
    }
    // Diffusion fluxes vs the fallback: see the same parameter in hydro.cpp (default
    // false = inviscid candidate).  The resistive energy flux has no face-local form, so
    // true is refused with resistivity.
    const bool diff_readd_ok =
        ((pcond == nullptr) || (pcond->alpha_aniso == 0.0 && !pcond->alpha_spitzer)) &&
        !(presist != nullptr && has_explicit_resistivity && peos->eos_data.is_ideal);
    mood_candidate_diffusion =
        pin->GetOrAddBoolean("mhd","mood_candidate_diffusion", false);
    if (mood_candidate_diffusion && !diff_readd_ok && (use_mood || use_fofc)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> mood_candidate_diffusion=true needs face-local diffusion "
        << "fluxes: not available for anisotropic/Spitzer conduction or resistivity"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Restrict revisions >= 2 to the cells a revised face can reach (mhd_mood.cpp).
    // Same answer bit for bit; skips the candidate update, the floor test (the GR
    // inversion) and the detection everywhere else.
    mood_touched_only = pin->GetOrAddBoolean("mhd","mood_touched_only",false);
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
    } else if (xorder.compare("fv4") == 0 || xorder.compare("fv5") == 0 ||
               xorder.compare("fv6") == 0 || xorder.compare("fv7") == 0 ||
               xorder.compare("fv8") == 0 || xorder.compare("fv9") == 0) {
      // Unlimited cell-polynomial reconstruction (fv.hpp); no limiter of its own.
      if (xorder.compare("fv4") == 0) {
        recon_method = ReconstructionMethod::fv4;
      } else if (xorder.compare("fv5") == 0) {
        recon_method = ReconstructionMethod::fv5;
      } else if (xorder.compare("fv6") == 0) {
        recon_method = ReconstructionMethod::fv6;
      } else if (xorder.compare("fv7") == 0) {
        recon_method = ReconstructionMethod::fv7;
      } else if (xorder.compare("fv8") == 0) {
        recon_method = ReconstructionMethod::fv8;
      } else {
        recon_method = ReconstructionMethod::fv9;
      }
      if (!use_mood) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "reconstruct=" << xorder
          << " has no built-in limiting and requires <mhd>/mood=true" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ng_base = ReconStencilRadius(recon_method) + 1;
      if (indcs.ng < ng_base) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least " << ng_base
          << " ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm") == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> reconstruct=ppm was renamed to fv4 "
        << "(unlimited 4th-order; also available: fv5 .. fv9)" << std::endl;
      std::exit(EXIT_FAILURE);
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
      // L-HLLD: low-Mach + carbuncle-controlled HLLD (Minoshima & Miyoshi 2021)
      } else if (rsolver.compare("lhlld") == 0) {
        rsolver_method = MHD_RSolver::lhlld;
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

    // select EMF averaging method (default ct_contact).  'uct' takes the UCT variant
    // that matches the Riemann solver: HLLD-type coefficients (from the five-wave fan)
    // with hlld/lhlld, HLL-type (from the outer speeds) with every other solver.
    // 'uct_hll' / 'uct_hlld' pin a variant explicitly: uct_hll works for Newtonian, SR
    // and GR; uct_hlld is Newtonian-only because it needs rsolver=hlld/lhlld.
    std::string emf_str = pin->GetOrAddString("mhd","emf","ct_contact");
    const bool hlld_fan = (rsolver_method == MHD_RSolver::hlld ||
                           rsolver_method == MHD_RSolver::lhlld);
    if (emf_str.compare("ct_contact") == 0) {
      emf_method = MHD_EMF::ct_contact;
    } else if (emf_str.compare("uct") == 0) {
      emf_method = hlld_fan ? MHD_EMF::uct_hlld : MHD_EMF::uct_hll;
    } else if (emf_str.compare("uct_hll") == 0) {
      emf_method = MHD_EMF::uct_hll;
    } else if (emf_str.compare("uct_hlld") == 0) {
      // the HLLD-type UCT coefficients come from the five-wave fan: hlld or lhlld
      if (!hlld_fan) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/emf = 'uct_hlld' requires rsolver = 'hlld' "
                  << "or 'lhlld'" << std::endl;
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
    // Use frozen candidates by default for the validated uniform periodic case.
    // An explicit false keeps the original recomposition path for comparisons.
    const bool frozen_supported = use_mood && evolution_t == "dynamic" &&
        emf_method != MHD_EMF::ct_contact && mood_edge_flag != 0 &&
        !mood_touched_only && pmy_pack->pmesh->multi_d &&
        !pmy_pack->pmesh->multilevel && pmy_pack->pmesh->strictly_periodic &&
        peos->eos_data.is_ideal && nscalars == 0 && mood_nad_bcc_avg &&
        pvisc == nullptr && presist == nullptr && pcond == nullptr &&
        !pin->DoesBlockExist("shearing_box") &&
        !pin->GetOrAddBoolean("mhd", "well_balanced", false) &&
        !(pmy_pack->pcoord->is_general_relativistic &&
          pmy_pack->pcoord->coord_data.bh_excise) &&
        IsUnlimitedFV(recon_method);
    mood_frozen_emf = pin->GetOrAddBoolean("mhd", "mood_frozen_emf", frozen_supported);
    if (mood_frozen_emf && !frozen_supported) {
      std::cerr << "Frozen EMF requires dynamic MOOD with periodic uniform "
                << "multidimensional UCT, FV4..FV9, ideal EOS, mood_nad_bcc=avg, "
                << "mood_edge=flag/auto, mood_touched_only=false, and no diffusion, "
                << "scalars, shear, excision or well balance\n";
      std::exit(EXIT_FAILURE);
    }
    if (mood_frozen_emf) {
      const auto &q = pmy_pack->pmesh->mb_indcs;
      if (q.ng > q.nx1 || q.ng > q.nx2 ||
          (pmy_pack->pmesh->three_d && q.ng > q.nx3)) {
        std::cerr << "Frozen EMF neighbor exchange requires block dimensions >= nghost\n";
        std::exit(EXIT_FAILURE);
      }
    }

    // Frozen base faces are reconstructed normally; edge composition samples them
    // transversely. These supports lie on different axes, so take their maximum.
    // L-HLLD also needs transverse velocity neighbors for its shock sensor.
    const int frozen_ng_need = std::max(3, ReconEdgeReach(recon_method) +
        (rsolver_method == MHD_RSolver::lhlld ? 1 : 0));
    if (emf_method != MHD_EMF::ct_contact) {
      int ng_need = ReconEdgeReach(recon_method) + ReconStencilRadius(recon_method);
      if (mood_frozen_emf) ng_need = frozen_ng_need;
      if (ng_need < 3) { ng_need = 3; }
      if (pmy_pack->pmesh->mb_indcs.ng < ng_need) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "UCT EMF with " << xorder
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
      n_fb_tiers = (recon_method == ReconstructionMethod::plm) ? 1 : 2;
      mood_max_revs = pin->GetOrAddInteger("mhd","mood_max_revs",n_fb_tiers);
      if (mood_max_revs < 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<mhd> mood_max_revs must be >= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // ct_contact composes edges from the two adjacent faces (reach 1).  UCT
      // reconstructs along the edge, so the light-cone halo must cover that reach.
      int h_f = 1;
      if (emf_method != MHD_EMF::ct_contact) {
        h_f = ReconEdgeReach(recon_method);
      }
      const bool ho_pad = is_uct && pmy_pack->pmesh->multi_d &&
          peos->eos_data.is_ideal && !pmy_pack->pcoord->is_special_relativistic &&
          !pmy_pack->pcoord->is_general_relativistic;
      const int pad_reach = ho_pad ? mhd_bcc::Reach(recon_method) : 1;
      // An edge revision affects the cell-centred field tested by PAD out to the
      // production face-to-cell interpolation reach, even when base EMFs are frozen.
      h_f += pad_reach - 1;
      // Light cone of the revisions.  Each revision's candidate over its halo needs
      // the faces revised by the previous one out to the composition reach h_f beyond
      // that halo (the staggered candidate composes corner EMFs over the halo), so the
      // halo grows by h_f per revision and the final one keeps h_f for the production
      // CornerE: halo0 = h_f*nrevs.  With ct_contact (h_f = 1) this is the former
      // h_f + nrevs - 1; UCT and production B interpolation have wider reach.
      mood_halo0 = mood_frozen_emf ? 0 : h_f*mood_max_revs;
      if (mood_frozen_emf) {
        // Detection is owned-cell only. Ghost cascade levels come from their
        // owners each revision; only the fixed magnetic averaging support remains.
        mood_halo0 = 0;
        mood_emf_halo = pad_reach-1;
      }
      // SED reads +/-3; the extra cell in ng_need covers the outer offset.
      // Otherwise the stencil is the reconstruction half-width.
      int stencil = ReconStencilRadius(recon_method);
      if (mood_sed && stencil < 2) { stencil = 2; }
      int ng_need = mood_halo0 + 1 + stencil;
      // the detector composes the UCT candidate over the halo: edge stencil reach past it
      if (mood_halo0 + h_f > ng_need) {
        ng_need = mood_halo0 + h_f;
      }
      // Cached PLM EMFs extend r-1 for averaging and read two more input cells;
      // L-HLLD's sensor reads one further cell. The frozen base bound covers both.
      // SED is evaluated on owned cells and reads immutable inputs at +/-3.
      if (mood_frozen_emf) {
        ng_need = frozen_ng_need;
      }
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

    // Well-balanced scheme (mhd_wb.cpp): perturbation over a STATIC magnetohydrostatic
    // equilibrium from the problem generator.  Modules whose meaning changes when u0 is a
    // perturbation are rejected here or in WBInitialize().
    well_balanced = pin->GetOrAddBoolean("mhd","well_balanced",false);
    wb_diagnostics = pin->GetOrAddBoolean("mhd","wb_diagnostics",false);
    if (well_balanced) {
      std::string wb_err;
      if (pmy_pack->pcoord->is_special_relativistic ||
          pmy_pack->pcoord->is_general_relativistic) {
        wb_err = "is implemented only for non-relativistic MHD";
      } else if (evolution_t.compare("dynamic") != 0) {
        wb_err = "requires <time>/evolution=dynamic";
      } else if (!peos->eos_data.is_ideal) {
        wb_err = "requires an ideal-gas EOS";
      } else if (pcond != nullptr || presist != nullptr) {
        wb_err = "cannot be used with conduction or resistivity (the equilibrium "
                 "would diffuse)";
      } else if (pin->DoesBlockExist("shearing_box")) {
        wb_err = "cannot be used with the shearing box / orbital advection";
      } else if (psrc != nullptr &&
                 (psrc->self_gravity || psrc->ism_cooling || psrc->rel_cooling)) {
        wb_err = "supports only const_accel source terms";
      } else if (pmy_pack->pmesh->multilevel &&
                 pin->GetOrAddBoolean("mesh_refinement","prolong_prims",false)) {
        // the coarse state is a perturbation; the primitive prolongation path would
        // invert it as a full state (the MeshRefinement object does not exist yet here)
        wb_err = "cannot be used with <mesh_refinement>/prolong_prims";
      }
      if (!wb_err.empty()) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<mhd> well_balanced=true " << wb_err << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (psrc != nullptr) psrc->pwb_pert = &w_pert;
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

      // per-direction velocity-compression field for the L-HLLD carbuncle sensor
      if (rsolver_method == MHD_RSolver::lhlld) {
        Kokkos::realloc(sdet, nmb, 3, ncells3, ncells2, ncells1);
      }

      if (mood_frozen_emf) {
        pbval_mood_prim = new MeshBoundaryValuesCC(ppack, pin, false);
        pbval_mood_prim->InitializeBuffers(nmhd+3);
        pbval_mood_candidate = new MeshBoundaryValuesCC(ppack, pin, false);
        pbval_mood_flags = new MeshBoundaryValuesCCInt(ppack, pin, mood_emf_halo+1);
        mood_diagnostics = pin->GetOrAddBoolean("mhd", "mood_diagnostics", false);
#if MPI_PARALLEL_ENABLED
        MPI_Comm_dup(MPI_COMM_WORLD, &mood_convergence_comm);
#endif
        const int ndir = pmy_pack->pmesh->three_d ? 3 : 2;
        pbval_mood_candidate->InitializeBuffers(ndir*nmhd+3);
        Kokkos::realloc(mood_prim, nmb, nmhd+3, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_candidate, nmb, ndir*nmhd+3, ncells3, ncells2, ncells1);
        mood_eplm = DvceEdgeFld4D<Real>("mood_eplm",nmb,ncells3,ncells2,ncells1);
        mood_edc = DvceEdgeFld4D<Real>("mood_edc",nmb,ncells3,ncells2,ncells1);
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
        Kokkos::realloc(bcc_nad, nmb, 3,    ncells3, ncells2, ncells1);
        Kokkos::deep_copy(fofc, false);
        if (nscalars > 0) {
          Kokkos::realloc(fofc_scal,    nmb, nscalars, ncells3, ncells2, ncells1);
          Kokkos::deep_copy(fofc_scal, false);
        }
      }
      // well-balanced equilibrium and perturbation arrays
      if (well_balanced || wb_diagnostics) {
        Kokkos::realloc(u_eq,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(w_eq,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(bcc_eq, nmb, 3, ncells3, ncells2, ncells1);
        Kokkos::realloc(b_eq.x1f, nmb, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(b_eq.x2f, nmb, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(b_eq.x3f, nmb, ncells3+1, ncells2, ncells1);
        Kokkos::deep_copy(u_eq, 0.0);
        Kokkos::deep_copy(w_eq, 0.0);
      }
      if (well_balanced) {
        Kokkos::realloc(w_pert,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(bcc_pert, nmb, 3, ncells3, ncells2, ncells1);
        Kokkos::realloc(w_eq_face.x1f, nmb, nmhd, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(w_eq_face.x2f, nmb, nmhd, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(w_eq_face.x3f, nmb, nmhd, ncells3+1, ncells2, ncells1);
        Kokkos::realloc(bcc_eq_face.x1f, nmb, 3, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(bcc_eq_face.x2f, nmb, 3, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(bcc_eq_face.x3f, nmb, 3, ncells3+1, ncells2, ncells1);
        Kokkos::realloc(f_eq.x1f, nmb, nmhd, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(f_eq.x2f, nmb, nmhd, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(f_eq.x3f, nmb, nmhd, ncells3+1, ncells2, ncells1);
        Kokkos::realloc(e3x1_eq, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(e2x1_eq, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(e1x2_eq, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(e3x2_eq, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(e2x3_eq, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(e1x3_eq, nmb, ncells3, ncells2, ncells1);
      }
      if (use_mood) {
        Kokkos::realloc(fb_level, nmb, ncells3, ncells2, ncells1);
        // NAD bound slots (mhd_mood.cpp): density, energy (ideal gas with
        // mood_nad_energy), B (|B| or 3 components), each also in full state with
        // well_balanced; velocity (|v| or 3 components); scalars
        const int nwb = well_balanced ? 2 : 1;
        int nbnd = ((peos->eos_data.is_ideal && mood_nad_energy) ? 2 : 1)*nwb;
        nbnd += ((mood_nad_b == 0) ? 1 : 3)*nwb;
        nbnd += (mood_nad_v == 1) ? 1 : (mood_nad_v == 2) ? 3 : 0;
        if (mood_nad_scalars) nbnd += nscalars;
        Kokkos::realloc(mood_val, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_qmn, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_qmx, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_tmx, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_cnt, 8);
        Kokkos::realloc(mood_stat, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_nadf, nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_crtol, nmb, ncells3, ncells2, ncells1);
        if (mood_touched_only) {
          Kokkos::realloc(mood_touched, nmb, ncells3, ncells2, ncells1);
          Kokkos::realloc(mood_tmp,     nmb, ncells3, ncells2, ncells1);
        }
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
#if MPI_PARALLEL_ENABLED
  if (mood_convergence_comm != MPI_COMM_NULL) MPI_Comm_free(&mood_convergence_comm);
#endif
  if (psbox_b != nullptr) {delete psbox_b;}
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_b != nullptr) {delete porb_b;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_mood_candidate;
  delete pbval_mood_flags;
  delete pbval_mood_prim;
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
