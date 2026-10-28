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
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    u1("cons1",1,1,1,1,1),
    u_sts0("u_sts0",1,1,1,1,1),
    u_sts1("u_sts1",1,1,1,1,1),
    u_sts2("u_sts2",1,1,1,1,1),
    u_sts_rhs("u_sts_rhs",1,1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    wl3d("wl3d",1,1,1,1,1),
    wr3d("wr3d",1,1,1,1,1),
    fofc("fofc",1,1,1,1),
    utest("utest",1,1,1,1,1),
    fb_level("fb_level",1,1,1,1),
    u_eq("u_eq",1,1,1,1,1),
    w_eq("w_eq",1,1,1,1,1),
    w_pert("w_pert",1,1,1,1,1),
    w_eq_face("w_eq_face",1,1,1,1,1),
    f_eq("f_eq",1,1,1,1,1),
    pmy_pack(ppack) {
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
  if (pin->DoesParameterExist("hydro","nu_iso") ||
      pin->DoesParameterExist("hydro","nu_aniso")) {
    pvisc = new Viscosity("hydro", ppack, pin);
    const bool active = (pvisc->nu_iso != 0.0 || pvisc->nu_aniso != 0.0);
    has_explicit_viscosity =
        active && pvisc->mode == parabolic::DiffusionSelection::explicit_only;
    has_sts_viscosity =
        active && pvisc->mode == parabolic::DiffusionSelection::sts_only;
    if (active) {
      ppack->RegisterParabolicProcess(
          {"hydro/viscosity", parabolic::ParabolicProcessOwner::hydro,
           pvisc->mode, &(pvisc->dtnew)});
    }
  } else {
    pvisc = nullptr;
  }

  // Thermal conduction (if requested in input file)
  if (pin->DoesParameterExist("hydro","alpha_iso") ||
      pin->DoesParameterExist("hydro","alpha_aniso") ||
      pin->DoesParameterExist("hydro","alpha_spitzer")) {
    if (peos->eos_data.is_ideal) {
      pcond = new Conduction("hydro", ppack, pin);
      const bool active =
          (pcond->alpha_iso != 0.0 || pcond->alpha_aniso != 0.0 || pcond->alpha_spitzer);
      has_explicit_conduction =
          active && pcond->mode == parabolic::DiffusionSelection::explicit_only;
      has_sts_conduction =
          active && pcond->mode == parabolic::DiffusionSelection::sts_only;
      if (active) {
        ppack->RegisterParabolicProcess(
            {"hydro/conductivity", parabolic::ParabolicProcessOwner::hydro,
             pcond->mode, &(pcond->dtnew)});
      }
    } else {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Thermal conduction in hydro requires ideal gas EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    pcond = nullptr;
  }

  // Source terms (if needed)
  if (pin->DoesBlockExist("hydro_srcterms")) {
    psrc = new SourceTerms("hydro_srcterms", ppack, pin);
  }

  has_any_sts_diffusion = has_sts_viscosity || has_sts_conduction;

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
    // PPMX/WENOZ/TENO) and all Newtonian (Advect/LLF/HLLE/HLLC/Roe) and relativistic
    // (LLF/HLLE/HLLC SR; LLF/HLLE GR) Riemann solvers.

    // FOFC: the split-kernel main flux kernels extend their face-normal range by one cell
    // when FOFC is enabled, so the self-contained first-order flux correction
    // (hydro_fofc.cpp) has the fluxes it needs over the [is-1,ie+2] etc. range.
    use_fofc = pin->GetOrAddBoolean("hydro","fofc",false);

    // MOOD a-posteriori fallback (hydro_mood.cpp).  Requires a flux-range extension of
    // mood_max_revs cells (see hydro_fluxes.cpp).  Mutually exclusive with FOFC.
    use_mood = pin->GetOrAddBoolean("hydro","mood",false);
    // NAD uses only each cell's neighboring bounds; no domain-wide scales.
    const std::string nadsc = pin->GetOrAddString("hydro","mood_nad_scale","relative");
    if (nadsc != "relative" && nadsc != "rcfl") {
      std::cerr << "<hydro> mood_nad_scale='" << nadsc
                << "' is unsupported; use relative or rcfl with mood_rtol/mood_atol\n";
      std::exit(EXIT_FAILURE);
    }
    mood_nad_rcfl = (nadsc == "rcfl");
    if (pin->DoesParameterExist("hydro", "mood_nad_theta")) {
      std::cerr << "<hydro> mood_nad_theta was removed with the global NAD scales\n";
      std::exit(EXIT_FAILURE);
    }
    mood_eps0  = pin->GetOrAddReal("hydro","mood_eps0",1.0e-12);
    // Well-balanced round-off floor relative to the equilibrium.  Default 0: any positive
    // value masks perturbations of that relative size from NAD (1e-12 hid the unlimited-
    // PPM ringing of an eta=1e-12 pulse), while 0 keeps the detector scale-invariant for
    // linear perturbations and was measured bitwise decomposition-invariant.
    mood_eps0_eq = pin->GetOrAddReal("hydro","mood_eps0_eq",0.0);
    mood_rtol = pin->GetOrAddReal("hydro","mood_rtol",
                                    mood_nad_rcfl ? 1.0e-4 : 1.0e-5);
    mood_atol = pin->GetOrAddReal("hydro","mood_atol",0.0);
    mood_nad_energy = pin->GetOrAddBoolean("hydro","mood_nad_energy",true);
    mood_nad_scalars = pin->GetOrAddBoolean("hydro","mood_nad_scalars",true);
    // DETECTION OFF -- the negative control for the whole cascade.  reconstruct=fv4..fv9
    // is unlimited and requires mood=true, so before this flag existed there was no way to
    // run the base scheme on its own and ask which of its properties the detector is
    // actually responsible for.  mood_detect=false keeps mood=true (so the guard above
    // is satisfied and the halos are sized as usual) but skips the detect/revise loop
    // entirely, leaving the pure unlimited scheme.  Expect it to be unstable on hard
    // problems: that is the point of the control, not a defect to work around.
    mood_detect = pin->GetOrAddBoolean("hydro","mood_detect",true);
    // One-sided closure of the unlimited fv4 at inflow/user (imposed-state) boundaries
    // (reconstruct/ppm_closure.hpp); false = the old centred stencil across the imposed
    // ghost state (the negative control).  fv4 only: the closure is 4th order.
    ppm_bc_closure = pin->GetOrAddBoolean("hydro","ppm_boundary_closure",true);

    mood_sed = pin->GetOrAddBoolean("hydro","mood_sed",true);
    // DEBUG / validation only.  If mood_debug_demote_x1max is given, every cell with
    // x1 < x1max is flagged and demoted at every revision of every stage, whatever the
    // detector says, so the revise pass runs over a chosen part of the box even on a
    // state the detector never flags.  Exists to test the flux bookkeeping on revised
    // faces (e.g. that diffusion fluxes survive revision); never for production runs.
    // A value at or below the mesh x1min (the default) switches it off, so an input
    // file can carry the parameter and a test can enable it from the command line.
    mood_debug_demote_x1max = pin->GetOrAddReal("hydro","mood_debug_demote_x1max",
                                                pmy_pack->pmesh->mesh_size.x1min);
    mood_debug_demote = (mood_debug_demote_x1max > pmy_pack->pmesh->mesh_size.x1min);

    // Explicit diffusion fluxes and the a-posteriori fallback.  false (default): the
    // diffusion fluxes are added AFTER MOOD/FOFC, so the detector judges the inviscid
    // candidate and every face, revised or not, then receives its diffusive flux.  true:
    // add them BEFORE the fallback (detector judges the full candidate) and re-add the
    // face-local diffusive flux on every face the revise pass overwrites (hydro_mood.cpp,
    // hydro_fofc.cpp; the face functions in diffusion/viscosity.hpp and
    // diffusion/conduction.hpp are the ones the bulk pass uses).  The two agree to ~1% in
    // demotion counts on viscous Shu-Osher tests at cell Reynolds numbers 0.4-4; the
    // inviscid candidate is cheaper and proved more robust against energy floors at
    // demoted shock faces when the time step is large, so it is the default.  Only the
    // isotropic viscous and heat fluxes have face-local forms, and the well-balanced FOFC
    // kernel writes faces in place, so those configurations cannot use true.
    const bool diff_readd_ok =
        ((pcond == nullptr) || (pcond->alpha_aniso == 0.0 && !pcond->alpha_spitzer)) &&
        !(pin->GetOrAddBoolean("hydro","well_balanced",false) && use_fofc);
    mood_candidate_diffusion =
        pin->GetOrAddBoolean("hydro","mood_candidate_diffusion", false);
    if (mood_candidate_diffusion && !diff_readd_ok && (use_mood || use_fofc)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<hydro> mood_candidate_diffusion=true needs face-local "
        << "diffusion fluxes: not available for anisotropic/Spitzer conduction or "
        << "well-balanced FOFC"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (use_mood && use_fofc) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<hydro> mood=true and fofc=true cannot be used together"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Well-balanced scheme (hydro_wb.cpp): evolve the perturbation over a static
    // equilibrium supplied by the problem generator.  Only the modules whose meaning is
    // unchanged when u0 holds a perturbation are allowed; the others are rejected here
    // or in WBInitialize() (modules constructed after Hydro).
    well_balanced = pin->GetOrAddBoolean("hydro","well_balanced",false);
    wb_diagnostics = pin->GetOrAddBoolean("hydro","wb_diagnostics",false);
    if (well_balanced) {
      std::string wb_err;
      if (pmy_pack->pcoord->is_special_relativistic ||
          pmy_pack->pcoord->is_general_relativistic) {
        wb_err = "is implemented only for non-relativistic hydro";
      } else if (evolution_t.compare("dynamic") != 0) {
        wb_err = "requires <time>/evolution=dynamic";
      } else if (pcond != nullptr) {
        wb_err = "cannot be used with thermal conduction (the equilibrium would conduct)";
      } else if (pin->DoesBlockExist("shearing_box")) {
        wb_err = "cannot be used with the shearing box / orbital advection";
      } else if (psrc != nullptr &&
                 (psrc->self_gravity || psrc->ism_cooling || psrc->rel_cooling)) {
        wb_err = "supports only const_accel source terms (not self-gravity or cooling)";
      }
      if (!wb_err.empty()) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<hydro> well_balanced=true " << wb_err << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // constant gravity acts on the perturbation density
      if (psrc != nullptr) psrc->pwb_pert = &w_pert;
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
               xorder.compare("teno") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2 (the +/-2 stencil requires at least 3 ghost zones)
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
    } else if (xorder.compare("fv4") == 0 || xorder.compare("fv5") == 0 ||
               xorder.compare("fv6") == 0 || xorder.compare("fv7") == 0 ||
               xorder.compare("fv8") == 0 || xorder.compare("fv9") == 0) {
      // Unlimited cell-polynomial reconstruction (fv.hpp); no limiter of its own, so it
      // is only stable behind the MOOD fallback.  Stencil half-width is 2 (fv4, fv5),
      // 3 (fv6, fv7) or 4 (fv8, fv9).
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
          << std::endl << "<hydro> reconstruct=" << xorder
          << " (unlimited) requires mood=true" << std::endl;
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
      // the boundary closure extrapolates from four ghost cells
      if (recon_method == ReconstructionMethod::fv4 && ppm_bc_closure && indcs.ng < 4) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<hydro> reconstruct=fv4 with ppm_boundary_closure=true needs "
          << "nghost >= 4" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm") == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<hydro> reconstruct=ppm was renamed to fv4 "
        << "(unlimited 4th-order; also available: fv5 .. fv9)" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> reconstruct = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Owner exchange freezes the high-order candidate before revisions. The flux
    // solver and the local detector retain their NR/SR/GR specializations.
    const bool frozen_supported = use_mood && evolution_t == "dynamic" &&
        !pmy_pack->pmesh->multilevel && pmy_pack->pmesh->strictly_periodic &&
        peos->eos_data.is_ideal && nscalars == 0 &&
        pvisc == nullptr && pcond == nullptr && !well_balanced &&
        !pin->DoesBlockExist("shearing_box") &&
        !(pmy_pack->pcoord->is_general_relativistic &&
          pmy_pack->pcoord->coord_data.bh_excise) &&
        IsUnlimitedFV(recon_method);
    mood_frozen_flux = pin->GetOrAddBoolean("hydro", "mood_frozen_flux", frozen_supported);
    if (mood_frozen_flux && !frozen_supported) {
      std::cerr << "Frozen hydro fluxes require dynamic MOOD with periodic uniform "
                << "FV4..FV9, ideal EOS, and no diffusion, scalars, shear, "
                << "excision or well balance\n";
      std::exit(EXIT_FAILURE);
    }
    if (mood_frozen_flux) {
      const auto &q = pmy_pack->pmesh->mb_indcs;
      if (q.ng > q.nx1 || (pmy_pack->pmesh->multi_d && q.ng > q.nx2) ||
          (pmy_pack->pmesh->three_d && q.ng > q.nx3)) {
        std::cerr << "Frozen flux neighbor exchange requires block dimensions >= nghost\n";
        std::exit(EXIT_FAILURE);
      }
    }

    // MOOD cascade: base -> plm -> dc, or plm -> dc if the base scheme is already PLM.
    if (use_mood) {
      n_fb_tiers = (recon_method == ReconstructionMethod::plm) ? 1 : 2;
      mood_max_revs = pin->GetOrAddInteger("hydro","mood_max_revs",n_fb_tiers);
      if (mood_max_revs < 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "<hydro> mood_max_revs must be >= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // radius+1 covers the reconstruction stencil; each revision consumes one more
      // ghost layer of the light-cone halo.
      int ng_need = ReconStencilRadius(recon_method) + 1 + mood_max_revs;
      if (mood_frozen_flux) {
        // Hydro has no transverse corner composition: only the face-normal
        // reconstruction radius plus one, or the PLM/SED revision halo.
        ng_need = std::max(ReconStencilRadius(recon_method) + 1, mood_max_revs + 3);
      }
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
      } else if (rsolver.compare("hllc") == 0 || rsolver.compare("lhllc") == 0) {
        // lhllc = L-HLLC, the low-Mach HLLC (hllc_hyd.hpp); same checks as hllc
        if (peos->eos_data.is_ideal) {
          rsolver_method = (rsolver.compare("hllc") == 0) ? Hydro_RSolver::hllc
                                                          : Hydro_RSolver::lhllc;
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
      if (has_any_sts_diffusion) {
        Kokkos::realloc(u_sts0,    nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(u_sts1,    nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(u_sts2,    nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(u_sts_rhs, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      }
      Kokkos::realloc(uflx.x1f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x2f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x3f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);

      // allocate global per-face L/R buffers for the split-kernel flux path.
      // Indexed by the GLOBAL cell/face index (m,n,k,j,i), so sized to the full
      // cell range (including ghost zones) in every dimension.
      Kokkos::realloc(wl3d, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(wr3d, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);

      if (mood_frozen_flux) {
        const int ndir = pmy_pack->pmesh->three_d ? 3 :
                         (pmy_pack->pmesh->multi_d ? 2 : 1);
        // NR/SR and Minkowski C2P use only the already-exchanged cell state.
        // Curved metrics additionally depend on locally computed coordinates.
        if (pmy_pack->pcoord->is_general_relativistic &&
            !pmy_pack->pcoord->coord_data.is_minkowski) {
          pbval_mood_prim = new MeshBoundaryValuesCC(ppack, pin, false);
          pbval_mood_prim->InitializeBuffers(nhydro);
          Kokkos::realloc(mood_prim, nmb, nhydro, ncells3, ncells2, ncells1);
        }
        pbval_mood_candidate = new MeshBoundaryValuesCC(ppack, pin, false);
        pbval_mood_candidate->InitializeBuffers(ndir*nhydro);
      }

      // allocate arrays of flags/candidate state used with FOFC and MOOD
      if (use_fofc || use_mood) {
        // MOOD tests passive scalars too, so the candidate must carry them
        int ntest = (use_mood && mood_nad_scalars) ? (nhydro + nscalars) : nhydro;
        Kokkos::realloc(fofc,  nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest, nmb, ntest, ncells3, ncells2, ncells1);
        Kokkos::deep_copy(fofc, false);
      }
      if (use_mood) {
        Kokkos::realloc(fb_level, nmb, ncells3, ncells2, ncells1);
        // NAD bound slots (hydro_mood.cpp): density, energy (ideal gas with
        // mood_nad_energy), the same in full state (well-balanced), scalars
        int nbnd = (peos->eos_data.is_ideal && mood_nad_energy) ? 2 : 1;
        if (well_balanced) nbnd *= 2;
        if (mood_nad_scalars) nbnd += nscalars;
        Kokkos::realloc(mood_val, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_qmn, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_qmx, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_tmx, nmb, nbnd, ncells3, ncells2, ncells1);
        Kokkos::realloc(mood_cnt, 8);
      }
      // well-balanced equilibrium and perturbation arrays.  u_eq/w_eq carry the scalar
      // slots (always zero) so the standard EOS kernels can be applied to them.
      if (well_balanced || wb_diagnostics) {
        Kokkos::realloc(u_eq,   nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(w_eq,   nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
        Kokkos::deep_copy(u_eq, 0.0);
        Kokkos::deep_copy(w_eq, 0.0);
      }
      if (well_balanced) {
        Kokkos::realloc(w_pert, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
        Kokkos::realloc(w_eq_face.x1f, nmb, nhydro, ncells3, ncells2, ncells1);
        Kokkos::realloc(w_eq_face.x2f, nmb, nhydro, ncells3, ncells2, ncells1);
        Kokkos::realloc(w_eq_face.x3f, nmb, nhydro, ncells3, ncells2, ncells1);
        Kokkos::realloc(f_eq.x1f, nmb, nhydro, ncells3, ncells2, ncells1);
        Kokkos::realloc(f_eq.x2f, nmb, nhydro, ncells3, ncells2, ncells1);
        Kokkos::realloc(f_eq.x3f, nmb, nhydro, ncells3, ncells2, ncells1);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

Hydro::~Hydro() {
  delete pbval_mood_candidate;
  delete pbval_mood_prim;
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_u;
  if (psrc != nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (pvisc != nullptr) {delete pvisc;}
  delete peos;
}

} // namespace hydro
