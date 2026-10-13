//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_wb.cpp
//! \brief Well-balanced (WB) scheme for non-relativistic hydro, following
//! Velasco-Romero & Teyssier (2025, MNRAS, "Spectral difference method with a posteriori
//! limiting II", sec. 2.2) and its spd implementation.
//!
//! The solution is split into a static equilibrium and a perturbation, U = U_eq + U'.
//! u0 holds U' and is what the time integrator, the boundary exchange, the physical BCs
//! and AMR restriction/prolongation operate on; w0 holds the primitives of the FULL
//! state.
//! At each face the reconstructed PERTURBATION states are added to a single-valued face
//! equilibrium, the Riemann problem is solved on the full states, and the Riemann
//! solver's own flux of the face equilibrium is subtracted:
//!   F' = RS(W_eq,f + W'_L, W_eq,f + W'_R) - RS(W_eq,f, W_eq,f).
//! Subtracting RS(W_eq,f, W_eq,f) rather than the analytic flux F(W_eq,f) is deliberate:
//! HLLC/HLLE do not return F(W) bit for bit when both states are equal, whereas RS-RS
//! cancels exactly, so an unperturbed equilibrium keeps U' == 0 exactly.  The equilibrium
//! residual (flux divergence minus source of U_eq, i.e. its truncation error) is dropped;
//! the gravity source acts on rho' = rho - rho_eq.

#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pgen/pgen.hpp"
#include "hydro.hpp"
#include "hydro/well_balance.hpp"
#include "reconstruct/recon.hpp"
#include "reconstruct/fv.hpp"
#include "reconstruct/ppm_closure.hpp"
#include "hydro/rsolvers/solve_face_hyd.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Hydro::WBInitialize()
//! \brief Called once by the ProblemGenerator after the pgen has set the (full) initial
//! condition.  Checks the modules constructed after Hydro, builds the equilibrium, and
//! (new runs only) converts u0 to the perturbation.  On restart u0 already holds U'.

void Hydro::WBInitialize(ProblemGenerator *ppgen, bool restart) {
  Mesh *pm = pmy_pack->pmesh;
  wb_eq_func = ppgen->user_eq_func;
  if (!well_balanced) {
    // wb_diagnostics only: equilibrium for the hydro_wpert output, scheme unchanged
    if (wb_eq_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> wb_diagnostics=true requires the problem "
                << "generator to enroll user_eq_func" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    WBSetEquilibrium();
    return;
  }
  std::string err;
  if (wb_eq_func == nullptr) {
    err = "requires the problem generator to enroll user_eq_func";
  } else if (pmy_pack->pmhd != nullptr) {
    err = "is not supported with Hydro and MHD together";
  } else if (pmy_pack->prad != nullptr) {
    err = "cannot be used with radiation";
  } else if (pmy_pack->ppart != nullptr) {
    err = "cannot be used with particles";
  } else if (pmy_pack->pionn != nullptr) {
    err = "cannot be used with ion-neutral coupling";
  } else if (pmy_pack->pturb != nullptr) {
    err = "cannot be used with the turbulence driver";
  } else if (pmy_pack->pgrav != nullptr) {
    err = "cannot be used with self-gravity";
  } else if (pm->pmr != nullptr && pm->pmr->prolong_prims) {
    err = "requires prolongation of conserved variables (prolong_primitives=false)";
  }
  for (int f=0; f<6; ++f) {
    if (pm->mesh_bcs[f] == BoundaryFlag::inflow) {
      err = "cannot be used with inflow boundaries (u_in is a full state)";
    }
  }
  if (!err.empty()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<hydro> well_balanced=true " << err << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ppgen->user_bcs && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> well_balanced=true: user boundary functions write u0, which "
              << "holds the PERTURBATION U' = U - U_eq" << std::endl;
  }

  WBSetEquilibrium();

  if (!restart) {
    auto &u0_ = u0;
    auto &ueq_ = u_eq;
    const int nhyd = nhydro;
    const int n1 = static_cast<int>(u0.extent(4));
    const int n2 = static_cast<int>(u0.extent(3));
    const int n3 = static_cast<int>(u0.extent(2));
    par_for("wb_to_pert", DevExeSpace(), 0, (pmy_pack->nmb_thispack-1), 0, nhyd-1,
            0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      u0_(m,n,k,j,i) -= ueq_(m,n,k,j,i);
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::WBSetEquilibrium()
//! \brief (Re)build every equilibrium array from the pgen's user_eq_func: u_eq, w_eq, the
//! face equilibrium w_eq_face and its Riemann flux f_eq.  Called at start-up, on restart
//! and after every mesh refinement / load balancing (block locations change).  Nothing
//! here is communicated: every block computes its equilibrium, ghost cells included,
//! from the global cell lattice, so neighbours agree bit for bit.

void Hydro::WBSetEquilibrium() {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int n1m1 = indcs.nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nhyd = nhydro;

  auto &ec0 = pm->ecounter;
  const int nfl0[3] = {ec0.neos_dfloor, ec0.neos_efloor, ec0.neos_tfloor};

  // (1) equilibrium primitive cell averages from the problem generator
  Kokkos::deep_copy(w_eq, 0.0);
  wb_eq_func(pm);

  // (2) u_eq = P2C(w_eq), then w_eq := C2P(u_eq): the WB inversion of U'=0 is then the
  //     same arithmetic on the same bits, so w0 == w_eq exactly at equilibrium.
  peos->PrimToCons(w_eq, u_eq, 0, n1m1, 0, n2m1, 0, n3m1);
  peos->ConsToPrim(u_eq, w_eq, false, 0, n1m1, 0, n2m1, 0, n3m1);

  // The equilibrium must be physically admissible EVERYWHERE, ghost zones included
  // (they hold the analytic continuation, not reflected data): a floor there would
  // silently alter U_eq.  Count the floors hit while building it, and keep them out of
  // the run's event counters.
  {
    auto &ec = pm->ecounter;
    const int nfloor = (ec.neos_dfloor - nfl0[0]) + (ec.neos_efloor - nfl0[1]) +
                       (ec.neos_tfloor - nfl0[2]);
    ec.neos_dfloor = nfl0[0]; ec.neos_efloor = nfl0[1]; ec.neos_tfloor = nfl0[2];
#if MPI_PARALLEL_ENABLED
    int nfl = nfloor;
    MPI_Allreduce(MPI_IN_PLACE, &nfl, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
    const int nfl = nfloor;
#endif
    if (nfl > 0) {
      if (well_balanced) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "well-balanced equilibrium is not admissible: " << nfl
                  << " density/pressure floors hit while building it (ghost zones "
                  << "included; e.g. p <= 0 above the domain)" << std::endl;
        std::exit(EXIT_FAILURE);
      } else if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "wb_diagnostics equilibrium hits " << nfl
                  << " floors (ghost zones included)" << std::endl;
      }
    }
  }

  // (3) the scheme is written for STATIC equilibria (v_eq = 0 exactly): the scalar
  //     fluxes, the energy source and the cancellation-free W' all assume it.
  {
    auto &weq_ = w_eq;
    Real vmax = 0.0;
    Kokkos::parallel_reduce("wb_vcheck",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0,0,0,0},
                                             {nmb1+1, n3m1+1, n2m1+1, n1m1+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &vm) {
        vm = fmax(vm, fabs(weq_(m,IVX,k,j,i)));
        vm = fmax(vm, fabs(weq_(m,IVY,k,j,i)));
        vm = fmax(vm, fabs(weq_(m,IVZ,k,j,i)));
      }, Kokkos::Max<Real>(vmax));
    if (vmax != 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> well_balanced=true requires a static "
                << "equilibrium (user_eq_func returned max|v_eq| = " << vmax << ")"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (!well_balanced) return;   // wb_diagnostics: cell averages are all that is needed

  // (4) single-valued face equilibrium: the unlimited central interpolation of the
  //     equilibrium cell averages at the order of the base scheme (fv6/fv8: their own
  //     interpolant, so that fvN(w_eq) + fvN(w') is fvN(w_eq + w') up to round-off;
  //     4th order for every other method), shared by every reconstruction method and
  //     MOOD tier.  Floored like a reconstructed state.  Valid on faces
  //     req .. n-req (FVFaceRadius), which has to cover every face the flux kernels
  //     visit (is-ext .. ie+1+ext, ext = the MOOD / FOFC halo).
  const int req = FVFaceRadius(recon_method);
  const auto rm_ = recon_method;
  {
    const int ext = use_mood ? mood_max_revs : (use_fofc ? 1 : 0);
    if (ng < req + ext) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> well_balanced=true needs nghost >= "
                << req + ext << " (face-equilibrium stencil " << req
                << " + fallback halo " << ext << "), but <mesh>/nghost=" << ng
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  {
    auto &weq_ = w_eq;
    auto &eos_ = peos->eos_data;
    auto wf1 = w_eq_face.x1f;
    par_for("wb_eqface_x1", DevExeSpace(), 0, nmb1, 0, nhyd-1, 0, n3m1, 0, n2m1,
            req, n1m1-req+1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      Real q = FVFaceAt(rm_, [&](const int o) {return weq_(m,n,k,j,i+o);});
      if (n == IDN) {q = fmax(q, eos_.dfloor);}
      if (eos_.is_ideal && n == IEN) {q = fmax(q, eos_.pfloor/(eos_.gamma - 1.0));}
      wf1(m,n,k,j,i) = q;
    });
    if (pm->multi_d) {
      auto wf2 = w_eq_face.x2f;
      par_for("wb_eqface_x2", DevExeSpace(), 0, nmb1, 0, nhyd-1, 0, n3m1, req, n2m1-req+1,
              0, n1m1,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        Real q = FVFaceAt(rm_, [&](const int o) {return weq_(m,n,k,j+o,i);});
        if (n == IDN) {q = fmax(q, eos_.dfloor);}
        if (eos_.is_ideal && n == IEN) {q = fmax(q, eos_.pfloor/(eos_.gamma - 1.0));}
        wf2(m,n,k,j,i) = q;
      });
    }
    if (pm->three_d) {
      auto wf3 = w_eq_face.x3f;
      par_for("wb_eqface_x3", DevExeSpace(), 0, nmb1, 0, nhyd-1, req, n3m1-req+1, 0, n2m1,
              0, n1m1,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        Real q = FVFaceAt(rm_, [&](const int o) {return weq_(m,n,k+o,j,i);});
        if (n == IDN) {q = fmax(q, eos_.dfloor);}
        if (eos_.is_ideal && n == IEN) {q = fmax(q, eos_.pfloor/(eos_.gamma - 1.0));}
        wf3(m,n,k,j,i) = q;
      });
    }
  }

  // (5) Riemann flux of the face equilibrium with the scheme's own solver
  switch (rsolver_method) {
    case Hydro_RSolver::llf:  WBEquilibriumFluxes<Hydro_RSolver::llf>();  break;
    case Hydro_RSolver::hlle: WBEquilibriumFluxes<Hydro_RSolver::hlle>(); break;
    case Hydro_RSolver::hllc: WBEquilibriumFluxes<Hydro_RSolver::hllc>(); break;
    case Hydro_RSolver::lhllc: WBEquilibriumFluxes<Hydro_RSolver::lhllc>(); break;
    case Hydro_RSolver::roe:  WBEquilibriumFluxes<Hydro_RSolver::roe>();  break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> well_balanced=true supports rsolver = "
                << "llf|hlle|hllc|lhllc|roe" << std::endl;
      std::exit(EXIT_FAILURE);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::WBEquilibriumFluxes()
//! \brief f_eq = RS(w_eq_face, w_eq_face) over every face with a valid face equilibrium.

template <Hydro_RSolver rsolver_method_>
void Hydro::WBEquilibriumFluxes() {
  Mesh *pm = pmy_pack->pmesh;
  RegionIndcs &indcs_ = pm->mb_indcs;
  const int ng = indcs_.ng;
  const int n1m1 = indcs_.nx1 + 2*ng - 1;
  const int n2m1 = (indcs_.nx2 > 1) ? (indcs_.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs_.nx3 > 1) ? (indcs_.nx3 + 2*ng - 1) : 0;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int is = indcs_.is, js = indcs_.js, ks = indcs_.ks;
  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  const int req = FVFaceRadius(recon_method);   // faces with a valid face equilibrium

  {
    auto wf = w_eq_face.x1f;
    auto fe = f_eq.x1f;
    par_for("wb_feq_x1", DevExeSpace(), 0, nmb1, 0, n3m1, 0, n2m1, req, n1m1-req+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      const int is_ = is, js_ = js, ks_ = ks;
      SolveFace<rsolver_method_, IVX>(eos, indcs, size, coord,
                                      m, k, j, i, is_, js_, ks_, wf, wf, fe);
    });
  }
  if (pm->multi_d) {
    auto wf = w_eq_face.x2f;
    auto fe = f_eq.x2f;
    par_for("wb_feq_x2", DevExeSpace(), 0, nmb1, 0, n3m1, req, n2m1-req+1, 0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      const int is_ = is, js_ = js, ks_ = ks;
      SolveFace<rsolver_method_, IVY>(eos, indcs, size, coord,
                                      m, k, j, i, is_, js_, ks_, wf, wf, fe);
    });
  }
  if (pm->three_d) {
    auto wf = w_eq_face.x3f;
    auto fe = f_eq.x3f;
    par_for("wb_feq_x3", DevExeSpace(), 0, nmb1, req, n3m1-req+1, 0, n2m1, 0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      const int is_ = is, js_ = js, ks_ = ks;
      SolveFace<rsolver_method_, IVZ>(eos, indcs, size, coord,
                                      m, k, j, i, is_, js_, ks_, wf, wf, fe);
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::WBFormPerturbation()
//! \brief Perturbation primitives W' from the stage-input state, over the full array (the
//! reconstruction stencils reach into the ghost zones).  For a static equilibrium:
//!   rho' = U'_rho (exact, no cancellation), v' = v, e' = e - e_eq;
//! passive scalars carry no equilibrium and stay full concentrations.

void Hydro::WBFormPerturbation() {
  auto &u0_ = u0;
  auto &w0_ = w0;
  auto &weq_ = w_eq;
  auto &wp_ = w_pert;
  const int nhyd = nhydro;
  const int nvars = nhydro + nscalars;
  const bool is_ideal = peos->eos_data.is_ideal;
  const int n1 = static_cast<int>(u0.extent(4));
  const int n2 = static_cast<int>(u0.extent(3));
  const int n3 = static_cast<int>(u0.extent(2));
  par_for("wb_wpert", DevExeSpace(), 0, (pmy_pack->nmb_thispack-1),
          0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    wp_(m,IDN,k,j,i) = u0_(m,IDN,k,j,i);
    wp_(m,IVX,k,j,i) = w0_(m,IVX,k,j,i);
    wp_(m,IVY,k,j,i) = w0_(m,IVY,k,j,i);
    wp_(m,IVZ,k,j,i) = w0_(m,IVZ,k,j,i);
    if (is_ideal) {
      wp_(m,IEN,k,j,i) = w0_(m,IEN,k,j,i) - weq_(m,IEN,k,j,i);
    }
    for (int n=nhyd; n<nvars; ++n) {
      wp_(m,n,k,j,i) = w0_(m,n,k,j,i);
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::WBCalculateFluxes()
//! \brief Well-balanced counterpart of CalculateFluxes (same ranges, so FOFC and MOOD see
//! the same halo): reconstruct W' without floors, add the face equilibrium, floor the
//! full states, solve, and subtract f_eq.  Scalar fluxes are upwinded on the FULL mass
//! flux F'_rho + F_eq,rho, since the scalars are stored as full densities.

template <Hydro_RSolver rsolver_method_>
void Hydro::WBCalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  const int nhyd = nhydro;
  int nvars = nhydro + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  // floors on the full face states wherever the non-WB reconstruction applies them
  const bool floors = (recon_method_ == ReconstructionMethod::ppmx ||
                       recon_method_ == ReconstructionMethod::wenoz ||
                       recon_method_ == ReconstructionMethod::teno ||
                       recon_method_ == ReconstructionMethod::fv4 ||
                       recon_method_ == ReconstructionMethod::fv6 ||
                       recon_method_ == ReconstructionMethod::fv8);

  int il1 = is, iu1 = ie+1, jl2 = js, ju2 = je+1, kl3 = ks, ku3 = ke+1;
  int itl = is, itu = ie, jtl = js, jtu = je, ktl = ks, ktu = ke;
  if (use_fofc || use_mood) {
    const int ext = (use_mood) ? mood_max_revs : 1;
    il1 = is-ext; iu1 = ie+1+ext;
    jl2 = js-ext; ju2 = je+1+ext;
    kl3 = ks-ext; ku3 = ke+1+ext;
    itl = is-ext; itu = ie+ext;
    if (pmy_pack->pmesh->multi_d) { jtl = js-ext; jtu = je+ext; }
    if (pmy_pack->pmesh->three_d) { ktl = ks-ext; ktu = ke+ext; }
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &wp_ = w_pert;
  auto wl_ = wl3d;
  auto wr_ = wr3d;

  // x1 direction
  {
    auto &flx1 = uflx.x1f;
    auto weqf = w_eq_face.x1f;
    auto feq = f_eq.x1f;
    ReconDispatch<IVX>(recon_method_, "hflux_x1_recon_wb", nmb1,
        ktl, ktu, jtl, jtu, il1-1, iu1, eos_, false, nvars, wp_, wl_, wr_);
    if (recon_method_ == ReconstructionMethod::fv4 && ppm_bc_closure) {
      PPMBoundaryClosure<IVX>("hflux_x1_ppm_closure_wb", nmb1, ktl, ktu, jtl, jtu, indcs_,
          pmy_pack->pmb->mb_bcs, eos_, false, nvars, wp_, wl_, wr_);
    }
    par_for("hflux_x1_rsolve_wb", DevExeSpace(), 0, nmb1, ktl, ktu, jtl, jtu, il1, iu1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // local copies of the solver inputs (required for CUDA 11.6+, as in hydro_fluxes)
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      auto wl = wl_;
      auto wr = wr_;
      auto flx = flx1;
      const int is_ = is, js_ = js, ks_ = ks;
      wellbalance::AddFaceEquilibrium(eos, floors, nhyd, m, k, j, i, weqf, wl, wr);
      SolveFace<rsolver_method_, IVX>(eos, indcs, size, coord,
                                      m, k, j, i, is_, js_, ks_, wl, wr, flx);
      for (int n=0; n<nhyd; ++n) {flx(m,n,k,j,i) -= feq(m,n,k,j,i);}
    });
    if (nvars > nhyd) {
      par_for("hflux_x1_scalars_wb", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real fd = flx1(m,IDN,k,j,i) + feq(m,IDN,k,j,i);
        for (int n=nhyd; n<nvars; ++n) {
          flx1(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
        }
      });
    }
  }

  // x2 direction
  if (pmy_pack->pmesh->multi_d) {
    auto &flx2 = uflx.x2f;
    auto weqf = w_eq_face.x2f;
    auto feq = f_eq.x2f;
    ReconDispatch<IVY>(recon_method_, "hflux_x2_recon_wb", nmb1,
        ktl, ktu, jl2-1, ju2, itl, itu, eos_, false, nvars, wp_, wl_, wr_);
    if (recon_method_ == ReconstructionMethod::fv4 && ppm_bc_closure) {
      PPMBoundaryClosure<IVY>("hflux_x2_ppm_closure_wb", nmb1, ktl, ktu, itl, itu, indcs_,
          pmy_pack->pmb->mb_bcs, eos_, false, nvars, wp_, wl_, wr_);
    }
    par_for("hflux_x2_rsolve_wb", DevExeSpace(), 0, nmb1, ktl, ktu, jl2, ju2, itl, itu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // local copies of the solver inputs (required for CUDA 11.6+, as in hydro_fluxes)
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      auto wl = wl_;
      auto wr = wr_;
      auto flx = flx2;
      const int is_ = is, js_ = js, ks_ = ks;
      wellbalance::AddFaceEquilibrium(eos, floors, nhyd, m, k, j, i, weqf, wl, wr);
      SolveFace<rsolver_method_, IVY>(eos, indcs, size, coord,
                                      m, k, j, i, is_, js_, ks_, wl, wr, flx);
      for (int n=0; n<nhyd; ++n) {flx(m,n,k,j,i) -= feq(m,n,k,j,i);}
    });
    if (nvars > nhyd) {
      par_for("hflux_x2_scalars_wb", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real fd = flx2(m,IDN,k,j,i) + feq(m,IDN,k,j,i);
        for (int n=nhyd; n<nvars; ++n) {
          flx2(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
        }
      });
    }
  }

  // x3 direction
  if (pmy_pack->pmesh->three_d) {
    auto &flx3 = uflx.x3f;
    auto weqf = w_eq_face.x3f;
    auto feq = f_eq.x3f;
    ReconDispatch<IVZ>(recon_method_, "hflux_x3_recon_wb", nmb1,
        kl3-1, ku3, jtl, jtu, itl, itu, eos_, false, nvars, wp_, wl_, wr_);
    if (recon_method_ == ReconstructionMethod::fv4 && ppm_bc_closure) {
      PPMBoundaryClosure<IVZ>("hflux_x3_ppm_closure_wb", nmb1, jtl, jtu, itl, itu, indcs_,
          pmy_pack->pmb->mb_bcs, eos_, false, nvars, wp_, wl_, wr_);
    }
    par_for("hflux_x3_rsolve_wb", DevExeSpace(), 0, nmb1, kl3, ku3, jtl, jtu, itl, itu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // local copies of the solver inputs (required for CUDA 11.6+, as in hydro_fluxes)
      auto eos = eos_;
      auto indcs = indcs_;
      auto size = size_;
      auto coord = coord_;
      auto wl = wl_;
      auto wr = wr_;
      auto flx = flx3;
      const int is_ = is, js_ = js, ks_ = ks;
      wellbalance::AddFaceEquilibrium(eos, floors, nhyd, m, k, j, i, weqf, wl, wr);
      SolveFace<rsolver_method_, IVZ>(eos, indcs, size, coord,
                                      m, k, j, i, is_, js_, ks_, wl, wr, flx);
      for (int n=0; n<nhyd; ++n) {flx(m,n,k,j,i) -= feq(m,n,k,j,i);}
    });
    if (nvars > nhyd) {
      par_for("hflux_x3_scalars_wb", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real fd = flx3(m,IDN,k,j,i) + feq(m,IDN,k,j,i);
        for (int n=nhyd; n<nvars; ++n) {
          flx3(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
        }
      });
    }
  }
  return;
}

// explicit instantiations (non-relativistic dynamic solvers only)
template void Hydro::WBEquilibriumFluxes<Hydro_RSolver::llf>();
template void Hydro::WBEquilibriumFluxes<Hydro_RSolver::hlle>();
template void Hydro::WBEquilibriumFluxes<Hydro_RSolver::hllc>();
template void Hydro::WBEquilibriumFluxes<Hydro_RSolver::lhllc>();
template void Hydro::WBEquilibriumFluxes<Hydro_RSolver::roe>();
template void Hydro::WBCalculateFluxes<Hydro_RSolver::llf>(Driver *d, int stage);
template void Hydro::WBCalculateFluxes<Hydro_RSolver::hlle>(Driver *d, int stage);
template void Hydro::WBCalculateFluxes<Hydro_RSolver::hllc>(Driver *d, int stage);
template void Hydro::WBCalculateFluxes<Hydro_RSolver::lhllc>(Driver *d, int stage);
template void Hydro::WBCalculateFluxes<Hydro_RSolver::roe>(Driver *d, int stage);

}  // namespace hydro
