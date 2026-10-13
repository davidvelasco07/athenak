//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_wb.cpp
//! \brief Well-balanced (WB) scheme for non-relativistic ideal MHD.  Extension of the
//! hydro scheme (hydro_wb.cpp) to a STATIC magnetohydrostatic equilibrium
//!   grad(p_eq) = rho_eq g + J_eq x B_eq,  v_eq = 0.
//!
//! - u0 holds U' = U - U_eq (E includes 1/2|B|^2); w0/bcc0 hold the full state.
//! - b0 keeps the FULL face field.  Because v_eq = 0 the equilibrium EMF vanishes, so CT
//!   stays exact provided (i) the face EMFs are perturbation EMFs (the face equilibrium
//!   EMF returned by the Riemann solver is subtracted, see f_eq/e*_eq), (ii) the UCT
//!   staggered-B edge reconstruction acts on b0 - b_eq (mhd_corner_e.cpp), and (iii) the
//!   CT update is written in perturbation form, b = b_eq + gam0*(b-b_eq) + gam1*(b1-b_eq)
//!   - dt*curl(E), since (2/3)X + (1/3)X != X bitwise (mhd_ct.cpp).
//! - Fluxes: reconstruct W' and B' = bcc0 - bcc_eq without floors, add the single-valued
//!   face equilibrium (4th-order interpolation of the equilibrium cell averages; the
//!   normal component is the face field itself), floor the full states, solve, subtract
//!   the Riemann flux and face EMFs of the face equilibrium (same solver).

#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen/pgen.hpp"
#include "mhd.hpp"
#include "hydro/well_balance.hpp"
#include "reconstruct/recon.hpp"
#include "reconstruct/fv.hpp"
#include "mhd/rsolvers/solve_face_mhd.hpp"
#include "mhd/mhd_wb.hpp"

namespace mhd {


//----------------------------------------------------------------------------------------
//! \fn void MHD::WBBuildFullState()
//! \brief Given primitive cell averages w and face fields b (full arrays), compute the
//! cell-centred field bcc and the conserved state u with EXACTLY the arithmetic of the
//! ConToPrim task, and overwrite w with ConsToPrim(u).  Used for the equilibrium and by
//! problem generators for the unperturbed initial state, so that U - U_eq is exactly
//! zero wherever no perturbation is added.

void MHD::WBBuildFullState(DvceArray5D<Real> &w, const DvceFaceFld4D<Real> &b,
                           DvceArray5D<Real> &bcc, DvceArray5D<Real> &u) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1m1 = indcs.nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
  DvceArray5D<Real> wtmp("wb_wtmp", w.extent(0), w.extent(1), w.extent(2), w.extent(3),
                         w.extent(4));
  Kokkos::deep_copy(bcc, 0.0);
  // pass 1: bcc from the face field (fluid part of this pass is discarded).  It converts
  // with bcc = 0 and inverts with the real field, so wherever B^2/2 > p/(gamma-1) the
  // discarded fluid state hits the energy floor: keep those floors out of the event
  // counters, or the admissibility check (WBSetEquilibrium) rejects a perfectly valid
  // strong-field equilibrium.
  auto &ec = pmy_pack->pmesh->ecounter;
  const int nfl_d = ec.neos_dfloor, nfl_e = ec.neos_efloor, nfl_t = ec.neos_tfloor;
  peos->PrimToCons(w, bcc, u, 0, n1m1, 0, n2m1, 0, n3m1);
  peos->ConsToPrim(u, b, wtmp, bcc, false, 0, n1m1, 0, n2m1, 0, n3m1);
  ec.neos_dfloor = nfl_d; ec.neos_efloor = nfl_e; ec.neos_tfloor = nfl_t;
  // pass 2: conserved state with that bcc, then primitives as the task would find them
  peos->PrimToCons(w, bcc, u, 0, n1m1, 0, n2m1, 0, n3m1);
  peos->ConsToPrim(u, b, w, bcc, false, 0, n1m1, 0, n2m1, 0, n3m1);
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::WBInitialize()
//! \brief Called by the ProblemGenerator constructor after the pgen (Mesh::pgen is not
//! assigned yet, hence the explicit pointer).  Builds the equilibrium and, on new runs,
//! converts u0 to the perturbation.

void MHD::WBInitialize(ProblemGenerator *ppgen, bool restart) {
  Mesh *pm = pmy_pack->pmesh;
  wb_eq_func = ppgen->user_eq_func;
  if (wb_eq_func == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> well_balanced/wb_diagnostics requires the problem "
              << "generator to enroll user_eq_func" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!well_balanced) {   // wb_diagnostics only
    WBSetEquilibrium();
    return;
  }
  std::string err;
  if (pmy_pack->phydro != nullptr) {
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
  }
  // Physical BCs, restriction and prolongation act on the face field: with SMR/AMR or a
  // boundary other than periodic/user, b0 is switched to the perturbation B' for the
  // boundary phase (WBFaceToPerturbation/WBFaceToFull, see mhd.hpp).  Periodic ghosts
  // of the full field and user ghosts holding the equilibrium are exact as they are.
  wb_b_comm_pert = pm->multilevel;
  for (int f=0; f<6; ++f) {
    if (pm->mesh_bcs[f] != BoundaryFlag::periodic && pm->mesh_bcs[f] != BoundaryFlag::user
        && pm->mesh_bcs[f] != BoundaryFlag::undef) {
      wb_b_comm_pert = true;
    }
  }
  if (!err.empty()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> well_balanced=true " << err << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ppgen->user_bcs && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<mhd> well_balanced=true: user boundary functions write u0 (the "
              << "PERTURBATION U') and b0 (the FULL face field)" << std::endl;
  }

  WBSetEquilibrium();

  if (!restart) {
    auto &u0_ = u0;
    auto &ueq_ = u_eq;
    const int n1 = static_cast<int>(u0.extent(4));
    const int n2 = static_cast<int>(u0.extent(3));
    const int n3 = static_cast<int>(u0.extent(2));
    par_for("wb_to_pert_mhd", DevExeSpace(), 0, (pmy_pack->nmb_thispack-1), 0, nmhd-1,
            0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      u0_(m,n,k,j,i) -= ueq_(m,n,k,j,i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::WBSetEquilibrium()
//! \brief (Re)build the equilibrium: the pgen fills w_eq (primitive cell averages) and
//! b_eq (face fields) over the full arrays; bcc_eq/u_eq/w_eq then follow from the task
//! arithmetic; then the face equilibrium and its Riemann flux/EMFs.

void MHD::WBSetEquilibrium() {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int n1m1 = indcs.nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nmhd_ = nmhd;

  auto &ec0 = pm->ecounter;
  const int nfl0[3] = {ec0.neos_dfloor, ec0.neos_efloor, ec0.neos_tfloor};
  Kokkos::deep_copy(w_eq, 0.0);
  Kokkos::deep_copy(b_eq.x1f, 0.0);
  Kokkos::deep_copy(b_eq.x2f, 0.0);
  Kokkos::deep_copy(b_eq.x3f, 0.0);
  wb_eq_func(pm);
  WBBuildFullState(w_eq, b_eq, bcc_eq, u_eq);

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

  // static equilibrium only
  {
    auto &weq_ = w_eq;
    Real vmax = 0.0;
    Kokkos::parallel_reduce("wb_vcheck_mhd",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0,0,0,0},
                                             {nmb1+1, n3m1+1, n2m1+1, n1m1+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &vm) {
        vm = fmax(vm, fabs(weq_(m,IVX,k,j,i)));
        vm = fmax(vm, fabs(weq_(m,IVY,k,j,i)));
        vm = fmax(vm, fabs(weq_(m,IVZ,k,j,i)));
      }, Kokkos::Max<Real>(vmax));
    if (vmax != 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd> well_balanced=true requires a static equilibrium "
                << "(user_eq_func returned max|v_eq| = " << vmax << ")" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (!well_balanced) return;

  // face equilibrium: central interpolation of the cell averages (fluid and cc-B) at the
  // order of the base scheme (fv6/fv8: their own interpolant, 4th order otherwise; see
  // Hydro::WBSetEquilibrium), valid on faces req .. n-req, which has to cover every
  // face the flux kernels visit (is-ext .. ie+1+ext, ext = the MOOD / FOFC halo)
  const int req = FVFaceRadius(recon_method);
  const auto rm_ = recon_method;
  {
    const int ext = use_mood ? mood_halo0 : (use_fofc ? 1 : 0);
    if (ng < req + ext) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd> well_balanced=true needs nghost >= "
                << req + ext << " (face-equilibrium stencil " << req
                << " + fallback halo " << ext << "), but <mesh>/nghost=" << ng
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  auto &weq_ = w_eq;
  auto &beq_ = bcc_eq;
  auto &eos_ = peos->eos_data;
  {
    auto wf = w_eq_face.x1f;
    auto bf = bcc_eq_face.x1f;
    par_for("wb_eqface_x1_mhd", DevExeSpace(), 0, nmb1, 0, n3m1, 0, n2m1, req, n1m1-req+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int n=0; n<nmhd_; ++n) {
        Real q = FVFaceAt(rm_, [&](const int o) {return weq_(m,n,k,j,i+o);});
        if (n == IDN) {q = fmax(q, eos_.dfloor);}
        if (n == IEN) {q = fmax(q, eos_.pfloor/(eos_.gamma - 1.0));}
        wf(m,n,k,j,i) = q;
      }
      for (int n=0; n<3; ++n) {
        bf(m,n,k,j,i) = FVFaceAt(rm_, [&](const int o) {return beq_(m,n,k,j,i+o);});
      }
    });
  }
  if (pm->multi_d) {
    auto wf = w_eq_face.x2f;
    auto bf = bcc_eq_face.x2f;
    par_for("wb_eqface_x2_mhd", DevExeSpace(), 0, nmb1, 0, n3m1, req, n2m1-req+1, 0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int n=0; n<nmhd_; ++n) {
        Real q = FVFaceAt(rm_, [&](const int o) {return weq_(m,n,k,j+o,i);});
        if (n == IDN) {q = fmax(q, eos_.dfloor);}
        if (n == IEN) {q = fmax(q, eos_.pfloor/(eos_.gamma - 1.0));}
        wf(m,n,k,j,i) = q;
      }
      for (int n=0; n<3; ++n) {
        bf(m,n,k,j,i) = FVFaceAt(rm_, [&](const int o) {return beq_(m,n,k,j+o,i);});
      }
    });
  }
  if (pm->three_d) {
    auto wf = w_eq_face.x3f;
    auto bf = bcc_eq_face.x3f;
    par_for("wb_eqface_x3_mhd", DevExeSpace(), 0, nmb1, req, n3m1-req+1, 0, n2m1, 0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int n=0; n<nmhd_; ++n) {
        Real q = FVFaceAt(rm_, [&](const int o) {return weq_(m,n,k+o,j,i);});
        if (n == IDN) {q = fmax(q, eos_.dfloor);}
        if (n == IEN) {q = fmax(q, eos_.pfloor/(eos_.gamma - 1.0));}
        wf(m,n,k,j,i) = q;
      }
      for (int n=0; n<3; ++n) {
        bf(m,n,k,j,i) = FVFaceAt(rm_, [&](const int o) {return beq_(m,n,k+o,j,i);});
      }
    });
  }

  switch (rsolver_method) {
    case MHD_RSolver::llf:  WBEquilibriumFluxes<MHD_RSolver::llf>();  break;
    case MHD_RSolver::hlle: WBEquilibriumFluxes<MHD_RSolver::hlle>(); break;
    case MHD_RSolver::hlld: WBEquilibriumFluxes<MHD_RSolver::hlld>(); break;
    case MHD_RSolver::lhlld: WBEquilibriumFluxes<MHD_RSolver::lhlld>(); break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd> well_balanced=true supports rsolver = "
                << "llf|hlle|hlld|lhlld" << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::WBEquilibriumFluxes()
//! \brief f_eq and the face equilibrium EMFs = RS(W_eq,f, W_eq,f) with the normal field
//! b_eq, over every face with a valid face equilibrium.  The UCT face outputs written
//! here are scratch (recomputed by every CalculateFluxes before use); the same uct_flag
//! as the flux path is passed so the solver runs the same arithmetic.

template <MHD_RSolver rsolver_method_>
void MHD::WBEquilibriumFluxes() {
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
  const bool use_uct = (emf_method != MHD_EMF::ct_contact);
  const int uct_fl = (emf_method == MHD_EMF::uct_hlld) ? 2 : (use_uct ? 1 : 0);
  // L-HLLD: the equilibrium is at rest, so its carbuncle sensor is zero; the stage flux
  // of an unperturbed equilibrium then sees the same theta and f - f_eq cancels exactly
  if constexpr (rsolver_method_ == MHD_RSolver::lhlld) { Kokkos::deep_copy(sdet, 0.0); }
  auto sdet_ = sdet;
  const int req = FVFaceRadius(recon_method);   // faces with a valid face equilibrium

  {
    auto wf = w_eq_face.x1f;  auto bf = bcc_eq_face.x1f;  auto bx_ = b_eq.x1f;
    auto fe = f_eq.x1f;  auto ey_ = e3x1_eq;  auto ez_ = e2x1_eq;
    auto aL1 = aL_x1f;  auto dl1 = dL_x1f;  auto dr1 = dR_x1f;
    auto vt1 = vy_x1f;  auto vt2 = vz_x1f;
    par_for("wb_feq_x1_mhd", DevExeSpace(), 0, nmb1, 0, n3m1, 0, n2m1, req, n1m1-req+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      auto eos = eos_;  auto indcs = indcs_;  auto size = size_;  auto coord = coord_;
      const int is_ = is, js_ = js, ks_ = ks;
      SolveFaceMHD<rsolver_method_, IVX>(eos, indcs, size, coord, m, k, j, i,
                                         is_, js_, ks_, wf, wf, bf, bf, bx_, fe, ey_, ez_,
                                         uct_fl, aL1, dl1, dr1, vt1, vt2, sdet_);
    });
  }
  if (pm->multi_d) {
    auto wf = w_eq_face.x2f;  auto bf = bcc_eq_face.x2f;  auto bx_ = b_eq.x2f;
    auto fe = f_eq.x2f;  auto ey_ = e1x2_eq;  auto ez_ = e3x2_eq;
    auto aL2 = aL_x2f;  auto dl2 = dL_x2f;  auto dr2 = dR_x2f;
    auto vt1 = vz_x2f;  auto vt2 = vx_x2f;
    par_for("wb_feq_x2_mhd", DevExeSpace(), 0, nmb1, 0, n3m1, req, n2m1-req+1, 0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      auto eos = eos_;  auto indcs = indcs_;  auto size = size_;  auto coord = coord_;
      const int is_ = is, js_ = js, ks_ = ks;
      SolveFaceMHD<rsolver_method_, IVY>(eos, indcs, size, coord, m, k, j, i,
                                         is_, js_, ks_, wf, wf, bf, bf, bx_, fe, ey_, ez_,
                                         uct_fl, aL2, dl2, dr2, vt1, vt2, sdet_);
    });
  }
  if (pm->three_d) {
    auto wf = w_eq_face.x3f;  auto bf = bcc_eq_face.x3f;  auto bx_ = b_eq.x3f;
    auto fe = f_eq.x3f;  auto ey_ = e2x3_eq;  auto ez_ = e1x3_eq;
    auto aL3 = aL_x3f;  auto dl3 = dL_x3f;  auto dr3 = dR_x3f;
    auto vt1 = vx_x3f;  auto vt2 = vy_x3f;
    par_for("wb_feq_x3_mhd", DevExeSpace(), 0, nmb1, req, n3m1-req+1, 0, n2m1, 0, n1m1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      auto eos = eos_;  auto indcs = indcs_;  auto size = size_;  auto coord = coord_;
      const int is_ = is, js_ = js, ks_ = ks;
      SolveFaceMHD<rsolver_method_, IVZ>(eos, indcs, size, coord, m, k, j, i,
                                         is_, js_, ks_, wf, wf, bf, bf, bx_, fe, ey_, ez_,
                                         uct_fl, aL3, dl3, dr3, vt1, vt2, sdet_);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::WBFaceToPerturbation()
//! \brief b0 <- b0 - b_eq over the full face arrays (ghosts included), so that the
//! boundary phase (restriction, prolongation, physical and user BCs) acts on B'.  A
//! no-op unless the run needs it (wb_b_comm_pert) and b0 is in its full state.  Shared
//! block faces stay bitwise consistent: every block performs the same (b - e) + e.

TaskStatus MHD::WBFaceToPerturbation(Driver *pdrive, int stage) {
  if (!well_balanced || !wb_b_comm_pert || wb_b0_is_pert) return TaskStatus::complete;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto b1 = b0.x1f, b2 = b0.x2f, b3 = b0.x3f;
  auto q1 = b_eq.x1f, q2 = b_eq.x2f, q3 = b_eq.x3f;
  const int n1 = static_cast<int>(b0.x1f.extent(3));
  const int n2 = static_cast<int>(b0.x2f.extent(2));
  const int n3 = static_cast<int>(b0.x3f.extent(1));
  par_for("wb_b_to_pert", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (k < n3-1 && j < n2-1) {
      b1(m,k,j,i) -= q1(m,k,j,i);
    }
    if (k < n3-1 && i < n1-1) {
      b2(m,k,j,i) -= q2(m,k,j,i);
    }
    if (j < n2-1 && i < n1-1) {
      b3(m,k,j,i) -= q3(m,k,j,i);
    }
  });
  wb_b0_is_pert = true;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::WBFaceToFull()
//! \brief b0 <- b0 + b_eq, the inverse of WBFaceToPerturbation, after the physical BCs.

TaskStatus MHD::WBFaceToFull(Driver *pdrive, int stage) {
  if (!well_balanced || !wb_b0_is_pert) return TaskStatus::complete;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto b1 = b0.x1f, b2 = b0.x2f, b3 = b0.x3f;
  auto q1 = b_eq.x1f, q2 = b_eq.x2f, q3 = b_eq.x3f;
  const int n1 = static_cast<int>(b0.x1f.extent(3));
  const int n2 = static_cast<int>(b0.x2f.extent(2));
  const int n3 = static_cast<int>(b0.x3f.extent(1));
  par_for("wb_b_to_full", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (k < n3-1 && j < n2-1) {
      b1(m,k,j,i) += q1(m,k,j,i);
    }
    if (k < n3-1 && i < n1-1) {
      b2(m,k,j,i) += q2(m,k,j,i);
    }
    if (j < n2-1 && i < n1-1) {
      b3(m,k,j,i) += q3(m,k,j,i);
    }
  });
  wb_b0_is_pert = false;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::WBFormPerturbation()
//! \brief W' and B' = bcc0 - bcc_eq of the stage-input state over the full arrays.

void MHD::WBFormPerturbation() {
  auto &u0_ = u0;
  auto &w0_ = w0;
  auto &weq_ = w_eq;
  auto &wp_ = w_pert;
  auto &bcc_ = bcc0;
  auto &bcceq_ = bcc_eq;
  auto &bp_ = bcc_pert;
  const int nmhd_ = nmhd;
  const int nvars = nmhd + nscalars;
  const int n1 = static_cast<int>(u0.extent(4));
  const int n2 = static_cast<int>(u0.extent(3));
  const int n3 = static_cast<int>(u0.extent(2));
  par_for("wb_wpert_mhd", DevExeSpace(), 0, (pmy_pack->nmb_thispack-1),
          0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    wp_(m,IDN,k,j,i) = u0_(m,IDN,k,j,i);
    wp_(m,IVX,k,j,i) = w0_(m,IVX,k,j,i);
    wp_(m,IVY,k,j,i) = w0_(m,IVY,k,j,i);
    wp_(m,IVZ,k,j,i) = w0_(m,IVZ,k,j,i);
    wp_(m,IEN,k,j,i) = w0_(m,IEN,k,j,i) - weq_(m,IEN,k,j,i);
    for (int n=nmhd_; n<nvars; ++n) {
      wp_(m,n,k,j,i) = w0_(m,n,k,j,i);
    }
    for (int n=0; n<3; ++n) {
      bp_(m,n,k,j,i) = bcc_(m,n,k,j,i) - bcceq_(m,n,k,j,i);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::WBCalculateFluxes()
//! \brief Well-balanced counterpart of CalculateFluxes (same face and transverse ranges).

template <MHD_RSolver rsolver_method_>
void MHD::WBCalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  const int nmhd_ = nmhd;
  int nvars = nmhd + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  const bool floors = (recon_method_ == ReconstructionMethod::ppmx ||
                       recon_method_ == ReconstructionMethod::wenoz ||
                       recon_method_ == ReconstructionMethod::teno ||
                       recon_method_ == ReconstructionMethod::fv4 ||
                       recon_method_ == ReconstructionMethod::fv6 ||
                       recon_method_ == ReconstructionMethod::fv8);

  const int mext = use_mood ? mood_halo0 : (use_fofc ? 1 : 0);
  int il1 = is-mext, iu1 = ie+1+mext;
  int jl2 = js-mext, ju2 = je+1+mext;
  int kl3 = ks-mext, ku3 = ke+1+mext;
  const auto emf_method_ = emf_method;
  const bool use_uct = (emf_method_ != MHD_EMF::ct_contact);
  int uct_ext = 1;
  if (use_uct) {
    // same transverse reach of the UCT face data as CalculateFluxes (mhd_fluxes.cpp):
    // the edge stencil, plus the MOOD light-cone halo over which the detector composes
    // the staggered candidate
    uct_ext = ReconEdgeReach(recon_method_);
    if (use_mood) {
      uct_ext += mood_halo0;
    }
  }
  const int ext = std::max(uct_ext, mext);
  const int uct_fl = (emf_method_ == MHD_EMF::uct_hlld) ? 2 : (use_uct ? 1 : 0);
  // L-HLLD carbuncle sensor from the stage-input (full) primitives
  if constexpr (rsolver_method_ == MHD_RSolver::lhlld) { LHLLDShockSensor(); }
  auto sdet_ = sdet;

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &wp_ = w_pert;
  auto &bp_ = bcc_pert;
  auto wl_ = wl3d;
  auto wr_ = wr3d;
  auto bl_ = bl3d;
  auto br_ = br3d;

  // x1 direction
  {
    auto &flx1 = uflx.x1f;
    auto &bx_ = b0.x1f;
    auto &e31 = e3x1;
    auto &e21 = e2x1;
    auto e31q = e3x1_eq;  auto e21q = e2x1_eq;
    auto weqf = w_eq_face.x1f;  auto beqf = bcc_eq_face.x1f;  auto feq = f_eq.x1f;
    auto aL1 = aL_x1f;  auto dl1 = dL_x1f;  auto dr1 = dR_x1f;
    auto vt1_1 = vy_x1f;  auto vt2_1 = vz_x1f;
    int jl = js, ju = je, kl = ks, ku = ke;
    if (pmy_pack->pmesh->multi_d) { jl = js-ext; ju = je+ext; }
    if (pmy_pack->pmesh->three_d) { kl = ks-ext; ku = ke+ext; }
    ReconDispatch<IVX>(recon_method_, "mflux_x1_recon_w_wb", nmb1,
        kl, ku, jl, ju, il1-1, iu1, eos_, false, nvars, wp_, wl_, wr_);
    ReconDispatch<IVX>(recon_method_, "mflux_x1_recon_b_wb", nmb1,
        kl, ku, jl, ju, il1-1, iu1, eos_, false, 3, bp_, bl_, br_);
    par_for("mflux_x1_rsolve_wb", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il1, iu1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      auto eos = eos_;  auto indcs = indcs_;  auto size = size_;  auto coord = coord_;
      auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
      auto bx = bx_;  auto flx = flx1;  auto eyl = e31;  auto ezl = e21;
      const int is_ = is, js_ = js, ks_ = ks;
      AddFaceEquilibriumMHD(eos, floors, nmhd_, m, k, j, i, weqf, beqf, wl, wr, bl, br);
      SolveFaceMHD<rsolver_method_, IVX>(eos, indcs, size, coord, m, k, j, i,
                                         is_, js_, ks_, wl, wr, bl, br, bx, flx, eyl, ezl,
                                         uct_fl, aL1, dl1, dr1, vt1_1, vt2_1, sdet_);
      for (int n=0; n<nmhd_; ++n) {flx(m,n,k,j,i) -= feq(m,n,k,j,i);}
      eyl(m,k,j,i) -= e31q(m,k,j,i);
      ezl(m,k,j,i) -= e21q(m,k,j,i);
    });
    if (nvars > nmhd_) {
      par_for("mflux_x1_scalars_wb", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real fd = flx1(m,IDN,k,j,i) + feq(m,IDN,k,j,i);
        for (int n = nmhd_; n < nvars; ++n) {
          flx1(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
        }
      });
    }
  }

  // x2 direction
  if (pmy_pack->pmesh->multi_d) {
    auto &flx2 = uflx.x2f;
    auto &by_ = b0.x2f;
    auto &e12 = e1x2;
    auto &e32 = e3x2;
    auto e12q = e1x2_eq;  auto e32q = e3x2_eq;
    auto weqf = w_eq_face.x2f;  auto beqf = bcc_eq_face.x2f;  auto feq = f_eq.x2f;
    auto aL2 = aL_x2f;  auto dl2 = dL_x2f;  auto dr2 = dR_x2f;
    auto vt1_2 = vz_x2f;  auto vt2_2 = vx_x2f;
    int kl = ks, ku = ke;
    if (pmy_pack->pmesh->three_d) { kl = ks-ext; ku = ke+ext; }
    ReconDispatch<IVY>(recon_method_, "mflux_x2_recon_w_wb", nmb1,
        kl, ku, jl2-1, ju2, is-ext, ie+ext, eos_, false, nvars, wp_, wl_, wr_);
    ReconDispatch<IVY>(recon_method_, "mflux_x2_recon_b_wb", nmb1,
        kl, ku, jl2-1, ju2, is-ext, ie+ext, eos_, false, 3, bp_, bl_, br_);
    par_for("mflux_x2_rsolve_wb", DevExeSpace(), 0, nmb1, kl, ku, jl2, ju2,
            is-ext, ie+ext,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      auto eos = eos_;  auto indcs = indcs_;  auto size = size_;  auto coord = coord_;
      auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
      auto bx = by_;  auto flx = flx2;  auto eyl = e12;  auto ezl = e32;
      const int is_ = is, js_ = js, ks_ = ks;
      AddFaceEquilibriumMHD(eos, floors, nmhd_, m, k, j, i, weqf, beqf, wl, wr, bl, br);
      SolveFaceMHD<rsolver_method_, IVY>(eos, indcs, size, coord, m, k, j, i,
                                         is_, js_, ks_, wl, wr, bl, br, bx, flx, eyl, ezl,
                                         uct_fl, aL2, dl2, dr2, vt1_2, vt2_2, sdet_);
      for (int n=0; n<nmhd_; ++n) {flx(m,n,k,j,i) -= feq(m,n,k,j,i);}
      eyl(m,k,j,i) -= e12q(m,k,j,i);
      ezl(m,k,j,i) -= e32q(m,k,j,i);
    });
    if (nvars > nmhd_) {
      par_for("mflux_x2_scalars_wb", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real fd = flx2(m,IDN,k,j,i) + feq(m,IDN,k,j,i);
        for (int n = nmhd_; n < nvars; ++n) {
          flx2(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
        }
      });
    }
  }

  // x3 direction
  if (pmy_pack->pmesh->three_d) {
    auto &flx3 = uflx.x3f;
    auto &bz_ = b0.x3f;
    auto &e23 = e2x3;
    auto &e13 = e1x3;
    auto e23q = e2x3_eq;  auto e13q = e1x3_eq;
    auto weqf = w_eq_face.x3f;  auto beqf = bcc_eq_face.x3f;  auto feq = f_eq.x3f;
    auto aL3 = aL_x3f;  auto dl3 = dL_x3f;  auto dr3 = dR_x3f;
    auto vt1_3 = vx_x3f;  auto vt2_3 = vy_x3f;
    ReconDispatch<IVZ>(recon_method_, "mflux_x3_recon_w_wb", nmb1,
        kl3-1, ku3, js-ext, je+ext, is-ext, ie+ext, eos_, false, nvars, wp_, wl_, wr_);
    ReconDispatch<IVZ>(recon_method_, "mflux_x3_recon_b_wb", nmb1,
        kl3-1, ku3, js-ext, je+ext, is-ext, ie+ext, eos_, false, 3, bp_, bl_, br_);
    par_for("mflux_x3_rsolve_wb", DevExeSpace(), 0, nmb1, kl3, ku3, js-ext, je+ext,
            is-ext, ie+ext,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      auto eos = eos_;  auto indcs = indcs_;  auto size = size_;  auto coord = coord_;
      auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
      auto bx = bz_;  auto flx = flx3;  auto eyl = e23;  auto ezl = e13;
      const int is_ = is, js_ = js, ks_ = ks;
      AddFaceEquilibriumMHD(eos, floors, nmhd_, m, k, j, i, weqf, beqf, wl, wr, bl, br);
      SolveFaceMHD<rsolver_method_, IVZ>(eos, indcs, size, coord, m, k, j, i,
                                         is_, js_, ks_, wl, wr, bl, br, bx, flx, eyl, ezl,
                                         uct_fl, aL3, dl3, dr3, vt1_3, vt2_3, sdet_);
      for (int n=0; n<nmhd_; ++n) {flx(m,n,k,j,i) -= feq(m,n,k,j,i);}
      eyl(m,k,j,i) -= e23q(m,k,j,i);
      ezl(m,k,j,i) -= e13q(m,k,j,i);
    });
    if (nvars > nmhd_) {
      par_for("mflux_x3_scalars_wb", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real fd = flx3(m,IDN,k,j,i) + feq(m,IDN,k,j,i);
        for (int n = nmhd_; n < nvars; ++n) {
          flx3(m,n,k,j,i) = (fd >= 0.0) ? fd*wl_(m,n,k,j,i) : fd*wr_(m,n,k,j,i);
        }
      });
    }
  }
}

// explicit instantiations (non-relativistic dynamic solvers only)
template void MHD::WBEquilibriumFluxes<MHD_RSolver::llf>();
template void MHD::WBEquilibriumFluxes<MHD_RSolver::hlle>();
template void MHD::WBEquilibriumFluxes<MHD_RSolver::hlld>();
template void MHD::WBEquilibriumFluxes<MHD_RSolver::lhlld>();
template void MHD::WBCalculateFluxes<MHD_RSolver::llf>(Driver *d, int stage);
template void MHD::WBCalculateFluxes<MHD_RSolver::hlle>(Driver *d, int stage);
template void MHD::WBCalculateFluxes<MHD_RSolver::hlld>(Driver *d, int stage);
template void MHD::WBCalculateFluxes<MHD_RSolver::lhlld>(Driver *d, int stage);

}  // namespace mhd
