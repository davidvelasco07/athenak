//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gmtf.cpp
//! \brief Problem generator for gravo-magneto-turbulent fragmentation

// C headers

// C++ headers
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>
#include <utility>
#include <random>
#include <vector>
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "gravity/gravity.hpp"
#include "particles/sink_positions.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "globals.hpp"
#include "driver/driver.hpp"
#include "particles/particles.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"

namespace {
Real sfe_term_ = -1.0;      // terminal star-formation efficiency; <= 0 disables
Real mtot0_ = -1.0;         // initial gas mass, set on the first history call
bool sfe_stop_announced_ = false;
Real njeans_ = 0.0, jeans_derefine_ = 2.5, sink_buffer_ = 4.0;
bool sink_protect_ = true, amr_diagnostics_ = false;
void GMTFRefinement(MeshBlockPack *pmbp);
void GMTFHistory(HistoryData *pdata, Mesh *pm);
void SeedSinks(ParameterInput *pin, MeshBlockPack *pmbp);
}  // namespace

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Enroll callbacks before the restart return; input settings must survive restart.
  sfe_term_ = pin->GetOrAddReal("problem", "sfe_term", -1.0);
  // The stopping test lives inside GMTFHistory, so sfe_term does nothing unless that
  // function is enrolled. Gating enrolment on user_hist alone made `sfe_term = 0.15` a
  // silent no-op in every deck that omitted the flag: the 256^3 and 512^3 GMTF runs each
  // ran to tlim, ~10x past the intended stop, and nothing in the output said why. A
  // parameter that is read must not be quietly ignorable, so enrol whenever EITHER the
  // extra history columns were asked for OR a stopping efficiency was set. The only side
  // effect of the second case is that such a run also writes <basename>.user.hst, which
  // carries m_sink, n_sink and SFE -- worth having whenever sfe_term is in play.
  //
  // BOTH members must be set. `user_hist_func` is only the callback; whether the history
  // output has a UserDefined entry to invoke it from is gated separately on the
  // `user_hist` flag (see HistoryOutput::HistoryOutput, which tests pgen->user_hist).
  // Setting the function pointer alone is a silent no-op -- the callback is simply never
  // reached. pgen.cpp initialises user_hist from the input before UserProblem runs and
  // validates it afterwards, so overriding it here is both safe and in time.
  if (pin->GetOrAddBoolean("problem", "user_hist", false) || sfe_term_ > 0.0) {
    user_hist = true;
    user_hist_func = GMTFHistory;
  }
  amr_diagnostics_ = pin->GetOrAddBoolean("problem", "amr_diagnostics", false);
  if (amr_diagnostics_) { user_hist = true; user_hist_func = GMTFHistory; }
  if (pmy_mesh_->adaptive || amr_diagnostics_) {
    njeans_ = pin->GetOrAddReal("problem", "njeans", 8.0);
    jeans_derefine_ = pin->GetOrAddReal("problem", "jeans_derefine", 2.5);
    sink_buffer_ = pin->GetOrAddReal("problem", "sink_buffer_cells", 4.0);
    sink_protect_ = pin->GetOrAddBoolean("problem", "sink_protect", true);
    auto pack = pmy_mesh_->pmb_pack;
    if (pack->phydro == nullptr || pack->pgrav == nullptr ||
        pack->phydro->peos->eos_data.is_ideal || pack->pgrav->four_pi_G <= 0.0 ||
        !std::isfinite(njeans_) || njeans_ <= 0.0 ||
        !std::isfinite(jeans_derefine_) || jeans_derefine_ <= 2.0 ||
        !std::isfinite(sink_buffer_) || (sink_protect_ && sink_buffer_ < 4.0)) {
      Kokkos::abort("GMTF Jeans AMR needs isothermal hydro, positive gravity/NJeans, "
                    "derefinement ratio > 2 and protected sink buffer >= 4 cells");
    }
    for (int face=0; face<6; ++face) {
      if (pmy_mesh_->mesh_bcs[face] != BoundaryFlag::periodic) {
        Kokkos::abort("GMTF Jeans AMR currently requires a periodic box");
      }
    }
    if (pmy_mesh_->adaptive) user_ref_func = GMTFRefinement;
  }
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  DvceArray5D<Real> u0;

  if (pmbp->phydro != nullptr) {
    // HYDRO -----------------------------------
    if (pmbp->phydro->peos->eos_data.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem requires isothermal eos" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmbp->phydro->peos->eos_data.iso_cs != 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem takes sound speed as unit velocity."
                << " set iso_sound_speed = 1.0 in the input file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    u0 = pmbp->phydro->u0;
  } else if (pmbp->pmhd != nullptr) {
    // MHD ------------------------------------
    if (pmbp->pmhd->peos->eos_data.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem requires isothermal eos" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmbp->pmhd->peos->eos_data.iso_cs != 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "this problem takes sound speed as unit velocity."
                << " set iso_sound_speed = 1.0 in the input file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    u0 = pmbp->pmhd->u0;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "this problem can only be run with Hydro and/or MHD, but no "
              << "<hydro> or <mhd> block in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;

  // capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;


  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    // Set initial conditions
    par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      // TODO(SMOON) SCALARS?
    });
  }

  // Initialize MHD variables ---------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &b0 = pmbp->pmhd->b0;
    Real mu_phi = pin->GetReal("problem", "mu_phi");
    Real lbox = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
    Real B0 = M_PI*lbox / mu_phi;

    // Set initial conditions
    par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = B0;
      if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
      if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
      if (k==ke) {b0.x3f(m,k+1,j,i) = B0;}
    });
  }

  // Add turbulent velocity perturbations ---------------------------------
  Real mach = pin->GetReal("problem", "Mach");
  int rseed = pin->GetInteger("problem", "rseed");
  int nlow = pin->GetInteger("problem", "nlow");
  int nhigh = pin->GetInteger("problem", "nhigh");
  Real expo = pin->GetReal("problem", "expo");

  if (mach <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Mach number must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nhigh <= nlow) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "problem/nhigh must be greater than problem/nlow" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // TODO(SMOON) Check if cs=1.0

  Real x1size = pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min;
  Real x2size = pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min;
  Real x3size = pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min;

  // The requirement is CUBIC CELLS, not a cubic domain. The multigrid solver separately
  // requires logically cubic MeshBlocks (multigrid.cpp:52), and with cubic cells a cubic
  // MeshBlock is also cubic in physical size -- but the global mesh may be any shape. That
  // matters for weak scaling: holding zones/GPU fixed on a cubic mesh forces the rank count
  // to be a perfect cube, so 128 and 256 GPUs are unreachable, while 2048x1024x1024 and
  // 2048x2048x1024 reach them exactly.
  Real dx1 = x1size/static_cast<Real>(pmbp->pmesh->mesh_indcs.nx1);
  Real dx2 = x2size/static_cast<Real>(pmbp->pmesh->mesh_indcs.nx2);
  Real dx3 = x3size/static_cast<Real>(pmbp->pmesh->mesh_indcs.nx3);
  if (!(dx1 == dx2 && dx1 == dx3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "this problem requires CUBIC CELLS: dx1,dx2,dx3 = " << dx1 << ", " << dx2
              << ", " << dx3 << std::endl
              << "  Scale the domain in the same ratio as the mesh, e.g. a 2048x1024x1024"
              << " mesh needs a 2:1:1 box." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Per-axis fundamental wavenumbers. The driving band is defined against the SHORTEST box
  // dimension, so nlow..nhigh always name the same PHYSICAL wavelengths whatever the aspect
  // ratio; on a cubic box every dk below is equal and this reduces exactly to the old code.
  const Real dkx = 2.0*M_PI/x1size;
  const Real dky = 2.0*M_PI/x2size;
  const Real dkz = 2.0*M_PI/x3size;
  const Real dk  = 2.0*M_PI/std::min(x1size, std::min(x2size, x3size));
  // widen the index ranges so |k| can still reach nhigh*dk along an elongated axis
  const int nix = static_cast<int>(std::ceil(nhigh*dk/dkx));
  const int niy = static_cast<int>(std::ceil(nhigh*dk/dky));
  const int niz = static_cast<int>(std::ceil(nhigh*dk/dkz));
  // exactly 1.0 on a cubic box, since dk is then the same expression as dkx/dky/dkz
  const Real rx = dkx/dk, ry = dky/dk, rz = dkz/dk;

  std::mt19937_64 gen(rseed);
  std::normal_distribution<Real> gauss(0.0, 1.0);

  std::vector<Real> host_kx, host_ky, host_kz;
  std::vector<Real> host_ax, host_ay, host_az;
  std::vector<Real> host_bx, host_by, host_bz;

  // Here, we utilize Hermitian symmetry to only keep non-redundant modes.
  // The points symmetric with respect to the origin are complex conjugates,
  // so we only need to keep one of them. Also, we discard DC component, i.e.,
  // kx=ky=kz=0, to avoid adding a net bulk velocity.
  for (int nkx = 0; nkx <= nix; ++nkx) {
    for (int nky = -niy; nky <= niy; ++nky) {
      for (int nkz = -niz; nkz <= niz; ++nkz) {
        if (nkx == 0) {
          // We are on the kx=0 plane
          if (nky < 0) continue;
          if (nky == 0 && nkz <= 0) continue;
        }

        Real kx = dkx*static_cast<Real>(nkx);
        Real ky = dky*static_cast<Real>(nky);
        Real kz = dkz*static_cast<Real>(nkz);
        Real kmag = std::sqrt(SQR(kx) + SQR(ky) + SQR(kz));
        // Select on PHYSICAL |k| measured in units of dk. Formed from the per-axis RATIOS
        // rather than as SQR(kmag/dk): on a cubic box rx = ry = rz = 1.0 exactly, so this
        // evaluates to nkx^2 + nky^2 + nkz^2 in exact integer-valued doubles and reproduces
        // the integer test it replaces bit-for-bit. Going through sqrt and back would let
        // rounding move a mode across the band edge and silently change the mode set.
        Real nmag2 = SQR(rx*static_cast<Real>(nkx)) + SQR(ry*static_cast<Real>(nky))
                   + SQR(rz*static_cast<Real>(nkz));
        if ( (SQR(static_cast<Real>(nlow)) <= nmag2) &&
             (nmag2 <= SQR(static_cast<Real>(nhigh))) ) {
          host_kx.push_back(kx);
          host_ky.push_back(ky);
          host_kz.push_back(kz);
          // Keep the unprojected isotropic field, equivalent to f_shear = 0.5 after
          // renormalization to the target Mach number.
          Real pcoeff = 1.0/std::pow(kmag, (expo + 2.0)/2.0);
          host_ax.push_back(pcoeff*gauss(gen));
          host_ay.push_back(pcoeff*gauss(gen));
          host_az.push_back(pcoeff*gauss(gen));
          host_bx.push_back(pcoeff*gauss(gen));
          host_by.push_back(pcoeff*gauss(gen));
          host_bz.push_back(pcoeff*gauss(gen));
        }
      }
    }
  }
  int nmode = static_cast<int>(host_kx.size());
  DualArray1D<Real> kx("kx", nmode);
  DualArray1D<Real> ky("ky", nmode);
  DualArray1D<Real> kz("kz", nmode);
  DualArray1D<Real> vxk_re("vxk_re", nmode);
  DualArray1D<Real> vyk_re("vyk_re", nmode);
  DualArray1D<Real> vzk_re("vzk_re", nmode);
  DualArray1D<Real> vxk_im("vxk_im", nmode);
  DualArray1D<Real> vyk_im("vyk_im", nmode);
  DualArray1D<Real> vzk_im("vzk_im", nmode);
  for (int n = 0; n < nmode; ++n) {
    kx.h_view(n) = host_kx[n];
    ky.h_view(n) = host_ky[n];
    kz.h_view(n) = host_kz[n];
    vxk_re.h_view(n) = host_ax[n];
    vyk_re.h_view(n) = host_ay[n];
    vzk_re.h_view(n) = host_az[n];
    vxk_im.h_view(n) = host_bx[n];
    vyk_im.h_view(n) = host_by[n];
    vzk_im.h_view(n) = host_bz[n];
  }

  auto sync_to_device = [](auto &arr) {
    arr.template modify<HostMemSpace>();
    arr.template sync<DevExeSpace>();
  };
  sync_to_device(kx);
  sync_to_device(ky);
  sync_to_device(kz);
  sync_to_device(vxk_re);
  sync_to_device(vyk_re);
  sync_to_device(vzk_re);
  sync_to_device(vxk_im);
  sync_to_device(vyk_im);
  sync_to_device(vzk_im);

  int nmb = pmbp->nmb_thispack;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int ng = indcs.ng;
  auto &size = pmbp->pmb->mb_size;
  DvceArray5D<Real> dv("vel_perturb", nmb, 3, nx3 + 2*ng, nx2 + 2*ng, nx1 + 2*ng);

  par_for("gmtf_build_turb", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    // Sum over all modes to construct the velocity perturbation in real space
    Real dv_x = 0.0;
    Real dv_y = 0.0;
    Real dv_z = 0.0;
    for (int n = 0; n < nmode; ++n) {
      Real kdotx = kx.d_view(n)*x1v + ky.d_view(n)*x2v + kz.d_view(n)*x3v;
      dv_x += vxk_re.d_view(n)*std::cos(kdotx) - vxk_im.d_view(n)*std::sin(kdotx);
      dv_y += vyk_re.d_view(n)*std::cos(kdotx) - vyk_im.d_view(n)*std::sin(kdotx);
      dv_z += vzk_re.d_view(n)*std::cos(kdotx) - vzk_im.d_view(n)*std::sin(kdotx);
    }
    dv(m,0,k,j,i) = dv_x;
    dv(m,1,k,j,i) = dv_y;
    dv(m,2,k,j,i) = dv_z;
  });

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const bool weighted = pmy_mesh_->multilevel;
  const Real volume = (pmy_mesh_->mesh_size.x1max-pmy_mesh_->mesh_size.x1min)*
                      (pmy_mesh_->mesh_size.x2max-pmy_mesh_->mesh_size.x2min)*
                      (pmy_mesh_->mesh_size.x3max-pmy_mesh_->mesh_size.x3min);
  if (weighted) {
    for (int v=0; v<3; ++v) {
      Real mean = 0.0;
      Kokkos::parallel_reduce("gmtf_amr_mean", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
      KOKKOS_LAMBDA(const int idx, Real &sum) {
        const int m=idx/nkji, k=(idx-m*nkji)/nji+ks;
        const int j=(idx-m*nkji-(k-ks)*nji)/nx1+js;
        const int i=(idx-m*nkji-(k-ks)*nji-(j-js)*nx1)+is;
        const auto b=size.d_view(m);
        sum += dv(m,v,k,j,i)*b.dx1*b.dx2*b.dx3;
      }, Kokkos::Sum<Real>(mean));
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE,&mean,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
      mean /= volume;
      par_for("gmtf_amr_zero_momentum", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m,int k,int j,int i) { dv(m,v,k,j,i) -= mean; });
    }
  }
  Real v2_sum = 0.0;
  Kokkos::parallel_reduce("gmtf_vrms",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &v2_sum_local) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    const auto b=size.d_view(m);
    const Real weight = weighted ? b.dx1*b.dx2*b.dx3 : 1.0;
    v2_sum_local += weight*(SQR(dv(m,0,k,j,i)) + SQR(dv(m,1,k,j,i))
                           + SQR(dv(m,2,k,j,i)));
  }, Kokkos::Sum<Real>(v2_sum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &v2_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Rescale the velocity perturbations to achieve the target Mach number
  long long Nx1 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx1);
  long long Nx2 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx2);
  long long Nx3 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx3);
  Real vrms = std::sqrt(v2_sum / (weighted ? volume : static_cast<Real>(Nx1*Nx2*Nx3)));
  par_for("gmtf_init_turb", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IM1,k,j,i) += mach/vrms*dv(m,0,k,j,i);
    u0(m,IM2,k,j,i) += mach/vrms*dv(m,1,k,j,i);
    u0(m,IM3,k,j,i) += mach/vrms*dv(m,2,k,j,i);
  });

  // Optional seeded sink population (benchmark mode); no-op unless nsink_seed > 0.
  SeedSinks(pin, pmbp);
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void GMTFHistory()
//! \brief gas mass/momenta, sink mass and count, and the star-formation efficiency.
//! Also applies the sfe_term stopping criterion: once SFE reaches it, tlim is pulled back
//! to the current time so the run ends at the close of this cycle by the NORMAL path --
//! final outputs are written and the usual "Terminating on time limit" message appears,
//! rather than aborting and losing the last dump.

// Jeans gas refinement OR finest-level sink protection. The density stencil includes
// ghosts, so a core approaching a block boundary requests refinement on both sides.
void GMTFRefinement(MeshBlockPack *pmbp) {
  auto pm = pmbp->pmesh;
  auto &flags = pm->pmr->refine_flag;
  const auto ix = pm->mb_indcs;
  const int ni=ix.nx1+2*ix.ng, nj=ix.nx2+2*ix.ng, nk=ix.nx3+2*ix.ng;
  const int nmb=pmbp->nmb_thispack, gids=pmbp->gids;
  auto u=pmbp->phydro->u0;
  auto sizes=pmbp->pmb->mb_size.d_view;
  const Real cs=pmbp->phydro->peos->eos_data.iso_cs;
  const Real factor=4.0*M_PI*M_PI*cs*cs/pmbp->pgrav->four_pi_G;
  const Real target2=njeans_*njeans_, derefine2=target2*jeans_derefine_*jeans_derefine_;
  const Real buffer=sink_buffer_;
  const bool protect=sink_protect_;
  int nsink=0;
  auto pos=GatherAllSinkPositions(pmbp,nsink);
  auto sp=pos.d_view;
  const Real lx=pm->mesh_size.x1max-pm->mesh_size.x1min;
  const Real ly=pm->mesh_size.x2max-pm->mesh_size.x2min;
  const Real lz=pm->mesh_size.x3max-pm->mesh_size.x3min;
  par_for_outer("GMTFJeansAMR",DevExeSpace(),0,0,0,nmb-1,
  KOKKOS_LAMBDA(TeamMember_t team,const int m) {
    Real rhomax;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,ni*nj*nk),
      [&](const int idx,Real &v) {
        const int k=idx/(ni*nj), j=(idx-k*ni*nj)/ni, i=idx-k*ni*nj-j*ni;
        v=Kokkos::fmax(v,u(m,IDN,k,j,i));
      },Kokkos::Max<Real>(rhomax));
    Kokkos::single(Kokkos::PerTeam(team),[&]() {
      const auto b=sizes(m);
      const Real dx=Kokkos::fmax(b.dx1,Kokkos::fmax(b.dx2,b.dx3));
      const Real j2=factor/(rhomax*dx*dx);
      int flag=(j2<target2) ? 1 : ((j2>derefine2) ? -1 : 0);
      if (protect) {
        for (int p=0;p<nsink;++p) {
          Real x=Kokkos::fabs(sp(3*p)-0.5*(b.x1min+b.x1max));
          Real y=Kokkos::fabs(sp(3*p+1)-0.5*(b.x2min+b.x2max));
          Real z=Kokkos::fabs(sp(3*p+2)-0.5*(b.x3min+b.x3max));
          x=Kokkos::fmin(x,lx-x); y=Kokkos::fmin(y,ly-y); z=Kokkos::fmin(z,lz-z);
          if (x<=0.5*(b.x1max-b.x1min)+buffer*b.dx1 &&
              y<=0.5*(b.x2max-b.x2min)+buffer*b.dx2 &&
              z<=0.5*(b.x3max-b.x3min)+buffer*b.dx3) flag=1;
        }
      }
      flags.d_view(m+gids)=flag;
    });
  });
  flags.template modify<DevExeSpace>();
  flags.template sync<HostMemSpace>();
}

void GMTFHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  pdata->nhist = 7;
  pdata->label[0] = "mass";     // gas
  pdata->label[1] = "1-mom";
  pdata->label[2] = "2-mom";
  pdata->label[3] = "3-mom";
  pdata->label[4] = "m_sink";
  pdata->label[5] = "n_sink";
  pdata->label[6] = "SFE";

  auto &u0 = pmbp->phydro->u0;
  auto &sz = pmbp->pmb->mb_size;
  auto &ix = pm->mb_indcs;
  const int is = ix.is, nx1 = ix.nx1, js = ix.js, nx2 = ix.nx2, ks = ix.ks, nx3 = ix.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  Real g[4] = {0.0, 0.0, 0.0, 0.0};
  for (int v = 0; v < 4; ++v) {
    Real sum = 0.0;
    Kokkos::parallel_reduce("gmtf_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &psum) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji + ks;
      const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
      const int i = (idx - m*nkji - (k-ks)*nji - (j-js)*nx1) + is;
      psum += u0(m, v, k, j, i)*sz.d_view(m).dx1*sz.d_view(m).dx2*sz.d_view(m).dx3;
    }, Kokkos::Sum<Real>(sum));
    g[v] = sum;
  }

  // sink mass on THIS rank; the history machinery MPI_SUM-reduces across ranks
  Real msink = 0.0;
  int nsink = 0;
  if (pmbp->ppart != nullptr) {
    nsink = pmbp->ppart->nprtcl_thispack;
    auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                  pmbp->ppart->prtcl_rdata);
    for (int p = 0; p < nsink; ++p) { msink += pr(IPM, p); }
  }

  for (int v = 0; v < 4; ++v) { pdata->hdata[v] = g[v]; }
  pdata->hdata[4] = msink;
  pdata->hdata[5] = static_cast<Real>(nsink);
  // SFE is a RATIO, so it cannot be formed from this rank's numbers and then summed.
  // Store the numerator here and rebuild it after the reduction below, where the global
  // totals are available; on one rank the two agree.
  pdata->hdata[6] = 0.0;
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) { pdata->hdata[n] = 0.0; }

  if (amr_diagnostics_) {
    pdata->nhist = 16;
    const char *extra[] = {"px_total", "py_total", "pz_total", "nblock", "nblock_max",
                           "j_bad_lo", "j_bad_max", "sink_lo", "vol_max"};
    for (int v=0; v<9; ++v) { pdata->label[7+v]=extra[v]; pdata->hdata[7+v]=0.0; }
    for (int v=0; v<3; ++v) pdata->hdata[7+v]=g[1+v];
    if (pmbp->ppart != nullptr) {
      auto pr=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pmbp->ppart->prtcl_rdata);
      auto pi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pmbp->ppart->prtcl_idata);
      for (int p=0;p<nsink;++p) {
        pdata->hdata[7]+=pr(IPM,p)*pr(IPVX,p);
        pdata->hdata[8]+=pr(IPM,p)*pr(IPVY,p);
        pdata->hdata[9]+=pr(IPM,p)*pr(IPVZ,p);
        const int m=pi(PGID,p)-pmbp->gids;
        if (m>=0 && m<pmbp->nmb_thispack &&
            pmbp->pmb->mb_lev.h_view(m)<pm->max_level) pdata->hdata[14]+=1.0;
      }
    }
    pdata->hdata[10]=pmbp->nmb_thispack;
    for (int m=0;m<pmbp->nmb_thispack;++m) {
      if (pmbp->pmb->mb_lev.h_view(m)==pm->max_level) {
        pdata->hdata[11]+=1.0;
        const auto b=sz.h_view(m);
        pdata->hdata[15]+=(b.x1max-b.x1min)*(b.x2max-b.x2min)*(b.x3max-b.x3min);
      }
    }
    const Real cs=pmbp->phydro->peos->eos_data.iso_cs;
    const Real fac=4.0*M_PI*M_PI*cs*cs/(pmbp->pgrav->four_pi_G*njeans_*njeans_);
    auto levels=pmbp->pmb->mb_lev.d_view;
    const int maxlevel=pm->max_level;
    for (int capped=0;capped<2;++capped) {
      Real count=0;
      Kokkos::parallel_reduce("gmtf_jeans_count",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
      KOKKOS_LAMBDA(const int idx,Real &sum) {
        const int m=idx/nkji, k=(idx-m*nkji)/nji+ks;
        const int j=(idx-m*nkji-(k-ks)*nji)/nx1+js;
        const int i=(idx-m*nkji-(k-ks)*nji-(j-js)*nx1)+is;
        const auto b=sz.d_view(m);
        const Real dx=Kokkos::fmax(b.dx1,Kokkos::fmax(b.dx2,b.dx3));
        if ((levels(m)==maxlevel)==(capped==1) && u0(m,IDN,k,j,i)>fac/(dx*dx)) sum+=1;
      },Kokkos::Sum<Real>(count));
      pdata->hdata[12+capped]=count;
    }
  }

  // ---- global totals, for the stopping criterion -------------------------------------
  Real loc[2] = {g[0], msink}, glb[2] = {g[0], msink};
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(loc, glb, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  const Real mtot = glb[0] + glb[1];
  if (mtot0_ < 0.0) { mtot0_ = mtot; }
  const Real sfe = (mtot > 0.0) ? glb[1]/mtot : 0.0;
  // report the GLOBAL value divided by nranks, since hst user data is MPI_SUM-reduced
  pdata->hdata[6] = sfe/static_cast<Real>(global_variable::nranks);

  if (sfe_term_ > 0.0 && sfe >= sfe_term_ && pm->pmy_driver != nullptr) {
    if (!sfe_stop_announced_) {
      sfe_stop_announced_ = true;
      if (global_variable::my_rank == 0) {
        std::cout << std::endl
                  << "### GMTF: SFE = " << sfe << " reached sfe_term = " << sfe_term_
                  << " at t = " << pm->time << " (cycle " << pm->ncycle << ")."
                  << std::endl
                  << "    M_sink = " << glb[1] << ", M_gas = " << glb[0]
                  << ", M_tot = " << mtot << " (initial " << mtot0_ << ")." << std::endl
                  << "    Stopping: beyond this the box has converted more gas than any"
                  << " feedback-free setup can represent." << std::endl << std::endl;
      }
    }
    // end the run at the close of this cycle, via the normal shutdown path
    pm->pmy_driver->tlim = pm->time;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Uniform11(std::uint64_t h)
//! \brief a hash word mapped to [-1, 1), using the top 53 bits.

Real Uniform11(std::uint64_t h) {
  return 2.0*(static_cast<Real>(h >> 11)/9007199254740992.0) - 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn std::uint64_t SplitMix64(std::uint64_t x)
//! \brief a fixed integer hash. Deterministic and platform-independent (no floating point,
//! no library RNG whose stream could differ), which is what makes the seed positions
//! identical on every rank and reproducible between runs and machines.

std::uint64_t SplitMix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30))*0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27))*0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

//----------------------------------------------------------------------------------------
//! \fn void SeedSinks(ParameterInput *pin, MeshBlockPack *pmbp)
//! \brief Lay down exactly <problem>/nsink_seed sink particles on a jittered cubic
//! lattice. No-op unless nsink_seed > 0.
//!
//! WHY THIS EXISTS. Sinks normally form from the gas, and past the first creation the
//! outcome is NOT decomposition-invariant: which cell first crosses the Larson-Penston
//! threshold differs between rank counts, and a sink in a different cell is an O(1)
//! change. A creation-driven scaling ladder therefore ends every rung with a different
//! sink count, and its rungs cannot be compared. Seeding turns the sink count into an
//! independent, controlled variable, which is what a cost model needs.
//!
//! Positions derive from GLOBAL quantities only -- the mesh bounds and a fixed integer
//! hash -- so every rank builds the identical list and then keeps the seeds its own blocks
//! contain. Deliberately NOT via ppc: ppc*(this rank's cells) rounds DOWN per rank, which
//! yields zero particles per rank on any sufficiently fine decomposition.
//!
//! Inputs (all <problem>):
//!   nsink_seed   number of sinks; <= 0 disables (default 0)
//!   sink_mass_cv mass of each sink in units of the 27-cell control-volume gas mass
//!                (default 10). Below 1 the accretion update is ill-posed -- see below.
//!   sink_mass    absolute mass override; used when > 0 (default -1, i.e. use sink_mass_cv)
//!   sink_min_sep required separation in cells; 0 disables the check (default 6.0)
//!   sink_jitter  lattice jitter as a fraction of the lattice spacing (default 0.2)
//!   sink_placement  "random" (default) or "lattice"
//!
//! WHY "random" IS THE DEFAULT. The jittered lattice guarantees a separation cheaply, but it
//! also fixes each sink's position RELATIVE TO THE MESHBLOCK GRID, and that controls
//! something the benchmark cares about a great deal: whether a sink's control volume can
//! reach a block face, which is the only way it ever emits a cross-rank record. Measured on
//! a 256^3/64^3 mesh, the lattice for nsink = 64 (spacing 64 cells) puts every sink 19.2
//! cells from the nearest face and the lattice for nsink = 4096 (spacing 16 cells) puts them
//! 4.8 cells away -- so NEITHER ever emits, at any rank count, while nsink = 256 and 1024
//! (spacings 36.6 and 23.3 cells) do. The measured ExchangeCVReset cost tracked that
//! commensurability exactly and not the sink count at all. Worse, a scaling ladder built on
//! such an nsink would report the cross-rank path as free however many ranks it ran on.
//! Rejection sampling makes face proximity a property of the domain rather than of nsink.

void SeedSinks(ParameterInput *pin, MeshBlockPack *pmbp) {
  const int nseed = pin->GetOrAddInteger("problem", "nsink_seed", 0);
  if (nseed <= 0) return;

  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem/nsink_seed > 0 requires a <particles> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->ppart->particle_type != ParticleType::sink) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem/nsink_seed > 0 requires particles/type = sink" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Mesh *pm = pmbp->pmesh;
  const Real x1min = pm->mesh_size.x1min, x2min = pm->mesh_size.x2min;
  const Real x3min = pm->mesh_size.x3min;
  const Real L1 = pm->mesh_size.x1max - x1min;
  const Real L2 = pm->mesh_size.x2max - x2min;
  const Real L3 = pm->mesh_size.x3max - x3min;

  // smallest cubic lattice that holds nseed points
  int n = 1;
  while (static_cast<std::int64_t>(n)*n*n < nseed) { ++n; }
  const std::int64_t ncell = static_cast<std::int64_t>(n)*n*n;
  const Real s1 = L1/n, s2 = L2/n, s3 = L3/n;

  const Real jit = pin->GetOrAddReal("problem", "sink_jitter", 0.2);
  if (jit < 0.0 || jit >= 0.5) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem/sink_jitter must be in [0, 0.5); got " << jit << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Separation guarantee. The accretion conflict radius is 5 cells (write half-width 2 +
  // read half-width 3), so sinks closer than that are placed in different colours and
  // AccreteMass serializes them. Which regime the run is in is a property of the
  // measurement, so it must be stated and checked, not discovered afterwards.
  const Real dx1 = L1/pm->mesh_indcs.nx1;
  const Real dx2 = L2/pm->mesh_indcs.nx2;
  const Real dx3 = L3/pm->mesh_indcs.nx3;
  const Real dxmax = std::max(dx1, std::max(dx2, dx3));
  const Real sep_guar = std::min(s1, std::min(s2, s3))*(1.0 - 2.0*jit);
  const Real minsep = pin->GetOrAddReal("problem", "sink_min_sep", 6.0);
  const std::string placement = pin->GetOrAddString("problem", "sink_placement", "random");
  if (placement != "random" && placement != "lattice") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem/sink_placement must be \"random\" or \"lattice\"; got \""
              << placement << "\"" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // The lattice inherits its separation from the spacing, so it must be checked up front.
  // Rejection sampling enforces the same bound directly and fails loudly if it cannot.
  if (placement == "lattice" && minsep > 0.0 && sep_guar < minsep*dxmax) {
    const Real lmin = std::min(L1, std::min(L2, L3));
    const int nmax = static_cast<int>(std::floor(lmin*(1.0 - 2.0*jit)/(minsep*dxmax)));
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem/nsink_seed = " << nseed << " cannot honour sink_min_sep = "
              << minsep << " cells." << std::endl
              << "  guaranteed separation " << sep_guar/dxmax << " cells < required "
              << minsep << std::endl
              << "  at this resolution and jitter the maximum is nsink_seed = "
              << (nmax > 0 ? nmax*nmax*nmax : 0)
              << " (a " << nmax << "^3 lattice)." << std::endl
              << "  Raise the resolution, lower sink_min_sep, or set sink_min_sep = 0 to"
              << " measure the serialized-colour regime deliberately." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::vector<Real> sx(nseed), sy(nseed), sz(nseed);

  if (placement == "lattice") {
    // Deterministic spread subset: order lattice cells by a hash of their index and take the
    // first nseed. Raster order would pack them into one corner slab and overload whichever
    // ranks own it, which would show up as a load-imbalance artefact in the scaling curves.
    std::vector<std::pair<std::uint64_t, std::int64_t>> ord;
    ord.reserve(static_cast<std::size_t>(ncell));
    for (std::int64_t c = 0; c < ncell; ++c) {
      ord.emplace_back(SplitMix64(static_cast<std::uint64_t>(c) ^ 0x5EEDULL), c);
    }
    std::sort(ord.begin(), ord.end());
    for (int p = 0; p < nseed; ++p) {
      const std::int64_t c = ord[p].second;
      const int i = static_cast<int>(c % n);
      const int j = static_cast<int>((c/n) % n);
      const int k = static_cast<int>(c/(static_cast<std::int64_t>(n)*n));
      const std::uint64_t hc = static_cast<std::uint64_t>(c);
      // jitter each axis from an independent hash word; bounded by |jit| < 0.5 of the
      // spacing, so a seed can never leave its own lattice cell and the separation bound
      // above holds by construction
      const Real u1 = Uniform11(SplitMix64(3*hc + 1));
      const Real u2 = Uniform11(SplitMix64(3*hc + 2));
      const Real u3 = Uniform11(SplitMix64(3*hc + 3));
      sx[p] = x1min + (i + 0.5 + jit*u1)*s1;
      sy[p] = x2min + (j + 0.5 + jit*u2)*s2;
      sz[p] = x3min + (k + 0.5 + jit*u3)*s3;
    }
  } else {
    // Rejection sampling against the SAME separation requirement, so the constraint is
    // enforced directly rather than inherited from a lattice -- and the resulting positions
    // carry no fixed relationship to the MeshBlock grid.
    const Real dmin = minsep*dxmax;
    const Real d2   = dmin*dmin;
    const std::int64_t maxtry = std::max<std::int64_t>(1000LL*nseed, 100000LL);
    std::int64_t tries = 0;
    int got = 0;
    while (got < nseed && tries < maxtry) {
      const std::uint64_t h = static_cast<std::uint64_t>(tries);
      const Real cx = x1min + 0.5*(Uniform11(SplitMix64(3*h + 11)) + 1.0)*L1;
      const Real cy = x2min + 0.5*(Uniform11(SplitMix64(3*h + 12)) + 1.0)*L2;
      const Real cz = x3min + 0.5*(Uniform11(SplitMix64(3*h + 13)) + 1.0)*L3;
      ++tries;
      bool ok = true;
      if (dmin > 0.0) {
        for (int q = 0; q < got; ++q) {
          // periodic minimum image: the box wraps, so two sinks either side of a face are
          // neighbours and their control volumes really do overlap
          Real ddx = cx - sx[q]; ddx -= L1*std::floor(ddx/L1 + 0.5);
          Real ddy = cy - sy[q]; ddy -= L2*std::floor(ddy/L2 + 0.5);
          Real ddz = cz - sz[q]; ddz -= L3*std::floor(ddz/L3 + 0.5);
          if (ddx*ddx + ddy*ddy + ddz*ddz < d2) { ok = false; break; }
        }
      }
      if (ok) { sx[got] = cx; sy[got] = cy; sz[got] = cz; ++got; }
    }
    if (got < nseed) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "could not place " << nseed << " sinks at sink_min_sep = " << minsep
                << " cells: placed " << got << " in " << tries << " attempts." << std::endl
                << "  The domain is too full at this separation. Lower sink_min_sep, raise"
                << " the resolution, or lower nsink_seed." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Ownership: lower-inclusive containment, matching Particles::SetGIDFromPosition. PGID
  // must name the block that actually holds the seed -- setting it to gids for every
  // particle (the obvious shortcut) makes cycle 1 deposit into the wrong block, which
  // injects a net momentum that is then conserved at the wrong value for the whole run.
  const int nmb = pmbp->nmb_thispack;
  const int gids = pmbp->gids;
  auto &mbsz = pmbp->pmb->mb_size;
  std::vector<int> own(nseed, -1);
  int nloc = 0;
  for (int p = 0; p < nseed; ++p) {
    for (int m = 0; m < nmb; ++m) {
      if (sx[p] >= mbsz.h_view(m).x1min && sx[p] < mbsz.h_view(m).x1max &&
          sy[p] >= mbsz.h_view(m).x2min && sy[p] < mbsz.h_view(m).x2max &&
          sz[p] >= mbsz.h_view(m).x3min && sz[p] < mbsz.h_view(m).x3max) {
        own[p] = m;
        ++nloc;
        break;
      }
    }
  }

  // How many sinks sit close enough to a MeshBlock face that their control volume can
  // reach across it? That is the ONLY population that ever emits a cross-rank record, so
  // it is the population the ExchangeCVReset cost is proportional to. Reported because a
  // seeding that silently drives it to zero makes the whole cross-rank path look free --
  // which is exactly what the jittered lattice did (see the note on sink_placement).
  int nface = 0;
  {
    const Real reach = 3.0;   // rctrl+1 read stencil, plus one cell of crossing slack
    for (int p = 0; p < nseed; ++p) {
      const int m = own[p];
      if (m < 0) continue;
      const Real dx[3] = {mbsz.h_view(m).dx1, mbsz.h_view(m).dx2, mbsz.h_view(m).dx3};
      const Real lo[3] = {mbsz.h_view(m).x1min, mbsz.h_view(m).x2min, mbsz.h_view(m).x3min};
      const Real hi[3] = {mbsz.h_view(m).x1max, mbsz.h_view(m).x2max, mbsz.h_view(m).x3max};
      const Real q[3]  = {sx[p], sy[p], sz[p]};
      for (int c = 0; c < 3; ++c) {
        if (std::min(q[c] - lo[c], hi[c] - q[c]) < reach*dx[c]) { ++nface; break; }
      }
    }
  }

  // Exact gate: every seed must be claimed by exactly one block, globally. A shortfall
  // means a seed fell through a block boundary (round-off at a face) or into a gap; a
  // surplus means two blocks claimed one. Either way the run would start with the wrong
  // number of sinks, which is precisely the variable this whole mode exists to control.
  int nglob = nloc;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nglob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nface, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (nglob != nseed) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "seeded sink ownership is inconsistent: " << nglob << " of " << nseed
              << " seeds were claimed by exactly one MeshBlock." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Sink mass. The scale that matters is the gas mass inside the 27-cell control volume,
  // NOT an absolute number: control-volume accretion computes dm as the difference between
  // the CV integral and its post-reset extrapolation, so |dm| is set by the CV contents. A
  // seeded sink does not sit at a density peak the way a created one does, so that
  // difference can be NEGATIVE, and the momentum update v_new = (m*v + dM)/(m + dm)
  // divides by m + dm. Once m is not comfortably larger than the CV mass the denominator
  // becomes small, sinks acquire velocities far above the gas, and the particle CFL
  // collapses dt -- which silently destroys any timing measurement. Hence a default
  // expressed as a MULTIPLE of the CV mass, which also tracks resolution correctly.
  const Real mcv = 27.0*dx1*dx2*dx3;   // gmtf initialises the gas at rho = 1
  const Real fcv = pin->GetOrAddReal("problem", "sink_mass_cv", 10.0);
  const Real mabs = pin->GetOrAddReal("problem", "sink_mass", -1.0);
  const Real msink = (mabs > 0.0) ? mabs : fcv*mcv;
  pmbp->ppart->ResizeForSeededParticles(nloc);

  if (nloc > 0) {
    auto pr_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_rdata);
    auto pi_h = Kokkos::create_mirror_view(pmbp->ppart->prtcl_idata);
    const bool has_prev = (pmbp->ppart->nrdata > IPX0);
    int slot = 0;
    for (int p = 0; p < nseed; ++p) {
      if (own[p] < 0) continue;
      pr_h(IPX, slot) = sx[p];   pr_h(IPVX, slot) = 0.0;
      pr_h(IPY, slot) = sy[p];   pr_h(IPVY, slot) = 0.0;
      pr_h(IPZ, slot) = sz[p];   pr_h(IPVZ, slot) = 0.0;
      pr_h(IPM, slot) = msink;
      pr_h(IPGX, slot) = 0.0; pr_h(IPGY, slot) = 0.0; pr_h(IPGZ, slot) = 0.0;
      if (has_prev) {
        pr_h(IPX0, slot) = sx[p]; pr_h(IPY0, slot) = sy[p]; pr_h(IPZ0, slot) = sz[p];
      }
      pi_h(PGID, slot) = gids + own[p];
      // tag = index in the GLOBAL seed list, so tags are unique across ranks by
      // construction. MergeSinks keys survivors by tag and FATALs on a duplicate.
      pi_h(PTAG, slot) = p;
      ++slot;
    }
    Kokkos::deep_copy(pmbp->ppart->prtcl_rdata, pr_h);
    Kokkos::deep_copy(pmbp->ppart->prtcl_idata, pi_h);
  }

  // ---- report, and flag anything that would invalidate the measurement ---------------
  if (global_variable::my_rank == 0) {
    // gmtf initialises the gas at rho = 1, so the initial gas mass is just the volume
    const Real mgas = L1*L2*L3;
    const Real fsink = nseed*msink/mgas;
    std::cout << "### GMTF: seeded " << nseed << " sink(s) on a " << n << "^3 lattice, "
              << "spacing " << std::min(s1, std::min(s2, s3))/dxmax << " cells, "
              << "guaranteed separation " << sep_guar/dxmax << " cells "
              << (sep_guar >= 5.0*dxmax ? "(> 5 => one accretion colour, fully parallel)"
                                        : "(< 5 => colours > 1, accretion serializes)")
              << std::endl
              << "###       placement = " << placement << "; " << nface << " of " << nseed
              << " sink(s) (" << (100.0*nface)/nseed << "%) can reach a MeshBlock face"
              << " => that is the cross-rank ExchangeCVReset population" << std::endl
              << "###       m_sink = " << msink << " each = " << msink/mcv
              << " x the control-volume gas mass; " << nseed*msink
              << " total = " << 100.0*fsink << "% of the initial gas mass" << std::endl;
    if (msink < mcv) {
      std::cout << "### WARNING: m_sink is below the control-volume gas mass, so accretion"
                << " is ill-posed for a seeded sink: dm can be negative and comparable to"
                << " m, the velocity update divides by (m + dm), and the particle CFL will"
                << " collapse dt. Raise problem/sink_mass_cv." << std::endl;
    }
    if (fsink > 0.1) {
      std::cout << "### WARNING: seeded sinks hold more than 10% of the gas mass. They"
                << " will dominate the potential and change the timestep, so runs with"
                << " different nsink_seed are no longer comparable." << std::endl;
    }
    if (pmbp->ppart->creation || pmbp->ppart->merging) {
      std::cout << "### WARNING: particles/creation or particles/merging is on with"
                << " nsink_seed > 0, so the sink count will NOT stay at " << nseed << "."
                << " A benchmark that varies N must set both to false." << std::endl;
    }
  }
}
}  // namespace
