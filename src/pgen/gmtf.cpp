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
#include <random>
#include <vector>
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
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
void GMTFHistory(HistoryData *pdata, Mesh *pm);
}  // namespace

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
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

  // Star-formation efficiency diagnostic and stopping criterion.
  //   SFE = M_sink / (M_sink + M_gas)
  // A pure gravo-turbulent box has no feedback -- no radiation, no outflows, no
  // supernovae -- and with periodic boundaries nothing opposes global collapse, so SFE
  // runs away to ~1 regardless of the physics being modelled. Real clouds convert only a
  // few per cent per free-fall time before feedback intervenes. sfe_term is therefore a
  // statement about the DOMAIN OF VALIDITY of this setup, not about the gas: past it the
  // run is integrating a cloud that could not exist. It was previously read from the
  // input and never used, so it silently did nothing.
  sfe_term_ = pin->GetOrAddReal("problem", "sfe_term", -1.0);
  if (pin->GetOrAddBoolean("problem", "user_hist", false)) {
    user_hist_func = GMTFHistory;
  }

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

  if (!(x1size == x2size && x1size == x3size)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "this problem assumes cubic domain" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real dk = 2.0*M_PI/x1size;

  std::mt19937_64 gen(rseed);
  std::normal_distribution<Real> gauss(0.0, 1.0);

  std::vector<Real> host_kx, host_ky, host_kz;
  std::vector<Real> host_ax, host_ay, host_az;
  std::vector<Real> host_bx, host_by, host_bz;

  // Here, we utilize Hermitian symmetry to only keep non-redundant modes.
  // The points symmetric with respect to the origin are complex conjugates,
  // so we only need to keep one of them. Also, we discard DC component, i.e.,
  // kx=ky=kz=0, to avoid adding a net bulk velocity.
  for (int nkx = 0; nkx <= nhigh; ++nkx) {
    for (int nky = -nhigh; nky <= nhigh; ++nky) {
      for (int nkz = -nhigh; nkz <= nhigh; ++nkz) {
        if (nkx == 0) {
          // We are on the kx=0 plane
          if (nky < 0) continue;
          if (nky == 0 && nkz <= 0) continue;
        }

        Real kx = dk*static_cast<Real>(nkx);
        Real ky = dk*static_cast<Real>(nky);
        Real kz = dk*static_cast<Real>(nkz);
        Real kmag = std::sqrt(SQR(kx) + SQR(ky) + SQR(kz));
        int nmag2 = SQR(nkx) + SQR(nky) + SQR(nkz);
        if ( (SQR(nlow) <= nmag2) && (nmag2 <= SQR(nhigh)) ) {
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
    v2_sum_local += SQR(dv(m,0,k,j,i)) + SQR(dv(m,1,k,j,i)) + SQR(dv(m,2,k,j,i));
  }, Kokkos::Sum<Real>(v2_sum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &v2_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Rescale the velocity perturbations to achieve the target Mach number
  long long Nx1 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx1);
  long long Nx2 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx2);
  long long Nx3 = static_cast<long long>(pmbp->pmesh->mesh_indcs.nx3);
  Real vrms = std::sqrt(v2_sum / static_cast<Real>(Nx1*Nx2*Nx3));
  par_for("gmtf_init_turb", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IM1,k,j,i) += mach/vrms*dv(m,0,k,j,i);
    u0(m,IM2,k,j,i) += mach/vrms*dv(m,1,k,j,i);
    u0(m,IM3,k,j,i) += mach/vrms*dv(m,2,k,j,i);
  });
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void GMTFHistory()
//! \brief gas mass/momenta, sink mass and count, and the star-formation efficiency.
//! Also applies the sfe_term stopping criterion: once SFE reaches it, tlim is pulled back
//! to the current time so the run ends at the close of this cycle by the NORMAL path --
//! final outputs are written and the usual "Terminating on time limit" message appears,
//! rather than aborting and losing the last dump.

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
}  // namespace
