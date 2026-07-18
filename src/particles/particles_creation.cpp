//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_creation.cpp
//! \brief sink-particle creation, following Moon & Ostriker 2025 (ApJ,
//! 10.3847/1538-4357/add477, section 3.4), who simplify Gong & Ostriker 2013.
//!
//! A sink is created in a cell when, simultaneously:
//!   1. the gas density exceeds the Larson-Penston threshold at half a cell,
//!          rho_thr = rho_LP(0.5 dx) = (8.86/pi) c_s^2 / (G dx^2)      [M&O eq. 53]
//!   2. the cell is the LOCAL MINIMUM of the gravitational potential (26-neighbour
//!      test). M&O drop GO13's converging-flow and boundedness checks: the LP
//!      threshold is so high that random compression essentially never reaches it
//!      without collapse, and the potential-minimum condition alone rejects
//!      disk-fragmentation false positives.
//! A candidate is also rejected within 2(rctrl+1) cells of an existing sink (the
//! reference merges overlapping control volumes instead; merging is a follow-up).
//!
//! The new particle is created MASSLESS at the cell centre with the local gas
//! velocity. The AccreteMass task that runs immediately afterwards (same task list,
//! ordered dependency) resets the new control volume and conservatively dumps
//! Delta M_sink = -Delta M_reset into the particle [M&O eq. 54] -- so the sink's
//! initial mass/momentum are exactly the LP-core excess over the reset state, with
//! no double counting and machine-precision total-mass conservation.
//!
//! Creation runs once per step (last RK stage), before accretion. Particle arrays
//! are resized on the host (Kokkos::resize preserves contents); serial/on-rank.

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "gravity/gravity.hpp"
#include "particles.hpp"

namespace particles {

namespace {
constexpr int MAX_NEW_SINKS = 32;   // per pack per step (host warns on overflow)
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::CreateSinks
//! \brief create sink particles where rho > rho_LP(dx/2) at a local potential minimum.

TaskStatus Particles::CreateSinks(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;
  if (particle_type != ParticleType::sink || !creation) return TaskStatus::complete;
  if (!(pmy_pack->pmesh->three_d)) return TaskStatus::complete;
  if (pmy_pack->pgrav == nullptr) return TaskStatus::complete;
  if (pmy_pack->phydro == nullptr && pmy_pack->pmhd == nullptr) {
    return TaskStatus::complete;
  }
  const bool is_mhd = (pmy_pack->pmhd != nullptr);
  auto u0 = is_mhd ? pmy_pack->pmhd->u0 : pmy_pack->phydro->u0;
  auto phi = pmy_pack->pgrav->phi;
  const Real cs = is_mhd ? pmy_pack->pmhd->peos->eos_data.iso_cs
                         : pmy_pack->phydro->peos->eos_data.iso_cs;
  const Real fpg = pmy_pack->pgrav->four_pi_G;
  // rho_thr = (8.86/pi) c_s^2/(G dx^2) with G = fpg/(4 pi)  ->  35.44 c_s^2/(fpg dx^2)
  const Real thr_fac = 4.0*8.86*cs*cs/fpg;

  auto &pr = prtcl_rdata;
  auto &pi = prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int gids = pmy_pack->gids;
  const int nmb = pmy_pack->nmb_thispack;
  const int npart = nprtcl_thispack;
  const int rctrl = 1;

  // ---- scan: candidate cells -> (m,k,j,i) buffer ----
  DvceArray2D<int> cand("sink_cand", MAX_NEW_SINKS, 4);
  DvceArray1D<int> ncand_d("ncand", 1);
  Kokkos::deep_copy(ncand_d, 0);

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  par_for("sink_create_scan", DevExeSpace(), 0, nmkji-1,
  KOKKOS_LAMBDA(const int idx) {
    const int m = idx/nkji;
    const int k = (idx - m*nkji)/nji + ks;
    const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
    const int i = (idx - m*nkji - (k-ks)*nji - (j-js)*nx1) + is;
    const Real dx1 = mbsize.d_view(m).dx1;
    const Real rho_thr = thr_fac/(dx1*dx1);
    const Real rho = u0(m, IDN, k, j, i);
    if (rho <= rho_thr) return;
    // local potential minimum over the 26 neighbours (phi ghosts valid: ExchangePhi
    // ran this stage)
    const Real p0 = phi(m, 0, k, j, i);
    for (int dk = -1; dk <= 1; ++dk) {
      for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          if (di == 0 && dj == 0 && dk == 0) continue;
          if (phi(m, 0, k+dk, j+dj, i+di) < p0) return;
        }
      }
    }
    // not too close to an existing sink (reference merges overlapping CVs instead)
    const Real xc = mbsize.d_view(m).x1min + (static_cast<Real>(i-is) + 0.5)*dx1;
    const Real yc = mbsize.d_view(m).x2min +
                    (static_cast<Real>(j-js) + 0.5)*mbsize.d_view(m).dx2;
    const Real zc = mbsize.d_view(m).x3min +
                    (static_cast<Real>(k-ks) + 0.5)*mbsize.d_view(m).dx3;
    const Real dmin = 2.0*(rctrl + 1)*dx1;
    for (int p = 0; p < npart; ++p) {
      const Real ddx = pr(IPX, p) - xc;
      const Real ddy = pr(IPY, p) - yc;
      const Real ddz = pr(IPZ, p) - zc;
      if (ddx*ddx + ddy*ddy + ddz*ddz < dmin*dmin) return;
    }
    const int slot = Kokkos::atomic_fetch_add(&ncand_d(0), 1);
    if (slot < MAX_NEW_SINKS) {
      cand(slot, 0) = m; cand(slot, 1) = k; cand(slot, 2) = j; cand(slot, 3) = i;
    }
  });

  auto ncand_h = Kokkos::create_mirror_view(ncand_d);
  Kokkos::deep_copy(ncand_h, ncand_d);
  int nnew = ncand_h(0);
  if (nnew <= 0) return TaskStatus::complete;
  if (nnew > MAX_NEW_SINKS) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in Particles::CreateSinks: " << nnew << " candidates, "
                << "capped at " << MAX_NEW_SINKS << " this step" << std::endl;
    }
    nnew = MAX_NEW_SINKS;
  }

  // host-side dedupe of same-step candidates closer than the exclusion radius
  auto cand_h = Kokkos::create_mirror_view(cand);
  Kokkos::deep_copy(cand_h, cand);
  auto msz_h = pmy_pack->pmb->mb_size.h_view;
  Real cx[MAX_NEW_SINKS], cy[MAX_NEW_SINKS], cz[MAX_NEW_SINKS];
  bool keep[MAX_NEW_SINKS];
  for (int n = 0; n < nnew; ++n) {
    const int m = cand_h(n, 0), k = cand_h(n, 1), j = cand_h(n, 2), i = cand_h(n, 3);
    cx[n] = msz_h(m).x1min + (static_cast<Real>(i-is) + 0.5)*msz_h(m).dx1;
    cy[n] = msz_h(m).x2min + (static_cast<Real>(j-js) + 0.5)*msz_h(m).dx2;
    cz[n] = msz_h(m).x3min + (static_cast<Real>(k-ks) + 0.5)*msz_h(m).dx3;
    keep[n] = true;
  }
  for (int n = 0; n < nnew; ++n) {
    if (!keep[n]) continue;
    for (int q = n+1; q < nnew; ++q) {
      const Real dmin = 2.0*(rctrl + 1)*msz_h(cand_h(n,0)).dx1;
      const Real d2 = (cx[n]-cx[q])*(cx[n]-cx[q]) + (cy[n]-cy[q])*(cy[n]-cy[q]) +
                      (cz[n]-cz[q])*(cz[n]-cz[q]);
      if (d2 < dmin*dmin) keep[q] = false;
    }
  }
  int nkeep = 0;
  for (int n = 0; n < nnew; ++n) if (keep[n]) ++nkeep;
  if (nkeep <= 0) return TaskStatus::complete;

  // ---- grow the particle arrays (Kokkos::resize preserves existing columns) ----
  const int npart_new = npart + nkeep;
  Kokkos::resize(prtcl_rdata, nrdata, npart_new);
  Kokkos::resize(prtcl_idata, nidata, npart_new);
  auto &prn = prtcl_rdata;
  auto &pin_ = prtcl_idata;

  // fill new slots: massless at the candidate cell centre, local gas velocity; the
  // AccreteMass reset that follows seeds mass/momentum conservatively (M&O eq. 54)
  DvceArray2D<int> cnew("cnew", nkeep, 4);
  auto cnew_h = Kokkos::create_mirror_view(cnew);
  int q = 0;
  for (int n = 0; n < nnew; ++n) {
    if (!keep[n]) continue;
    for (int c = 0; c < 4; ++c) cnew_h(q, c) = cand_h(n, c);
    if (global_variable::my_rank == 0) {
      std::printf("CreateSinks: new sink #%d at (%.5f, %.5f, %.5f) cycle=%d\n",
                  npart + q, cx[n], cy[n], cz[n], pmy_pack->pmesh->ncycle);
    }
    ++q;
  }
  Kokkos::deep_copy(cnew, cnew_h);
  const int tag0 = 1000000 + created_total_;
  created_total_ += nkeep;

  par_for("sink_create_fill", DevExeSpace(), 0, nkeep-1,
  KOKKOS_LAMBDA(const int n) {
    const int m = cnew(n, 0), k = cnew(n, 1), j = cnew(n, 2), i = cnew(n, 3);
    const int p = npart + n;
    const Real x = mbsize.d_view(m).x1min + (static_cast<Real>(i-is) + 0.5)*mbsize.d_view(m).dx1;
    const Real y = mbsize.d_view(m).x2min + (static_cast<Real>(j-js) + 0.5)*mbsize.d_view(m).dx2;
    const Real z = mbsize.d_view(m).x3min + (static_cast<Real>(k-ks) + 0.5)*mbsize.d_view(m).dx3;
    const Real rho = u0(m, IDN, k, j, i);
    prn(IPX, p) = x;  prn(IPY, p) = y;  prn(IPZ, p) = z;
    prn(IPVX, p) = u0(m, IM1, k, j, i)/rho;
    prn(IPVY, p) = u0(m, IM2, k, j, i)/rho;
    prn(IPVZ, p) = u0(m, IM3, k, j, i)/rho;
    prn(IPM, p) = 0.0;
    prn(IPGX, p) = 0.0; prn(IPGY, p) = 0.0; prn(IPGZ, p) = 0.0;
    prn(IPX0, p) = x; prn(IPY0, p) = y; prn(IPZ0, p) = z;
    pin_(PGID, p) = gids + m;
    pin_(PTAG, p) = tag0 + n;
  });

  nprtcl_thispack = npart_new;
  return TaskStatus::complete;
}

}  // namespace particles
