//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid_bvals.cpp
//! \brief implementation of MultigridBoundaryValues: boundary communication for the
//!        multigrid solver (fill coarse, prolongate, pack/send, recv/unpack, init recv)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../coordinates/cell_locations.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/nghbr_index.hpp"
#include "../parameter_input.hpp"
#include "multigrid.hpp"

//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::MultigridBoundaryValues()
//! \brief Constructor for multigrid boundary values object
//----------------------------------------------------------------------------------------

MultigridBoundaryValues::MultigridBoundaryValues(
    MeshBlockPack *pmbp, ParameterInput *pin,
    bool coarse, Multigrid *pmg)
  : MeshBoundaryValuesCC(pmbp, pin, coarse), pmy_mg(pmg) {
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::RemapIndicesForMG()
//! \brief Remap isame/icoar/ifine indices from hydro coordinates (ng ghost cells) to MG
//! coordinates (ngh_ ghost cells). Must be called AFTER InitializeBuffers.

void MultigridBoundaryValues::RemapIndicesForMG() {
  int ng  = pmy_pack->pmesh->mb_indcs.ng;
  int ngh = pmy_mg->GetGhostCells();
  int nx1 = pmy_pack->pmesh->mb_indcs.nx1;
  int nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  int nx3 = pmy_pack->pmesh->mb_indcs.nx3;
  int nnghbr = pmy_pack->pmb->nnghbr;

  if (ng != ngh) {
    int is_h = ng, ie_h = ng + nx1 - 1;
    int js_h = ng, je_h = ng + nx2 - 1;
    int ks_h = ng, ke_h = ng + nx3 - 1;
    int is_m = ngh, ie_m = ngh + nx1 - 1;
    int js_m = ngh, je_m = ngh + nx2 - 1;
    int ks_m = ngh, ke_m = ngh + nx3 - 1;
    int ng1_m = ngh - 1;

    auto remap_send = [](int &lo, int &hi,
                         int s_h, int e_h,
                         int s_m, int e_m, int ng1) {
      if (lo == s_h && hi == e_h) {
        lo = s_m; hi = e_m;
      } else if (lo > s_h) {
        lo = e_m - ng1; hi = e_m;
      } else {
        lo = s_m; hi = s_m + ng1;
      }
    };
    auto remap_recv = [](int &lo, int &hi,
                         int s_h, int e_h,
                         int s_m, int e_m, int ng_m) {
      if (lo >= s_h && hi <= e_h) {
        lo = s_m; hi = e_m;
      } else if (lo > e_h) {
        lo = e_m + 1; hi = e_m + ng_m;
      } else {
        lo = s_m - ng_m; hi = s_m - 1;
      }
    };

    for (int n = 0; n < nnghbr; ++n) {
      auto &si = sendbuf[n].isame[0];
      remap_send(si.bis, si.bie, is_h, ie_h, is_m, ie_m, ng1_m);
      remap_send(si.bjs, si.bje, js_h, je_h, js_m, je_m, ng1_m);
      remap_send(si.bks, si.bke, ks_h, ke_h, ks_m, ke_m, ng1_m);
      sendbuf[n].isame_ndat = (si.bie-si.bis+1)*(si.bje-si.bjs+1)*(si.bke-si.bks+1);

      auto &ri = recvbuf[n].isame[0];
      remap_recv(ri.bis, ri.bie, is_h, ie_h, is_m, ie_m, ngh);
      remap_recv(ri.bjs, ri.bje, js_h, je_h, js_m, je_m, ngh);
      remap_recv(ri.bks, ri.bke, ks_h, ke_h, ks_m, ke_m, ngh);
      recvbuf[n].isame_ndat = (ri.bie-ri.bis+1)*(ri.bje-ri.bjs+1)*(ri.bke-ri.bks+1);
    }
  }

  // Recompute icoar/ifine indices from scratch using MG mesh parameters.
  // These are needed for inter-level (fine-coarse) boundary communication.
  int ng1_m = ngh - 1;
  int cnx1 = nx1 / 2, cnx2 = nx2 / 2, cnx3 = nx3 / 2;
  int is_m = ngh, ie_m = ngh + nx1 - 1;
  int js_m = ngh, je_m = ngh + nx2 - 1;
  int ks_m = ngh, ke_m = ngh + nx3 - 1;
  int cis_m = ngh, cie_m = ngh + cnx1 - 1;
  int cjs_m = ngh, cje_m = ngh + cnx2 - 1;
  int cks_m = ngh, cke_m = ngh + cnx3 - 1;

  // Recover (ox1,ox2,ox3,f1,f2) from the buffer index n and recompute icoar/ifine.
  // We iterate the same way InitializeBuffers does.
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  auto compute_send_icoar = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3) {
    auto &ic = buf.icoar[0];
    ic.bis = (ox1 > 0) ? (cie_m - ng1_m) : cis_m;
    ic.bie = (ox1 < 0) ? (cis_m + ng1_m) : cie_m;
    ic.bjs = (ox2 > 0) ? (cje_m - ng1_m) : cjs_m;
    ic.bje = (ox2 < 0) ? (cjs_m + ng1_m) : cje_m;
    ic.bks = (ox3 > 0) ? (cke_m - ng1_m) : cks_m;
    ic.bke = (ox3 < 0) ? (cks_m + ng1_m) : cke_m;
    buf.icoar_ndat = (ic.bie-ic.bis+1)*(ic.bje-ic.bjs+1)*(ic.bke-ic.bks+1);
  };

  auto compute_send_ifine = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3, int f1, int f2) {
    auto &ifn = buf.ifine[0];
    ifn.bis = (ox1 > 0) ? (ie_m - ng1_m) : is_m;
    ifn.bie = (ox1 < 0) ? (is_m + ng1_m) : ie_m;
    ifn.bjs = (ox2 > 0) ? (je_m - ng1_m) : js_m;
    ifn.bje = (ox2 < 0) ? (js_m + ng1_m) : je_m;
    ifn.bks = (ox3 > 0) ? (ke_m - ng1_m) : ks_m;
    ifn.bke = (ox3 < 0) ? (ks_m + ng1_m) : ke_m;
    if (ox1 == 0) {
      if (f1 == 1) {
        ifn.bis += cnx1 - ngh;
      } else {
        ifn.bie -= cnx1 - ngh;
      }
    }
    if (ox2 == 0 && nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          ifn.bjs += cnx2 - ngh;
        } else {
          ifn.bje -= cnx2 - ngh;
        }
      } else {
        if (f2 == 1) {
          ifn.bjs += cnx2 - ngh;
        } else {
          ifn.bje -= cnx2 - ngh;
        }
      }
    }
    if (ox3 == 0 && nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          ifn.bks += cnx3 - ngh;
        } else {
          ifn.bke -= cnx3 - ngh;
        }
      } else {
        if (f2 == 1) {
          ifn.bks += cnx3 - ngh;
        } else {
          ifn.bke -= cnx3 - ngh;
        }
      }
    }
    buf.ifine_ndat =
        (ifn.bie-ifn.bis+1)
        * (ifn.bje-ifn.bjs+1)
        * (ifn.bke-ifn.bks+1);
  };

  auto compute_recv_icoar = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3,
                                int f1, int f2) {
    auto &ic = buf.icoar[0];
    if (ox1 == 0) {
      ic.bis = cis_m; ic.bie = cie_m;
      if (f1 == 0) {
        ic.bie += ngh;
      } else {
        ic.bis -= ngh;
      }
    } else if (ox1 > 0) {
      ic.bis = cie_m + 1; ic.bie = cie_m + ngh;
    } else {
      ic.bis = cis_m - ngh; ic.bie = cis_m - 1;
    }
    if (ox2 == 0) {
      ic.bjs = cjs_m; ic.bje = cje_m;
      if (nx2 > 1) {
        if (ox1 != 0) {
          if (f1 == 0) {
            ic.bje += ngh;
          } else {
            ic.bjs -= ngh;
          }
        } else {
          if (f2 == 0) {
            ic.bje += ngh;
          } else {
            ic.bjs -= ngh;
          }
        }
      }
    } else if (ox2 > 0) {
      ic.bjs = cje_m + 1; ic.bje = cje_m + ngh;
    } else {
      ic.bjs = cjs_m - ngh; ic.bje = cjs_m - 1;
    }
    if (ox3 == 0) {
      ic.bks = cks_m; ic.bke = cke_m;
      if (nx3 > 1) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 0) {
            ic.bke += ngh;
          } else {
            ic.bks -= ngh;
          }
        } else {
          if (f2 == 0) {
            ic.bke += ngh;
          } else {
            ic.bks -= ngh;
          }
        }
      }
    } else if (ox3 > 0) {
      ic.bks = cke_m + 1; ic.bke = cke_m + ngh;
    } else {
      ic.bks = cks_m - ngh; ic.bke = cks_m - 1;
    }
    buf.icoar_ndat =
        (ic.bie-ic.bis+1)
        * (ic.bje-ic.bjs+1)
        * (ic.bke-ic.bks+1);
  };

  auto compute_recv_ifine = [&](MeshBoundaryBuffer &buf,
                                int ox1, int ox2, int ox3,
                                int f1, int f2) {
    auto &ifn = buf.ifine[0];
    if (ox1 == 0) {
      ifn.bis = is_m; ifn.bie = ie_m;
      if (f1 == 1) {
        ifn.bis += cnx1;
      } else {
        ifn.bie -= cnx1;
      }
    } else if (ox1 > 0) {
      ifn.bis = ie_m + 1; ifn.bie = ie_m + ngh;
    } else {
      ifn.bis = is_m - ngh; ifn.bie = is_m - 1;
    }
    if (ox2 == 0) {
      ifn.bjs = js_m; ifn.bje = je_m;
      if (nx2 > 1) {
        if (ox1 != 0) {
          if (f1 == 1) {
            ifn.bjs += cnx2;
          } else {
            ifn.bje -= cnx2;
          }
        } else {
          if (f2 == 1) {
            ifn.bjs += cnx2;
          } else {
            ifn.bje -= cnx2;
          }
        }
      }
    } else if (ox2 > 0) {
      ifn.bjs = je_m + 1; ifn.bje = je_m + ngh;
    } else {
      ifn.bjs = js_m - ngh; ifn.bje = js_m - 1;
    }
    if (ox3 == 0) {
      ifn.bks = ks_m; ifn.bke = ke_m;
      if (nx3 > 1) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 1) {
            ifn.bks += cnx3;
          } else {
            ifn.bke -= cnx3;
          }
        } else {
          if (f2 == 1) {
            ifn.bks += cnx3;
          } else {
            ifn.bke -= cnx3;
          }
        }
      }
    } else if (ox3 > 0) {
      ifn.bks = ke_m + 1; ifn.bke = ke_m + ngh;
    } else {
      ifn.bks = ks_m - ngh; ifn.bke = ks_m - 1;
    }
    buf.ifine_ndat =
        (ifn.bie-ifn.bis+1)
        * (ifn.bje-ifn.bjs+1)
        * (ifn.bke-ifn.bks+1);
  };

  // Iterate over all buffer directions (mirrors InitializeBuffers order)
  // x1 faces
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy=0; fy<nfy; fy++) {
        int idx = NeighborIndex(n,0,0,fy,fz);
        compute_send_icoar(sendbuf[idx], n, 0, 0);
        compute_send_ifine(sendbuf[idx], n, 0, 0, fy, fz);
        compute_recv_icoar(recvbuf[idx], n, 0, 0, fy, fz);
        compute_recv_ifine(recvbuf[idx], n, 0, 0, fy, fz);
      }
    }
  }
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,m,0,fx,fz);
          compute_send_icoar(sendbuf[idx], 0, m, 0);
          compute_send_ifine(sendbuf[idx], 0, m, 0, fx, fz);
          compute_recv_icoar(recvbuf[idx], 0, m, 0, fx, fz);
          compute_recv_ifine(recvbuf[idx], 0, m, 0, fx, fz);
        }
      }
    }
    // x1x2 edges
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int idx = NeighborIndex(n,m,0,fz,0);
          compute_send_icoar(sendbuf[idx], n, m, 0);
          compute_send_ifine(sendbuf[idx], n, m, 0, fz, 0);
          compute_recv_icoar(recvbuf[idx], n, m, 0, fz, 0);
          compute_recv_ifine(recvbuf[idx], n, m, 0, fz, 0);
        }
      }
    }
  }
  if (pmy_pack->pmesh->three_d) {
    // x3 faces
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,0,l,fx,fy);
          compute_send_icoar(sendbuf[idx], 0, 0, l);
          compute_send_ifine(sendbuf[idx], 0, 0, l, fx, fy);
          compute_recv_icoar(recvbuf[idx], 0, 0, l, fx, fy);
          compute_recv_ifine(recvbuf[idx], 0, 0, l, fx, fy);
        }
      }
    }
    // x3x1 edges
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int idx = NeighborIndex(n,0,l,fy,0);
          compute_send_icoar(sendbuf[idx], n, 0, l);
          compute_send_ifine(sendbuf[idx], n, 0, l, fy, 0);
          compute_recv_icoar(recvbuf[idx], n, 0, l, fy, 0);
          compute_recv_ifine(recvbuf[idx], n, 0, l, fy, 0);
        }
      }
    }
    // x2x3 edges
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int idx = NeighborIndex(0,m,l,fx,0);
          compute_send_icoar(sendbuf[idx], 0, m, l);
          compute_send_ifine(sendbuf[idx], 0, m, l, fx, 0);
          compute_recv_icoar(recvbuf[idx], 0, m, l, fx, 0);
          compute_recv_ifine(recvbuf[idx], 0, m, l, fx, 0);
        }
      }
    }
    // corners
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int idx = NeighborIndex(n,m,l,0,0);
          compute_send_icoar(sendbuf[idx], n, m, l);
          compute_send_ifine(sendbuf[idx], n, m, l, 0, 0);
          compute_recv_icoar(recvbuf[idx], n, m, l, 0, 0);
          compute_recv_ifine(recvbuf[idx], n, m, l, 0, 0);
        }
      }
    }
  }

  int nvar = pmy_mg->nvar_;
  int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  if (pmy_pack->pmesh->multilevel) {
    for (int n = 0; n < nnghbr; ++n) {
      int smax = std::max(sendbuf[n].isame_ndat,
                   std::max(sendbuf[n].icoar_ndat, sendbuf[n].ifine_ndat));
      if (nvar * smax > sendbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(sendbuf[n].vars, nmb, nvar * smax);
      }
      int rmax = std::max(recvbuf[n].isame_ndat,
                   std::max(recvbuf[n].icoar_ndat, recvbuf[n].ifine_ndat));
      if (nvar * rmax > recvbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(recvbuf[n].vars, nmb, nvar * rmax);
      }
    }

    int cbnx3 = cnx3 + 2*ngh;
    int cbnx2 = cnx2 + 2*ngh;
    int cbnx1 = cnx1 + 2*ngh;
    Kokkos::realloc(coarse_buf_, nmb, nvar, cbnx3, cbnx2, cbnx1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::ComputePerLevelIndices()
//! \brief Pre-compute isame/icoar/ifine send and recv indices for every MG level and
//! every neighbor direction.  This replaces the fragile runtime shift logic in
//! PackAndSendMG / RecvAndUnpackMG with exact, pre-computed values.  Uses the same index
//! formulas as buffs_cc.cpp but parameterized by MG ngh and the per-level cell count.
//! Must be called AFTER InitializeBuffers (so the 56 buffer slots exist).

void MultigridBoundaryValues::ComputePerLevelIndices() {
  int ngh    = pmy_mg->GetGhostCells();
  int ng1    = ngh - 1;
  int nx_max = pmy_mg->GetSize();            // finest-level cell count per direction
  int nlevel = pmy_mg->GetNumberOfLevels();

  bool md = pmy_pack->pmesh->multi_d;
  bool td = pmy_pack->pmesh->three_d;
  bool ml = pmy_pack->pmesh->multilevel;

  int nfx = ml ? 2 : 1;
  int nfy = (ml && md) ? 2 : 1;
  int nfz = (ml && td) ? 2 : 1;

  // Helper lambdas that mirror buffs_cc.cpp formulas but with MG parameters.
  // ncells = active cells in this direction at the current MG level.
  auto compute_send = [&](MGPerLevelIndcs &out,
                          int ox1, int ox2, int ox3, int f1, int f2,
                          int ncells) {
    int is_m  = ngh, ie_m  = ngh + ncells - 1;
    int js_m  = ngh, je_m  = ngh + ncells - 1;
    int ks_m  = ngh, ke_m  = ngh + ncells - 1;
    int cnx   = ncells / 2;
    int cis_m = ngh, cie_m = ngh + cnx - 1;
    int cjs_m = ngh, cje_m = ngh + cnx - 1;
    int cks_m = ngh, cke_m = ngh + cnx - 1;

    // -- isame (same-level send) --
    if (f1 == 0 && f2 == 0) {
      auto &s = out.isame;
      s.bis = (ox1 > 0) ? (ie_m - ng1) : is_m;
      s.bie = (ox1 < 0) ? (is_m + ng1) : ie_m;
      s.bjs = (ox2 > 0) ? (je_m - ng1) : js_m;
      s.bje = (ox2 < 0) ? (js_m + ng1) : je_m;
      s.bks = (ox3 > 0) ? (ke_m - ng1) : ks_m;
      s.bke = (ox3 < 0) ? (ks_m + ng1) : ke_m;
      out.isame_ndat = (s.bie-s.bis+1)*(s.bje-s.bjs+1)*(s.bke-s.bks+1);
    }

    // -- icoar (send to coarser) --
    // Face neighbors (nface==1): indices point to coarse_buf_ ghost cells
    // where face-aligned 2x2 averages are stored by FillCoarseMG.
    // Edge/corner neighbors (nface>1): indices point to coarse_buf_ interior
    // where volume averages are stored by FillCoarseMG.
    {
      auto &c = out.icoar;
      int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
      if (nface == 1) {
        c.bis = (ox1 > 0) ? (cie_m + 1)   : (ox1 < 0) ? (cis_m - ngh) : cis_m;
        c.bie = (ox1 > 0) ? (cie_m + ngh)  : (ox1 < 0) ? (cis_m - 1)  : cie_m;
        c.bjs = (ox2 > 0) ? (cje_m + 1)   : (ox2 < 0) ? (cjs_m - ngh) : cjs_m;
        c.bje = (ox2 > 0) ? (cje_m + ngh)  : (ox2 < 0) ? (cjs_m - 1)  : cje_m;
        c.bks = (ox3 > 0) ? (cke_m + 1)   : (ox3 < 0) ? (cks_m - ngh) : cks_m;
        c.bke = (ox3 > 0) ? (cke_m + ngh)  : (ox3 < 0) ? (cks_m - 1)  : cke_m;
      } else {
        c.bis = (ox1 > 0) ? (cie_m - ng1) : cis_m;
        c.bie = (ox1 < 0) ? (cis_m + ng1) : cie_m;
        c.bjs = (ox2 > 0) ? (cje_m - ng1) : cjs_m;
        c.bje = (ox2 < 0) ? (cjs_m + ng1) : cje_m;
        c.bks = (ox3 > 0) ? (cke_m - ng1) : cks_m;
        c.bke = (ox3 < 0) ? (cks_m + ng1) : cke_m;
      }
      out.icoar_ndat = (c.bie-c.bis+1)*(c.bje-c.bjs+1)*(c.bke-c.bks+1);
    }

    // -- ifine (send to finer) --
    {
      auto &f = out.ifine;
      f.bis = (ox1 > 0) ? (ie_m - ng1) : is_m;
      f.bie = (ox1 < 0) ? (is_m + ng1) : ie_m;
      f.bjs = (ox2 > 0) ? (je_m - ng1) : js_m;
      f.bje = (ox2 < 0) ? (js_m + ng1) : je_m;
      f.bks = (ox3 > 0) ? (ke_m - ng1) : ks_m;
      f.bke = (ox3 < 0) ? (ks_m + ng1) : ke_m;
      if (ox1 == 0) {
        if (f1 == 1) {
          f.bis += cnx - ngh;
        } else {
          f.bie -= cnx - ngh;
        }
      }
      if (ox2 == 0 && md) {
        if (ox1 != 0) {
          if (f1 == 1) {
            f.bjs += cnx - ngh;
          } else {
            f.bje -= cnx - ngh;
          }
        } else {
          if (f2 == 1) {
            f.bjs += cnx - ngh;
          } else {
            f.bje -= cnx - ngh;
          }
        }
      }
      if (ox3 == 0 && td) {
        if (ox1 != 0 && ox2 != 0) {
          if (f1 == 1) {
            f.bks += cnx - ngh;
          } else {
            f.bke -= cnx - ngh;
          }
        } else {
          if (f2 == 1) {
            f.bks += cnx - ngh;
          } else {
            f.bke -= cnx - ngh;
          }
        }
      }
      out.ifine_ndat =
          (f.bie-f.bis+1)
          * (f.bje-f.bjs+1)
          * (f.bke-f.bks+1);
    }
  };

  auto compute_recv = [&](MGPerLevelIndcs &out,
                          int ox1, int ox2, int ox3, int f1, int f2,
                          int ncells) {
    int is_m  = ngh, ie_m  = ngh + ncells - 1;
    int js_m  = ngh, je_m  = ngh + ncells - 1;
    int ks_m  = ngh, ke_m  = ngh + ncells - 1;
    int cnx   = ncells / 2;
    int cis_m = ngh, cie_m = ngh + cnx - 1;
    int cjs_m = ngh, cje_m = ngh + cnx - 1;
    int cks_m = ngh, cke_m = ngh + cnx - 1;

    // -- isame (same-level recv) --
    if (f1 == 0 && f2 == 0) {
      auto &s = out.isame;
      if (ox1 == 0) {
        s.bis = is_m; s.bie = ie_m;
      } else if (ox1 > 0) {
        s.bis = ie_m + 1; s.bie = ie_m + ngh;
      } else {
        s.bis = is_m - ngh; s.bie = is_m - 1;
      }
      if (ox2 == 0) {
        s.bjs = js_m; s.bje = je_m;
      } else if (ox2 > 0) {
        s.bjs = je_m + 1; s.bje = je_m + ngh;
      } else {
        s.bjs = js_m - ngh; s.bje = js_m - 1;
      }
      if (ox3 == 0) {
        s.bks = ks_m; s.bke = ke_m;
      } else if (ox3 > 0) {
        s.bks = ke_m + 1; s.bke = ke_m + ngh;
      } else {
        s.bks = ks_m - ngh; s.bke = ks_m - 1;
      }
      out.isame_ndat =
          (s.bie-s.bis+1)
          * (s.bje-s.bjs+1)
          * (s.bke-s.bks+1);
    }

    // -- icoar (recv from coarser, matches send-to-finer) --
    {
      auto &c = out.icoar;
      if (ox1 == 0) {
        c.bis = cis_m; c.bie = cie_m;
        if (f1 == 0) {
          c.bie += ngh;
        } else {
          c.bis -= ngh;
        }
      } else if (ox1 > 0) {
        c.bis = cie_m + 1; c.bie = cie_m + ngh;
      } else {
        c.bis = cis_m - ngh; c.bie = cis_m - 1;
      }

      if (ox2 == 0) {
        c.bjs = cjs_m; c.bje = cje_m;
        if (md) {
          if (ox1 != 0) {
            if (f1 == 0) {
              c.bje += ngh;
            } else {
              c.bjs -= ngh;
            }
          } else {
            if (f2 == 0) {
              c.bje += ngh;
            } else {
              c.bjs -= ngh;
            }
          }
        }
      } else if (ox2 > 0) {
        c.bjs = cje_m + 1; c.bje = cje_m + ngh;
      } else {
        c.bjs = cjs_m - ngh; c.bje = cjs_m - 1;
      }

      if (ox3 == 0) {
        c.bks = cks_m; c.bke = cke_m;
        if (td) {
          if (ox1 != 0 && ox2 != 0) {
            if (f1 == 0) {
              c.bke += ngh;
            } else {
              c.bks -= ngh;
            }
          } else {
            if (f2 == 0) {
              c.bke += ngh;
            } else {
              c.bks -= ngh;
            }
          }
        }
      } else if (ox3 > 0) {
        c.bks = cke_m + 1; c.bke = cke_m + ngh;
      } else {
        c.bks = cks_m - ngh; c.bke = cks_m - 1;
      }
      out.icoar_ndat =
          (c.bie-c.bis+1)
          * (c.bje-c.bjs+1)
          * (c.bke-c.bks+1);
    }

    // -- ifine (recv from finer, matches send-to-coarser) --
    {
      auto &fn = out.ifine;
      if (ox1 == 0) {
        fn.bis = is_m; fn.bie = ie_m;
        if (f1 == 1) {
          fn.bis += cnx;
        } else {
          fn.bie -= cnx;
        }
      } else if (ox1 > 0) {
        fn.bis = ie_m + 1; fn.bie = ie_m + ngh;
      } else {
        fn.bis = is_m - ngh; fn.bie = is_m - 1;
      }

      if (ox2 == 0) {
        fn.bjs = js_m; fn.bje = je_m;
        if (md) {
          if (ox1 != 0) {
            if (f1 == 1) {
              fn.bjs += cnx;
            } else {
              fn.bje -= cnx;
            }
          } else {
            if (f2 == 1) {
              fn.bjs += cnx;
            } else {
              fn.bje -= cnx;
            }
          }
        }
      } else if (ox2 > 0) {
        fn.bjs = je_m + 1; fn.bje = je_m + ngh;
      } else {
        fn.bjs = js_m - ngh; fn.bje = js_m - 1;
      }

      if (ox3 == 0) {
        fn.bks = ks_m; fn.bke = ke_m;
        if (td) {
          if (ox1 != 0 && ox2 != 0) {
            if (f1 == 1) {
              fn.bks += cnx;
            } else {
              fn.bke -= cnx;
            }
          } else {
            if (f2 == 1) {
              fn.bks += cnx;
            } else {
              fn.bke -= cnx;
            }
          }
        }
      } else if (ox3 > 0) {
        fn.bks = ke_m + 1; fn.bke = ke_m + ngh;
      } else {
        fn.bks = ks_m - ngh; fn.bke = ks_m - 1;
      }
      out.ifine_ndat =
          (fn.bie-fn.bis+1)
          * (fn.bje-fn.bjs+1)
          * (fn.bke-fn.bks+1);
    }
  };

  // Fill indices for each MG level and each neighbor direction.
  for (int lev = 0; lev < nlevel && lev < kMaxMGLevels; ++lev) {
    int shift = nlevel - 1 - lev;
    int ncells = nx_max >> shift;
    if (ncells < 1) ncells = 1;

    // x1 faces
    for (int n = -1; n <= 1; n += 2) {
      for (int fz = 0; fz < nfz; fz++) {
        for (int fy = 0; fy < nfy; fy++) {
          int idx = NeighborIndex(n, 0, 0, fy, fz);
          compute_send(send_mg_indcs_[idx][lev], n, 0, 0, fy, fz, ncells);
          compute_recv(recv_mg_indcs_[idx][lev], n, 0, 0, fy, fz, ncells);
        }
      }
    }
    if (md) {
      // x2 faces
      for (int m = -1; m <= 1; m += 2) {
        for (int fz = 0; fz < nfz; fz++) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, m, 0, fx, fz);
            compute_send(send_mg_indcs_[idx][lev], 0, m, 0, fx, fz, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, m, 0, fx, fz, ncells);
          }
        }
      }
      // x1x2 edges
      for (int m = -1; m <= 1; m += 2) {
        for (int n = -1; n <= 1; n += 2) {
          for (int fz = 0; fz < nfz; fz++) {
            int idx = NeighborIndex(n, m, 0, fz, 0);
            compute_send(send_mg_indcs_[idx][lev], n, m, 0, fz, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, m, 0, fz, 0, ncells);
          }
        }
      }
    }
    if (td) {
      // x3 faces
      for (int l = -1; l <= 1; l += 2) {
        for (int fy = 0; fy < nfy; fy++) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, 0, l, fx, fy);
            compute_send(send_mg_indcs_[idx][lev], 0, 0, l, fx, fy, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, 0, l, fx, fy, ncells);
          }
        }
      }
      // x3x1 edges
      for (int l = -1; l <= 1; l += 2) {
        for (int n = -1; n <= 1; n += 2) {
          for (int fy = 0; fy < nfy; fy++) {
            int idx = NeighborIndex(n, 0, l, fy, 0);
            compute_send(send_mg_indcs_[idx][lev], n, 0, l, fy, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, 0, l, fy, 0, ncells);
          }
        }
      }
      // x2x3 edges
      for (int l = -1; l <= 1; l += 2) {
        for (int m = -1; m <= 1; m += 2) {
          for (int fx = 0; fx < nfx; fx++) {
            int idx = NeighborIndex(0, m, l, fx, 0);
            compute_send(send_mg_indcs_[idx][lev], 0, m, l, fx, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], 0, m, l, fx, 0, ncells);
          }
        }
      }
      // corners
      for (int l = -1; l <= 1; l += 2) {
        for (int m = -1; m <= 1; m += 2) {
          for (int n = -1; n <= 1; n += 2) {
            int idx = NeighborIndex(n, m, l, 0, 0);
            compute_send(send_mg_indcs_[idx][lev], n, m, l, 0, 0, ncells);
            compute_recv(recv_mg_indcs_[idx][lev], n, m, l, 0, 0, ncells);
          }
        }
      }
    }
  }

  int nvar = pmy_mg->nvar_;
  int nmb = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  int nnghbr = pmy_pack->pmb->nnghbr;
  int finest = nlevel - 1;

  if (pmy_pack->pmesh->multilevel) {
    for (int n = 0; n < nnghbr; ++n) {
      int smax = std::max(send_mg_indcs_[n][finest].isame_ndat,
                   std::max(send_mg_indcs_[n][finest].icoar_ndat,
                            send_mg_indcs_[n][finest].ifine_ndat));
      if (nvar * smax > sendbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(sendbuf[n].vars, nmb, nvar * smax);
      }
      int rmax = std::max(recv_mg_indcs_[n][finest].isame_ndat,
                   std::max(recv_mg_indcs_[n][finest].icoar_ndat,
                            recv_mg_indcs_[n][finest].ifine_ndat));
      if (nvar * rmax > recvbuf[n].vars.extent_int(1)) {
        Kokkos::realloc(recvbuf[n].vars, nmb, nvar * rmax);
      }
    }

    int cnx_f = nx_max / 2;
    int cbn = cnx_f + 2*ngh;
    Kokkos::realloc(coarse_buf_, nmb, nvar, cbn, cbn, cbn);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::FillCoarseMG()
//! \brief Restrict MG data interior into coarse_buf_ interior so that the prolongation
//! kernel has gradient context from the block's own data.

void MultigridBoundaryValues::FillCoarseMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return;
  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;
  if (ncells < 2) return;
  int cnc = ncells / 2;
  int nmb = pmy_pack->nmb_thispack;
  auto cbuf = coarse_buf_;

  // Volume-average restriction: fill coarse_buf_ interior
  Kokkos::parallel_for("FillCoarseMG",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, DevExeSpace>(
      {0, 0, 0, 0}, {nmb * nvar, cnc, cnc, cnc}),
    KOKKOS_LAMBDA(const int mv, const int ck, const int cj, const int ci) {
      int m = mv / nvar;
      int v = mv - m * nvar;
      int fi = ngh + 2*ci;
      int fj = ngh + 2*cj;
      int fk = ngh + 2*ck;
      cbuf(m, v, ngh + ck, ngh + cj, ngh + ci) = 0.125 * (
        u(m,v,fk,  fj,  fi) + u(m,v,fk,  fj,  fi+1) +
        u(m,v,fk,  fj+1,fi) + u(m,v,fk,  fj+1,fi+1) +
        u(m,v,fk+1,fj,  fi) + u(m,v,fk+1,fj,  fi+1) +
        u(m,v,fk+1,fj+1,fi) + u(m,v,fk+1,fj+1,fi+1));
    });

  if (!(pmy_pack->pmesh->multilevel)) return;

  // Face-aligned restriction: store 2x2 face averages in coarse_buf_ ghost cells.
  // These are consumed by PackAndSendMG for fine-to-coarse face sends.
  // face_id: 0=x-left, 1=x-right, 2=y-left, 3=y-right, 4=z-left, 5=z-right
  Kokkos::parallel_for("FillCoarseMG_faces",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>, DevExeSpace>(
      {0, 0, 0, 0}, {nmb * nvar, 6, cnc, cnc}),
    KOKKOS_LAMBDA(const int mv, const int face, const int c1, const int c0) {
      int m = mv / nvar;
      int v = mv - m * nvar;
      if (face < 2) {
        int fj = ngh + 2*c0;
        int fk = ngh + 2*c1;
        int fi = (face == 0) ? ngh : ngh + ncells - 1;
        int ci = (face == 0) ? ngh - 1 : ngh + cnc;
        cbuf(m, v, ngh + c1, ngh + c0, ci) = 0.25 * (
          u(m,v,fk,  fj,  fi) + u(m,v,fk,  fj+1,fi) +
          u(m,v,fk+1,fj,  fi) + u(m,v,fk+1,fj+1,fi));
      } else if (face < 4) {
        int fi = ngh + 2*c0;
        int fk = ngh + 2*c1;
        int fj = (face == 2) ? ngh : ngh + ncells - 1;
        int cj = (face == 2) ? ngh - 1 : ngh + cnc;
        cbuf(m, v, ngh + c1, cj, ngh + c0) = 0.25 * (
          u(m,v,fk,  fj,fi  ) + u(m,v,fk,  fj,fi+1) +
          u(m,v,fk+1,fj,fi  ) + u(m,v,fk+1,fj,fi+1));
      } else {
        int fi = ngh + 2*c0;
        int fj = ngh + 2*c1;
        int fk = (face == 4) ? ngh : ngh + ncells - 1;
        int ck = (face == 4) ? ngh - 1 : ngh + cnc;
        cbuf(m, v, ck, ngh + c1, ngh + c0) = 0.25 * (
          u(m,v,fk,fj,  fi  ) + u(m,v,fk,fj,  fi+1) +
          u(m,v,fk,fj+1,fi  ) + u(m,v,fk,fj+1,fi+1));
      }
    });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::ProlongateFCMG()
//! \brief Prolongate from coarse_buf_ to fine ghost cells using the same flux-conserving
//! formulas as FillFineCoarseMGGhosts.  For face neighbors at coarser level, uses
//! gradient-based prolongation.  For edge/corner neighbors at coarser level, uses simple
//! injection.  For finer neighbors, restriction was already done inline in unpack.

TaskStatus MultigridBoundaryValues::ProlongateFCMG(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;
  if (ncells < 2) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr_d = pmy_pack->pmb->nghbr.d_view;
  auto mblev_d = pmy_pack->pmb->mb_lev.d_view;
  auto mbgid_d = pmy_pack->pmb->mb_gid.d_view;
  auto fc_cx = pmy_mg->fc_childx_;
  auto fc_cy = pmy_mg->fc_childy_;
  auto fc_cz = pmy_mg->fc_childz_;
  auto cbuf = coarse_buf_;

  int nvar_l = nvar;
  int ngh_l = ngh;
  int ncells_l = ncells;
  int half = ncells / 2;
  constexpr Real ot = 1.0/3.0;

  Kokkos::parallel_for("ProlongateFCMG",
    Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m) {
      int m_lev = mblev_d(m);
      int child_x = fc_cx(m);
      int child_y = fc_cy(m);
      int child_z = fc_cz(m);

      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
            int f2_max = (nface == 1) ? 1 : 0;
            int f1_max = (nface <= 2) ? 1 : 0;

            for (int f2 = 0; f2 <= f2_max; ++f2) {
              for (int f1 = 0; f1 <= f1_max; ++f1) {
                int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
                if (n < 0 || n >= 56) continue;
                if (nghbr_d(m, n).gid < 0) continue;
                int nlev = nghbr_d(m, n).lev;

                // From finer face neighbor: apply FC correction.
                // Ghost cells already contain restricted face avg from unpack.
                if (nlev > m_lev && nface == 1) {
                  int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                  int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                  int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (ox1 != 0) {
                    sub_y = f1; sub_z = f2;
                  } else if (ox2 != 0) {
                    sub_x = f1; sub_z = f2;
                  } else {
                    sub_x = f1; sub_y = f2;
                  }

                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l + sub_x*half;
                    gie = ngh_l + sub_x*half + half - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l + sub_y*half;
                    gje = ngh_l + sub_y*half + half - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l + sub_z*half;
                    gke = ngh_l + sub_z*half + half - 1;
                  }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          Real avg = u(m,v,gk,gj,gi);
                          u(m,v,gk,gj,gi) =
                              ot*(4.0*avg
                              - u(m,v,gk+ok,gj+oj,gi+oi));
                        }
                      }
                    }
                  }
                  continue;
                }

                if (nlev >= m_lev) continue;

                // Face neighbor from coarser: flux-conserving prolongation
                // from coarse_buf_ into fine ghost cells of u
                if (nface == 1) {
                  if (ox1 != 0) {
                    int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int si  = (ox1 < 0) ? ngh_l - 1 : ngh_l + half;
                    int sj0 = ngh_l;
                    int sk0 = ngh_l;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int sj = sj0; sj < sj0 + half; ++sj) {
                          int fj = ngh_l + 2*(sj - sj0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = cbuf(m,v,sk,sj,si);
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+half) ? sj+1 : sj;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+half) ? sk+1 : sk;
                          Real gy = 0.125*(cbuf(m,v,sk,sjp,si)-cbuf(m,v,sk,sjm,si));
                          Real gz = 0.125*(cbuf(m,v,skp,sj,si)-cbuf(m,v,skm,sj,si));
                          u(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u(m,v,fk  ,fj  ,fi));
                          u(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u(m,v,fk  ,fj+1,fi));
                          u(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u(m,v,fk+1,fj  ,fi));
                          u(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u(m,v,fk+1,fj+1,fi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sj  = (ox2 < 0) ? ngh_l - 1 : ngh_l + half;
                    int si0 = ngh_l;
                    int sk0 = ngh_l;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = cbuf(m,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+half) ? si+1 : si;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+half) ? sk+1 : sk;
                          Real gx = 0.125*(cbuf(m,v,sk,sj,sip)-cbuf(m,v,sk,sj,sim));
                          Real gz = 0.125*(cbuf(m,v,skp,sj,si)-cbuf(m,v,skm,sj,si));
                          u(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u(m,v,fk  ,fj,fi  ));
                          u(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u(m,v,fk  ,fj,fi+1));
                          u(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u(m,v,fk+1,fj,fi  ));
                          u(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u(m,v,fk+1,fj,fi+1));
                        }
                      }
                    }
                  } else {
                    int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sk  = (ox3 < 0) ? ngh_l - 1 : ngh_l + half;
                    int si0 = ngh_l;
                    int sj0 = ngh_l;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fj = ngh_l + 2*(sj - sj0);
                          Real cc = cbuf(m,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+half) ? si+1 : si;
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+half) ? sj+1 : sj;
                          Real gx = 0.125*(cbuf(m,v,sk,sj,sip)-cbuf(m,v,sk,sj,sim));
                          Real gy = 0.125*(cbuf(m,v,sk,sjp,si)-cbuf(m,v,sk,sjm,si));
                          u(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u(m,v,fk,fj  ,fi  ));
                          u(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u(m,v,fk,fj  ,fi+1));
                          u(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u(m,v,fk,fj+1,fi  ));
                          u(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u(m,v,fk,fj+1,fi+1));
                        }
                      }
                    }
                  }
                } else {
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l;
                    gie = ngh_l + ncells_l - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l;
                    gje = ngh_l + ncells_l - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l;
                    gke = ngh_l + ncells_l - 1;
                  }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int ci, cj, ck;
                          if (ox1 < 0)      ci = ngh_l - 1;
                          else if (ox1 > 0) ci = ngh_l + half;
                          else              ci = ngh_l + (gi - ngh_l)/2;
                          if (ox2 < 0)      cj = ngh_l - 1;
                          else if (ox2 > 0) cj = ngh_l + half;
                          else              cj = ngh_l + (gj - ngh_l)/2;
                          if (ox3 < 0)      ck = ngh_l - 1;
                          else if (ox3 > 0) ck = ngh_l + half;
                          else              ck = ngh_l + (gk - ngh_l)/2;

                          u(m, v, gk, gj, gi) = cbuf(m, v, ck, cj, ci);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts()
//! \brief Fill ghost cells at fine-coarse boundaries.
//! Faces use flux-conserving prolongation/restriction matching Athena++ formulas.
//! Edges and corners use simple injection/restriction. Same-rank only.

TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;

  if (ncells < 1) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int my_rank = global_variable::my_rank;
  auto nghbr_d = pmy_pack->pmb->nghbr.d_view;
  auto mblev_d = pmy_pack->pmb->mb_lev.d_view;
  auto mbgid_d = pmy_pack->pmb->mb_gid.d_view;

#ifndef NDEBUG
  {
    static bool warned_cross_rank_fc = false;
    if (!warned_cross_rank_fc) {
      auto &nghbr_h = pmy_pack->pmb->nghbr;
      auto &mblev_h = pmy_pack->pmb->mb_lev;
      for (int m = 0; m < nmb && !warned_cross_rank_fc; ++m) {
        for (int n = 0; n < nnghbr && !warned_cross_rank_fc; ++n) {
          if (nghbr_h.h_view(m,n).gid >= 0
              && nghbr_h.h_view(m,n).lev != mblev_h.h_view(m)
              && nghbr_h.h_view(m,n).rank != my_rank) {
            std::cout << "### MG WARNING: cross-rank fine-coarse neighbor detected "
                      << "(m=" << m << " n=" << n << " rank=" << nghbr_h.h_view(m,n).rank
                      << "). FillFineCoarseMGGhosts skips cross-rank neighbors."
                      << std::endl;
            warned_cross_rank_fc = true;
          }
        }
      }
    }
  }
#endif
  auto fc_cx = pmy_mg->fc_childx_;
  auto fc_cy = pmy_mg->fc_childy_;
  auto fc_cz = pmy_mg->fc_childz_;

  int nmb_l = nmb;
  int nnghbr_l = nnghbr;
  int my_rank_l = my_rank;
  int nvar_l = nvar;
  int ngh_l = ngh;
  int ncells_l = ncells;
  constexpr Real ot = 1.0/3.0;

  Kokkos::parallel_for("FillFCMGGhosts",
    Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m) {
      int m_lev = mblev_d(m);
      int child_x = fc_cx(m);
      int child_y = fc_cy(m);
      int child_z = fc_cz(m);
      int half = ncells_l / 2;

      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);
            int f2_max = (nface == 1) ? 1 : 0;
            int f1_max = (nface <= 2) ? 1 : 0;

            for (int f2 = 0; f2 <= f2_max; ++f2) {
              for (int f1 = 0; f1 <= f1_max; ++f1) {
                int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
                if (n < 0 || n >= nnghbr_l) continue;
                if (nghbr_d(m, n).gid < 0) continue;

                int nlev = nghbr_d(m, n).lev;
                if (nlev == m_lev) continue;
                if (nghbr_d(m, n).rank != my_rank_l) continue;

                int dm = nghbr_d(m, n).gid - mbgid_d(0);
                if (dm < 0 || dm >= nmb_l) continue;

                if (nlev < m_lev && nface == 1) {
                  // Coarser neighbor, face: flux-conserving prolongation
                  if (ox1 != 0) {
                    int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int si  = (ox1 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int sj0 = ngh_l + child_y * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int sj = sj0; sj < sj0 + half; ++sj) {
                          int fj = ngh_l + 2*(sj - sj0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real gy = 0.125*(u(dm,v,sk,sjp,si)-u(dm,v,sk,sjm,si));
                          Real gz = 0.125*(u(dm,v,skp,sj,si)-u(dm,v,skm,sj,si));
                          u(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u(m,v,fk  ,fj  ,fi));
                          u(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u(m,v,fk  ,fj+1,fi));
                          u(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u(m,v,fk+1,fj  ,fi));
                          u(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u(m,v,fk+1,fj+1,fi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sj  = (ox2 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real gx = 0.125*(u(dm,v,sk,sj,sip)-u(dm,v,sk,sj,sim));
                          Real gz = 0.125*(u(dm,v,skp,sj,si)-u(dm,v,skm,sj,si));
                          u(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u(m,v,fk  ,fj,fi  ));
                          u(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u(m,v,fk  ,fj,fi+1));
                          u(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u(m,v,fk+1,fj,fi  ));
                          u(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u(m,v,fk+1,fj,fi+1));
                        }
                      }
                    }
                  } else {
                    int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sk  = (ox3 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sj0 = ngh_l + child_y * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fj = ngh_l + 2*(sj - sj0);
                          Real cc = u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          Real gx = 0.125*(u(dm,v,sk,sj,sip)-u(dm,v,sk,sj,sim));
                          Real gy = 0.125*(u(dm,v,sk,sjp,si)-u(dm,v,sk,sjm,si));
                          u(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u(m,v,fk,fj  ,fi  ));
                          u(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u(m,v,fk,fj  ,fi+1));
                          u(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u(m,v,fk,fj+1,fi  ));
                          u(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u(m,v,fk,fj+1,fi+1));
                        }
                      }
                    }
                  }

                } else if (nlev > m_lev && nface == 1) {
                  // Finer neighbor, face: flux-conserving restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (ox1 != 0) {
                    sub_y = f1; sub_z = f2;
                  }
                  if (ox2 != 0) {
                    sub_x = f1; sub_z = f2;
                  }
                  if (ox3 != 0) {
                    sub_x = f1; sub_y = f2;
                  }

                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l + sub_x*half;
                    gie = ngh_l + sub_x*half + half - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l + sub_y*half;
                    gje = ngh_l + sub_y*half + half - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l + sub_z*half;
                    gke = ngh_l + sub_z*half + half - 1;
                  }

                  int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                  int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                  int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                  if (ox1 != 0) {
                    int fi = (ox1 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gj = gjs; gj <= gje; ++gj) {
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = 0.25*(u(dm,v,fk0,fj0,fi)+u(dm,v,fk0,fj0+1,fi)
                                           +u(dm,v,fk0+1,fj0,fi)+u(dm,v,fk0+1,fj0+1,fi));
                          u(m,v,gk,gj,gis) = ot*(4.0*favg - u(m,v,gk+ok,gj+oj,gis+oi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fj = (ox2 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = 0.25*(u(dm,v,fk0,fj,fi0)+u(dm,v,fk0,fj,fi0+1)
                                           +u(dm,v,fk0+1,fj,fi0)+u(dm,v,fk0+1,fj,fi0+1));
                          u(m,v,gk,gjs,gi) = ot*(4.0*favg - u(m,v,gk+ok,gjs+oj,gi+oi));
                        }
                      }
                    }
                  } else {
                    int fk = (ox3 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          Real favg = 0.25*(u(dm,v,fk,fj0,fi0)+u(dm,v,fk,fj0,fi0+1)
                                           +u(dm,v,fk,fj0+1,fi0)+u(dm,v,fk,fj0+1,fi0+1));
                          u(m,v,gks,gj,gi) = ot*(4.0*favg - u(m,v,gks+ok,gj+oj,gi+oi));
                        }
                      }
                    }
                  }

                } else if (nlev < m_lev) {
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l;
                    gie = ngh_l + ncells_l - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l;
                    gje = ngh_l + ncells_l - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l;
                    gke = ngh_l + ncells_l - 1;
                  }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int si, sj, sk;
                          if (ox1 < 0)      si = ngh_l + ncells_l - 1;
                          else if (ox1 > 0) si = ngh_l;
                          else
                            si = ngh_l + child_x*half + (gi - ngh_l)/2;
                          if (ox2 < 0)      sj = ngh_l + ncells_l - 1;
                          else if (ox2 > 0) sj = ngh_l;
                          else
                            sj = ngh_l + child_y*half + (gj - ngh_l)/2;
                          if (ox3 < 0)      sk = ngh_l + ncells_l - 1;
                          else if (ox3 > 0) sk = ngh_l;
                          else
                            sk = ngh_l + child_z*half + (gk - ngh_l)/2;

                          u(m, v, gk, gj, gi) = u(dm, v, sk, sj, si);
                        }
                      }
                    }
                  }

                } else {
                  // Finer neighbor, edge/corner: simple restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (nface == 2) {
                    if (ox1 == 0) sub_x = f1;
                    if (ox2 == 0) sub_y = f1;
                    if (ox3 == 0) sub_z = f1;
                  }
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0) {
                    gis = 0; gie = ngh_l - 1;
                  } else if (ox1 > 0) {
                    gis = ngh_l + ncells_l;
                    gie = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gis = ngh_l + sub_x*half;
                    gie = ngh_l + sub_x*half + half - 1;
                  }
                  if (ox2 < 0) {
                    gjs = 0; gje = ngh_l - 1;
                  } else if (ox2 > 0) {
                    gjs = ngh_l + ncells_l;
                    gje = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gjs = ngh_l + sub_y*half;
                    gje = ngh_l + sub_y*half + half - 1;
                  }
                  if (ox3 < 0) {
                    gks = 0; gke = ngh_l - 1;
                  } else if (ox3 > 0) {
                    gks = ngh_l + ncells_l;
                    gke = ngh_l + ncells_l + ngh_l - 1;
                  } else {
                    gks = ngh_l + sub_z*half;
                    gke = ngh_l + sub_z*half + half - 1;
                  }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0, fi1, fj0, fj1, fk0, fk1;
                          if (ox1 < 0) {
                            fi0 = ngh_l+ncells_l-2; fi1 = ngh_l+ncells_l-1;
                          } else if (ox1 > 0) {
                            fi0 = ngh_l; fi1 = ngh_l + 1;
                          } else {
                            fi0 = ngh_l+2*(gi-(ngh_l+sub_x*half)); fi1 = fi0+1;
                          }
                          if (ox2 < 0) {
                            fj0 = ngh_l+ncells_l-2; fj1 = ngh_l+ncells_l-1;
                          } else if (ox2 > 0) {
                            fj0 = ngh_l; fj1 = ngh_l + 1;
                          } else {
                            fj0 = ngh_l+2*(gj-(ngh_l+sub_y*half)); fj1 = fj0+1;
                          }
                          if (ox3 < 0) {
                            fk0 = ngh_l+ncells_l-2; fk1 = ngh_l+ncells_l-1;
                          } else if (ox3 > 0) {
                            fk0 = ngh_l; fk1 = ngh_l + 1;
                          } else {
                            fk0 = ngh_l+2*(gk-(ngh_l+sub_z*half)); fk1 = fk0+1;
                          }
                          u(m, v, gk, gj, gi) = 0.125 * (
                            u(dm,v,fk0,fj0,fi0) + u(dm,v,fk0,fj0,fi1) +
                            u(dm,v,fk0,fj1,fi0) + u(dm,v,fk0,fj1,fi1) +
                            u(dm,v,fk1,fj0,fi0) + u(dm,v,fk1,fj0,fi1) +
                            u(dm,v,fk1,fj1,fi0) + u(dm,v,fk1,fj1,fi1));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
  });

#if MPI_PARALLEL_ENABLED
  // Cross-rank fine-coarse ghost fill via synchronous MPI.
  // Each rank sends its block's data at the current MG level for every cross-rank
  // FC neighbor pair, then applies prolongation/restriction using the received data.
  {
    auto &nghbr_h = pmy_pack->pmb->nghbr;
    auto &mblev_h = pmy_pack->pmb->mb_lev;

    struct FCPair {
      int m, n, ox1, ox2, ox3, f1, f2, nface;
      int remote_rank, remote_gid, nlev, m_lev;
    };
    std::vector<FCPair> pairs;
    for (int m = 0; m < nmb; ++m) {
      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nf_ = (ox1 != 0 ? 1 : 0)
                      + (ox2 != 0 ? 1 : 0)
                      + (ox3 != 0 ? 1 : 0);
            int f2_max = (nf_ == 1) ? 1 : 0;
            int f1_max = (nf_ <= 2) ? 1 : 0;
            for (int f2_ = 0; f2_ <= f2_max; ++f2_) {
              for (int f1_ = 0; f1_ <= f1_max; ++f1_) {
                int nn = NeighborIndex(ox1, ox2, ox3, f1_, f2_);
                if (nn < 0 || nn >= nnghbr) continue;
                if (nghbr_h.h_view(m,nn).gid < 0) continue;
                int nlev_n = nghbr_h.h_view(m,nn).lev;
                if (nlev_n == mblev_h.h_view(m)) continue;
                if (nghbr_h.h_view(m,nn).rank == my_rank) continue;
                FCPair p;
                p.m = m; p.n = nn;
                p.ox1 = ox1; p.ox2 = ox2; p.ox3 = ox3;
                p.f1 = f1_; p.f2 = f2_;
                p.nface = (ox1 != 0 ? 1 : 0)
                          + (ox2 != 0 ? 1 : 0)
                          + (ox3 != 0 ? 1 : 0);
                p.remote_rank = nghbr_h.h_view(m,nn).rank;
                p.remote_gid = nghbr_h.h_view(m,nn).gid;
                p.nlev = nlev_n;
                p.m_lev = mblev_h.h_view(m);
                pairs.push_back(p);
              }
            }
          }
        }
      }
    }

    if (!pairs.empty()) {
      Kokkos::fence();
      int ntot = ncells + 2*ngh;
      int blk_sz = nvar * ntot * ntot * ntot;

      int np = static_cast<int>(pairs.size());
      std::vector<std::vector<Real>> sdata(np), rdata(np);
      std::vector<MPI_Request> sreqs(np, MPI_REQUEST_NULL);
      std::vector<MPI_Request> rreqs(np, MPI_REQUEST_NULL);

      for (int p = 0; p < np; ++p) {
        rdata[p].resize(blk_sz);
        int tag = CreateBvals_MPI_Tag(pairs[p].m, pairs[p].n + 64);
        MPI_Irecv(rdata[p].data(), blk_sz, MPI_ATHENA_REAL,
                  pairs[p].remote_rank, tag, comm_vars, &rreqs[p]);
      }
      for (int p = 0; p < np; ++p) {
        sdata[p].resize(blk_sz);
        int bm = pairs[p].m;
        int idx = 0;
        for (int v = 0; v < nvar; ++v)
          for (int k = 0; k < ntot; ++k)
            for (int j = 0; j < ntot; ++j)
              for (int i = 0; i < ntot; ++i)
                sdata[p][idx++] = u(bm, v, k, j, i);
        int dn = nghbr_h.h_view(bm, pairs[p].n).dest;
        int rlid = pairs[p].remote_gid
                   - pmy_pack->pmesh->gids_eachrank[pairs[p].remote_rank];
        int stag = CreateBvals_MPI_Tag(rlid, dn + 64);
        MPI_Isend(sdata[p].data(), blk_sz, MPI_ATHENA_REAL,
                  pairs[p].remote_rank, stag, comm_vars, &sreqs[p]);
      }

      MPI_Waitall(np, rreqs.data(), MPI_STATUSES_IGNORE);

      constexpr Real ot_h = 1.0/3.0;
      for (int p = 0; p < np; ++p) {
        auto &pr = pairs[p];
        int ml = pr.m;
        // Compute child offsets from global LogicalLocation data
        int gid_ml = pmy_pack->pmb->mb_gid.h_view(ml);
        LogicalLocation &loc_ml = pmy_pack->pmesh->lloc_eachmb[gid_ml];
        int root_level = pmy_pack->pmesh->root_level;
        int cx = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx1 & 1) : 0;
        int cy = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx2 & 1) : 0;
        int cz = (loc_ml.level > root_level) ?
                 static_cast<int>(loc_ml.lx3 & 1) : 0;
        int hl = ncells / 2;
        int nl = pr.nlev, mlev = pr.m_lev;
        int nf = pr.nface;
        int ox1 = pr.ox1, ox2 = pr.ox2, ox3 = pr.ox3;
        int f1_ = pr.f1, f2_ = pr.f2;

        auto R = [&](int v, int k, int j, int i) -> Real {
          return rdata[p][((v*ntot + k)*ntot + j)*ntot + i];
        };

        if (nl < mlev && nf == 1) {
          if (ox1 != 0) {
            int fig = (ox1 < 0) ? ngh - 1 : ngh + ncells;
            int fi = (ox1 < 0) ? ngh : ngh + ncells - 1;
            int si = (ox1 < 0) ? ngh + ncells - 1 : ngh;
            int sj0 = ngh + cy*hl;
            int sk0 = ngh + cz*hl;
            for (int v = 0; v < nvar; ++v) {
              for (int sk = sk0; sk < sk0+hl; ++sk) {
                for (int sj = sj0; sj < sj0+hl; ++sj) {
                  int fj = ngh + 2*(sj - sj0);
                  int fk = ngh + 2*(sk - sk0);
                  Real cc = R(v, sk, sj, si);
                  int sjm = (sj > ngh) ? sj-1 : sj;
                  int sjp = (sj < ngh+ncells-1) ? sj+1 : sj;
                  int skm = (sk > ngh) ? sk-1 : sk;
                  int skp = (sk < ngh+ncells-1) ? sk+1 : sk;
                  Real gy = 0.125 *
                      (R(v,sk,sjp,si) - R(v,sk,sjm,si));
                  Real gz = 0.125 *
                      (R(v,skp,sj,si) - R(v,skm,sj,si));
                  u(ml,v,fk,fj,fig) =
                      ot_h*(2.0*(cc-gy-gz) + u(ml,v,fk,fj,fi));
                  u(ml,v,fk,fj+1,fig) =
                      ot_h*(2.0*(cc+gy-gz) + u(ml,v,fk,fj+1,fi));
                  u(ml,v,fk+1,fj,fig) =
                      ot_h*(2.0*(cc-gy+gz) + u(ml,v,fk+1,fj,fi));
                  u(ml,v,fk+1,fj+1,fig) =
                      ot_h*(2.0*(cc+gy+gz) + u(ml,v,fk+1,fj+1,fi));
                }
              }
            }
          } else if (ox2 != 0) {
            int fjg = (ox2 < 0) ? ngh - 1 : ngh + ncells;
            int fj = (ox2 < 0) ? ngh : ngh + ncells - 1;
            int sj = (ox2 < 0) ? ngh + ncells - 1 : ngh;
            int si0 = ngh + cx*hl;
            int sk0 = ngh + cz*hl;
            for (int v = 0; v < nvar; ++v) {
              for (int sk = sk0; sk < sk0+hl; ++sk) {
                for (int si = si0; si < si0+hl; ++si) {
                  int fi = ngh + 2*(si - si0);
                  int fk = ngh + 2*(sk - sk0);
                  Real cc = R(v, sk, sj, si);
                  int sim = (si > ngh) ? si-1 : si;
                  int sip = (si < ngh+ncells-1) ? si+1 : si;
                  int skm = (sk > ngh) ? sk-1 : sk;
                  int skp = (sk < ngh+ncells-1) ? sk+1 : sk;
                  Real gx = 0.125 *
                      (R(v,sk,sj,sip) - R(v,sk,sj,sim));
                  Real gz = 0.125 *
                      (R(v,skp,sj,si) - R(v,skm,sj,si));
                  u(ml,v,fk,fjg,fi) =
                      ot_h*(2.0*(cc-gx-gz) + u(ml,v,fk,fj,fi));
                  u(ml,v,fk,fjg,fi+1) =
                      ot_h*(2.0*(cc+gx-gz) + u(ml,v,fk,fj,fi+1));
                  u(ml,v,fk+1,fjg,fi) =
                      ot_h*(2.0*(cc-gx+gz) + u(ml,v,fk+1,fj,fi));
                  u(ml,v,fk+1,fjg,fi+1) =
                      ot_h*(2.0*(cc+gx+gz) + u(ml,v,fk+1,fj,fi+1));
                }
              }
            }
          } else {
            int fkg = (ox3 < 0) ? ngh - 1 : ngh + ncells;
            int fk = (ox3 < 0) ? ngh : ngh + ncells - 1;
            int sk = (ox3 < 0) ? ngh + ncells - 1 : ngh;
            int si0 = ngh + cx*hl;
            int sj0 = ngh + cy*hl;
            for (int v = 0; v < nvar; ++v) {
              for (int sj = sj0; sj < sj0+hl; ++sj) {
                for (int si = si0; si < si0+hl; ++si) {
                  int fi = ngh + 2*(si - si0);
                  int fj = ngh + 2*(sj - sj0);
                  Real cc = R(v, sk, sj, si);
                  int sim = (si > ngh) ? si-1 : si;
                  int sip = (si < ngh+ncells-1) ? si+1 : si;
                  int sjm = (sj > ngh) ? sj-1 : sj;
                  int sjp = (sj < ngh+ncells-1) ? sj+1 : sj;
                  Real gx = 0.125 *
                      (R(v,sk,sj,sip) - R(v,sk,sj,sim));
                  Real gy = 0.125 *
                      (R(v,sk,sjp,si) - R(v,sk,sjm,si));
                  u(ml,v,fkg,fj,fi) =
                      ot_h*(2.0*(cc-gx-gy) + u(ml,v,fk,fj,fi));
                  u(ml,v,fkg,fj,fi+1) =
                      ot_h*(2.0*(cc+gx-gy) + u(ml,v,fk,fj,fi+1));
                  u(ml,v,fkg,fj+1,fi) =
                      ot_h*(2.0*(cc-gx+gy) + u(ml,v,fk,fj+1,fi));
                  u(ml,v,fkg,fj+1,fi+1) =
                      ot_h*(2.0*(cc+gx+gy) + u(ml,v,fk,fj+1,fi+1));
                }
              }
            }
          }
        } else if (nl > mlev && nf == 1) {
          int sx = 0, sy = 0, sz = 0;
          if (ox1 != 0) {
            sy = f1_; sz = f2_;
          }
          if (ox2 != 0) {
            sx = f1_; sz = f2_;
          }
          if (ox3 != 0) {
            sx = f1_; sy = f2_;
          }
          int gis, gie, gjs, gje, gks, gke;
          if (ox1 < 0) {
            gis = 0; gie = ngh - 1;
          } else if (ox1 > 0) {
            gis = ngh + ncells;
            gie = ngh + ncells + ngh - 1;
          } else {
            gis = ngh + sx*hl;
            gie = ngh + sx*hl + hl - 1;
          }
          if (ox2 < 0) {
            gjs = 0; gje = ngh - 1;
          } else if (ox2 > 0) {
            gjs = ngh + ncells;
            gje = ngh + ncells + ngh - 1;
          } else {
            gjs = ngh + sy*hl;
            gje = ngh + sy*hl + hl - 1;
          }
          if (ox3 < 0) {
            gks = 0; gke = ngh - 1;
          } else if (ox3 > 0) {
            gks = ngh + ncells;
            gke = ngh + ncells + ngh - 1;
          } else {
            gks = ngh + sz*hl;
            gke = ngh + sz*hl + hl - 1;
          }
          int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
          int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
          int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;
          if (ox1 != 0) {
            int fi = (ox1 > 0) ? ngh : ngh + ncells - 1;
            for (int v = 0; v < nvar; ++v) {
              for (int gk = gks; gk <= gke; ++gk) {
                for (int gj = gjs; gj <= gje; ++gj) {
                  int fj0 = ngh + 2*(gj - (ngh + sy*hl));
                  int fk0 = ngh + 2*(gk - (ngh + sz*hl));
                  Real fa = 0.25 *
                      (R(v,fk0,fj0,fi) + R(v,fk0,fj0+1,fi)
                       + R(v,fk0+1,fj0,fi)
                       + R(v,fk0+1,fj0+1,fi));
                  u(ml,v,gk,gj,gis) =
                      ot_h*(4.0*fa
                            - u(ml,v,gk+ok,gj+oj,gis+oi));
                }
              }
            }
          } else if (ox2 != 0) {
            int fj = (ox2 > 0) ? ngh : ngh + ncells - 1;
            for (int v = 0; v < nvar; ++v) {
              for (int gk = gks; gk <= gke; ++gk) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int fi0 = ngh + 2*(gi - (ngh + sx*hl));
                  int fk0 = ngh + 2*(gk - (ngh + sz*hl));
                  Real fa = 0.25 *
                      (R(v,fk0,fj,fi0) + R(v,fk0,fj,fi0+1)
                       + R(v,fk0+1,fj,fi0)
                       + R(v,fk0+1,fj,fi0+1));
                  u(ml,v,gk,gjs,gi) =
                      ot_h*(4.0*fa
                            - u(ml,v,gk+ok,gjs+oj,gi+oi));
                }
              }
            }
          } else {
            int fk = (ox3 > 0) ? ngh : ngh + ncells - 1;
            for (int v = 0; v < nvar; ++v) {
              for (int gj = gjs; gj <= gje; ++gj) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int fi0 = ngh + 2*(gi - (ngh + sx*hl));
                  int fj0 = ngh + 2*(gj - (ngh + sy*hl));
                  Real fa = 0.25 *
                      (R(v,fk,fj0,fi0) + R(v,fk,fj0,fi0+1)
                       + R(v,fk,fj0+1,fi0)
                       + R(v,fk,fj0+1,fi0+1));
                  u(ml,v,gks,gj,gi) =
                      ot_h*(4.0*fa
                            - u(ml,v,gks+ok,gj+oj,gi+oi));
                }
              }
            }
          }
        } else if (nl < mlev) {
          int gis, gie, gjs, gje, gks, gke;
          if (ox1 < 0) {
            gis = 0; gie = ngh - 1;
          } else if (ox1 > 0) {
            gis = ngh + ncells;
            gie = ngh + ncells + ngh - 1;
          } else {
            gis = ngh; gie = ngh + ncells - 1;
          }
          if (ox2 < 0) {
            gjs = 0; gje = ngh - 1;
          } else if (ox2 > 0) {
            gjs = ngh + ncells;
            gje = ngh + ncells + ngh - 1;
          } else {
            gjs = ngh; gje = ngh + ncells - 1;
          }
          if (ox3 < 0) {
            gks = 0; gke = ngh - 1;
          } else if (ox3 > 0) {
            gks = ngh + ncells;
            gke = ngh + ncells + ngh - 1;
          } else {
            gks = ngh; gke = ngh + ncells - 1;
          }
          for (int v = 0; v < nvar; ++v) {
            for (int gk = gks; gk <= gke; ++gk) {
              for (int gj = gjs; gj <= gje; ++gj) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int si, sj, sk;
                  if (ox1 < 0) {
                    si = ngh + ncells - 1;
                  } else if (ox1 > 0) {
                    si = ngh;
                  } else {
                    si = ngh + cx*hl + (gi - ngh)/2;
                  }
                  if (ox2 < 0) {
                    sj = ngh + ncells - 1;
                  } else if (ox2 > 0) {
                    sj = ngh;
                  } else {
                    sj = ngh + cy*hl + (gj - ngh)/2;
                  }
                  if (ox3 < 0) {
                    sk = ngh + ncells - 1;
                  } else if (ox3 > 0) {
                    sk = ngh;
                  } else {
                    sk = ngh + cz*hl + (gk - ngh)/2;
                  }
                  u(ml,v,gk,gj,gi) = R(v,sk,sj,si);
                }
              }
            }
          }
        } else {
          int sx = 0, sy = 0, sz = 0;
          if (nf == 2) {
            if (ox1 == 0) sx = f1_;
            if (ox2 == 0) sy = f1_;
            if (ox3 == 0) sz = f1_;
          }
          int gis, gie, gjs, gje, gks, gke;
          if (ox1 < 0) {
            gis = 0; gie = ngh - 1;
          } else if (ox1 > 0) {
            gis = ngh + ncells;
            gie = ngh + ncells + ngh - 1;
          } else {
            gis = ngh + sx*hl;
            gie = ngh + sx*hl + hl - 1;
          }
          if (ox2 < 0) {
            gjs = 0; gje = ngh - 1;
          } else if (ox2 > 0) {
            gjs = ngh + ncells;
            gje = ngh + ncells + ngh - 1;
          } else {
            gjs = ngh + sy*hl;
            gje = ngh + sy*hl + hl - 1;
          }
          if (ox3 < 0) {
            gks = 0; gke = ngh - 1;
          } else if (ox3 > 0) {
            gks = ngh + ncells;
            gke = ngh + ncells + ngh - 1;
          } else {
            gks = ngh + sz*hl;
            gke = ngh + sz*hl + hl - 1;
          }
          for (int v = 0; v < nvar; ++v) {
            for (int gk = gks; gk <= gke; ++gk) {
              for (int gj = gjs; gj <= gje; ++gj) {
                for (int gi = gis; gi <= gie; ++gi) {
                  int fi0, fi1, fj0, fj1, fk0, fk1;
                  if (ox1 < 0) {
                    fi0 = ngh + ncells - 2;
                    fi1 = ngh + ncells - 1;
                  } else if (ox1 > 0) {
                    fi0 = ngh; fi1 = ngh + 1;
                  } else {
                    fi0 = ngh + 2*(gi - (ngh + sx*hl));
                    fi1 = fi0 + 1;
                  }
                  if (ox2 < 0) {
                    fj0 = ngh + ncells - 2;
                    fj1 = ngh + ncells - 1;
                  } else if (ox2 > 0) {
                    fj0 = ngh; fj1 = ngh + 1;
                  } else {
                    fj0 = ngh + 2*(gj - (ngh + sy*hl));
                    fj1 = fj0 + 1;
                  }
                  if (ox3 < 0) {
                    fk0 = ngh + ncells - 2;
                    fk1 = ngh + ncells - 1;
                  } else if (ox3 > 0) {
                    fk0 = ngh; fk1 = ngh + 1;
                  } else {
                    fk0 = ngh + 2*(gk - (ngh + sz*hl));
                    fk1 = fk0 + 1;
                  }
                  u(ml,v,gk,gj,gi) = 0.125 * (
                      R(v,fk0,fj0,fi0) + R(v,fk0,fj0,fi1)
                      + R(v,fk0,fj1,fi0) + R(v,fk0,fj1,fi1)
                      + R(v,fk1,fj0,fi0) + R(v,fk1,fj0,fi1)
                      + R(v,fk1,fj1,fi0) + R(v,fk1,fj1,fi1));
                }
              }
            }
          }
        }
      }
      MPI_Waitall(np, sreqs.data(), MPI_STATUSES_IGNORE);
    }
  }
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::PackAndSend()
//! \brief Pack restricted fluxes of multigrid variables at fine/coarse
//! boundaries into boundary buffers and send to neighbors.
//! Adapts to different block sizes per level.

TaskStatus MultigridBoundaryValues::PackAndSendMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = u.extent_int(1);

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  int lev_ = pmy_mg->GetCurrentLevel();
  int nlev_total = pmy_mg->GetNumberOfLevels();
  int shift_ps = nlev_total - 1 - lev_;
  int ncells_ps = pmy_mg->GetSize() >> shift_ps;
  bool skip_fc_this_level = (ncells_ps < 2);
  auto &smgi = send_mg_indcs_;
  auto cbuf = coarse_buf_;

#if MPI_PARALLEL_ENABLED
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0
          && nghbr.h_view(m,n).rank != my_rank) {
        int nlev_h = nghbr.h_view(m,n).lev;
        int mlev_h = mblev.h_view(m);
        bool is_fc = (nlev_h != mlev_h);
        if (is_fc && skip_fc_this_level) continue;
        MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
      }
    }
  }
#endif

  {
  int nmnv = nmb * nnghbr * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("PackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank() / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
    const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

    if (nghbr.d_view(m, n).gid < 0) {
      tmember.team_barrier();
      return;
    }

    int nlev = nghbr.d_view(m, n).lev;
    int mlev = mblev.d_view(m);

    bool is_fc = (nlev != mlev);
    if (is_fc && skip_fc_this_level) {
      tmember.team_barrier();
      return;
    }

    int il, iu, jl, ju, kl, ku;
    bool is_coarser = (nlev < mlev);

    if (nlev == mlev) {
      il = smgi[n][lev_].isame.bis; iu = smgi[n][lev_].isame.bie;
      jl = smgi[n][lev_].isame.bjs; ju = smgi[n][lev_].isame.bje;
      kl = smgi[n][lev_].isame.bks; ku = smgi[n][lev_].isame.bke;
    } else if (is_coarser) {
      il = smgi[n][lev_].icoar.bis; iu = smgi[n][lev_].icoar.bie;
      jl = smgi[n][lev_].icoar.bjs; ju = smgi[n][lev_].icoar.bje;
      kl = smgi[n][lev_].icoar.bks; ku = smgi[n][lev_].icoar.bke;
    } else {
      il = smgi[n][lev_].ifine.bis; iu = smgi[n][lev_].ifine.bie;
      jl = smgi[n][lev_].ifine.bjs; ju = smgi[n][lev_].ifine.bje;
      kl = smgi[n][lev_].ifine.bks; ku = smgi[n][lev_].ifine.bke;
    }

    int ni = iu - il + 1;
    int nj = ju - jl + 1;
    int nk = ku - kl + 1;
    int nkj = nk * nj;

    int dm = nghbr.d_view(m, n).gid - mbgid.d_view(0);
    int dn = nghbr.d_view(m, n).dest;

    if (is_coarser) {
      // Restricted data is pre-computed in coarse_buf_ by FillCoarseMG:
      //   face neighbors  -> face-aligned 2x2 avg in ghost cells
      //   edge/corner     -> volume 2x2x2 avg in interior cells
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
        if (nghbr.d_view(m, n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = cbuf(m, v, k, j, i);
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = cbuf(m, v, k, j, i);
          });
        }
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        if (nghbr.d_view(m, n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = u(m, v, k, j, i);
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = u(m, v, k, j, i);
          });
        }
      });
    }
    tmember.team_barrier();
  });
  }

  #if MPI_PARALLEL_ENABLED
  bool has_cross_rank = false;
  for (int m=0; m<nmb && !has_cross_rank; ++m) {
    for (int n=0; n<nnghbr && !has_cross_rank; ++n) {
      if (nghbr.h_view(m,n).gid >= 0 && nghbr.h_view(m,n).rank != my_rank) {
        int nlev_h = nghbr.h_view(m,n).lev;
        int mlev_h = mblev.h_view(m);
        bool is_fc_h = (nlev_h != mlev_h);
        if (is_fc_h && skip_fc_this_level) continue;
        has_cross_rank = true;
      }
    }
  }
  if (has_cross_rank) Kokkos::fence();

  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid < 0) continue;
      int nlev = nghbr.h_view(m,n).lev;
      int mlev = pmy_pack->pmb->mb_lev.h_view(m);
      bool is_fc_mpi = (nlev != mlev);
      if (is_fc_mpi && skip_fc_this_level) continue;
      {
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          int data_size;
          if (nlev < mlev) {
            data_size = nvar * send_mg_indcs_[n][lev_].icoar_ndat;
          } else if (nlev == mlev) {
            data_size = nvar * send_mg_indcs_[n][lev_].isame_ndat;
          } else {
            data_size = nvar * send_mg_indcs_[n][lev_].ifine_ndat;
          }

          MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);

          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);
          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {
            no_errors = false;
          }
          pmy_mg->pmy_driver_->mg_timers_.msg_count++;
          pmy_mg->pmy_driver_->mg_timers_.bytes_sent +=
              data_size * static_cast<int64_t>(sizeof(Real));
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValuesCC::RecvAndUnpackMG()
//! \brief Receive and unpack cell-centered multigrid variables.
//! Handles ghost-cell filling at each multigrid level independently.

TaskStatus MultigridBoundaryValues::RecvAndUnpackMG(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  int shift_ru = pmy_mg->GetNumberOfLevels() - 1 - pmy_mg->GetCurrentLevel();
  int ncells_ru = pmy_mg->GetSize() >> shift_ru;
  bool skip_fc_this_level = (ncells_ru < 2);

  #if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  bool bflag = false;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0
          && nghbr.h_view(m,n).rank != global_variable::my_rank) {
        int nlev_h = nghbr.h_view(m,n).lev;
        int mlev_h = mblev.h_view(m);
        bool is_fc_h = (nlev_h != mlev_h);
        if (is_fc_h && skip_fc_this_level) continue;
        {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "MPI error in testing non-blocking receives"
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }
          if (!static_cast<bool>(test)) {
            bflag = true;
          }
        }
      }
    }
  }
  if (bflag) {
    return TaskStatus::incomplete;
  }
#endif

  //----- STEP 2: buffers have all completed, so unpack
  int nvar = u.extent_int(1);
  int ngh = pmy_mg->GetGhostCells();
  int lev_ = pmy_mg->GetCurrentLevel();
  auto cbuf = coarse_buf_;
  auto &rmgi = recv_mg_indcs_;

  {
  int nmnv = nmb * nnghbr * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("UnpackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank() / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
    const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

    if (nghbr.d_view(m, n).gid < 0) {
      tmember.team_barrier();
      return;
    }

    int nlev = nghbr.d_view(m, n).lev;
    int mlev = mblev.d_view(m);

    bool is_fc = (nlev != mlev);
    if (is_fc && skip_fc_this_level) {
      tmember.team_barrier();
      return;
    }

    bool from_coarser = (nlev < mlev);

    int il, iu, jl, ju, kl, ku;

    if (nlev == mlev) {
      il = rmgi[n][lev_].isame.bis; iu = rmgi[n][lev_].isame.bie;
      jl = rmgi[n][lev_].isame.bjs; ju = rmgi[n][lev_].isame.bje;
      kl = rmgi[n][lev_].isame.bks; ku = rmgi[n][lev_].isame.bke;
    } else if (from_coarser) {
      il = rmgi[n][lev_].icoar.bis; iu = rmgi[n][lev_].icoar.bie;
      jl = rmgi[n][lev_].icoar.bjs; ju = rmgi[n][lev_].icoar.bje;
      kl = rmgi[n][lev_].icoar.bks; ku = rmgi[n][lev_].icoar.bke;
    } else {
      il = rmgi[n][lev_].ifine.bis; iu = rmgi[n][lev_].ifine.bie;
      jl = rmgi[n][lev_].ifine.bjs; ju = rmgi[n][lev_].ifine.bje;
      kl = rmgi[n][lev_].ifine.bks; ku = rmgi[n][lev_].ifine.bke;
    }

    int ni = iu - il + 1;
    int nj = ju - jl + 1;
    int nk = ku - kl + 1;
    int nkj = nk * nj;

    if (from_coarser) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        [&](const int i) {
          cbuf(m, v, k, j, i) = rbuf[n].vars(m,
              (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
        });
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        [&](const int i) {
          u(m, v, k, j, i) = rbuf[n].vars(m,
              (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
        });
      });
    }
    tmember.team_barrier();
  });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus MultigridBoundaryValues::InitRecvMG(const int nvars) {
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int lev_ = pmy_mg->GetCurrentLevel();
  int shift_ir = pmy_mg->GetNumberOfLevels() - 1 - lev_;
  int ncells_ir = pmy_mg->GetSize() >> shift_ir;
  bool skip_fc_ir = (ncells_ir < 2);

  // Initialize communications of variables
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        int nlev = nghbr.h_view(m,n).lev;
        int mlev = mblev.h_view(m);
        bool is_fc_ir = (nlev != mlev);
        if (is_fc_ir && skip_fc_ir) continue;
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          int data_size;
          if (nlev < mlev) {
            data_size = nvars * recv_mg_indcs_[n][lev_].icoar_ndat;
          } else if (nlev == mlev) {
            data_size = nvars * recv_mg_indcs_[n][lev_].isame_ndat;
          } else {
            data_size = nvars * recv_mg_indcs_[n][lev_].ifine_ndat;
          }

          auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, Kokkos::ALL);

          MPI_Wait(&(recvbuf[n].vars_req[m]), MPI_STATUS_IGNORE);

          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(recvbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {
            no_errors = false;
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__
       << " at line " << __LINE__ << std::endl
       << "MPI error in posting non-blocking receives"
       << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}
