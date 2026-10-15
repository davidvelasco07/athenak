#ifndef EOS_MHD_BCC_HPP_
#define EOS_MHD_BCC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Production and candidate PAD must use the same face-to-cell magnetic field.
#include "athena.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "reconstruct/teno.hpp"
#include "reconstruct/fv.hpp"

namespace mhd_bcc {

inline constexpr int Reach(ReconstructionMethod recon) {
  switch (recon) {
    case ReconstructionMethod::dc: return 1;
    case ReconstructionMethod::plm: return 2;
    case ReconstructionMethod::fv4: return 2;
    case ReconstructionMethod::fv6: return 3;
    case ReconstructionMethod::fv8: return 4;
    default: return 3;
  }
}

KOKKOS_INLINE_FUNCTION
Real FaceToCenterInterp(const ReconstructionMethod recon,
                        const Real &bf_im3, const Real &bf_im2, const Real &bf_im1,
                        const Real &bf_i, const Real &bf_ip1, const Real &bf_ip2,
                        const Real &bf_ip3, const Real &bf_ip4) {
  // bf values are face-centered; cell center i is the midpoint between faces i and i+1.
  // Limited methods reconstruct L/R dual-grid states and average them; unlimited fvN
  // methods use the matching central interpolant of the dual-grid face averages.
  Real qL, qR, dum;
  switch (recon) {
    case ReconstructionMethod::dc:
      return 0.5*(bf_i + bf_ip1);
    case ReconstructionMethod::plm:
      PLM(bf_im1, bf_i, bf_ip1, qL, dum);    // center face i → qL at cell center
      PLM(bf_i, bf_ip1, bf_ip2, dum, qR);    // center face i+1 → qR at cell center
      return 0.5*(qL + qR);
    case ReconstructionMethod::ppm4:
    case ReconstructionMethod::ppmx:
      PPM4(bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, qL, dum);  // center face i → qL at cc
      PPM4(bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3, dum, qR);  // center face i+1 → qR at cc
      return 0.5*(qL + qR);
    case ReconstructionMethod::wenoz:
      WENOZ(bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, qL, dum);  // center face i → qL at cc
      WENOZ(bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3, dum, qR);  // center face i+1 → qR at cc
      return 0.5*(qL + qR);
    case ReconstructionMethod::teno:
      TENO(bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, qL, dum);   // center face i → qL at cc
      TENO(bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3, dum, qR);   // center face i+1 → qR at cc
      return 0.5*(qL + qR);
    case ReconstructionMethod::fv4:
      return FV4Face(bf_im1, bf_i, bf_ip1, bf_ip2);
    case ReconstructionMethod::fv6:
      return FV6Face(bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3);
    case ReconstructionMethod::fv8:
      return FV8Face(bf_im3, bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3, bf_ip4);
  }
  return 0.5*(bf_i + bf_ip1);  // fallback
}

// Match the production interpolation's existing boundary reductions. The bounds
// describe the production C2P range, not the smaller candidate detection halo.
KOKKOS_INLINE_FUNCTION
Real CellCenteredComponent(const ReconstructionMethod recon,
    const DvceArray4D<Real> &b, const int m, const int k, const int j, const int i,
    const int dk, const int dj, const int di, const int lo, const int hi) {
  const int c = dk*k + dj*j + di*i;
  if (c > lo+2 && c < hi-2) {
    return FaceToCenterInterp(recon,
        b(m,k-3*dk,j-3*dj,i-3*di), b(m,k-2*dk,j-2*dj,i-2*di),
        b(m,k-dk,j-dj,i-di), b(m,k,j,i), b(m,k+dk,j+dj,i+di),
        b(m,k+2*dk,j+2*dj,i+2*di), b(m,k+3*dk,j+3*dj,i+3*di),
        b(m,k+4*dk,j+4*dj,i+4*di));
  }
  if (c > lo+1 && c < hi-1) {
    const auto r6 = (recon == ReconstructionMethod::fv8)
        ? ReconstructionMethod::fv6 : recon;
    return FaceToCenterInterp(r6,
        b(m,k-2*dk,j-2*dj,i-2*di), b(m,k-2*dk,j-2*dj,i-2*di),
        b(m,k-dk,j-dj,i-di), b(m,k,j,i), b(m,k+dk,j+dj,i+di),
        b(m,k+2*dk,j+2*dj,i+2*di), b(m,k+3*dk,j+3*dj,i+3*di),
        b(m,k+3*dk,j+3*dj,i+3*di));
  }
  if (c > lo && c < hi) {
    return FV4Face(b(m,k-dk,j-dj,i-di), b(m,k,j,i),
        b(m,k+dk,j+dj,i+di), b(m,k+2*dk,j+2*dj,i+2*di));
  }
  return 0.5*(b(m,k,j,i) + b(m,k+dk,j+dj,i+di));
}

}  // namespace mhd_bcc
#endif  // EOS_MHD_BCC_HPP_
