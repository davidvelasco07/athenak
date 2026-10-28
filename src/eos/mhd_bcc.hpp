#ifndef EOS_MHD_BCC_HPP_
#define EOS_MHD_BCC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Production conversion and MOOD use the same magnetic volume averages for fvN.
#include "athena.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "reconstruct/teno.hpp"
#include "reconstruct/fv.hpp"

namespace mhd_bcc {

KOKKOS_INLINE_FUNCTION
bool IsFV(ReconstructionMethod recon) {
  return IsUnlimitedFV(recon);
}

// fv5/fv7/fv9 integrate the same even-order polynomial as fv4/fv6/fv8 (same radius).
inline constexpr int Reach(ReconstructionMethod recon) {
  switch (recon) {
    case ReconstructionMethod::dc: return 1;
    case ReconstructionMethod::plm: return 2;
    case ReconstructionMethod::fv4: return 2;
    case ReconstructionMethod::fv5: return 2;
    case ReconstructionMethod::fv6: return 3;
    case ReconstructionMethod::fv7: return 3;
    case ReconstructionMethod::fv8: return 4;
    case ReconstructionMethod::fv9: return 4;
    default: return 3;
  }
}

// Integrate the degree N-1 polynomial through N face-area averages along the
// component's normal direction over the central cell. The tangential directions
// are already averaged, so this integral is a volume average. These are not the
// cell-average-to-interface weights in reconstruct/fv.hpp.
KOKKOS_INLINE_FUNCTION
Real FV4VolumeAverage(Real bm1, Real b0, Real bp1, Real bp2) {
  return (13.0*(b0 + bp1) - (bm1 + bp2))/24.0;
}

KOKKOS_INLINE_FUNCTION
Real FV6VolumeAverage(Real bm2, Real bm1, Real b0, Real bp1, Real bp2, Real bp3) {
  return (802.0*(b0 + bp1) - 93.0*(bm1 + bp2) + 11.0*(bm2 + bp3))/1440.0;
}

KOKKOS_INLINE_FUNCTION
Real FV8VolumeAverage(Real bm3, Real bm2, Real bm1, Real b0, Real bp1, Real bp2,
                     Real bp3, Real bp4) {
  return (68323.0*(b0 + bp1) - 9531.0*(bm1 + bp2) + 1879.0*(bm2 + bp3)
          - 191.0*(bm3 + bp4))/120960.0;
}

KOKKOS_INLINE_FUNCTION
Real FaceToCenterInterp(const ReconstructionMethod recon,
                        const Real &bf_im3, const Real &bf_im2, const Real &bf_im1,
                        const Real &bf_i, const Real &bf_ip1, const Real &bf_ip2,
                        const Real &bf_ip3, const Real &bf_ip4) {
  // bf values are face-centered; cell center i is the midpoint between faces i and i+1.
  // Preserve the existing limited-method interpolation. fvN uses the volume
  // integrals below instead of treating face-area averages as dual-cell averages.
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
    case ReconstructionMethod::fv5:
      return FV4VolumeAverage(bf_im1, bf_i, bf_ip1, bf_ip2);
    case ReconstructionMethod::fv6:
    case ReconstructionMethod::fv7:
      return FV6VolumeAverage(bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2, bf_ip3);
    case ReconstructionMethod::fv8:
    case ReconstructionMethod::fv9:
      return FV8VolumeAverage(bf_im3, bf_im2, bf_im1, bf_i, bf_ip1, bf_ip2,
                              bf_ip3, bf_ip4);
  }
  return 0.5*(bf_i + bf_ip1);  // fallback
}

// Bounds describe the full cell array used by production C2P, with faces lo..hi+1
// in the component's normal direction. Reduce the order near array boundaries.
// Load only the stencil actually used: candidate CT need not fill a wider one.
KOKKOS_INLINE_FUNCTION
Real CellCenteredComponent(const ReconstructionMethod recon,
    const DvceArray4D<Real> &b, const int m, const int k, const int j, const int i,
    const int dk, const int dj, const int di, const int lo, const int hi) {
  const int c = dk*k + dj*j + di*i;
  if (IsFV(recon)) {
    if ((recon == ReconstructionMethod::fv8 || recon == ReconstructionMethod::fv9) &&
        c > lo+2 && c < hi-2) {
      return FV8VolumeAverage(
          b(m,k-3*dk,j-3*dj,i-3*di), b(m,k-2*dk,j-2*dj,i-2*di),
          b(m,k-dk,j-dj,i-di), b(m,k,j,i), b(m,k+dk,j+dj,i+di),
          b(m,k+2*dk,j+2*dj,i+2*di), b(m,k+3*dk,j+3*dj,i+3*di),
          b(m,k+4*dk,j+4*dj,i+4*di));
    }
    if (recon != ReconstructionMethod::fv4 && recon != ReconstructionMethod::fv5 &&
        c > lo+1 && c < hi-1) {
      return FV6VolumeAverage(
          b(m,k-2*dk,j-2*dj,i-2*di), b(m,k-dk,j-dj,i-di), b(m,k,j,i),
          b(m,k+dk,j+dj,i+di), b(m,k+2*dk,j+2*dj,i+2*di),
          b(m,k+3*dk,j+3*dj,i+3*di));
    }
    if (c > lo && c < hi) {
      return FV4VolumeAverage(b(m,k-dk,j-dj,i-di), b(m,k,j,i),
          b(m,k+dk,j+dj,i+di), b(m,k+2*dk,j+2*dj,i+2*di));
    }
    return 0.5*(b(m,k,j,i) + b(m,k+dk,j+dj,i+di));
  }
  // Keep the pre-existing non-FV conversion behavior.
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
