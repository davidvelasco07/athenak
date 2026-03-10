#ifndef DIFFUSION_CURRENT_DENSITY_HPP_
#define DIFFUSION_CURRENT_DENSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file current_density.hpp
//  \brief Inlined function to compute current density in 1-D pencils in i-direction

#include "athena.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \fn CurrentDensity()
//  \brief Calculates the three components of the current density at cell edges
//  Each component of J is centered identically to the edge-electric-field
//               _____________
//               |\           \
//               | \           \
//               |  \___________\
//               |   |           |
//               \   |           |
//              J2*  *J3         |
//   x2 x3         \ |           |
//    \ |           \|_____*_____|
//     \|__x1             J1

KOKKOS_INLINE_FUNCTION
void CurrentDensity(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceFaceFld4D<Real> &b, const RegionSize &size,
     ScrArray1D<Real> &j1, ScrArray1D<Real> &j2, ScrArray1D<Real> &j3) {
  par_for_inner(member, il, iu, [&](const int i) {
    j1(i) = 0.0;
    j2(i) = -(b.x3f(m,k,j,i) - b.x3f(m,k,j,i-1))/size.dx1;
    j3(i) =  (b.x2f(m,k,j,i) - b.x2f(m,k,j,i-1))/size.dx1;
  });
  member.team_barrier();

  if (b.x1f.extent_int(2) > 1) {  // proxy for nx2gt1: 2D problems
    par_for_inner(member, il, iu, [&](const int i) {
      j1(i) += (b.x3f(m,k,j,i) - b.x3f(m,k,j-1,i))/size.dx2;
      j3(i) -= (b.x1f(m,k,j,i) - b.x1f(m,k,j-1,i))/size.dx2;
    });
    member.team_barrier();
  }

  if (b.x1f.extent_int(1) > 1) {  // proxy for nx3gt1: 3D problems
    par_for_inner(member, il, iu, [&](const int i) {
      j1(i) -= (b.x2f(m,k,j,i) - b.x2f(m,k-1,j,i))/size.dx3;
      j2(i) += (b.x1f(m,k,j,i) - b.x1f(m,k-1,j,i))/size.dx3;
    });
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn CurrentDensityFourthOrder()
//  \brief Calculates the three components of current density at cell edges using
//  4th-order stencils.  Replaces each 2-point face difference with the 4-point formula
//    (15(f_+ - f_-) - (f_++ - f_--)) / (12*dx)
//  The face-centered arrays b.x1f, b.x2f, b.x3f are ALREADY centered at the correct
//  half-integer positions for each component of J, so no additional averaging is needed.

KOKKOS_INLINE_FUNCTION
void CurrentDensityFourthOrder(TeamMember_t const &member, const int m, const int k,
     const int j, const int il, const int iu, const DvceFaceFld4D<Real> &b,
     const RegionSize &size, ScrArray1D<Real> &j1, ScrArray1D<Real> &j2,
     ScrArray1D<Real> &j3) {
  // J3 at edge (i-1/2, j-1/2, k) = dB2/dx - dB1/dy
  //   b.x2f(m,k,j,i) at (i, j-1/2, k): 4-pt x-diff gives dB2/dx at (i-1/2, j-1/2, k)
  //   b.x1f(m,k,j,i) at (i-1/2, j, k): 4-pt y-diff gives dB1/dy at (i-1/2, j-1/2, k)
  // J2 at edge (i-1/2, k-1/2, j) = -dB3/dx  [+dB1/dz added in 3D below]
  //   b.x3f(m,k,j,i) at (i, j, k-1/2): 4-pt x-diff gives dB3/dx at (i-1/2, j, k-1/2)
  par_for_inner(member, il, iu, [&](const int i) {
    j1(i) = 0.0;
    j2(i) = -(15.0*(b.x3f(m,k,j,i  ) - b.x3f(m,k,j,i-1))
                -  (b.x3f(m,k,j,i+1) - b.x3f(m,k,j,i-2))) / (12.0*size.dx1);
    j3(i) =  (15.0*(b.x2f(m,k,j,i  ) - b.x2f(m,k,j,i-1))
                -  (b.x2f(m,k,j,i+1) - b.x2f(m,k,j,i-2))) / (12.0*size.dx1);
  });
  member.team_barrier();

  if (b.x1f.extent_int(2) > 1) {  // proxy for nx2gt1: 2D problems
    // J1 at edge (i, j-1/2, k-1/2) += dB3/dy  [b.x3f at (i, j, k-1/2)]
    // J3 at edge (i-1/2, j-1/2, k) -= dB1/dy  [b.x1f at (i-1/2, j, k)]
    par_for_inner(member, il, iu, [&](const int i) {
      j1(i) +=  (15.0*(b.x3f(m,k,j,i) - b.x3f(m,k,j-1,i))
                    -  (b.x3f(m,k,j+1,i) - b.x3f(m,k,j-2,i))) / (12.0*size.dx2);
      j3(i) -= (15.0*(b.x1f(m,k,j,i) - b.x1f(m,k,j-1,i))
                   -  (b.x1f(m,k,j+1,i) - b.x1f(m,k,j-2,i))) / (12.0*size.dx2);
    });
    member.team_barrier();
  }

  if (b.x1f.extent_int(1) > 1) {  // proxy for nx3gt1: 3D problems
    // J1 at edge (i, j-1/2, k-1/2) -= dB2/dz  [b.x2f at (i, j-1/2, k)]
    // J2 at edge (i-1/2, k-1/2, j) += dB1/dz  [b.x1f at (i-1/2, j, k)]
    par_for_inner(member, il, iu, [&](const int i) {
      j1(i) -= (15.0*(b.x2f(m,k,j,i) - b.x2f(m,k-1,j,i))
                   -  (b.x2f(m,k+1,j,i) - b.x2f(m,k-2,j,i))) / (12.0*size.dx3);
      j2(i) += (15.0*(b.x1f(m,k,j,i) - b.x1f(m,k-1,j,i))
                   -  (b.x1f(m,k+1,j,i) - b.x1f(m,k-2,j,i))) / (12.0*size.dx3);
    });
  }
  return;
}

#endif // DIFFUSION_CURRENT_DENSITY_HPP_
