#ifndef PARTICLES_SINK_CONVERGENCE_HPP_
#define PARTICLES_SINK_CONVERGENCE_HPP_
#include "athena.hpp"
namespace particles {
// Strict inflow on each pair of CV faces. Uniform face areas cancel within an axis.
// Use conserved variables so the check sees the same state as the proposed reset.
template<class View>
KOKKOS_INLINE_FUNCTION bool SinkConverging(const View &u, int m, int i, int j, int k,
                                          Real vx, Real vy, Real vz, int mode) {
  if (mode == 0) return true;
  const Real vref[3] = {vx, vy, vz};
  for (int axis=0; axis<3; ++axis) {
    Real inflow = 0.0;
    for (int a=-1; a<=1; ++a) for (int b=-1; b<=1; ++b) {
      int l[3]={i,j,k}, r[3]={i,j,k};
      l[axis]--; r[axis]++;
      l[(axis+1)%3]+=a; r[(axis+1)%3]+=a;
      l[(axis+2)%3]+=b; r[(axis+2)%3]+=b;
      const Real dl=u(m,IDN,l[2],l[1],l[0]), dr=u(m,IDN,r[2],r[1],r[0]);
      if (!(dl > 0.0 && dr > 0.0)) return false;
      const Real ml=u(m,IM1+axis,l[2],l[1],l[0]);
      const Real mr=u(m,IM1+axis,r[2],r[1],r[0]);
      inflow += mode == 1 ? (ml-dl*vref[axis])-(mr-dr*vref[axis]) : ml/dl-mr/dr;
    }
    if (!(inflow > 0.0)) return false;
  }
  return true;
}
}  // namespace particles
#endif
