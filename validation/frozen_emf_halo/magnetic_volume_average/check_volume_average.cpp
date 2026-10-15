// Validate actual device operators against analytic volume integrals, including
// direction mapping, reduced stencils, smooth convergence and transverse averages.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "eos/mhd_bcc.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
Real Power(Real x, int p) {
  Real y = 1.0;
  for (int n=0; n<p; ++n) y *= x;
  return y;
}

KOKKOS_INLINE_FUNCTION
Real IntegralAverage(Real x, Real h, int p) {
  return (Power(x+0.5*h,p+1)-Power(x-0.5*h,p+1))/((p+1)*h);
}

ReconstructionMethod Method(int order) {
  return order == 4 ? ReconstructionMethod::fv4 :
         order == 6 ? ReconstructionMethod::fv6 : ReconstructionMethod::fv8;
}

void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void PolynomialChecks(int &count, Real &max_error) {
  for (int order : {4,6,8}) for (int axis=0; axis<3; ++axis) {
    const int di = axis == 0, dj = axis == 1, dk = axis == 2;
    const auto recon = Method(order);
    for (int cells : {1,2,3,4,6,8,16}) {
      auto b = DvceArray4D<Real>("faces",1,dk ? cells+1 : 1,
                                dj ? cells+1 : 1,di ? cells+1 : 1);
      auto out = DvceArray1D<Real>("averages",cells);
      const Real h = 1.0/cells;
      for (int p=0; p<order; ++p) {
        Kokkos::parallel_for("polynomial_faces",Kokkos::RangePolicy<>(0,cells+1),
        KOKKOS_LAMBDA(int c) {
          b(0,dk*c,dj*c,di*c) = Power((c-0.5*cells)*h,p);
        });
        Kokkos::parallel_for("polynomial_volumes",Kokkos::RangePolicy<>(0,cells),
        KOKKOS_LAMBDA(int c) {
          out(c) = mhd_bcc::CellCenteredComponent(
              recon,b,0,dk*c,dj*c,di*c,dk,dj,di,0,cells-1);
        });
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
        for (int c=0; c<cells; ++c) {
          int radius = std::min(order/2,std::min(c+1,cells-c));
          if (p >= 2*radius) continue;
          const Real exact = IntegralAverage((c+0.5-0.5*cells)*h,h,p);
          const Real err = std::abs(host(c)-exact);
          max_error = std::max(max_error,err);
          Require(std::isfinite(host(c)) && err < 5.e-13,"polynomial integral failed");
          ++count;
        }
      }
    }
  }
}

void TensorChecks(int &count, Real &max_error) {
  constexpr int n = 12, c = 6;
  const Real h = 0.125;
  for (int order : {4,6,8}) for (int axis=0; axis<3; ++axis) {
    const auto recon = Method(order);
    const int di = axis == 0, dj = axis == 1, dk = axis == 2;
    auto b = DvceArray4D<Real>("area_averages",1,n+1,n+1,n+1);
    auto out = DvceArray1D<Real>("volume_average",1);
    for (int p=0; p<order; ++p) {
      Kokkos::parallel_for("tensor_faces",Kokkos::RangePolicy<>(0,(n+1)*(n+1)*(n+1)),
      KOKKOS_LAMBDA(int idx) {
        const int i = idx%(n+1), j = (idx/(n+1))%(n+1), k = idx/((n+1)*(n+1));
        const int pos[3] = {i,j,k};
        Real value = 1.0;
        for (int a=0; a<3; ++a) {
          const Real x = (pos[a]-c)*h;
          if (a == axis) value *= Power(x-0.5*h,p);
          else value *= 1.0+0.3*IntegralAverage(x,h,2)+0.2*IntegralAverage(x,h,3);
        }
        b(0,k,j,i) = value;
      });
      Kokkos::parallel_for("tensor_volume",Kokkos::RangePolicy<>(0,1),
      KOKKOS_LAMBDA(int idx) {
        out(0) = mhd_bcc::CellCenteredComponent(recon,b,0,c,c,c,dk,dj,di,0,n-1);
      });
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
      const Real transverse = 1.0+0.3*IntegralAverage(0.0,h,2);
      const Real exact = IntegralAverage(0.0,h,p)*transverse*transverse;
      const Real err = std::abs(host(0)-exact);
      max_error = std::max(max_error,err);
      Require(std::isfinite(host(0)) && err < 5.e-13,"face-area to volume integral failed");
      ++count;
    }
  }
}

void ConvergenceChecks() {
  constexpr int n = 16, c = 8;
  const Real pi = std::acos(-1.0), x0 = 0.137;
  for (int order : {4,6,8}) {
    const auto recon = Method(order);
    auto b = DvceArray4D<Real>("sine_faces",1,1,1,n+1);
    auto out = DvceArray1D<Real>("sine_average",1);
    Real previous = 0.0, rate = 0.0;
    for (Real h : {0.2,0.1,0.05}) {
      Kokkos::parallel_for("sine_faces",Kokkos::RangePolicy<>(0,n+1),
      KOKKOS_LAMBDA(int i) { b(0,0,0,i) = sin(2*pi*(x0+(i-c-0.5)*h)); });
      Kokkos::parallel_for("sine_average",Kokkos::RangePolicy<>(0,1),
      KOKKOS_LAMBDA(int idx) {
        out(0) = mhd_bcc::CellCenteredComponent(recon,b,0,0,0,c,0,0,1,0,n-1);
      });
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
      const Real exact = sin(2*pi*x0)*sin(pi*h)/(pi*h);
      const Real error = std::abs(host(0)-exact);
      if (previous > 0.0) rate = std::log2(previous/error);
      previous = error;
    }
    Require(rate > order-0.2,"smooth integration convergence order failed");
    std::cout << "FV" << order << " sine rate=" << rate
              << " final_error=" << previous << '\n';
  }
}
}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc,argv);
  int status = 0;
  try {
    int count = 0;
    Real max_error = 0.0;
    PolynomialChecks(count,max_error);
    TensorChecks(count,max_error);
    std::cout << std::setprecision(17);
    ConvergenceChecks();
    std::cout << "PASS integral checks=" << count << " maximum_error=" << max_error << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  Kokkos::finalize();
  return status;
}
