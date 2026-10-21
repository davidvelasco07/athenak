// Compact provenance labels advected by the actual hydro mass flux and RK integrator.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "tracers/passive_tracers.hpp"
#include "pgen/pgen.hpp"

namespace {
std::vector<double> initial_mass;
KOKKOS_INLINE_FUNCTION
Real Fraction(int f,Real x,Real y,Real z,Real radius,int ndim) {
  const Real cx = Real(0.13)+Real((37*f)%97)/Real(97)*Real(0.74);
  const Real cy = Real(0.13)+Real((53*f)%89)/Real(89)*Real(0.74);
  const Real cz = Real(0.13)+Real((29*f)%83)/Real(83)*Real(0.74);
  Real dx = fabs(x-cx), dy = fabs(y-cy), dz = fabs(z-cz);
  dx = fmin(dx,Real(1)-dx); dy = fmin(dy,Real(1)-dy); dz = fmin(dz,Real(1)-dz);
  return dx*dx+(ndim > 1 ? dy*dy : 0)+(ndim > 2 ? dz*dz : 0) < radius*radius ?
         Real(1)+Real(f%7)/Real(8) : Real(0);
}
std::vector<double> Mass(Mesh *mesh) {
  const auto view = mesh->pmb_pack->phydro->ptracers->GetView();
  const auto ind = mesh->mb_indcs;
  const int nc = ind.nx1*ind.nx2*ind.nx3, nb = mesh->pmb_pack->nmb_thispack;
  const auto size = mesh->pmb_pack->pmb->mb_size.d_view;
  const bool multi = mesh->multi_d, three = mesh->three_d;
  std::vector<double> mass(mesh->pmb_pack->phydro->ptracers->Count());
  for (int f = 0; f < static_cast<int>(mass.size()); ++f) {
    double sum = 0;
    Kokkos::parallel_reduce("tracer_mass",Kokkos::RangePolicy<DevExeSpace>(0,nb*nc),
        KOKKOS_LAMBDA(int c,double &total) {
      const int b = c/nc, cell = c%nc;
      const int k = cell/(ind.nx1*ind.nx2)+ind.ks;
      const int j = (cell/ind.nx1)%ind.nx2+ind.js, i = cell%ind.nx1+ind.is;
      const Real volume = size(b).dx1*(multi ? size(b).dx2 : 1)*
                          (three ? size(b).dx3 : 1);
      total += view.Read(0,b,f,k,j,i)*volume;
    },sum);
    mass[f] = sum;
  }
  return mass;
}
void Errors(ParameterInput *pin,Mesh *mesh) {
  auto *hydro = mesh->pmb_pack->phydro;
  auto &tracer = *hydro->ptracers;
  tracer.Finish();
  const auto view = tracer.GetView();
  const auto ind = mesh->mb_indcs;
  const int nc = ind.nx1*ind.nx2*ind.nx3, nb = mesh->pmb_pack->nmb_thispack;
  const int nf = tracer.Count(), nh = hydro->nhydro;
  const bool compare = tracer.CompareDense();
  const auto native = hydro->u0;
  Real error = 0, minimum = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("tracer_compare",Kokkos::RangePolicy<DevExeSpace>(0,nb*nf*nc),
      KOKKOS_LAMBDA(int c,Real &maximum) {
    const int f = (c/nc)%nf, b = c/(nc*nf), cell = c%nc;
    const int k = cell/(ind.nx1*ind.nx2)+ind.ks;
    const int j = (cell/ind.nx1)%ind.nx2+ind.js, i = cell%ind.nx1+ind.is;
    const Real q = view.Read(0,b,f,k,j,i);
    const Real difference = compare ? fabs(q-native(b,nh+f,k,j,i)) : Real(0);
    if (!Kokkos::isfinite(q)) maximum = std::numeric_limits<Real>::max();
    else if (difference > maximum) maximum = difference;
  },Kokkos::Max<Real>(error));
  Kokkos::parallel_reduce("tracer_min",Kokkos::RangePolicy<DevExeSpace>(0,nb*nf*nc),
      KOKKOS_LAMBDA(int c,Real &value) {
    const int f = (c/nc)%nf, b = c/(nc*nf), cell = c%nc;
    const int k = cell/(ind.nx1*ind.nx2)+ind.ks;
    const int j = (cell/ind.nx1)%ind.nx2+ind.js, i = cell%ind.nx1+ind.is;
    const Real q = view.Read(0,b,f,k,j,i);
    if (q < value) value = q;
  },Kokkos::Min<Real>(minimum));
  const auto mass = Mass(mesh);
  double mass_error = 0;
  for (int f = 0; f < nf; ++f) {
    if (initial_mass[f] <= 0) throw std::runtime_error("Tracer test label unresolved");
    mass_error = std::max(mass_error,fabs(mass[f]/initial_mass[f]-1));
  }
  const auto memory = tracer.Memory();
  const double tolerance = sizeof(Real) == 4 ? 3e-5 : 3e-11;
  std::ofstream out("tracers.json");
  out << std::setprecision(17) << "{\"mode\":\"" << tracer.Mode()
      << "\",\"compare_native\":" << (compare ? "true" : "false")
      << ",\"blocks\":" << nb << ",\"fields\":" << nf
      << ",\"cycles\":" << mesh->ncycle << ",\"time\":" << mesh->time
      << ",\"max_native_error\":" << error << ",\"relative_mass_error\":" << mass_error
      << ",\"minimum\":" << minimum << ",\"payload_bytes\":" << memory.reserved_payload
      << ",\"metadata_bytes\":" << memory.device_metadata
      << ",\"dense_payload_bytes\":" << memory.dense_payload
      << ",\"activations\":" << tracer.activations << ",\"deallocations\":" << tracer.deallocations
      << ",\"peak_pairs\":" << tracer.peak_pairs
      << ",\"profiled_tracer_seconds\":" << tracer.profiled_seconds << "}\n";
  if (error > tolerance || minimum < -tolerance || mass_error > tolerance) {
    throw std::runtime_error("Tracer native-reference, positivity, or mass check failed");
  }
}
}
void ProblemGenerator::SparseTracers(ParameterInput *pin,const bool restart) {
  if (restart) throw std::runtime_error("Sparse tracer restart is not implemented");
  auto *pack = pmy_mesh_->pmb_pack;
  if (!pack->phydro || !pack->phydro->ptracers) {
    throw std::runtime_error("sparse_tracers pgen requires hydro and passive_tracers");
  }
  auto &tracer = *pack->phydro->ptracers;
  if (!tracer.CompareDense() && pack->phydro->nscalars != 0) {
    throw std::runtime_error("Tracer demonstration requires nscalars=0 unless comparing");
  }
  const auto ind = pmy_mesh_->mb_indcs;
  const int nf = tracer.Count(), nh = pack->phydro->nhydro;
  const int ndim = pmy_mesh_->three_d ? 3 : (pmy_mesh_->multi_d ? 2 : 1);
  const Real radius = pin->GetOrAddReal("problem","radius",0.035);
  const Real vx = pin->GetOrAddReal("problem","vx",0.7);
  const Real vy = ndim > 1 ? pin->GetOrAddReal("problem","vy",0.35) : 0;
  const Real vz = ndim > 2 ? pin->GetOrAddReal("problem","vz",0.2) : 0;
  const Real contrast = pin->GetOrAddReal("problem","density_contrast",0.2);
  if (radius <= 0 || radius >= 0.5 || fabs(contrast) >= 0.9) {
    throw std::invalid_argument("Invalid tracer radius/density contrast");
  }
  const auto extent = pmy_mesh_->mesh_size;
  if (extent.x1min != 0 || extent.x1max != 1 ||
      (ndim > 1 && (extent.x2min != 0 || extent.x2max != 1)) ||
      (ndim > 2 && (extent.x3min != 0 || extent.x3max != 1))) {
    throw std::runtime_error("Tracer demonstration requires a unit periodic domain");
  }
  std::vector<sparse::BlockField> pairs;
  for (int b = 0; b < pack->nmb_thispack; ++b) {
    const auto size = pack->pmb->mb_size.h_view(b);
    for (int f = 0; f < nf; ++f) {
      bool live = false;
      for (int k = 0; k < ind.nx3 && !live; ++k) {
        for (int j = 0; j < ind.nx2 && !live; ++j) {
          for (int i = 0; i < ind.nx1 && !live; ++i) {
            live = Fraction(f,size.x1min+(i+0.5)*size.dx1,
                size.x2min+(j+0.5)*size.dx2,size.x3min+(k+0.5)*size.dx3,radius,ndim) != 0;
          }
        }
      }
      if (live) pairs.push_back({b,f});
    }
  }
  tracer.Initialize(pairs);
  auto gas = pack->phydro->u0;
  const auto size = pack->pmb->mb_size.d_view;
  const bool ideal = pin->GetString("hydro","eos") == "ideal";
  const Real gamma = pin->GetOrAddReal("hydro","gamma",Real(5)/3);
  const int cells = ind.nx1*ind.nx2*ind.nx3;
  Kokkos::parallel_for("tracer_gas_initial",Kokkos::RangePolicy<DevExeSpace>(0,pack->nmb_thispack*cells),
      KOKKOS_LAMBDA(int c) {
    const int b = c/cells, cell = c%cells;
    const int k = cell/(ind.nx1*ind.nx2)+ind.ks;
    const int j = (cell/ind.nx1)%ind.nx2+ind.js, i = cell%ind.nx1+ind.is;
    const Real x = size(b).x1min+(i-ind.is+0.5)*size(b).dx1;
    const Real rho = Real(1)+contrast*sin(Real(2)*M_PI*x);
    gas(b,IDN,k,j,i) = rho; gas(b,IM1,k,j,i) = rho*vx;
    gas(b,IM2,k,j,i) = rho*vy; gas(b,IM3,k,j,i) = rho*vz;
    if (ideal) gas(b,IEN,k,j,i) = Real(1)/(gamma-Real(1))+
                                 Real(0.5)*rho*(vx*vx+vy*vy+vz*vz);
  });
  const auto view = tracer.GetView();
  const bool compare = tracer.CompareDense();
  using Policy = Kokkos::RangePolicy<DevExeSpace,
      Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
  Kokkos::parallel_for("tracer_labels_initial",Policy(0,pack->nmb_thispack*nf*cells),
      KOKKOS_LAMBDA(int c) {
    const int b = c/(nf*cells), f = (c/cells)%nf, cell = c%cells;
    const int k = cell/(ind.nx1*ind.nx2)+ind.ks;
    const int j = (cell/ind.nx1)%ind.nx2+ind.js, i = cell%ind.nx1+ind.is;
    const Real q = gas(b,IDN,k,j,i)*Fraction(f,
        size(b).x1min+(i-ind.is+0.5)*size(b).dx1,
        size(b).x2min+(j-ind.js+0.5)*size(b).dx2,
        size(b).x3min+(k-ind.ks+0.5)*size(b).dx3,radius,ndim);
    const int slot = view.ordinary ? b*nf+f : view.slots(b,f);
    if (slot >= 0) view.At(0,slot,k,j,i) = q;
    if (compare) gas(b,nh+f,k,j,i) = q;
  });
  initial_mass = Mass(pmy_mesh_);
  tracer.WriteFrame(0,0);
  pgen_final_func = Errors;
}
