#include "tracers/passive_tracers.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "reconstruct/plm.hpp"

namespace tracers {
namespace {
using Light = Kokkos::Experimental::WorkItemProperty::HintLightWeight_t;
using Range = Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<std::int64_t>,Light>;
using Team = Kokkos::TeamPolicy<DevExeSpace,Light>;
sparse::Registry Registry(int nf) {
  if (nf <= 0 || nf > 100000) throw std::invalid_argument("Invalid tracer count");
  std::vector<sparse::Field> fields;
  for (int f = 0; f < nf; ++f) fields.push_back({f,"tracer_"+std::to_string(f),0});
  return sparse::Registry(std::move(fields));
}
KOKKOS_INLINE_FUNCTION
Real Fraction(View view,sparse::LocalMesh mesh,DvceArray5D<Real> gas,
              int b,int f,int k,int j,int i) {
  if (!mesh.Locate(b,k,j,i)) return 0;
  return view.Read(0,b,f,k,j,i)/gas(b,IDN,k,j,i);
}
}

PassiveTracers::PassiveTracers(MeshBlockPack *pack,ParameterInput *pin)
    : pack_(pack),mesh_(sparse::MakeLocalMesh(pack)) {
  if (!pack->pmesh->strictly_periodic || pack->pcoord->is_special_relativistic ||
      pack->pcoord->is_general_relativistic ||
      pin->GetString("time","evolution") == "static") {
    throw std::runtime_error("Passive tracer pilot requires periodic Newtonian hydro");
  }
  const auto integrator = pin->GetString("time","integrator");
  if (integrator != "rk1" && integrator != "rk2" && integrator != "rk3") throw std::runtime_error("Unsupported tracer integrator");
  auto &ind = pack->pmesh->mb_indcs;
  if (ind.nx1 < ind.ng || (pack->pmesh->multi_d && ind.nx2 < ind.ng) ||
      (pack->pmesh->three_d && ind.nx3 < ind.ng)) {
    throw std::runtime_error("Tracer blocks must be at least nghost cells wide");
  }
  nf_ = pin->GetInteger("passive_tracers","nfields");
  mode_ = pin->GetOrAddString("passive_tracers","storage","sparse");
  compare_ = pin->GetOrAddBoolean("passive_tracers","compare_dense",false);
  profile_ = pin->GetOrAddBoolean("passive_tracers","profile",false);
  frame_interval_ = pin->GetOrAddInteger("passive_tracers","frame_interval",0);
  frames_ = pin->GetOrAddString("passive_tracers","frames","tracer_frames");
  if (mode_ == "native") compare_ = true;
  if (mode_ != "dense" && mode_ != "forced" && mode_ != "sparse" && mode_ != "native") {
    throw std::invalid_argument("Tracer storage must be native, dense, forced, or sparse");
  }
  const int nb = pack->nmb_thispack;
  auto fields = Registry(nf_);
  if (std::int64_t(nb)*nf_ > std::numeric_limits<int>::max() || frame_interval_ < 0) {
    throw std::invalid_argument("Invalid tracer shape or frame interval");
  }
  nr_ = 3+(pack->pmesh->multi_d ? 1 : 0)+(pack->pmesh->three_d ? 1 : 0);
  budget_ = pin->GetOrAddInteger("passive_tracers","capacity",nb*nf_);
  const int n1 = ind.nx1+2*ind.ng;
  const int n2 = pack->pmesh->multi_d ? ind.nx2+2*ind.ng : 1;
  const int n3 = pack->pmesh->three_d ? ind.nx3+2*ind.ng : 1;
  if (mode_ == "native") return;
  if (mode_ == "dense") {
    dense_ = DvceArray6D<Real>("passive_dense",nr_,nb,nf_,n3,n2,n1);
    Kokkos::deep_copy(dense_,0);
  } else {
    vars_ = std::make_unique<sparse::Variables>("passive",std::move(fields),
                                              nb,n3,n2,n1,nr_,mode_ == "forced");
    requests_ = DvceArray2D<int>("passive_requests",nb,nf_);
    Kokkos::deep_copy(requests_,0);
  }
}
PassiveTracers::~PassiveTracers() { if (started_) Finish(); }

void PassiveTracers::Initialize(const std::vector<sparse::BlockField> &pairs) {
  if (initialized_) throw std::logic_error("Tracers already initialized");
  if (vars_) vars_->SetAllocation(pairs);
  initialized_ = true;
}
View PassiveTracers::GetView() const {
  if (!initialized_) throw std::logic_error("Initialize tracers in the problem generator");
  View result{};
  result.nf = nf_;
  result.capacity = pack_->nmb_thispack*nf_;
  result.nhydro = pack_->phydro->nhydro;
  result.reference = mode_ == "native";
  result.ordinary = !vars_;
  if (result.reference) result.native = pack_->phydro->u0;
  else if (!vars_) result.dense = dense_;
  else {
    const auto v = started_ ? vars_->GetDevicePoolView().fields : vars_->GetDeviceView();
    result.data = v.data; result.slots = v.slot_of;
    result.blocks = v.block_of; result.fields = v.field_of;
    result.capacity = vars_->Capacity();
  }
  return result;
}

void PassiveTracers::Advance(Driver *driver,int stage) {
  if (mode_ == "native") return;
  if (!initialized_) throw std::logic_error("Initialize tracers in the problem generator");
  if (profile_) Kokkos::fence();
  Kokkos::Timer timer;
  if (vars_ && !started_) {
    vars_->BeginDeviceAllocation(mode_ == "forced" ? pack_->nmb_thispack*nf_ : budget_);
    started_ = true;
  }
  auto view = GetView();
  const auto mesh = mesh_;
  const auto ind = pack_->pmesh->mb_indcs;
  const bool multi = pack_->pmesh->multi_d, three = pack_->pmesh->three_d;
  const int nx = ind.nx1, ny = ind.nx2, nz = ind.nx3, cells = nx*ny*nz;
  const int ng = ind.ng, n1 = nx+2*ng, n2 = multi ? ny+2*ng : 1;
  const int n3 = three ? nz+2*ng : 1;
  const auto requests = requests_;
  const auto gas = pack_->phydro->w0;
  // Preserve every live RK register; only current state requires a receiver halo.
  if (mode_ == "sparse") {
    if (epoch_ == std::numeric_limits<int>::max()) {
      Kokkos::deep_copy(requests,0); epoch_ = 0;
    }
    const int epoch = ++epoch_;
    Kokkos::parallel_for("tracer_request_stencil",Team(view.capacity,Kokkos::AUTO),
        KOKKOS_LAMBDA(const Team::member_type &team) {
      const int slot = team.league_rank(), b = view.Block(slot);
      if (b < 0) return;
      int live = 0, saved = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,cells),
          [&](int c,int &count) {
        if (view.At(0,slot,c/(nx*ny)+ind.ks,(c/nx)%ny+ind.js,c%nx+ind.is) != 0) ++count;
      },live);
      if (stage > 1) Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,cells),
          [&](int c,int &count) {
        if (view.At(1,slot,c/(nx*ny)+ind.ks,(c/nx)%ny+ind.js,c%nx+ind.is) != 0) ++count;
      },saved);
      Kokkos::single(Kokkos::PerTeam(team),[&]() {
        const int f = view.Field(slot);
        if (live || saved) Kokkos::atomic_exchange(&requests(b,f),epoch);
        if (live) {
          const int neighbors[6] = {12,14,10,16,4,22};
          const int count = three ? 6 : (multi ? 4 : 2);
          for (int d = 0; d < count; ++d) {
            const int other = mesh.neighbors(b,neighbors[d]);
            if (other >= 0) Kokkos::atomic_exchange(&requests(other,f),epoch);
          }
        }
      });
    });
    vars_->ApplyDeviceMask(requests,epoch);
  }
  // Begin/Apply never relocate retained slots within this session.
  if (stage == 1) {
    Kokkos::parallel_for("tracer_copy_rk",Range(0,std::int64_t(view.capacity)*cells),
        KOKKOS_LAMBDA(std::int64_t c) {
      const int slot = c/cells;
      if (view.Block(slot) < 0) return;
      const int cell = c%cells, k = cell/(nx*ny)+ind.ks;
      const int j = (cell/nx)%ny+ind.js, i = cell%nx+ind.is;
      view.At(1,slot,k,j,i) = view.At(0,slot,k,j,i);
    });
  }
  const auto fx = pack_->phydro->uflx.x1f, fy = pack_->phydro->uflx.x2f;
  const auto fz = pack_->phydro->uflx.x3f;
  const bool plm = pack_->phydro->recon_method == ReconstructionMethod::plm;
  const int storage_cells = n1*n2*n3;
  // Identical mass-flux upwinding in all storage modes, matching native scalars.
  Kokkos::parallel_for("tracer_face_flux",Range(0,std::int64_t(view.capacity)*storage_cells),
      KOKKOS_LAMBDA(std::int64_t c) {
    const int slot = c/storage_cells, b = view.Block(slot);
    if (b < 0) return;
    const int f = view.Field(slot), cell = c%storage_cells;
    const int k = cell/(n1*n2), j = (cell/n1)%n2, i = cell%n1;
    const int ndim = three ? 3 : (multi ? 2 : 1);
    for (int d = 0; d < ndim; ++d) {
      const bool face = i >= ind.is && i <= ind.ie+(d == 0) &&
                        j >= ind.js && j <= ind.je+(d == 1) &&
                        k >= ind.ks && k <= ind.ke+(d == 2);
      if (!face) continue;
      const int di = d == 0, dj = d == 1, dk = d == 2;
      const Real mass = d == 0 ? fx(b,IDN,k,j,i) :
                        (d == 1 ? fy(b,IDN,k,j,i) : fz(b,IDN,k,j,i));
      const int up = mass >= 0 ? 1 : 0;
      const int iu = i-up*di, ju = j-up*dj, ku = k-up*dk;
      Real q = Fraction(view,mesh,gas,b,f,ku,ju,iu);
      if (plm) {
        const Real qm = Fraction(view,mesh,gas,b,f,ku-dk,ju-dj,iu-di);
        const Real qp = Fraction(view,mesh,gas,b,f,ku+dk,ju+dj,iu+di);
        Real left, right;
        PLM(qm,q,qp,left,right);
        q = mass >= 0 ? left : right;
      }
      view.At(2+d,slot,k,j,i) = mass*q;
    }
  });
  const Real gam0 = driver->gam0[stage-1], gam1 = driver->gam1[stage-1];
  const Real dt = driver->beta[stage-1]*pack_->pmesh->dt;
  const auto size = pack_->pmb->mb_size.d_view;
  Kokkos::parallel_for("tracer_rk_update",Range(0,std::int64_t(view.capacity)*cells),
      KOKKOS_LAMBDA(std::int64_t c) {
    const int slot = c/cells, b = view.Block(slot);
    if (b < 0) return;
    const int cell = c%cells, k = cell/(nx*ny)+ind.ks;
    const int j = (cell/nx)%ny+ind.js, i = cell%nx+ind.is;
    Real div = (view.At(2,slot,k,j,i+1)-view.At(2,slot,k,j,i))/size(b).dx1;
    if (multi) div += (view.At(3,slot,k,j+1,i)-view.At(3,slot,k,j,i))/size(b).dx2;
    if (three) div += (view.At(4,slot,k+1,j,i)-view.At(4,slot,k,j,i))/size(b).dx3;
    view.At(0,slot,k,j,i) = gam0*view.At(0,slot,k,j,i)+gam1*view.At(1,slot,k,j,i)-dt*div;
  });
  // Only the final RK stage makes the old register dead. No threshold pruning.
  if (mode_ == "sparse" && stage == driver->nexp_stages) {
    if (epoch_ == std::numeric_limits<int>::max()) {
      Kokkos::deep_copy(requests,0); epoch_ = 0;
    }
    const int epoch = ++epoch_;
    Kokkos::parallel_for("tracer_prune_completed_step",Team(view.capacity,Kokkos::AUTO),
        KOKKOS_LAMBDA(const Team::member_type &team) {
      const int slot = team.league_rank(), b = view.Block(slot);
      if (b < 0) return;
      int live = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,cells),
          [&](int c,int &count) {
        if (view.At(0,slot,c/(nx*ny)+ind.ks,(c/nx)%ny+ind.js,c%nx+ind.is) != 0) ++count;
      },live);
      Kokkos::single(Kokkos::PerTeam(team),[&]() {
        if (live) Kokkos::atomic_exchange(&requests(b,view.Field(slot)),epoch);
      });
    });
    vars_->ApplyDeviceMask(requests,epoch);
  }
  if (profile_) { Kokkos::fence(); profiled_seconds += timer.seconds(); }
}

void PassiveTracers::Finish() {
  if (!started_) return;
  const auto stats = vars_->EndDeviceAllocation();
  activations += stats.activations; deallocations += stats.deallocations;
  peak_pairs = std::max(peak_pairs,stats.peak_pairs);
  started_ = false;
}
sparse::MemoryUsage PassiveTracers::Memory() const {
  if (mode_ == "native") {
    const auto &gas = *pack_->phydro;
    const std::size_t one = std::size_t(pack_->nmb_thispack)*nf_*
        gas.u0.extent(2)*gas.u0.extent(3)*gas.u0.extent(4)*sizeof(Real);
    // Native tracers occupy u0,u1,w0,wl,wr and the three existing flux arrays.
    return {8*one,8*one,8*one,mesh_.neighbors.size()*sizeof(int)};
  }
  if (vars_) {
    auto result = vars_->Memory();
    result.device_metadata += (requests_.size()+mesh_.neighbors.size())*sizeof(int);
    return result;
  }
  return {dense_.size()*sizeof(Real),dense_.size()*sizeof(Real),
          dense_.size()*sizeof(Real),mesh_.neighbors.size()*sizeof(int)};
}

void PassiveTracers::WriteFrame(int cycle,Real time) const {
  if (frame_interval_ <= 0 || (cycle%frame_interval_ != 0)) return;
  const auto view = GetView();
  const auto ind = pack_->pmesh->mb_indcs;
  const int nx = ind.nx1, ny = ind.nx2, nz = ind.nx3, cells = nx*ny*nz;
  const auto gas = pack_->phydro->u0;
  DvceArray2D<Real> frame("tracer_frame",nf_,pack_->nmb_thispack*cells);
  Kokkos::parallel_for("tracer_frame",Range(0,frame.size()),
      KOKKOS_LAMBDA(std::int64_t c) {
    const int f = c/(frame.extent_int(1)), rest = c%frame.extent_int(1);
    const int b = rest/cells, cell = rest%cells;
    const int k = cell/(nx*ny)+ind.ks, j = (cell/nx)%ny+ind.js, i = cell%nx+ind.is;
    frame(f,rest) = view.Read(0,b,f,k,j,i)/gas(b,IDN,k,j,i);
  });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),frame);
  std::filesystem::create_directories(frames_);
  const auto stem = frames_+"/frame_"+std::to_string(cycle);
  std::ofstream out(stem+".bin",std::ios::binary);
  for (int f = 0; f < nf_; ++f) for (int b = 0; b < pack_->nmb_thispack; ++b) {
    for (int cell = 0; cell < cells; ++cell) {
      const float value = host(f,b*cells+cell);
      out.write(reinterpret_cast<const char*>(&value),sizeof(float));
    }
  }
  std::ofstream meta(stem+".json");
  meta.precision(17);
  meta << "{\"cycle\":" << cycle << ",\"time\":" << time << ",\"fields\":" << nf_
       << ",\"block_cells\":[" << nx << ',' << ny << ',' << nz << "],\"blocks\":[";
  for (int b = 0; b < pack_->nmb_thispack; ++b) {
    const auto loc = pack_->pmesh->lloc_eachmb[pack_->pmb->mb_gid.h_view(b)];
    if (b) meta << ',';
    meta << '[' << loc.lx1 << ',' << loc.lx2 << ',' << loc.lx3 << ']';
  }
  meta << "],\"allocated\":[";
  auto slots = Kokkos::create_mirror_view_and_copy(HostMemSpace(),view.slots);
  for (int b = 0; b < pack_->nmb_thispack; ++b) for (int f = 0; f < nf_; ++f) {
    if (b || f) meta << ',';
    meta << (view.ordinary || slots(b,f) >= 0 ? 1 : 0);
  }
  meta << "]}\n";
  if (!out || !meta) throw std::runtime_error("Failed to write tracer visualization frame");
}
}  // namespace tracers
