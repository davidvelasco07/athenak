#include "nr_radiation/sc/beam_storage.hpp"
#include "nr_radiation/nr_radiation.hpp"
#include "nr_radiation/sc/sc_interp.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>

namespace nr_radiation {
namespace {
using Range = Kokkos::RangePolicy<DevExeSpace,
    Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
void Geometry(std::ostream &out,MeshBlockPack *pack) {
  const auto ind = pack->pmesh->mb_indcs;
  out << "\"block_cells\":[" << ind.nx1 << ',' << ind.nx2 << ',' << ind.nx3
      << "],\"blocks\":[";
  for (int b = 0; b < pack->nmb_thispack; ++b) {
    const auto loc = pack->pmesh->lloc_eachmb[pack->pmb->mb_gid.h_view(b)];
    if (b) out << ',';
    out << '[' << loc.lx1 << ',' << loc.lx2 << ',' << loc.lx3 << ']';
  }
  out << ']';
}
}
BeamStorage::BeamStorage(MeshBlockPack *pack,const std::string &mode,int nang,
    const std::vector<int> &rays,const std::vector<sparse::BlockField> &pairs)
    : pack_(pack),mesh_(sparse::MakeLocalMesh(pack)),nang_(nang),nselected_(rays.size()) {
  std::vector<int> angles = rays;
  if (mode == "forced") {
    angles.resize(nang);
    for (int a = 0; a < nang; ++a) angles[a] = a;
  }
  nselected_ = angles.size();
  ray_ids_ = DvceArray1D<int>("beam_ray_ids",angles.size());
  ray_index_ = DvceArray1D<int>("beam_ray_index",nang);
  auto ids = Kokkos::create_mirror_view(ray_ids_);
  auto index = Kokkos::create_mirror_view(ray_index_);
  for (int a = 0; a < nang; ++a) index(a) = -1;
  for (int a = 0; a < nselected_; ++a) { ids(a) = angles[a]; index(angles[a]) = a; }
  Kokkos::deep_copy(ray_ids_,ids); Kokkos::deep_copy(ray_index_,index);
  const auto ind = pack->pmesh->mb_indcs;
  const int n1 = ind.nx1+2*ind.ng, n2 = ind.nx2+2*ind.ng;
  if (mode == "selected") {
    dense_ = DvceArray5D<Real>("beam_selected_dense",pack->nmb_thispack,nselected_,1,n2,n1);
    Kokkos::deep_copy(dense_,0);
  } else {
    std::vector<sparse::Field> fields;
    for (int a = 0; a < nang; ++a) fields.push_back({a,"ordinate_"+std::to_string(a),0});
    vars_ = std::make_unique<sparse::Variables>("beam_intensity",
        sparse::Registry(std::move(fields)),pack->nmb_thispack,1,n2,n1,1,mode == "forced");
    vars_->SetAllocation(pairs);
  }
}
IntensityView BeamStorage::GetView() const {
  IntensityView out{};
  out.ray_ids = ray_ids_; out.ray_index = ray_index_;
  out.angle_count = nselected_;
  if (vars_) {
    const auto v = vars_->GetDeviceView();
    out.data = v.data; out.slots = v.slot_of; out.blocks = v.block_of;
    out.fields = v.field_of; out.active = v.active_slots;
    out.mode = 2; out.work_count = v.active_count;
  } else {
    out.mode = 1; out.dense = dense_; out.work_count = pack_->nmb_thispack*nselected_;
  }
  return out;
}
void BeamStorage::ExchangeGhosts() {
  const auto view = GetView();
  const auto mesh = mesh_;
  const auto ind = pack_->pmesh->mb_indcs;
  const int n1 = ind.nx1+2*ind.ng, n2 = ind.nx2+2*ind.ng, cells = n1*n2;
  // Each destination ghost is written once; all source reads are interior cells.
  Kokkos::parallel_for("beam_local_ghost_exchange",Range(0,view.work_count*cells),
      KOKKOS_LAMBDA(int c) {
    const int w = c/cells, cell = c%cells;
    const int b = view.Block(w), a = view.Angle(w), i = cell%n1, j = cell/n1;
    if (i >= ind.is && i <= ind.ie && j >= ind.js && j <= ind.je) return;
    int other = b, k = ind.ks, jj = j, ii = i;
    const Real value = mesh.Locate(other,k,jj,ii) ?
                       Real(view(other,a,k,jj,ii)) : Real(0);
    view(b,a,ind.ks,j,i) = value;
  });
}
sparse::MemoryUsage BeamStorage::Memory() const {
  auto memory = vars_ ? vars_->Memory() : sparse::MemoryUsage{
      dense_.size()*sizeof(Real),dense_.size()*sizeof(Real),
      std::size_t(pack_->nmb_thispack)*nang_*dense_.extent(2)*dense_.extent(3)*
          dense_.extent(4)*sizeof(Real),0};
  memory.device_metadata += (ray_ids_.size()+ray_index_.size()+mesh_.neighbors.size())*sizeof(int);
  return memory;
}
IntensityView SC::GetIntensityView() const {
  if (beam_storage) return beam_storage->GetView();
  IntensityView out{};
  out.dense = ir; out.angle_count = nang_tot;
  out.work_count = pmy_pack->nmb_thispack*nang_tot;
  return out;
}
void SC::ConfigureBeamStorage(ParameterInput *pin) {
  if (intensity_storage == "dense") return;
  if (beam_storage) throw std::logic_error("Beam storage already configured");
  const auto mesh = pmy_pack->pmesh;
  const auto ind = mesh->mb_indcs;
  const int nx = mesh->mesh_indcs.nx1, ny = mesh->mesh_indcs.nx2;
  const int iang = pin->GetInteger("problem","iang"), na = pang->nang;
  struct Seed {int angle,face,cell;};
  std::vector<Seed> seeds;
  const char *keys[4] = {"ihor1","ihor2","ivert1","ivert2"};
  const int oct[4] = {0,1,0,2};
  for (int n = 0; n < 4; ++n) {
    const int cell = pin->GetInteger("problem",keys[n]);
    if (cell < 0) continue;
    if (cell >= (n < 2 ? nx : ny)) throw std::out_of_range("Beam seed outside domain");
    seeds.push_back({oct[n]*na+iang,n < 2 ? 2 : 0,cell});
  }
  std::set<int> unique;
  for (const auto &seed : seeds) unique.insert(seed.angle);
  std::vector<int> rays(unique.begin(),unique.end());
  if (rays.empty()) throw std::runtime_error("Beam pilot needs at least one injected ray");
  std::vector<sparse::BlockField> pairs;
  const int stride = nx+2;
  const auto size = pmy_pack->pmb->mb_size.h_view(0);
  for (int angle : rays) {
    std::vector<unsigned char> support((nx+2)*(ny+2),0);
    auto at = [&](int i,int j) -> unsigned char& { return support[(j+1)*stride+i+1]; };
    for (const auto &seed : seeds) if (seed.angle == angle) {
      if (seed.face == 2) at(seed.cell,-1) = 1;
      else at(-1,seed.cell) = 1;
    }
    const Real mux = pang->mu.h_view(angle/na,angle%na,0);
    const Real muy = pang->mu.h_view(angle/na,angle%na,1);
    const int sx = mux > 0 ? 1 : -1, sy = muy > 0 ? 1 : -1;
    const auto inv = ComputeSCAngleInv(mux,muy,0,size.dx1,size.dx2,size.dx3,2,sx,sy,0);
    // Reachability of BOTH actual interpolation donors, not a geometric thin ray.
    // Includes zero-weight donors conservatively and all required ghost footprints.
    for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
      const int i = sx > 0 ? x : nx-1-x, j = sy > 0 ? y : ny-1-y;
      at(i,j) = at(i-sx,j-sy) || (inv.axis == 0 ? at(i,j-sy) : at(i-sx,j));
    }
    for (int b = 0; b < pmy_pack->nmb_thispack; ++b) {
      const auto loc = mesh->lloc_eachmb[pmy_pack->pmb->mb_gid.h_view(b)];
      const int x0 = loc.lx1*ind.nx1, y0 = loc.lx2*ind.nx2;
      bool live = false;
      for (int j = -ind.ng; j < ind.nx2+ind.ng && !live; ++j) {
        for (int i = -ind.ng; i < ind.nx1+ind.ng && !live; ++i) {
          const int x = x0+i, y = y0+j;
          if (x >= -1 && x <= nx && y >= -1 && y <= ny) live = at(x,y);
        }
      }
      if (live) pairs.push_back({b,angle});
    }
  }
  beam_storage = std::make_unique<BeamStorage>(pmy_pack,intensity_storage,nang_tot,rays,pairs);
}

void SC::ComputeBeamMoments(bool full) {
  const auto view = GetIntensityView();
  const auto ind = pmy_pack->pmesh->mb_indcs;
  const int extra = full ? 1 : 0;
  const int nx = ind.nx1+2*extra, ny = ind.nx2+2*extra, cells = nx*ny;
  const auto weights = pang->wmu.d_view;
  const auto mu = pang->mu.d_view;
  const int na = pang->nang;
  const auto mom = moments;
  Kokkos::parallel_for("beam_angular_moments",Range(0,pmy_pack->nmb_thispack*cells),
      KOKKOS_LAMBDA(int c) {
    const int b = c/cells, cell = c%cells;
    const int i = ind.is-extra+cell%nx, j = ind.js-extra+cell/nx, k = ind.ks;
    Real J = 0,H1 = 0,H2 = 0,H3 = 0,K11 = 0,K22 = 0,K33 = 0,K12 = 0,K13 = 0,K23 = 0;
    for (int n = 0; n < view.angle_count; ++n) {
      const int a = view.GlobalAngle(n), oct = a/na, local = a%na;
      const Real wI = weights(local)*Real(view(b,a,k,j,i));
      J += wI;
      if (full) {
        const Real x = mu(oct,local,0), y = mu(oct,local,1), z = mu(oct,local,2);
        H1 += x*wI; H2 += y*wI; H3 += z*wI;
        K11 += x*x*wI; K22 += y*y*wI; K33 += z*z*wI;
        K12 += x*y*wI; K13 += x*z*wI; K23 += y*z*wI;
      }
    }
    if (!full) mom(b,0,k,j,i) = J;
    else {
      mom(b,1,k,j,i) = H1; mom(b,2,k,j,i) = H2; mom(b,3,k,j,i) = H3;
      mom(b,4,k,j,i) = K11; mom(b,5,k,j,i) = K22; mom(b,6,k,j,i) = K33;
      mom(b,7,k,j,i) = K12; mom(b,8,k,j,i) = K13; mom(b,9,k,j,i) = K23;
    }
  });
}
void SC::WriteBeamFrame(int iteration) {
  if (beam_frame_interval <= 0 || iteration%beam_frame_interval != 0) return;
  const auto ind = pmy_pack->pmesh->mb_indcs;
  const int nx = ind.nx1, ny = ind.nx2, cells = nx*ny;
  const auto mom = moments;
  DvceArray1D<Real> values("beam_frame",pmy_pack->nmb_thispack*cells);
  Kokkos::parallel_for("beam_frame",Range(0,values.size()),KOKKOS_LAMBDA(int c) {
    values(c) = mom(c/cells,0,ind.ks,ind.js+(c%cells)/nx,ind.is+c%nx);
  });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),values);
  std::filesystem::create_directories(beam_frame_directory);
  const auto stem = beam_frame_directory+"/frame_"+
      std::to_string(pmy_pack->pmesh->ncycle*10000+iteration);
  std::ofstream out(stem+".bin",std::ios::binary);
  for (int c = 0; c < values.extent_int(0); ++c) {
    const float value = host(c); out.write(reinterpret_cast<const char*>(&value),sizeof(float));
  }
  const auto view = GetIntensityView();
  auto slots = Kokkos::create_mirror_view_and_copy(HostMemSpace(),view.slots);
  auto index = Kokkos::create_mirror_view_and_copy(HostMemSpace(),view.ray_index);
  std::ofstream meta(stem+".json");
  meta << "{\"iteration\":" << iteration << ",\"fields\":1,\"mask_fields\":" << nang_tot << ',';
  Geometry(meta,pmy_pack);
  meta << ",\"allocated\":[";
  for (int b = 0; b < pmy_pack->nmb_thispack; ++b) for (int a = 0; a < nang_tot; ++a) {
    if (b || a) meta << ',';
    meta << (view.mode == 0 || (view.mode == 1 ? index(a) >= 0 : slots(b,a) >= 0) ? 1 : 0);
  }
  meta << "]}\n";
  if (!out || !meta) throw std::runtime_error("Failed to write SC visualization frame");
}
void SC::DumpBeamState() {
  if (!beam_storage && !beam_dump_final && !beam_diagnostics) return;
  const auto view = GetIntensityView();
  sparse::MemoryUsage memory = beam_storage ? beam_storage->Memory() :
      sparse::MemoryUsage{ir.size()*sizeof(Real),ir.size()*sizeof(Real),ir.size()*sizeof(Real),0};
  std::ofstream report("beam.json");
  report << std::setprecision(17) << "{\"mode\":\"" << intensity_storage
      << "\",\"blocks\":" << pmy_pack->nmb_thispack << ",\"angles\":" << nang_tot
      << ",\"pairs\":" << view.work_count << ",\"sweeps\":" << niter_last
      << ",\"residual\":" << resid_last << ",\"converged\":" << (converged ? "true" : "false")
      << ",\"payload_bytes\":" << memory.reserved_payload
      << ",\"metadata_bytes\":" << memory.device_metadata
      << ",\"dense_payload_bytes\":" << memory.dense_payload << "}\n";
  if (!beam_dump_final) return;
  const auto ind = pmy_pack->pmesh->mb_indcs;
  const int cells = ind.nx1*ind.nx2;
  DvceArray2D<Real> result("beam_final",pmy_pack->nmb_thispack*(nang_tot+10),cells);
  const auto mom = moments;
  const int na = nang_tot;
  Kokkos::parallel_for("beam_final",Range(0,result.size()),KOKKOS_LAMBDA(int c) {
    const int pair = c/cells, cell = c%cells, b = pair/(na+10), a = pair%(na+10);
    const int k = ind.ks, j = cell/ind.nx1+ind.js, i = cell%ind.nx1+ind.is;
    result(pair,cell) = a < na ? Real(view(b,a,k,j,i)) : mom(b,a-na,k,j,i);
  });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),result);
  std::ofstream data("beam_state.bin",std::ios::binary);
  for (int pair = 0; pair < result.extent_int(0); ++pair) for (int c = 0; c < cells; ++c) {
    const double value = host(pair,c);
    data.write(reinterpret_cast<const char*>(&value),sizeof(double));
  }
}
}  // namespace nr_radiation
