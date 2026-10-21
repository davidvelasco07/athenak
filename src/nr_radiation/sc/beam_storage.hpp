#ifndef NR_RADIATION_SC_BEAM_STORAGE_HPP_
#define NR_RADIATION_SC_BEAM_STORAGE_HPP_
// Fixed-support, non-emitting SC beam pilot. IDs always refer to original ordinates.
#include <memory>
#include <string>
#include <vector>
#include "sparse/sparse_variables.hpp"
#include "sparse/local_mesh.hpp"
namespace nr_radiation {
struct IntensityRef {
  Real *value;
  KOKKOS_INLINE_FUNCTION operator Real() const { return value ? *value : Real(0); }
  KOKKOS_INLINE_FUNCTION void operator=(Real x) const {
    if (value) *value = x;
    else if (x != 0) Kokkos::abort("Nonzero write outside fixed SC beam support");
  }
};
struct IntensityView {
  DvceArray5D<Real> dense, data;
  DvceArray2D<const int> slots;
  DvceArray1D<const int> blocks, fields, active;
  DvceArray1D<const int> ray_ids, ray_index;
  int mode = 0, work_count = 0, angle_count = 0; // 0 full dense, 1 selected, 2 pool
  KOKKOS_INLINE_FUNCTION IntensityRef operator()(int b,int a,int k,int j,int i) const {
    if (mode == 2) {
      const int s = slots(b,a);
      return {s < 0 ? nullptr : &data(0,s,k,j,i)};
    }
    const int idx = mode == 1 ? ray_index(a) : a;
    return {idx < 0 ? nullptr : &dense(b,idx,k,j,i)};
  }
  KOKKOS_INLINE_FUNCTION int Block(int w) const {
    return mode == 2 ? blocks(active(w)) : w/angle_count;
  }
  KOKKOS_INLINE_FUNCTION int Angle(int w) const {
    return mode == 2 ? fields(active(w)) : GlobalAngle(w%angle_count);
  }
  KOKKOS_INLINE_FUNCTION int GlobalAngle(int n) const {
    return mode == 0 ? n : ray_ids(n);
  }
};
class BeamStorage {
 public:
  BeamStorage(MeshBlockPack *pack,const std::string &mode,int nang,
              const std::vector<int> &rays,const std::vector<sparse::BlockField> &pairs);
  IntensityView GetView() const;
  void ExchangeGhosts();
  sparse::MemoryUsage Memory() const;
 private:
  MeshBlockPack *pack_;
  sparse::LocalMesh mesh_;
  std::unique_ptr<sparse::Variables> vars_;
  DvceArray5D<Real> dense_;
  DvceArray1D<int> ray_ids_,ray_index_;
  int nang_,nselected_;
};
}  // namespace nr_radiation
#endif
