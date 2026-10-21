#ifndef TRACERS_PASSIVE_TRACERS_HPP_
#define TRACERS_PASSIVE_TRACERS_HPP_
#include <memory>
#include <string>
#include "sparse/sparse_variables.hpp"
#include "sparse/local_mesh.hpp"
class Driver;
class ParameterInput;

namespace tracers {
// Same arithmetic/access contract for an ordinary dense array and sparse slot pool.
struct View {
  DvceArray6D<Real> dense;
  DvceArray5D<Real> native;
  DvceArray5D<Real> data;
  DvceArray2D<const int> slots;
  DvceArray1D<const int> blocks, fields;
  int nf, capacity, nhydro;
  bool ordinary, reference;
  KOKKOS_INLINE_FUNCTION int Block(int slot) const {
    return ordinary ? slot/nf : blocks(slot);
  }
  KOKKOS_INLINE_FUNCTION int Field(int slot) const {
    return ordinary ? slot%nf : fields(slot);
  }
  KOKKOS_INLINE_FUNCTION Real Read(int r,int b,int f,int k,int j,int i) const {
    if (reference) {
      if (r != 0) Kokkos::abort("Native reference view exposes only current state");
      return native(b,nhydro+f,k,j,i);
    }
    if (ordinary) return dense(r,b,f,k,j,i);
    const int slot = slots(b,f);
    return slot < 0 ? Real(0) : data(r,slot,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION Real &At(int r,int slot,int k,int j,int i) const {
    if (reference) {
      if (r != 0) Kokkos::abort("Native reference view exposes only current state");
      return native(slot/nf,nhydro+slot%nf,k,j,i);
    }
    return ordinary ? dense(r,slot/nf,slot%nf,k,j,i) : data(r,slot,k,j,i);
  }
};

class PassiveTracers {
 public:
  PassiveTracers(MeshBlockPack *pack,ParameterInput *pin);
  ~PassiveTracers();
  // Pgen allocates an exact initial set and writes conserved rho*C to register 0.
  void Initialize(const std::vector<sparse::BlockField> &pairs);
  View GetView() const;
  void Advance(Driver *driver,int stage);  // after gas fluxes, before gas RK update
  void Finish();
  void WriteFrame(int cycle,Real time) const;
  int Count() const { return nf_; }
  bool CompareDense() const { return compare_; }
  const std::string &Mode() const { return mode_; }
  sparse::MemoryUsage Memory() const;
  std::int64_t activations = 0, deallocations = 0;
  int peak_pairs = 0;
  double profiled_seconds = 0;
 private:
  MeshBlockPack *pack_;
  sparse::LocalMesh mesh_;
  std::unique_ptr<sparse::Variables> vars_;
  DvceArray6D<Real> dense_;
  DvceArray2D<int> requests_;
  int nf_, nr_, budget_, epoch_ = 0, frame_interval_;
  bool compare_, profile_, started_ = false, initialized_ = false;
  std::string mode_, frames_;
};
}  // namespace tracers
#endif
