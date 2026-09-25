#ifndef PARTICLES_SINK_MERGE_HPP_
#define PARTICLES_SINK_MERGE_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace particles {
// Host-only, deterministic global pair reduction. Candidate weights never become mass.
template<class T> struct SinkMergeRecord {
  std::array<T,3> x, v, dx;
  T mass, weight;
  int tag;
  bool alive = true, changed = false;
};
template<class T> T SinkMinimumImage(T d, T length, bool periodic) {
  return periodic ? d-length*std::floor(d/length+T(0.5)) : d;
}
template<class T> bool SinkContact(const SinkMergeRecord<T> &a,
    const SinkMergeRecord<T> &b, const std::array<T,3> &origin,
    const std::array<T,3> &length, bool periodic, bool cells, bool faces) {
  int touches=0;
  for (int c=0; c<3; ++c) {
    T x=a.x[c], y=b.x[c];
    if (cells) {
      x=origin[c]+(std::floor((x-origin[c])/a.dx[c])+T(0.5))*a.dx[c];
      y=origin[c]+(std::floor((y-origin[c])/b.dx[c])+T(0.5))*b.dx[c];
    }
    const T distance=std::abs(SinkMinimumImage(x-y,length[c],periodic));
    const T width=T(1.5)*(a.dx[c]+b.dx[c]);
    if (distance > width) return false;
    if (distance == width) {
      if (!faces || ++touches > 1) return false;
    }
  }
  return true;
}
template<class T, class Locate> void ResolveSinkMergers(
    std::vector<SinkMergeRecord<T>> &s, const std::array<T,3> &origin,
    const std::array<T,3> &length, bool periodic, bool cells, bool faces,
    bool bound, T grav, Locate locate) {
  std::sort(s.begin(),s.end(),[](const auto &a,const auto &b){return a.tag<b.tag;});
  for (;;) {
    bool merged=false;
    for (size_t i=0; i<s.size() && !merged; ++i) {
      if (!s[i].alive) continue;
      for (size_t j=i+1; j<s.size(); ++j) {
        if (!s[j].alive || !SinkContact(s[i],s[j],origin,length,periodic,cells,faces)) continue;
        auto &a=s[i]; auto &b=s[j];
        const bool anew=a.mass==T(0), bnew=b.mass==T(0);
        // A new candidate is not a physical body: geometric rejection must not
        // depend on a binding energy involving its zero mass.
        if (bound && !anew && !bnew) {
          T r2=0, v2=0;
          for(int c=0;c<3;++c) {
            T d=SinkMinimumImage(a.x[c]-b.x[c],length[c],periodic);
            r2+=d*d; v2+=(a.v[c]-b.v[c])*(a.v[c]-b.v[c]);
          }
          if (!(T(0.5)*v2*std::sqrt(r2)<grav*(a.mass+b.mass))) continue;
        }
        if (anew != bnew) {
          (anew ? a : b).alive=false;
          (anew ? b : a).changed=true;
        } else {
          // Most massive survives; equal masses/weights use the lowest stable tag.
          size_t keep=i, drop=j;
          if (b.mass>a.mass) {keep=j;drop=i;}
          auto &u=s[keep]; auto &v=s[drop];
          T wu=anew?u.weight:u.mass, wv=anew?v.weight:v.mass, total=wu+wv;
          for(int c=0;c<3;++c) {
            u.x[c]+=wv/total*SinkMinimumImage(v.x[c]-u.x[c],length[c],periodic);
            if(periodic) u.x[c]-=length[c]*std::floor((u.x[c]-origin[c])/length[c]);
            u.v[c]=anew?T(0):(wu*u.v[c]+wv*v.v[c])/total;
          }
          u.mass+=v.mass; u.weight+=v.weight; u.changed=true; v.alive=false;
          locate(u); // Recompute containing level/cell after EVERY merger.
        }
        merged=true;
        break; // Restart global scan: the new COM may contact an earlier particle.
      }
    }
    if (!merged) break;
  }
}
}  // namespace particles
#endif
