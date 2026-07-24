#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.hpp
//  \brief definitions for MHD class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Viscosity;
class Resistivity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class OrbitalAdvectionFC;
class ShearingBoxCC;
class ShearingBoxFC;
class Driver;

// function ptr for user-defined MHD boundary functions enrolled in problem generator
namespace mhd {
using MHDBoundaryFnPtr = void (*)(int m, Mesh* pm, MHD* pmhd, DvceArray5D<Real> &u);
}

// constants that enumerate MHD Riemann Solver options
enum class MHD_RSolver {advect, llf, hlle, hlld, lhlld, roe,   // non-relativistic
                        llf_sr, hlle_sr,                // SR
                        llf_gr, hlle_gr};                       // GR

// constants that enumerate EMF (corner electric field) averaging options
enum class MHD_EMF {ct_contact, uct_hll, uct_hlld};

//----------------------------------------------------------------------------------------
//! \struct MHDTaskIDs
//  \brief container to hold TaskIDs of all mhd tasks

struct MHDTaskIDs {
  TaskID savest;
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID srctrms;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID efld;
  TaskID efldsrc;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID sendb_oa;
  TaskID recvb_oa;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID sendb_shr;
  TaskID recvb_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD {
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  ReconstructionMethod recon_method;
  MHD_RSolver rsolver_method;
  MHD_EMF emf_method;
  EquationOfState *peos;   // chosen EOS

  int nmhd;                // number of mhd variables (5/4 for ideal/isothermal EOS)
  int nscalars;            // number of passive scalars
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceFaceFld4D<Real> b0;  // face-centered magnetic fields
  DvceArray5D<Real> bcc0;  // cell-centered magnetic fields

  DvceArray5D<Real> coarse_u0;    // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;    // primitive variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld4D<Real> coarse_b0;  // face-centered B-field on 2x coarser grid

  // Objects containing boundary communication buffers and routines for u and b
  MeshBoundaryValuesCC *pbval_u;
  MeshBoundaryValuesFC *pbval_b;
  MHDBoundaryFnPtr MHDBoundaryFunc[6];

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  OrbitalAdvectionFC *porb_b = nullptr;
  ShearingBoxCC *psbox_u = nullptr;
  ShearingBoxFC *psbox_b = nullptr;

  // Object(s) for extra physics (viscosity, resistivity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Resistivity *presist = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables, second register
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  // candidate face-centered B for the MHD MOOD detector: the genuine staggered CT update of
  // b0 with the candidate corner EMFs, averaged to cell centers to fill bcctest (so the
  // detector sees the real evolved field, not the FOFC face-E proxy). Allocated when mood.
  DvceFaceFld4D<Real> b0_test;
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  // UCT data stored at cell faces by Riemann solvers (only allocated when UCT is used)
  // x1-faces: flux weight, diffusion coefficients, upwind transverse velocities
  DvceArray5D<Real> sdet;  // per-direction velocity compression (LHLLD carbuncle sensor)
  DvceArray4D<Real> aL_x1f, dL_x1f, dR_x1f, vy_x1f, vz_x1f;
  // x2-faces: flux weight, diffusion coefficients, upwind transverse velocities
  DvceArray4D<Real> aL_x2f, dL_x2f, dR_x2f, vx_x2f, vz_x2f;
  // x3-faces: flux weight, diffusion coefficients, upwind transverse velocities
  DvceArray4D<Real> aL_x3f, dL_x3f, dR_x3f, vx_x3f, vy_x3f;
  // global per-face L/R buffers for the split-kernel flux path: primitives (nmhd+nscalars
  // components) and reconstructed cell-centered B-field (3 components)
  DvceArray5D<Real> wl_split, wr_split;
  DvceArray5D<Real> bl_split, br_split;
  Real dtnew;

  // following used for time derivatives in computation of jcon
  bool wbcc_saved = false;
  DvceArray5D<Real> wsaved;
  DvceArray5D<Real> bccsaved;

  // following used for FOFC algorithm
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC

  // following used for the MOOD a-posteriori fallback ("FB") scheme.  Shares the
  // fofc/utest/bcctest arrays: fofc marks cells flagged (newly demoted) in the current
  // revision iteration, utest/bcctest hold the candidate update.
  bool use_mood = false;    // flag to enable MOOD fallback
  int mood_max_revs = 1;    // max revision iterations per RK stage
  int mood_nad_scale;       // NAD tolerance scale: 0=relative, 1=grange, 2=gdu, 3=gcfl
  Real mood_nad_theta;      // grange Mach-softening exponent
  bool mood_nad_energy;     // include the energy variable in NAD (density always on)
  int mood_nad_b;           // B in NAD: 0=|B| magnitude, 1=components (of bcctest)
  int mood_nad_v;           // velocity in NAD: 0=off, 1=|v| magnitude, 2=components
  bool uct_diag;            // print max|d| (UCT dissipation coeff) + EMF-NaN diagnostic
  int mood_edge_flag;       // 1: explicitly demote UCT edge reconstruction at edges
                            // adjacent to demoted cells; false: implicit blending of
                            // revised face data through the corner composition
  Real mood_rtol;           // NAD tolerance as a fraction of the selected scale
  Real mood_eps0;           // round-off floor (relative to the violated bound)
  Real mood_atol;           // absolute floor of the NAD tolerance
  bool mood_sed;            // exempt smooth extrema from NAD detection
  int n_fb_tiers;           // # of fallback tiers below base scheme (2, or 1 if plm)
  int mood_halo0;           // first-iteration detection halo (light-cone start width)
  DvceArray4D<int> fb_level;    // per-cell cascade level (0=base, 1=plm, 2=dc)
  DvceArray4D<Real> bmag_ref;   // |B| of stage-input bcc0 (only for mood_nad_b=0)

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions...
  void SetSaveWBcc();
  void AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_timeintegrator" task list
  TaskStatus SaveMHDState(Driver *d, int stage);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus MHDSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus CornerE(Driver *d, int stage);
  // Compose edge-centered corner EMFs (efld) from the current face-centered E-fields and
  // UCT face coefficients over an arbitrary edge-point range [il,iu]x[jl,ju]x[kl,ku].
  // Newtonian-ideal UCT path only (factored out of CornerE so the MOOD detector can build
  // the genuine staggered candidate B over its light-cone halo — see mhd_mood.cpp).
  void ComposeCornerEMF(int il, int iu, int jl, int ju, int kl, int ku);
  TaskStatus EFieldSrc(Driver *d, int stage);
  TaskStatus SendE(Driver *d, int stage);
  TaskStatus RecvE(Driver *d, int stage);
  TaskStatus CT(Driver *d, int stage);
  TaskStatus SendB_OA(Driver *d, int stage);
  TaskStatus RecvB_OA(Driver *d, int stage);
  TaskStatus RestrictB(Driver *d, int stage);
  TaskStatus SendB(Driver *d, int stage);
  TaskStatus RecvB(Driver *d, int stage);
  TaskStatus SendB_Shr(Driver *d, int stage);
  TaskStatus RecvB_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <MHD_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

  // MOOD a-posteriori fallback, templated over Riemann solver (fallback tiers re-solve
  // flagged faces with the same solver as the base scheme)
  template <MHD_RSolver T>
  void MOODLoop(Driver *d, int stage);

  DvceArray5D<Real> utest, bcctest;  // scratch arrays for FOFC/MOOD

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
