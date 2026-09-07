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
#include "diffusion/sts_types.hpp"
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
enum class MHD_RSolver {advect, llf, hlle, hlld, roe,   // non-relativistic
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
  DvceArray5D<Real> u_sts0;   // conserved variables at start of STS sweep
  DvceArray5D<Real> u_sts1;   // previous STS stage state
  DvceArray5D<Real> u_sts2;   // second previous STS stage state
  DvceArray5D<Real> u_sts_rhs;  // cached first-stage RKL2 operator contribution
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  DvceFaceFld4D<Real> b_sts0;  // face-centered fields at start of STS sweep
  DvceFaceFld4D<Real> b_sts1;  // previous STS stage fields
  DvceFaceFld4D<Real> b_sts2;  // second previous STS stage fields
  DvceFaceFld4D<Real> b_sts_rhs;  // cached first-stage RKL2 operator contribution
  // candidate face-centered B for MHD MOOD+UCT detector (genuine staggered CT update)
  DvceFaceFld4D<Real> b0_test;
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  // UCT data stored at cell faces by Riemann solvers (only allocated when UCT is used)
  DvceArray4D<Real> aL_x1f, dL_x1f, dR_x1f, vy_x1f, vz_x1f;
  DvceArray4D<Real> aL_x2f, dL_x2f, dR_x2f, vx_x2f, vz_x2f;
  DvceArray4D<Real> aL_x3f, dL_x3f, dR_x3f, vx_x3f, vy_x3f;
  // global per-face L/R buffers for the split-kernel flux path: primitives (nmhd+nscalars
  // components) and reconstructed cell-centered B-field (3 components)
  DvceArray5D<Real> wl3d, wr3d;
  DvceArray5D<Real> bl3d, br3d;
  Real dtnew;

  // following used for time derivatives in computation of jcon
  bool wbcc_saved = false;
  DvceArray5D<Real> wsaved;
  DvceArray5D<Real> bccsaved;

  // following used for FOFC algorithm
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  DvceArray5D<bool> fofc_scal;  // flag to indicate if FOFC for scalar is needed
  bool use_fofc = false;   // flag to enable FOFC

  // MOOD a-posteriori fallback ("FB"). Shares fofc/utest/bcctest arrays.
  bool use_mood = false;
  int mood_max_revs = 1;
  int mood_nad_scale;
  Real mood_nad_theta;
  bool mood_nad_energy;
  bool mood_nad_scalars;    // include passive-scalar concentrations in NAD
  int mood_nad_b;           // 0=|B|, 1=components
  int mood_nad_v;           // 0=off, 1=|v|, 2=components
  bool uct_diag;            // print max|d| (UCT dissipation) + EMF-NaN diagnostic
  int mood_edge_flag;       // 1: demote UCT edge recon at demoted cells; 0: blend only
  Real mood_rtol;
  Real mood_eps0;
  Real mood_atol;
  bool mood_sed;
  int n_fb_tiers;
  int mood_halo0;
  DvceArray4D<int> fb_level;
  DvceArray4D<Real> bmag_ref;

  bool has_explicit_viscosity = false;
  bool has_explicit_conduction = false;
  bool has_explicit_resistivity = false;
  bool has_sts_viscosity = false;
  bool has_sts_conduction = false;
  bool has_sts_resistivity = false;
  bool has_any_sts_diffusion = false;
  bool has_any_sts_cell_update = false;
  bool has_any_sts_field_update = false;

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions...
  void SetSaveWBcc();
  void AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_timeintegrator" task list
  TaskStatus SaveMHDState(Driver *d, int stage);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus InitRecvParabolic(Driver *d, int stage);
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
  // Compose edge-centered corner EMFs from face-centered E and UCT face coefficients
  // over [il,iu]x[jl,ju]x[kl,ku]. Newtonian UCT path (also used by MOOD candidate B).
  void ComposeCornerEMF(int il, int iu, int jl, int ju, int kl, int ku);
  TaskStatus EField(Driver *d, int stage);
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
  void RecomputeTimeStepFromCurrentState(Driver *pdrive);
  TaskStatus ClearSTSFlux(Driver *d, int stage);
  TaskStatus ClearSTSEField(Driver *d, int stage);
  TaskStatus STSFluxes(Driver *d, int stage);
  TaskStatus STSEField(Driver *d, int stage);
  TaskStatus STSUpdateU(Driver *d, int stage);
  TaskStatus STSUpdateB(Driver *d, int stage);
  TaskStatus STSRefreshTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <MHD_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

  // MOOD a-posteriori fallback (same Riemann solver as base scheme)
  template <MHD_RSolver T>
  void MOODLoop(Driver *d, int stage);

  DvceArray5D<Real> utest, bcctest;  // scratch arrays for FOFC

 private:
  void AddSelectedDiffusionFluxes(parabolic::DiffusionSelection selection);
  void AddSelectedDiffusionEMF(parabolic::DiffusionSelection selection);
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
