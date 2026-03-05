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
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  // UCT data stored at cell faces by Riemann solvers (only allocated when UCT is used)
  // x1-faces: flux weight, diffusion coefficients, upwind transverse velocities
  DvceArray4D<Real> aL_x1f, dL_x1f, dR_x1f, vy_x1f, vz_x1f;
  // x2-faces: flux weight, diffusion coefficients, upwind transverse velocities
  DvceArray4D<Real> aL_x2f, dL_x2f, dR_x2f, vx_x2f, vz_x2f;
  // x3-faces: flux weight, diffusion coefficients, upwind transverse velocities
  DvceArray4D<Real> aL_x3f, dL_x3f, dR_x3f, vx_x3f, vy_x3f;
  Real dtnew;

  // following used for time derivatives in computation of jcon
  bool wbcc_saved = false;
  DvceArray5D<Real> wsaved;
  DvceArray5D<Real> bccsaved;

  // following used for FOFC algorithm
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC

  // following used for Mignone 4th-order scheme
  bool use_mignone = false;
  DvceArray5D<Real> u0_c;      // pointwise conserved variables
  DvceArray5D<Real> w0_c;      // pointwise primitives (cell-center values)
  DvceArray5D<Real> bcc0_c;    // pointwise cell-centered B (3 components)
  DvceFaceFld5D<Real> uflx_f;  // pointwise face fluxes (before surface averaging)
  DvceFaceFld4D<Real> b0_c;    // pointwise face-centered B (for UCT CornerE)

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions...
  void SetSaveWBcc();
  void AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // 4th-order IC setup (called from Driver::InitBoundaryValuesAndPrimitives)
  void InitMignoneIC();
  // 4th-order reference solution setup (called from LinearWaveErrors after pgen sets b1/w0)
  void InitMignoneRef();
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

  DvceArray5D<Real> utest, bcctest;  // scratch arrays for FOFC

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
