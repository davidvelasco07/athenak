#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

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
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class ShearingBoxCC;
class Driver;
class Mesh;
class ProblemGenerator;

// constants that enumerate Hydro Riemann Solver options
enum class Hydro_RSolver {advect, llf, hlle, hllc, lhllc, roe,    // non-relativistic
                          llf_sr, hlle_sr, hllc_sr,        // SR
                          llf_gr, hlle_gr};                // GR

//----------------------------------------------------------------------------------------
//! \struct HydroTaskIDs
//  \brief container to hold TaskIDs of all hydro tasks

struct HydroTaskIDs {
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
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro {
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  ReconstructionMethod recon_method;
  Hydro_RSolver rsolver_method;
  EquationOfState *peos;  // chosen EOS

  int nhydro;             // number of hydro variables (5/4 for ideal/isothermal EOS)
  int nscalars;           // number of passive scalars
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables

  DvceArray5D<Real> coarse_u0;  // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;  // primitive variables on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  ShearingBoxCC *psbox_u = nullptr;

  // Object(s) for extra physics (viscosity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables at intermediate step
  DvceArray5D<Real> u_sts0;   // conserved variables at start of STS sweep
  DvceArray5D<Real> u_sts1;   // previous STS stage state
  DvceArray5D<Real> u_sts2;   // second previous STS stage state
  DvceArray5D<Real> u_sts_rhs;  // cached first-stage RKL2 operator contribution
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  Real dtnew;

  bool has_explicit_viscosity = false;
  bool has_explicit_conduction = false;
  bool has_sts_viscosity = false;
  bool has_sts_conduction = false;
  bool has_any_sts_diffusion = false;

  // Global per-face L/R primitive buffers used by the split-kernel flux path
  // (PLM + LLF|HLLC). Shaped (nmb, nvars, nf3, nf2, nf1) where
  //   nf1 = nx1+1,
  //   nf2 = (nx2>1) ? nx2+1 : 1,
  //   nf3 = (nx3>1) ? nx3+1 : 1.
  // The buffer is sized to the max active face range across all three directions,
  // and reused sequentially per direction.  Indexing convention is
  //   wl(m, n, k-ks, j-js, i-is)
  // where for direction d the face-normal axis is face-indexed and the two
  // transverse axes are cell-indexed (origin-shifted by ks/js/is).
  DvceArray5D<Real> wl3d;
  DvceArray5D<Real> wr3d;

  // following used for FOFC
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC
  DvceArray5D<Real> utest;  // scratch array for FOFC

  // following used for the MOOD a-posteriori fallback ("FB") scheme.  Shares the
  // fofc/utest arrays: fofc marks cells flagged (newly demoted) in the current
  // revision iteration, utest holds the candidate update.
  bool use_mood = false;    // flag to enable MOOD fallback
  // Immutable owner flux exchange before the local fallback revisions (NR/SR/GR).
  bool mood_frozen_flux = false;
  MeshBoundaryValuesCC *pbval_mood_prim = nullptr;
  MeshBoundaryValuesCC *pbval_mood_candidate = nullptr;
  DvceArray5D<Real> mood_prim, mood_coarse;
  int mood_max_revs = 1;    // max revision iterations per RK stage
  bool mood_nad_energy;     // include the energy variable in NAD (density always on)
  bool mood_nad_scalars;    // include passive-scalar concentrations in NAD
  bool mood_detect;         // false = run the unlimited base with NO detection
  bool ppm_bc_closure = true;  // one-sided fv4 closure at imposed-state boundaries
                            // (reconstruct/ppm_closure.hpp; false = centred stencil)
                            // (negative control for the cascade; see hydro.cpp)
  bool mood_nad_rcfl;       // scale relative bounds by the cell-local Courant factor
  Real mood_rtol;           // NAD tolerance as a fraction of the selected scale
  Real mood_eps0;           // round-off floor (relative to the violated bound)
  Real mood_eps0_eq;        // well-balanced: round-off floor relative to the equilibrium
                            // (default 0, see hydro.cpp)
  Real mood_atol;           // absolute floor of the NAD tolerance
  bool mood_sed;            // exempt smooth extrema from NAD detection
  bool mood_euler_predictor = false;  // trial state: RK stage combination (default)
                            // or full-dt forward Euler from the stage input (true)
  bool mood_candidate_diffusion = false;  // true: explicit diffusion fluxes BEFORE the
                            // MOOD/FOFC fallback (detector sees the full candidate),
                            // re-added face-locally on revised faces; false: after it
  bool mood_debug_demote = false;  // DEBUG: force demotion of cells with x1 < x1max
  Real mood_debug_demote_x1max = 0.0;  // (validation of revised-face bookkeeping only)
  int n_fb_tiers;           // # of fallback tiers below base scheme (2, or 1 if base=plm)
  DvceArray4D<int> fb_level;    // per-cell cascade level (0=base, 1=plm, 2=dc)
  // relaxed-DMP bounds of the NAD reference quantities over the cell neighbourhood,
  // computed once per stage by separable sweeps (utils/mood_bounds.hpp): per-cell
  // values, min, max and max-scratch, one slot per tested quantity (val doubles as the
  // min-scratch; see MoodNeighborhoodBounds for which pair holds the result)
  DvceArray5D<Real> mood_val, mood_qmn, mood_qmx, mood_tmx;
  DvceArray1D<int> mood_cnt;    // detection counters (atomics; see hydro_mood.cpp)

  // following used for the well-balanced (WB) scheme (hydro_wb.cpp).  With
  // well_balanced=true, u0 holds the PERTURBATION U' = U - U_eq over a static
  // equilibrium supplied by the problem generator (user_eq_func); w0 holds the
  // primitives of the FULL state.  Passive scalars carry no equilibrium (full in u0).
  bool well_balanced = false;
  // build u_eq/w_eq WITHOUT changing the scheme, so hydro_wpert can measure a non-WB run
  // against the same equilibrium (the negative control of the well-balanced tests)
  bool wb_diagnostics = false;
  DvceArray5D<Real> u_eq;         // equilibrium conserved cell averages (incl. ghosts)
  DvceArray5D<Real> w_eq;         // = ConsToPrim(u_eq), so w0 == w_eq bitwise at U'=0
  DvceArray5D<Real> w_pert;       // perturbation primitives W' (+ full scalars)
  DvceFaceFld5D<Real> w_eq_face;  // single-valued equilibrium primitive state on faces
  DvceFaceFld5D<Real> f_eq;       // RS(w_eq_face, w_eq_face) with this Riemann solver
  void (*wb_eq_func)(Mesh *pm) = nullptr;  // pgen's user_eq_func (set in WBInitialize)

  // container to hold names of TaskIDs
  HydroTaskIDs id;

  // functions...
  void AssembleHydroTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus InitRecvParabolic(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendMOODPrimitives(Driver *d, int stage);
  TaskStatus RecvMOODPrimitives(Driver *d, int stage);
  TaskStatus SendMOODCandidate(Driver *d, int stage);
  TaskStatus RecvMOODCandidate(Driver *d, int stage);
  TaskStatus ApplyFallback(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus HydroSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ClearSTSFlux(Driver *d, int stage);
  TaskStatus STSFluxes(Driver *d, int stage);
  TaskStatus STSUpdate(Driver *d, int stage);
  TaskStatus STSRefreshTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <Hydro_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

  // MOOD a-posteriori fallback, templated over Riemann solver (fallback tiers re-solve
  // flagged faces with the same solver as the base scheme)
  template <Hydro_RSolver T>
  void MOODLoop(Driver *d, int stage);

  // well-balanced scheme
  // after the pgen: build equilibrium, shift u0.  Called from the ProblemGenerator
  // constructor, i.e. before Mesh::pgen is assigned, hence the explicit pointer.
  void WBInitialize(ProblemGenerator *ppgen, bool restart);
  void WBSetEquilibrium();           // (re)build u_eq..f_eq (start-up, restart, AMR)
  void WBFormPerturbation();         // w_pert from u0/w0, once per stage before fluxes
  template <Hydro_RSolver T>
  void WBEquilibriumFluxes();        // f_eq = RS(w_eq_face, w_eq_face)
  template <Hydro_RSolver T>
  void WBCalculateFluxes(Driver *d, int stage);  // WB counterpart of CalculateFluxes

 private:
  void AddSelectedDiffusionFluxes(parabolic::DiffusionSelection selection);
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
