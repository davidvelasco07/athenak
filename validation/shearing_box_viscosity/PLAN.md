# Viscosity in the shearing box — plan for `feature/fallback-validation`

Audited at fbd6978e5 (2026-10-19). Compared against upstream `main` (ac3067529) and the
Athena++ tree in `~/athenak/athena`.

## 1. What exists

| Piece | State on this branch |
|---|---|
| `src/diffusion/viscosity.{hpp,cpp}` | Isotropic Navier–Stokes stress in flux form, ideal-gas energy flux `-σ·v`, explicit or RKL2 super-time-stepping (`viscosity_integrator = explicit|sts`). Shared by Hydro and MHD. `nu_aniso` is a no-op. |
| Where it is applied | `Hydro::Fluxes` → `AddSelectedDiffusionFluxes(explicit_only)` right after the Riemann fluxes and **before** `ApplyFallback` (MOOD/FOFC). Same in MHD. |
| `src/shearing_box/` | Orbital advection (3D) + shear-periodic x1 remap (CC and FC), Coriolis/tidal source terms, energy source `qΩ(ρ v'x v'y − Bx By)`, stratified option. No hydro flux remap across the shear boundary (conservation there is at truncation level; pre-existing). |
| Velocity convention | **The stored velocity is the perturbation** `v' = v − v_K`, `v_K = −qΩ x ŷ`. Evidence: source-term coefficients `2Ω` / `(2−q)Ω` (shearing_box_srcterms.cpp:44-45), epicycle IC sets `IM2 = 0` (pgen/tests/shwave.cpp:112), the shear-periodic remap applies no velocity jump. In 2D (x1–x2 = r–z) the azimuth is **x3** and there is no orbital advection (∂/∂z = 0 kills the background advection). |
| Diffusion ↔ shearing box | Nothing. `src/diffusion/` never reads `<shearing_box>`; neither does upstream main. Athena++ avoids the problem by handing the diffusion operator `porb->w_orb` (total velocity); AthenaK has no such array. |
| Tests | `tst/test_suite/sbox/` (hydro/MHD shwave, MRI3D GPU), `tst/test_suite/diffusion/` (1D viscous pulse, STS). Nothing combines the two. |

## 2. What is missing or wrong today

Setting `hydro/nu_iso` in a shearing-box run compiles and runs, but:

1. **Background shear is absent from the stress.** `σ_xy` is built from `∂v'y/∂x` instead of
   `∂v'y/∂x − qΩ`. Lost: the flux term `ρν qΩ` (its divergence is non-zero wherever ρ varies),
   and the stress that enters the energy flux.
2. **Viscous heating of the background shear is absent.** With `E' = ½ρv'² + e` the energy
   equation in the orbital-advection frame is
   `∂E'/∂t + ∇·[(E'+P)v' − σ·v'] = (existing OA sources) − qΩ σ_xy`,
   `σ_xy = ρν(∂v'y/∂x + ∂v'x/∂y − qΩ)`.
   For `v' = 0` this is `ρν q²Ω²` = `(9/4)ρνΩ²` for Keplerian shear, the textbook disc heating
   rate. Today an ideal-gas viscous shearing box does not heat at all. (2D r–z: `σ_xz`.)
3. **MOOD and FOFC drop diffusion fluxes on revised faces** (stated in the header of
   `hydro_mood.cpp`). Harmless-ish elsewhere; in the shearing box the dropped flux contains
   the O(1) constant `ρν qΩ`, so one revised face injects momentum `ρν qΩ/Δx` into its two
   cells. For ν ~ 1e-3 c_s H at 64 cells/H that is ~10% of the pressure flux. Must be fixed
   for this branch's purpose (fv4/6/8 + MOOD).
4. **STS is incompatible with the shearing box.** The `parabolic_stagen` task list has
   `SendU/RecvU` but no `InitRecv(time)`, `SendU_Shr/RecvU_Shr`, so STS stages see
   plain-periodic x1 ghosts, offset by the shear. `STSUpdate` would also skip the energy
   source of item 2. Nothing refuses the combination today.
5. No test exercises viscosity + shearing box.
6. Minor: `Viscosity::AddViscousFluxes` tests `nu_iso` twice; the second should be
   `nu_aniso` (harmless because `AddViscousFluxAniso` is empty). `mood_frozen_flux` is already
   disabled when `pvisc` or `<shearing_box>` exist, so no action there.

## 3. Implementation plan

### Phase 1 — correct explicit viscosity in the shearing box (hydro + MHD)

Files: `src/diffusion/viscosity.{hpp,cpp}`, `src/hydro/hydro_tasks.cpp`, `src/mhd/mhd_tasks.cpp`,
`src/hydro/hydro.cpp`, `src/mhd/mhd.cpp`.

1. `Viscosity` ctor: if `pin->DoesBlockExist("shearing_box")`, read `qshear`, `omega0`;
   store `shear_rate = -qshear*omega0` (= `dv_K/dx`) and `shear_dir` (`IVY` in 3D,
   `IVZ` in 2D r–z, mirroring `three_d || shearing_box_r_phi` in the source terms).
   Otherwise `shear_rate = 0`.
2. `AddViscousFluxIso`: add `shear_rate` to the `(x, shear_dir)` stress component.
   3D: x1-flux `fvy += shear_rate`; x2-flux `fvx += shear_rate`.
   2D r–z: x1-flux `fvz += shear_rate` (no x3 flux in 2D).
   The energy flux keeps its form (`w0` is `v'`, `fv` is now the full stress). Adding
   `+0.0` leaves non-shearing runs bit-identical.
3. New `Viscosity::AddShearingBoxHeating(w0, eos, bdt, u0)` (ideal gas only):
   `u0(IEN) += bdt * (−qΩ) * σ_xy,cc` with cell-centred central differences for
   `∂v'y/∂x`, `∂v'x/∂y` (second order, like the existing `qΩ ρ v'x v'y` source). Call it in
   `Hydro::HydroSrcTerms` and `MHD::MHDSrcTerms` right after `psbox_u->SourceTermsCC` when
   `has_explicit_viscosity`.
4. Startup guards in `hydro.cpp` / `mhd.cpp`: `<shearing_box>` + `viscosity_integrator=sts`
   → fatal error (until Phase 3); `<shearing_box>` + `nu_aniso != 0` → fatal error.
   Fix the `nu_aniso` typo in `AddViscousFluxes`.
5. `NewTimeStep` unchanged (the shear does not alter the diffusion stability limit).

### Phase 2 — MOOD/FOFC keep diffusion fluxes on revised faces

Recommended: **move `AddSelectedDiffusionFluxes(explicit_only)` after `ApplyFallback`** in
`Hydro::Fluxes` and the MHD counterpart. The fallback cascade then works on the purely
hyperbolic flux and the viscous/conductive flux is added once, on every face, at the end.
Consequences:
- Conservative and correct on revised faces; one-line move per module.
- The MOOD detector judges the inviscid candidate. Diffusion only smooths, so this can only
  flag more, never fewer, cells than the full candidate would.
- Runs where MOOD never revised a face, and all runs without diffusion, stay bit-identical
  (gate T3).
- Update the header comment in `hydro_mood.cpp` / `mhd_mood.cpp`.
Alternative (rejected): mask revised faces and re-add diffusion only there. More code, no gain.

### Phase 3 (only if needed) — STS in the shearing box

Add `psbox_u->InitRecv(time)`, `SendU_Shr/RecvU_Shr`, `ClearSend/ClearRecv` to the
`parabolic_stagen` lists (hydro, MHD CC and FC), and the Phase-1 energy source to `STSUpdate`.
Orbital advection itself is not needed in the parabolic stages. Decision: skip unless the
viscous time step binds, i.e. unless the cell Reynolds number `Δx c_s/ν` drops below ~6.

### Phase 4 — tests (`tst/test_suite/sbox/`, inputs in `tst/inputs/`)

- **T1 exact heating gate** (ideal gas, uniform ρ, `v' = 0`, 3D, `nu_iso > 0`, 1 and 4 ranks):
  total momentum stays exactly zero (bitwise) and the history total energy grows by exactly
  `ρ ν q²Ω² V` per unit time (RK integrates a constant source exactly, so the gate is 1e-14).
  Run with `ppmx` and with `fv4 + MOOD + rk3`. Negative control: disable the heating term
  and confirm the test fails.
- **T2 viscous vortical shwave**: the existing JG05 test (`ipert=2`) with `hydro/nu_iso = ν`.
  A single shearing-wave mode is an exact viscous solution with amplitude factor
  `exp(−ν Φ(t))`, `Φ(t) = (kx0²+ky²) t + kx0 qΩ ky t² + (qΩ ky)² t³/3`, `kx(t) = kx0 + qΩ ky t`.
  With the stock box (`nwx=−8, nwy=2`, L=0.5) `Φ(5) ≈ 1.8e4`, so `ν = 5e-5` gives e^−0.9
  by t=5 while `dt_visc ≈ 0.18 ≫ dt_hyp ≈ 2e-3`. Check second-order convergence 32→64 as the
  current test does, with `ppmx/rk2` and `fv4+MOOD/rk3`, 4 MPI ranks (confirm
  `Number of parallel ranks = 4` in the log).
- **T3 non-regression gates** against a reference binary built in an isolated detached
  worktree: (a) shearing-box runs with `nu_iso = 0` bit-identical; (b) `test_diffusion_visc_*`
  unchanged; (c) MOOD runs without diffusion bit-identical after the Phase-2 reorder.
- **T4 MHD smoke**: `mri3d_unstratified` with `nu_iso` (and `eta`) for a few orbits, plus T1
  repeated through MHD with a uniform `Bz`.

### Effort

Phase 1 ≈ 60 lines; Phase 2 ≈ 10 lines + gates; tests ≈ half a day of CPU runs (sequential
launches only). Phase 3 is a separate piece of work.

## 4. Decisions needed

1. Target use: hydro or MHD, unstratified or stratified, ideal or isothermal gas? (Isothermal
   makes the energy terms moot but they get implemented anyway.)
2. Is STS wanted (Phase 3), or is refusing STS + shearing box at startup acceptable?
3. Phase 2 approach: accept that MOOD detects on the inviscid candidate?

## 5. Status

- 2026-10-19 — **Phase 1 implemented** (uncommitted at the time of writing): shear term in
  `AddViscousFluxIso`, `Viscosity::AddShearingBoxHeating` called from the hydro and MHD
  source-term tasks, STS/`nu_aniso` guards in the shearing box, `nu_aniso` typo fixed.
  Test `tst/test_suite/sbox/test_sbox_vischeat_mpicpu.py` + `tst/inputs/hydro_sbox_vischeat.athinput`
  (T1): passes on 4 ranks for ppmx/rk2 and fv4+MOOD/rk3 (momentum exactly 0, heating rate
  to 5e-11 of ΔE, limited by RK round-off); the pre-change binary fails it (ΔE = 0).
  Bit-identity gates vs a HEAD reference built in an isolated worktree: inviscid shearing
  box (T3a) and 1D viscous diffusion (T3b) identical. MHD smoke (mri3d + nu_iso, 30 cycles) runs.
  Not done: Phase 2 (MOOD/FOFC revised faces), Phase 3 (STS), T2 viscous shwave.
- 2026-10-19 — **Phase 2 implemented** (uncommitted): `AddSelectedDiffusionFluxes` now runs
  after `ApplyFallback` in `Hydro::Fluxes` and `MHD::Fluxes`, so MOOD/FOFC-revised faces
  keep the viscous/conduction fluxes; the detector judges the hyperbolic candidate.
  Debug knob `<hydro>/mood_debug_demote_x1max` (off unless > mesh x1min) forces every
  cell with x1 < x1max to be demoted each revision, to exercise the revise pass on a quiet
  state. Gates (B1 = Phase 1 + knob, B2 = Phase 2):
  * forced-demotion heating check (fv4+MOOD/rk3, x1max=0, t=0.2): B1 gives a momentum
    kick max|mom| = 7.1e-4 and heating off by 0.27%; B2 gives momentum exactly 0 and
    heating to 1e-10 of ΔE;
  * Shu–Osher fv4+MOOD without diffusion (2700–3300 demotions per cycle): B1 and B2
    bit-identical (23 output files);
  * uniform viscous fv4+MOOD on 4 ranks (no revisions): B1 and B2 bit-identical;
  * heating pytest passes on B2 for both cases.
  Not done: Phase 3 (STS), T2 viscous shwave (now meaningful with MOOD).
- 2026-10-19 — **T2 viscous vortical shwave run** (`tst/test_suite/sbox/test_sbox_viscshwave_mpicpu.py`,
  figure `figures/viscous_shwave_amplitude_error.png`, script `scripts/plot_viscous_shwave.py`,
  ν = 1e-4, 4 ranks). Findings:
  * 32² is unresolved (4 cells per initial x-wavelength): both schemes damp the wave
    numerically before t = 0.2, so the test uses 64² and 128².
  * Mean |δv_x − analytic|/amp: ppmx/rk2 0.020 (64²) and 0.021 (128²); fv4+MOOD/rk3 0.35
    (64²) and 0.017 (128²). The ~2% floor is compressibility: the IC is the incompressible
    mode, and with c_s = Ω = 1 the true compressible mode differs at O(Ω/(c_s k)). The
    inviscid runs at 128² are off by 0.03 (ppmx) and 0.06 (fv4) vs the inviscid solution.
  * fv4 with `mood_detect=false` is identical to fv4+MOOD at both resolutions: MOOD never
    fires on this problem, so the 64² fv4 undershoot is the unlimited base scheme.
  * **T2 is blind to the −qΩ stress term**: with uniform ρ that term is a constant flux
    with zero divergence (isothermal: no heating). The pre-change HEAD binary gives
    0.0199 vs 0.0202 for the new one at ppmx 64². T2 validates viscous diffusion across
    the sheared, remapped x1 boundary and under orbital advection, not the shear term.
  * Side findings (pre-existing, same on HEAD): the `log` event output aborts under MPI
    with MPI_ERR_TRUNCATE in this setup; the cycle-0 event log reports 14112 density-floor
    hits (ghost planes before the first boundary fill).
  * Proposed dedicated gate for the −qΩ momentum term (T5, not run): ideal gas, uniform P,
    ρ = ρ₀(1 + ε sin kx), v' = 0 (an exact hydrodynamic steady state with HLLC). The
    viscous force per unit mass −νqΩ ∂_x ln ρ ŷ drives epicycles: v'_y = (F/κ) sin κt,
    v'_x = (2ΩF/κ²)(1 − cos κt), κ² = 2(2−q)Ω², so KE_y(t) = ½(sin κt/κ)² (νqΩ)² ∫(∂_xρ)²/ρ dV
    up to O(νk²t) viscous damping of v' and O(v'/c_s) nonlinearity. Second order in Δx via
    the face-averaged density. A binary without the −qΩ term gives KE_y ≡ 0.
- 2026-10-19 — **T5 viscous forced epicycle run** (`tst/test_suite/sbox/test_sbox_viscepicycle_mpicpu.py`,
  input `tst/inputs/hydro_sbox_viscepicycle.athinput`, pgen `shwave.cpp` ipert=5, figure
  `figures/viscous_epicycle_key.png`, script `scripts/plot_viscous_epicycle.py`).
  Correction to the §2 sketch: with c_s k ≈ 16 Ω the response is not a free epicycle but
  a secular v_y = F t in geostrophic balance plus a small acoustic-epicyclic oscillation
  (ω² = κ² + c_s²k²); the test integrates the linear system (Coriolis + pressure + viscous
  damping) by RK4 and compares KE_y(t) = ¼ρ₀V b(t)². Results (ε = 0.1, ν = 1e-4, 4 ranks,
  max relative error for t ≥ 0.5): ppmx/rk2 1.8% (32²), 0.6% (64²); fv4+MOOD/rk3 the same
  to 4 digits. Pre-change HEAD binary (+ the same pgen): KE_y ≡ 0 — the −qΩ stress term
  is what the test measures. Gate = 3%.
- 2026-10-20 — **GPU check staged on Vista** (GH200, `gh` partition). Serial-code audit of the
  diff: every added loop is a `par_for`/`par_for_inner`/`KOKKOS_LAMBDA`; host code added is
  parameter parsing only. Trees rsynced to `$WORK/sbox_visc/{new,ref}` (ref = HEAD fbd6978e5 +
  the shwave ipert=5 generator), built on the login node with `nvhpc-hpcx-cuda13/26.1`,
  `$HOME/athenak-wb/kokkos` (4.7.2, sm_90 wrapper), `-DKokkos_ARCH_HOPPER90=ON`, ~10 min each.
  Job `1084006` (1 node, 45 min): T1 (ppmx, fv4+MOOD, forced demotion, ref), T5 (32²/64², ref),
  T2 (128²), perf A/B new vs ref at 256³ (viscous ppmx, inviscid ppmx, viscous fv4+MOOD) and
  MRI3D 128×256×64 with ν, plus kokkos-tools per-kernel timing. Evaluate with
  `scratchpad/gpu/eval_gpu.py <copied run dir>`.
- 2026-10-20 — **Scheme set changed on request: no PPMX.** Tests, inputs, figures and the GPU
  job (resubmitted as Vista `1084109`) now use PLM/rk2, WENO-Z/rk3 and FV4+MOOD/rk3.
  T5: PLM, WENO-Z and FV4 identical to 4 digits (1.8% at 32², 0.6% at 64²). T2: WENO-Z 0.31/0.016
  amp at 64²/128² (same as FV4); PLM 2.0/1.3 amp (inviscid 3.9 amp at 128²) — numerical
  diffusion dominates ν = 1e-4, so PLM is drawn in the figure but not gated in
  `test_sbox_viscshwave_mpicpu.py`. Open design question raised by the user: MOOD detects on the
  candidate WITHOUT the diffusion flux since Phase 2 (source terms were never in it); at cell
  Reynolds numbers near one this can demote spuriously. Proposed fix: add diffusion before the
  fallback and re-add a face-local diffusive flux on revised faces (hydro, MHD, FOFC), with
  `mood_candidate_diffusion=false` as the negative control. Awaiting the user's decision.
- 2026-10-20 — **Full-flux candidate implemented** (`mood_candidate_diffusion`, default true,
  hydro and MHD). The isotropic viscous and heat fluxes are now per-face device functions
  (`ViscousFluxX1/2/3` in `diffusion/viscosity.hpp`, `HeatFluxIsoX1/2/3` in
  `diffusion/conduction.hpp`) used by the bulk kernels and re-applied inside the MOOD revise
  kernels and the FOFC flux-replacement kernels (hydro and MHD) on every face they rewrite.
  With the flag true the diffusion fluxes are added before `ApplyFallback` (detector sees
  the full candidate); false adds them after (negative control, the Phase 2 ordering).
  Anisotropic/Spitzer conduction, the resistive energy flux and well-balanced FOFC have no
  face-local form and require false. Call sites keep the `flx -= nud*fv` expression shape so
  the refactored bulk kernels are bit-identical to the old ones (gates: T1 PLM, T2 WENO-Z 64²,
  T5 PLM 32², 1D viscous and conductive pulses, Shu–Osher fv4+MOOD — all IDENTICAL after the
  first attempt with separated products differed at 1e-12 from lost FMA contraction).
  Forced-demotion heating gate: momentum exactly 0 and heating to 1e-10 in BOTH modes.
  MHD MOOD/FOFC + viscosity and hydro FOFC + viscosity run in both modes (smoke).
  **Finding (viscous Shu–Osher, fv4+MOOD, 400 cells, ν = 0.01 ⇒ cell Re ≈ 1):**

  | ν | candidate | nad_energy | demotions (logged cycles) |
  |---|---|---|---|
  | 0 | either | on | 13660 |
  | 0.01 | full | on | 33444 |
  | 0.01 | inviscid | on | 3081 |
  | 0.01 | full | off | 3730 |
  | 0.01 | inviscid | off | 6853 |
  | 0.003 | full / inviscid | on | 7702 / 7767 |
  | 0.03 | full / inviscid | on | 5552 / 5553 |

  For density and velocity the full candidate flags FEWER cells (6853 → 3730), as expected
  for a smoothing operator; but the viscous energy flux is not a diffusion of E: viscous
  heating creates genuine new extrema in E/P at the shock, and the energy DMP flags them,
  so with `mood_nad_energy=true` the full candidate demotes 10× more. At cell Re ≳ 3 the two
  modes agree. The default stays true as requested; the energy-DMP interaction is the open
  design question.
- 2026-10-20 — Tests extended: `test_sbox_vischeat_mpicpu.py` now has five cases (PLM/rk2,
  WENO-Z/rk3, FV4+MOOD/rk3, and FV4+MOOD with forced demotion of x < 0 in both candidate
  modes); all three pytests pass on 4 ranks with the final build. Vista: final source
  rebuilt (`new/build_gh`), GPU job resubmitted as **1084226** (gh,gh-dev) with the
  candidate-mode variants, the viscous Shu–Osher demotion counts and the perf A/B.
- 2026-10-20 — **CORRECTION of the Shu–Osher reading above.** The ν = 0.01 divergence between
  the candidate modes was not the energy DMP: at CFL 0.8 (the Shu–Osher input default) the ν = 0.01 runs with detection
  on hit the ENERGY FLOOR (31 hits inviscid-candidate, 352 full-candidate, all before cycle
  165, none in the no-detection, MOOD-free, ν = 0.003 or ν = 0.03 runs), the floor injects
  energy, and the solutions run away (shock at x = 0.93 instead of 0.87, pressure spikes to
  120). With CFL 0.15 no floors fire and the two modes agree: 8749 vs 8743 demotions. The
  `nad_energy=false` numbers were floor-contaminated too. Mechanism: the explicit viscous
  flux at a demoted (first-order) shock face at cell Re ≈ 1 with dt = min(dt_hyp, dt_visc)
  drives e < 0 in the shocked cell; the unlimited FV4 face does not. Rsolver (LLF vs HLLC)
  makes no difference. Open: whether the diffusion time step should combine with the
  hyperbolic one (1/dt = 1/dt_hyp + 1/dt_visc) instead of min(). Figures:
  `figures/maps_shwave_vx.png`, `maps_epicycle.png`, `maps_kick.png`, `profiles_shu_osher.png`
  (ν = 0.01, CFL 0.3, all four cases), `profiles_shu_osher_nu.png` (ν and CFL panel);
  script `scripts/plot_maps.py`, data via `run_maps.sh` (bin slices; tab is 1D-only).
- 2026-10-20 — **Harmonic explicit time step implemented**: `<time>/diffusion_dt_combine =
  min | harmonic` (default min, bit-identical gate passed on Shu–Osher ν=0.01 and T1). With
  harmonic, 1/dt = 1/dt_hyp + Σ 1/dt_diff over the explicit (non-STS) diffusion processes.
  Viscous Shu–Osher, FV4+MOOD, CFL 0.8 (the input default; earlier notes said 0.3 in error):

  | ν | rule | efloor | demotions full / inviscid candidate | dt_last | cycles |
  |---|---|---|---|---|---|
  | 0.01 | min | 352 / 31 | 33444 / 3081 | 4.0e-4 (after runaway) | 816 / 766 |
  | 0.01 | harmonic | 0 | 3685 / 3684 | 3.9e-4 | 1185 |
  | 0.003 | harmonic | 0 | 8394 / 8571 | 6.2e-4 | 750 |
  | 0.03 | harmonic | 0 | 6768 / 6778 | 1.9e-4 | 2439 |
  | 0.01, no detection | min | 0 | 0 | 7.5e-4 | 629 |

  The floors were a time-step effect; with the harmonic rule the two candidate modes agree
  to ≤ 2% at every viscosity. Cost: at cell Re ≈ 1 the harmonic rule roughly halves dt (the
  two limits are comparable), at cell Re ≳ 4 it changes little. Not made the default.
- 2026-10-20 — **Harmonic dt option REMOVED on the user's instruction** (min rule is the AthenaK
  convention; the Shu–Osher floors came from running that input at its CFL 0.8 rather than
  the usual 0.3–0.4). **Default flipped to `mood_candidate_diffusion = false`** (inviscid
  candidate: diffusion added after the fallback) pending the test battery; `true` (full
  candidate with face-local re-add) stays available. Rationale: equal detector behaviour
  where both are stable, cheaper (no per-face re-add), and more robust against energy
  floors at demoted shock faces when the step is large. Running: diffusion + sbox pytest
  battery with the new default, Shu–Osher ν = 0.01 at CFL 0.4/0.3 in both modes, and cost
  measurements (forced-demotion 64³ worst case; LW implode 400² fv4+MOOD).
- 2026-10-20 — **Decision: `mood_candidate_diffusion = false` (inviscid candidate) is the
  default.** Evidence:
  * Shu–Osher ν = 0.01 at the usual CFL 0.4 / 0.3 (min dt rule): no energy floors in either
    mode; demotions 3895 / 3895 (CFL 0.4) and 4841 / 4832 (CFL 0.3), full / inviscid
    candidate. The earlier floors were the CFL 0.8 of that input, as the user suspected.
  * Cost, serial CPU, zone-cycles/s: forced demotion of half the box (64³, every stage,
    worst case) full 5.87e5 vs inviscid 6.36e5 (−8%); LW implode 400² fv4+MOOD with ~10%
    of cells demoted per cycle: 1.918e6 vs 1.925e6 (−0.4%); 1D Shu–Osher: within run-to-run
    noise (0.26–0.28 s). The full candidate pays a per-revised-face recomputation; the
    inviscid candidate pays nothing extra.
  * Test battery with the new default (CPU): resist, ambipolar, sts controller, sts mpicpu,
    hydro + MHD shwave, vischeat (5 cases), viscshwave, viscepicycle all PASS.
    `test_diffusion_visc_cpu` and `test_diffusion_conduct_cpu` FAIL, but they fail
    identically on the pre-change HEAD binary (error 4.9e-8 vs threshold 1.5e-10, not
    converging 64→128, byte-identical `-errs.dat` from both binaries): a pre-existing
    broken test, unrelated to this work.
  * GPU job 1084226 (still queued) carries the full-candidate-default build B5; it exercises
    both modes explicitly in the forced-demotion runs and in p4/p4b, so its results apply.
- 2026-10-20 — **GPU results, Vista GH200, job 1084226** (build B5: full-flux candidate code,
  default then true; both modes run explicitly). All gates pass on the device: T1 momentum
  exactly 0 and heating to 3e-12 of ΔE for PLM, WENO-Z, FV4+MOOD and both forced-demotion
  modes (pre-change binary: ΔE = 0); T5 1.8% (32²) / 0.6% (64²) for all three schemes
  (pre-change: KE_y = 0); T2 at 128²: WENO-Z 0.016 amp, FV4 0.017 amp, PLM 1.3 amp. The
  viscous Shu–Osher at CFL 0.8 reproduces the CPU floor-driven counts (33444 / 3081).
  Performance, zone-cycles/s on one GH200, 40 cycles, new / ref: viscous PLM 256³
  5.36e8 / 5.36e8; inviscid PLM 6.98e8 / 7.00e8; viscous WENO-Z 2.91e8 / 2.91e8; viscous
  FV4+MOOD 2.59e8 / 2.54e8 (candidate true 2.593e8, false 2.591e8); MRI3D 128×256×64 viscous
  WENO-Z 1.85e8 / 1.02e8 (sub-second runs; re-measured in job 2, below). kokkos-tools
  (viscous PLM 256³): visc1/2/3 are device `[for]` kernels at 42–48 ms per 40 calls (ref
  48–51 ms, old team-scratch form), `visc_sbox_heat` 11.6 ms per 40 calls = 1.7% of the
  step; total 0.679 s new vs 0.682 s ref. Follow-up perf job submitted (p5 MRI 300 cycles
  alternating binaries ×2, PLM control 200 cycles, forced-demotion 256³ both modes).
- 2026-10-20 — **GPU perf re-measure, job 1084694** (300 cycles, alternating binaries): MRI3D
  viscous WENO-Z new / ref = 1.673e8 / 1.609e8 and 1.923e8 / 1.911e8 zone-cycles/s — equal
  within noise; the 1.8× of job 1084226 was a sub-second-run artefact (first runs of a
  sequence are also ~15% slower: warm-up). Hydro viscous PLM 256³: 5.29e8 / 5.32e8.
  Forced demotion of half the box, FV4+MOOD 256³, 100 cycles: full candidate 1.506e8 vs
  inviscid candidate 1.646e8 (−9%, same as the CPU worst case). Roofline job submitted
  (Nsight Compute per-kernel FP64/FP32 FLOPs + DRAM bytes, MRI3D viscous, PLM/rk2,
  WENO-Z/rk3, FV4+MOOD/rk3); analysis script `scripts/roofline_mri3d.py`.
- 2026-10-20 — **Apollo front node (A100-PCIe-40GB, CUDA 12.4, Kokkos 4.7.2 from
  `~/athenak-rp-pr/kokkos`, AMPERE80, no MPI, gcc-toolset-12 host compiler; build 5 min).**
  Viscous MRI3D 128×256×64, 100 cycles, zone-cycles/s: PLM/rk2 1.26e8, WENO-Z/rk3 8.03e7,
  FV4+MOOD/rk3 2.73e7 (nghost 7). Nsight Compute cannot run there: `ERR_NVGPUCTRPERM`,
  the driver restricts GPU performance counters to admins (needs
  `NVreg_RestrictProfilingToAdminUsers=0` from the Apollo admins). The ncu roofline job on
  Vista (1084708) stays queued; its MRI input was missing `mood = false`, fixed in place
  before it runs. Trap: an `mhd/mood=true` override needs the parameter in the input file.
- 2026-10-20 — **Per-kernel GPU time on Apollo's A100 (nsys trace, 20 cycles, viscous MRI3D
  128×256×64, 2 MeshBlocks)**, `figures/kernel_breakdown_mri3d_a100.png`,
  `scripts/plot_kernel_breakdown.py` (groups nvcc-demangled Kokkos lambdas by the calling
  method; par_for helpers are skipped). GPU kernel time per cycle: PLM/rk2 18.0 ms,
  WENO-Z/rk3 27.5 ms, FV4+MOOD/rk3 84.7 ms. Shares: PLM — FC boundary pack/unpack 29%,
  reconstruction 11%, Riemann 10%, viscosity 8%, RK update 5%, CornerE 4%; WENO-Z —
  reconstruction 22%, FC pack/unpack 25%, Riemann 10%, viscosity 7%; FV4+MOOD — MOODLoop
  54%, FC pack/unpack 20%, recon+Riemann 8%, viscosity 2%. Viscosity is 7–8% of the step
  for the base schemes. Roofline (FLOP/byte) still needs the Vista ncu job.
- 2026-10-20 — **Why the MOOD loop is slow (A100 trace, viscous MRI3D FV4+MOOD/rk3, 85 ms/cycle):**
  MOODLoop = 45 ms/cycle, of which the detection `parallel_reduce` is 32 ms/cycle: 5.4 ms
  per call, 120 calls per 20 cycles = 2 revisions × 3 stages. Per revision: candidate 0.62 ms,
  PAD c2p 0.2 ms, detection 5.4 ms, three revise sweeps 0.44–0.53 ms each (full-face sweeps
  with a per-face flag test). The whole hyperbolic flux computation (recon + Riemann) is
  6.4 ms/cycle, so detection alone is 5× the fluxes and the loop 7×. Per cell the detector
  reads a 27-point neighbourhood for every tested quantity (up to two passes), the
  candidate arrays and the candidate B, in a flattened 1D RangePolicy over the
  halo-extended range. `mood_touched_only=true` changed nothing (2.69e7 vs 2.75e7
  zone-cycles/s; detection still 120 full calls) because `touched_only = mood_touched_only
  && !ho_pad` is disabled by the PAD mode in use; `mood_max_revs=1` gives 3.35e7 (+22%).

- 2026-10-20 — **MOOD detection: relaxed-DMP bounds once per stage, separable sweeps**
  (uncommitted). The reference quantities the NAD test bounds (stage-input w0/w_pert/u0,
  |B|/B components, |v|/v components, scalar concentrations) do not change during the
  revision loop, yet `mood_detect`/`mmood_detect` re-scanned their 27-cell min/max per
  tested quantity, per cell, per revision (MHD default comps/comps = 8 passes = 216 neighbour
  reads per cell per revision). New `src/utils/mood_bounds.hpp::MoodNeighborhoodBounds()`:
  per-cell values are gathered once per stage into `mood_val` (kernels `mood_bnd_val` /
  `mmood_bnd_val`, one slot per tested quantity, WB full-state copies as extra slots) and the
  3^d-box min/max is built by three 3-point sweeps (x1 → x2 → x3, kernels `mood_bnd_x[123]`,
  every slot in one kernel, so four launches per stage however many quantities are tested;
  `mood_val` doubles as the min scratch and `mood_tmx` is the max scratch, so the result sits in
  (`mood_qmn`,`mood_qmx`) in 1D/3D and in (`mood_val`,`mood_tmx`) in 2D — the callers take the
  pair the helper names), computed over the widest detection halo + 1 cell. The detection kernels then read `bmn/bmx(m,slot,k,j,i)`;
  PAD, SED (lazy, on the reference arrays) and all tolerances are untouched. In MHD the bounds
  are built under `initialize`, so the per-revision calls of the frozen-EMF path reuse them.
  `fmin`/`fmax` are exact and association-free (a NaN neighbour is ignored unless the whole box
  is NaN, in both forms; a signed zero cannot change a decision), so the detector decisions
  are bitwise identical by construction — and gated as such. Memory: 4·nslots cell arrays
  (hydro default 8, MHD default isothermal 28) allocated with `use_mood` in `hydro.cpp`/`mhd.cpp`.
  (A first version swept one slot per kernel, 3·nslots+1 launches per stage; the A100 profile
  showed the launches, not the bandwidth, eating the hydro gain, hence the fused form. Both
  versions passed the full gate battery.)
  **Gates** (`scratchpad/sep/gates.py`; reference = isolated build of the pre-change tree,
  `scratchpad/sep/ref_src`, assembled from the worktree plus Apollo's byte-identical
  `~/athenak-sbox-visc/src` copies of the seven touched files; every output file — `tab`/`bin`
  dumps, `hst`, event `log` — compared byte for byte, final cycle/time equal):

  | case | ranks | files | MOOD events (ref log) | result |
  |---|---|---|---|---|
  | Shu–Osher fv4+MOOD/rk3, ν=0, CFL 0.3 | 1 | 3 | 29 246 | identical |
  | Shu–Osher ν=0.01 (inviscid candidate) | 1 | 3 | 4 832 | identical |
  | LW implode 200², fv4+MOOD/rk3 | 1 / 4 | 3 / 2 | 22 024 954 | identical |
  | scalar advection, 1D, nscalars=1 (sine; detector silent) | 1 | 3 | 0 | identical |
  | Orszag–Tang 200², fv4+MOOD/rk3, comps/comps, ct_contact | 1 / 4 | 3 / 2 | 2 810 654 | identical |
  | Orszag–Tang, |v| mode, no energy pass, nscalars=1 | 1 | 3 | 2 424 452 | identical |
  | viscous MRI3D 64×128×16, fv4+MOOD/rk3, isothermal | 1 / 4 | 3 / 2 | 2 033 729 | identical |
  | negative control: ref with `mood_detect=false` vs ref | 1 | 3 | 0 | 2 of 3 files differ |

  Serial wall times on the Mac moved by +0…+15% (OT 31.8→36.6 s, implode 44.7→47.1 s,
  MRI 12.4→13.4 s; measured while other jobs shared the machine — see the clean re-timing
  below): on a CPU the old 27-point loop is served from L1, while the sweeps stream the
  bound arrays. Performance is a GPU question (A100 below); CPU runs are the identity gates.
  Clean serial re-timing (idle Mac, interleaved ×2): OT 31.4/32.7 → 36.7/37.1 s (+15%),
  implode 45.8/45.9 → 49.0/48.4 s (+6%), MRI 12.6/12.5 → 13.5/13.4 s (+7%).
  Second battery (all identical, detector firing): OT mag/mag + `emf=uct_hlld`/hlld (2 721 880
  events), square-wave scalar advection nscalars=2 (16 705), WB hydro pulse (42 711, 5 files
  incl. `wpert`), WB MHD `uct_hlld` comps (66 860) and mag/mag (61 616), WB MHD on 4 ranks.
  WB+MHD+`uct_hlld`+fv4 needs `nghost=12` with `mood_max_revs=2` (startup check).
  **GPU bit identity (Apollo A100, viscous MRI3D 64×128×16, 60 cycles, 4 `bin` dumps):** md5
  identical ref vs new. **A100 kernel sums (nsys, 50 cycles, 128×256×64 FV4+MOOD/rk3; both GPUs
  were shared with another user's jobs at 100% utilisation, so absolute times are inflated
  ~1.6× and wall-clock throughput is meaningless — only same-condition ratios are quoted):**
  detection `parallel_reduce` 66.8 → 27.5 ms/cycle (−59%), the 7 bound slots cost 3 sweeps ×
  ~1.4–1.5 ms/cycle (21 small kernels per cycle) plus the gather, total GPU kernel time 137 →
  102 ms/cycle (−26%). Contended wall-clock: MRI +4%, hydro shearing box (uniform state,
  2 slots) −6%. Clean GH200 numbers: Vista job (`sep_new`/`sep_ref` builds, `job_sep.slurm`).
  **Fused-slot version (final), same A100 under the same contention, nsys 50 cycles:** MRI3D
  detection reduce 48.1 → 21.0 ms/cycle (−56%), gather + three sweeps ≈ 1.1 + 1.4 + 1.4 + 1.2 ms
  (7 slots), total GPU kernel time 108.5 → 85.7 ms/cycle (−21%); hydro shearing box (2 slots,
  uniform state, one revision per stage so only 3 detect calls/cycle): detection 1.59 → 0.83
  ms/cycle but the four bound kernels add 1.3 ms, total 18.06 → 18.62 (+3%) — a wash where
  detection was already cheap, a net gain once a second revision runs. Mac serial re-timing of
  the fused version: implode +3%, OT +6%, OT mag/uct −2%, MRI ±0, WB ±0.
  **Clean GH200 numbers (Vista job 1085486, idle GPU, separable-bounds binaries `sep_new` vs
  pre-change `sep_ref`, 200 cycles, zone-cycles/s):** viscous MRI3D 128×256×64 FV4+MOOD/rk3
  4.36/4.38/4.37e7 → 4.93/4.93/4.94e7 (**+13% end-to-end**); same with `mood_nad_b=mag`,
  `mood_nad_v=mag` 5.28e7 → 5.70e7 (+8%); hydro shearing box, uniform state (2 slots, one
  revision) 1.970e8 → 1.928e8 (−2%). Context, new binary: PLM/rk2 2.10e8, WENO-Z/rk3 1.33e8, so
  FV4+MOOD is 4.3× PLM per zone-cycle here (was 4.8×). GPU gate on GH200: 4 `bin` dumps
  md5-identical. The job's nsys section hit the 30-min limit (no GH200 kernel sums; the
  `uct_hlld` gate variant needs `nghost=12` and did not run).
- 2026-10-21 — **Detection counters: parallel_reduce → par_for + atomics** (uncommitted). The
  detection kernels were multi-value `Kokkos::parallel_reduce`s over a flattened 1D range: hydro
  summed 2 integers, MHD 5 (`ndemoted` incl. halo, `ndemoted_int`, `nnad`, `npad`, `ndc`).
  Audit of what each count is for:
  - `ndemoted` (incl. the ghost halo): the loop-exit test (`== 0`) of the non-frozen path —
    needed, but only as "any cell demoted", not as a number.
  - `ndemoted_int` (interior only): feeds `ecounter.nmood`, the `mood` column of the event
    `log` output — the one diagnostic a production run normally keeps.
  - `nnad`, `npad`, `ndc`: used only by the frozen-EMF path (`mood_frozen_emf`): the
    pending/PAD consensus bits exchanged between ranks (mhd_tasks.cpp) and the
    `mood_diagnostics` printout. Dead in the default (non-frozen) path, yet counted there.
  Change: both kernels are plain `par_for`s over `(m,k,j,i)` (no flattened-index divisions),
  flagged cells bump device counters `mood_cnt` with `Kokkos::atomic_inc`, and the host reads
  the small view back once per revision. MHD counts `nnad/npad/ndc` only when `frozen`.
  Integer atomics are exact, so every count and decision is unchanged (gated below). Contention
  is proportional to the number of flagged cells, i.e. negligible unless most of the box is
  demoted.
  **Result:** gates 16/16 byte-identical again (same battery, negative control differs; Mac
  serial: implode ±0, OT +5%, MRI −2% vs the pre-change binary). A100 (shared GPU, nsys 50
  cycles, MRI3D 128×256×64 FV4+MOOD/rk3): detection kernel 20.95 → 20.43 ms/cycle, i.e. the
  five-value reduction epilogue was NOT where the time went — the remaining ~3.4 ms per call
  (≈2 ms on an idle GPU) is the per-cell body: 8 detection passes reading 16 bound values plus
  the candidate state, PAD/NaN checks and the rcfl scale (~50 double reads per cell, within
  ~2.5× of the bandwidth bound). Hydro (uniform box): 0.83 → 0.69 ms/cycle. Wall-clock during
  a quiet stretch of the shared GPU: pre-change 2.75e7 zone-cycles/s (= the idle value measured
  earlier today) vs 3.09e7 with bounds+atomics, +12.6% end-to-end for this viscous MRI setup.
  Kept anyway: the kernel is simpler (4D `par_for`, no flattened-index divisions), the
  frozen-EMF-only counters cost nothing on the default path, and the hydro kernel is slightly
  cheaper. Trap: Apollo's default `nsys` (CUDA 12.4, 2023.4.4) twice produced only a `.qdstrm`
  whose import failed for this run; `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys` profiled and
  reported it fine, and `nsys stats` silently skips when an (empty) output CSV already exists.
  Next lever if the detector still matters: fewer passes (`mood_nad_v=off`/`mag`, `mood_nad_b=mag`
  halve the pass count) or a pass-split/SoA layout of the detection body.
- 2026-10-21 — **MHD detection passes as independent threads** (uncommitted; user: the passes
  "should be able to run in parallel to each other"). `mmood_detect` is now three kernels:
  `mmood_detect_prep` (per cell: masked/tier filter, PAD+NaN danger, rcfl band scale →
  `mood_stat`, `mood_crtol`), `mmood_detect_pass` (per (cell, pass) over density, energy, B,
  v and scalar passes: candidate value, precomputed bounds, band, SED → `mood_nadf` via
  `atomic_store`), `mmood_detect_combine` (per cell: OR, demote, count). The sequential loop
  stopped at the first failing pass; evaluating the rest changes nothing, so the trouble set
  and counts are identical — gates 16/16 byte-identical, GPU gate md5-identical. Hydro keeps
  the fused kernel (two passes).
  **A100 (shared GPU, same-condition ratios, nsys 50 cycles, MRI3D 128×256×64 FV4+MOOD/rk3):**
  detection 55.9 ms/cycle (reference) → 9.1 + 1.0 + 0.9 = 11.1 ms/cycle for the three kernels
  (−80%; the fused bounds-only kernel was −56%), total GPU kernel time 120.4 → 80.2 ms/cycle
  (−33%). The MOOD loop is now candidate 3.7, pass 9.1, revise 2.9+2.8+2.4, PAD 1.1 ms/cycle.
  **Cost on a serial CPU:** the per-pass kernel re-reads the cell state once per pass and the
  early exit is gone: Mac OT 31.5 → 39.3 s (+25%), OT mag/uct +5%, MRI +9%, implode (hydro)
  ±0. CPU runs are the identity gates, GPU the production target, so the split is the default;
  an execution-space switch (fused on Serial/OpenMP) is a one-line follow-up if CPU MOOD runs
  ever matter.
  **Follow-up the same day (user: on a CPU the pass index should cost nothing).** The CPU
  loss was not the parallel index but what came with it: with `pass` as the second index of
  the 5-index `par_for` each pass re-streamed the block; moving it inside the row loop
  `(m,k,j,pass,i)` did not help (OT still +27%), because each (cell, pass) thread redoes the
  cell's fixed work — index decode, status/band reads, candidate density, B average — seven
  times, and the three-kernel structure pays the decode three times per cell. Final form: the
  detection logic lives once in `struct MoodNadPassCtx` (`Prep`, `CandidateDensity`, `Test`,
  `Combine` in mhd_mood.cpp); GPU backends (`KOKKOS_ENABLE_CUDA/HIP/SYCL`) launch prep,
  per-(cell,pass) and combine kernels, CPU backends one per-cell kernel that calls the three in
  sequence with the pass loop stopping at the first failure. Gates 16/16 identical at every
  step (five rebuilds). Mac serial vs pre-change: MRI ±0, OT +8% (bounds+atomics alone were
  +5%), WB/implode ±0. A100 (shared, same-session ratios): detection 55.4 → 9.9 + 0.5 + 0.9 ms
  per cycle, total 119.9 → 80.4 (−33%); the inner-pass index order costs the pass kernel
  ~8% vs pass-outer (row of 130 cells is not a multiple of the warp), <1% of the step.
  Clean GH200 comparison of pass-outer vs pass-inner vs reference: Vista job 1087205.
  **Vista job 1087205 (idle GH200, 200 cycles ×3 interleaved, zone-cycles/s):** viscous MRI3D
  128×256×64 FV4+MOOD/rk3: reference 4.37/4.39/4.37e7, pass-outer split 5.35/5.35/5.34e7
  (**+22%**), pass-inner split 5.28/5.26/5.26e7 (+20.5%); with mag/mag detection (4 passes)
  5.27e7 → 5.87e7 / 5.84e7 (+11%). Gate: 4 `bin` dumps md5-identical across the three
  binaries. Since the CPU path no longer depends on the GPU index order, the GPU launch is
  back to the pass-outer 5-index `par_for` (final). Bounds alone had given +13% here, so the
  pass-parallel detector adds another +9 points end to end. (The job's nsys section again hit
  the time limit; the A100 kernel sums stand in for the breakdown.)
