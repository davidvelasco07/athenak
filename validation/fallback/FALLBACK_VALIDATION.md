> Historical results: this report records the former global-scale detector. The current code supports local relative/rcfl NAD and the validated frozen-EMF method where supported; see [current validation](../frozen_emf_halo/neighbor_exchange/README.md). Rerun this suite before treating its recorded results as current.

# MOOD Fallback Validation Report

_Generated 2026-10-07 03:02 UTC_
_Commit `516bb4d9a`_

## Summary

- Records analyzed: **932**
- FV4 linear-wave zero demotions: **True**
- FV6 linear-wave zero demotions: **True**
- FV8 linear-wave zero demotions: **True**
- All runs completed (rc=0): **False**

## Method

We compare unlimited central finite-volume face interpolation at 4th, 6th and 8th order (`reconstruct=fv4|fv6|fv8`) with a-posteriori MOOD fallback (RK3) against PLM+RK2, WENOZ+RK3, and TENO-5+RK3. Detection uses NAD (`mood_nad_scale=gcfl`, `mood_rtol=1e-5`; `mood_nad_v=off` for MHD *linear waves* only -- see the note below) plus PAD/NaN. Passive-scalar concentrations are included in the relaxed-DMP test, so scalar discontinuities can trigger fallback even on a uniform-density kinematic background. NAD demotions are floored at PLM.

**Each scheme carries its own a-posteriori protection.** MOOD and FOFC are mutually exclusive in the code, and unlimited FV4/FV6/FV8 *require* `mood=true`, so the like-for-like comparison gives the unlimited schemes the MOOD cascade and gives PLM / WENO-Z / TENO first-order flux correction (`fofc=true`). Benchmarking MOOD against unprotected competitors would overstate it: on the current sheet, TENO fails outright without FOFC and survives with it. `fofc` is a per-scheme property of the manifest so it cannot silently differ per case. MOOD is unsupported with BH excision. Newtonian MHD uses `emf=uct_hlld` for all schemes. Relativistic MHD defaults to `ct_contact` and may opt into `uct_hll` per case (`uct_hlld` needs `rsolver=hlld`, which SR/GR reject); the `blast_grmhd` / `blast_grmhd_uct` pair runs both.

**`returncode == 0` is not evidence of success.** A run whose state has been wiped still prints "Terminating on time limit": on the current sheet, unprotected TENO reported rc=0 with an all-NaN final dump. Every stress run is therefore health-gated on the EOS floor/failure counters and the timestep history (`dt_collapse` when $\Delta t_{\min} \le 0$ or $< 10^{-6}\Delta t_{\max}$).

The report is organized by physics. Each section covers **1D / 2D / 3D** linear waves (convergence mosaics with shared axes), plus 1D shocks and available multi-D stress cases.

## Hydrodynamics

### 1D linear waves

![Linear-wave mosaic hydro 1d](figures/lwave_hydro_1d_mosaic.png)

![FV4 demotions hydro](figures/lwave_nmood_hydro.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going acoustic | 256 | 5.497982e-12 | 0.04 | 0 | yes |
| fv4_fb | transverse shear z | 256 | 1.81143e-13 | 2.11 | 0 | yes |
| fv4_fb | right-going acoustic | 256 | 5.497744e-12 | 0.04 | 0 | yes |
| fv6_fb | left-going acoustic | 256 | 5.49782e-12 | 0.01 | 0 | yes |
| fv6_fb | transverse shear z | 256 | 1.737696e-13 | -0.66 | 0 | yes |
| fv6_fb | right-going acoustic | 256 | 5.497812e-12 | 0.01 | 0 | yes |
| fv8_fb | left-going acoustic | 256 | 5.498006e-12 | 0.01 | 0 | yes |
| fv8_fb | transverse shear z | 256 | 1.734038e-13 | -0.68 | 0 | yes |
| fv8_fb | right-going acoustic | 256 | 5.498035e-12 | 0.01 | 0 | yes |
| plm | left-going acoustic | 256 | 1.228226e-09 | 2.02 | 0 | yes |
| plm | transverse shear z | 256 | 5.517613e-10 | 2.03 | 0 | yes |
| plm | right-going acoustic | 256 | 1.228227e-09 | 2.02 | 0 | yes |
| teno | left-going acoustic | 256 | 5.497806e-12 | 0.02 | 0 | yes |
| teno | transverse shear z | 256 | 1.813326e-13 | -0.56 | 0 | yes |
| teno | right-going acoustic | 256 | 5.497714e-12 | 0.02 | 0 | yes |
| wenoz | left-going acoustic | 256 | 5.497799e-12 | 0.02 | 0 | yes |
| wenoz | transverse shear z | 256 | 1.861057e-13 | -0.60 | 0 | yes |
| wenoz | right-going acoustic | 256 | 5.497752e-12 | 0.02 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic hydro 2d](figures/lwave_hydro_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going acoustic | 128 | 2.838811e-11 | 3.60 | 0 | yes |
| fv4_fb | right-going acoustic | 128 | 2.83881e-11 | 3.60 | 0 | yes |
| fv6_fb | left-going acoustic | 128 | 1.365538e-11 | 2.84 | 0 | yes |
| fv6_fb | right-going acoustic | 128 | 1.365543e-11 | 2.84 | 0 | yes |
| fv8_fb | left-going acoustic | 128 | 1.36365e-11 | 2.84 | 0 | yes |
| fv8_fb | right-going acoustic | 128 | 1.363651e-11 | 2.84 | 0 | yes |
| plm | left-going acoustic | 128 | 1.802754e-08 | 1.84 | 0 | yes |
| plm | right-going acoustic | 128 | 1.802754e-08 | 1.84 | 0 | yes |
| teno | left-going acoustic | 128 | 1.462497e-11 | 3.19 | 0 | yes |
| teno | right-going acoustic | 128 | 1.462486e-11 | 3.19 | 0 | yes |
| wenoz | left-going acoustic | 128 | 1.462485e-11 | 3.19 | 0 | yes |
| wenoz | right-going acoustic | 128 | 1.462489e-11 | 3.19 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic hydro 3d](figures/lwave_hydro_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going acoustic | 64 | 4.202751e-10 | 3.78 | 0 | yes |
| fv4_fb | right-going acoustic | 64 | 4.202752e-10 | 3.78 | 0 | yes |
| fv6_fb | left-going acoustic | 64 | 2.363084e-10 | 2.97 | 0 | yes |
| fv6_fb | right-going acoustic | 64 | 2.363084e-10 | 2.97 | 0 | yes |
| fv8_fb | left-going acoustic | 64 | 2.365988e-10 | 2.98 | 0 | yes |
| fv8_fb | right-going acoustic | 64 | 2.365988e-10 | 2.98 | 0 | yes |
| plm | left-going acoustic | 64 | 7.883148e-08 | 1.49 | 0 | yes |
| plm | right-going acoustic | 64 | 7.883148e-08 | 1.49 | 0 | yes |
| teno | left-going acoustic | 64 | 2.894962e-10 | 3.60 | 0 | yes |
| teno | right-going acoustic | 64 | 2.894962e-10 | 3.60 | 0 | yes |
| wenoz | left-going acoustic | 64 | 2.894976e-10 | 3.62 | 0 | yes |
| wenoz | right-going acoustic | 64 | 2.894977e-10 | 3.62 | 0 | yes |

### Shocks

![shock_shu_osher_dens](figures/shock_shu_osher_dens.png)

![shock_sod_dens](figures/shock_sod_dens.png)

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| shu_osher | 1d | fv4_fb | 256 | 0 | 27576 | 6.583750009536743 |
| shu_osher | 1d | fv4_fb | 512 | 0 | 67195 | 8.699394702911377 |
| shu_osher | 1d | fv4_fb | 1024 | 0 | 150202 | 12.121439456939697 |
| shu_osher | 1d | fv6_fb | 256 | 0 | 30597 | 2.9991588592529297 |
| shu_osher | 1d | fv6_fb | 512 | 0 | 75992 | 6.212106943130493 |
| shu_osher | 1d | fv6_fb | 1024 | 0 | 179179 | 14.689929246902466 |
| shu_osher | 1d | fv8_fb | 256 | 0 | 34559 | 2.95504093170166 |
| shu_osher | 1d | fv8_fb | 512 | 0 | 78911 | 7.0734336376190186 |
| shu_osher | 1d | fv8_fb | 1024 | 0 | 174677 | 14.276941061019897 |
| shu_osher | 1d | plm | 256 | 0 | 0 | 4.85801362991333 |
| shu_osher | 1d | plm | 512 | 0 | 0 | 5.158158779144287 |
| shu_osher | 1d | plm | 1024 | 0 | 0 | 6.00898551940918 |
| shu_osher | 1d | teno | 256 | 0 | 0 | 5.015064477920532 |
| shu_osher | 1d | teno | 512 | 0 | 0 | 5.576136589050293 |
| shu_osher | 1d | teno | 1024 | 0 | 0 | 6.819214582443237 |
| shu_osher | 1d | wenoz | 256 | 0 | 0 | 4.977221250534058 |
| shu_osher | 1d | wenoz | 512 | 0 | 0 | 5.588316202163696 |
| shu_osher | 1d | wenoz | 1024 | 0 | 0 | 6.855767488479614 |
| sod | 1d | fv4_fb | 256 | 0 | 22045 | 6.614264249801636 |
| sod | 1d | fv4_fb | 512 | 0 | 48894 | 6.407215118408203 |
| sod | 1d | fv4_fb | 1024 | 0 | 105257 | 8.302880048751831 |
| sod | 1d | fv6_fb | 256 | 0 | 24379 | 1.628000020980835 |
| sod | 1d | fv6_fb | 512 | 0 | 54241 | 3.1809728145599365 |
| sod | 1d | fv6_fb | 1024 | 0 | 122256 | 7.341597080230713 |
| sod | 1d | fv8_fb | 256 | 0 | 25960 | 1.5586950778961182 |
| sod | 1d | fv8_fb | 512 | 0 | 56740 | 3.2086760997772217 |
| sod | 1d | fv8_fb | 1024 | 0 | 126665 | 6.708297967910767 |
| sod | 1d | plm | 256 | 0 | 0 | 4.611562728881836 |
| sod | 1d | plm | 512 | 0 | 0 | 4.9116575717926025 |
| sod | 1d | plm | 1024 | 0 | 0 | 5.254113435745239 |
| sod | 1d | teno | 256 | 0 | 0 | 4.681830883026123 |
| sod | 1d | teno | 512 | 0 | 0 | 4.949263095855713 |
| sod | 1d | teno | 1024 | 0 | 0 | 5.4772748947143555 |
| sod | 1d | wenoz | 256 | 0 | 0 | 4.689580202102661 |
| sod | 1d | wenoz | 512 | 0 | 0 | 4.921968936920166 |
| sod | 1d | wenoz | 1024 | 0 | 0 | 5.66727352142334 |

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| implode_hydro | 2d | fv6_fb | clean | 0 | 182542306 | 345.72102880477905 |
| fu_jet_m80 | 2d | fv6_fb | **non-finite** | 44 | 218845191 | 58.73589515686035 |
| blast_hydro | 2d | fv6_fb | clean | 0 | 22852700 | 144.23295998573303 |
| blast_hydro | 2d | teno | **rc!=0** | 0 | 0 | 3.127929210662842 |
| ha_jet | 2d | fv6_fb | **floored** | 12 | 10222131 | 205.2997190952301 |
| slotted_cyl | 2d | fv8_fb | clean | 0 | 2377665 | 43.80655097961426 |
| blast_hydro | 2d | wenoz | **rc!=0** | 0 | 0 | 3.156033754348755 |
| blast_hydro | 2d | fv8_fb | clean | 0 | 23668572 | 131.51153302192688 |
| blast_hydro | 2d | plm | **rc!=0** | 0 | 0 | 3.1566178798675537 |
| ha_jet | 2d | fv8_fb | **floored** | 6 | 10656490 | 208.3671441078186 |
| slotted_cyl | 2d | fv6_fb | clean | 0 | 2590992 | 43.63608407974243 |
| implode_hydro | 2d | fv8_fb | clean | 0 | 189683550 | 352.6362769603729 |
| fu_jet_m80 | 2d | fv8_fb | **floored** | 410 | 11131376 | 106.75207614898682 |
| blast_hydro | 2d | fv4_fb | clean | 0 | 4646336 | 5.930562496185303 |

![blast_hydro stress mosaic](figures/stress_blast_hydro_mosaic.png)

![fu_jet_m80 stress mosaic](figures/stress_fu_jet_m80_mosaic.png)

![ha_jet stress mosaic](figures/stress_ha_jet_mosaic.png)

![implode_hydro stress mosaic](figures/stress_implode_hydro_mosaic.png)

The slotted-cylinder test advects a bounded passive scalar through one full rotation on a $200^2$ mesh. Scalar-aware NAD confines fallback to the discontinuity and suppresses unlimited-FV ringing.

![slotted_cyl stress mosaic](figures/stress_slotted_cyl_mosaic.png)

### NAD tolerance study

Relaxed-DMP tolerance sweeps keep the base scheme, grid, Riemann solver, and integrator fixed. The Liska--Wendroff implosion probes interacting shocks and contacts; the slotted disk isolates the passive-scalar detector.

#### Liska--Wendroff implosion

![Liska--Wendroff implosion snapshot mosaic](figures/tolerance_implode_hydro_mosaic.png)

![Liska--Wendroff implosion demotion curve](figures/tolerance_implode_hydro_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 62205225 | 23.972003698349 |
| 1e-03 | 200 | 0 | 159450725 | 24.19423484802246 |
| 1e-04 | 200 | 0 | 174105603 | 23.430827140808105 |
| 1e-05 | 200 | 0 | 175861689 | 23.174686908721924 |
| 1e-06 | 200 | 0 | 176301945 | 24.914732456207275 |

#### Slotted disk

![Slotted disk snapshot mosaic](figures/tolerance_slotted_cyl_mosaic.png)

![Slotted disk demotion curve](figures/tolerance_slotted_cyl_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 61122 | 5.782958030700684 |
| 1e-03 | 200 | 0 | 650314 | 4.287592649459839 |
| 1e-04 | 200 | 0 | 1410463 | 4.295799970626831 |
| 1e-05 | 200 | 0 | 2440854 | 4.31870698928833 |
| 1e-06 | 200 | 0 | 3129071 | 4.2786924839019775 |

## Magnetohydrodynamics

### 1D linear waves

![Linear-wave mosaic mhd 1d](figures/lwave_mhd_1d_mosaic.png)

![FV4 demotions mhd](figures/lwave_nmood_mhd.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going fast magnetosonic | 256 | 1.009816e-11 | 2.01 | 0 | yes |
| fv4_fb | left-going Alfvén | 256 | 3.103081e-13 | 1.84 | 0 | yes |
| fv4_fb | left-going slow magnetosonic | 256 | 3.846367e-12 | 0.04 | 0 | yes |
| fv4_fb | entropy/contact | 256 | 5.838054e-13 | 1.02 | 0 | yes |
| fv4_fb | right-going slow magnetosonic | 256 | 3.847842e-12 | 0.03 | 0 | yes |
| fv4_fb | right-going Alfvén | 256 | 3.113863e-13 | 1.83 | 0 | yes |
| fv4_fb | right-going fast magnetosonic | 256 | 1.009795e-11 | 2.01 | 0 | yes |
| fv6_fb | left-going fast magnetosonic | 256 | 1.009669e-11 | 2.00 | 0 | yes |
| fv6_fb | left-going Alfvén | 256 | 3.063597e-13 | -0.76 | 0 | yes |
| fv6_fb | left-going slow magnetosonic | 256 | 3.845113e-12 | -0.01 | 0 | yes |
| fv6_fb | entropy/contact | 256 | 5.826382e-13 | -1.01 | 0 | yes |
| fv6_fb | right-going slow magnetosonic | 256 | 3.844561e-12 | -0.01 | 0 | yes |
| fv6_fb | right-going Alfvén | 256 | 3.056019e-13 | -0.77 | 0 | yes |
| fv6_fb | right-going fast magnetosonic | 256 | 1.009744e-11 | 2.00 | 0 | yes |
| fv8_fb | left-going fast magnetosonic | 256 | 1.009731e-11 | 2.00 | 0 | yes |
| fv8_fb | left-going Alfvén | 256 | 3.076449e-13 | -0.77 | 0 | yes |
| fv8_fb | left-going slow magnetosonic | 256 | 3.845886e-12 | -0.01 | 0 | yes |
| fv8_fb | entropy/contact | 256 | 5.821558e-13 | -1.00 | 0 | yes |
| fv8_fb | right-going slow magnetosonic | 256 | 3.843513e-12 | -0.01 | 0 | yes |
| fv8_fb | right-going Alfvén | 256 | 3.0761e-13 | -0.78 | 0 | yes |
| fv8_fb | right-going fast magnetosonic | 256 | 1.009686e-11 | 2.00 | 0 | yes |
| plm | left-going fast magnetosonic | 256 | 1.465083e-09 | 2.02 | 0 | yes |
| plm | left-going Alfvén | 256 | 7.800711e-10 | 2.03 | 0 | yes |
| plm | left-going slow magnetosonic | 256 | 9.08071e-10 | 2.05 | 0 | yes |
| plm | entropy/contact | 256 | 8.156857e-10 | 2.04 | 0 | yes |
| plm | right-going slow magnetosonic | 256 | 9.080752e-10 | 2.05 | 0 | yes |
| plm | right-going Alfvén | 256 | 7.800713e-10 | 2.03 | 0 | yes |
| plm | right-going fast magnetosonic | 256 | 1.465099e-09 | 2.02 | 0 | yes |
| teno | left-going fast magnetosonic | 256 | 1.009767e-11 | 2.00 | 0 | yes |
| teno | left-going Alfvén | 256 | 3.091141e-13 | -0.67 | 0 | yes |
| teno | left-going slow magnetosonic | 256 | 3.846018e-12 | -0.01 | 0 | yes |
| teno | entropy/contact | 256 | 5.550204e-13 | -0.99 | 0 | yes |
| teno | right-going slow magnetosonic | 256 | 3.846053e-12 | -0.01 | 0 | yes |
| teno | right-going Alfvén | 256 | 3.092114e-13 | -0.66 | 0 | yes |
| teno | right-going fast magnetosonic | 256 | 1.009824e-11 | 2.00 | 0 | yes |
| wenoz | left-going fast magnetosonic | 256 | 1.009766e-11 | 2.00 | 0 | yes |
| wenoz | left-going Alfvén | 256 | 3.076772e-13 | -0.66 | 0 | yes |
| wenoz | left-going slow magnetosonic | 256 | 3.846868e-12 | -0.01 | 0 | yes |
| wenoz | entropy/contact | 256 | 5.518025e-13 | -0.99 | 0 | yes |
| wenoz | right-going slow magnetosonic | 256 | 3.84438e-12 | -0.01 | 0 | yes |
| wenoz | right-going Alfvén | 256 | 3.081676e-13 | -0.65 | 0 | yes |
| wenoz | right-going fast magnetosonic | 256 | 1.009757e-11 | 2.00 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic mhd 2d](figures/lwave_mhd_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going fast magnetosonic | 128 | 4.075089e-10 | 2.18 | 0 | yes |
| fv4_fb | right-going fast magnetosonic | 128 | 4.075085e-10 | 2.18 | 0 | yes |
| fv6_fb | left-going fast magnetosonic | 128 | 3.660084e-10 | 2.01 | 0 | yes |
| fv6_fb | right-going fast magnetosonic | 128 | 3.660084e-10 | 2.01 | 0 | yes |
| fv8_fb | left-going fast magnetosonic | 128 | 3.660463e-10 | 2.01 | 0 | yes |
| fv8_fb | right-going fast magnetosonic | 128 | 3.660463e-10 | 2.01 | 0 | yes |
| plm | left-going fast magnetosonic | 128 | 2.15449e-08 | 1.81 | 0 | yes |
| plm | right-going fast magnetosonic | 128 | 2.154531e-08 | 1.81 | 0 | yes |
| teno | left-going fast magnetosonic | 128 | 3.659035e-10 | 2.01 | 0 | yes |
| teno | right-going fast magnetosonic | 128 | 3.659035e-10 | 2.01 | 0 | yes |
| wenoz | left-going fast magnetosonic | 128 | 3.659035e-10 | 2.01 | 0 | yes |
| wenoz | right-going fast magnetosonic | 128 | 3.659036e-10 | 2.01 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic mhd 3d](figures/lwave_mhd_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going fast magnetosonic | 64 | 1.811123e-09 | 2.70 | 0 | yes |
| fv4_fb | right-going fast magnetosonic | 64 | 1.811123e-09 | 2.70 | 0 | yes |
| fv6_fb | left-going fast magnetosonic | 64 | 1.403449e-09 | 2.06 | 0 | yes |
| fv6_fb | right-going fast magnetosonic | 64 | 1.403449e-09 | 2.06 | 0 | yes |
| fv8_fb | left-going fast magnetosonic | 64 | 1.40647e-09 | 2.10 | 0 | yes |
| fv8_fb | right-going fast magnetosonic | 64 | 1.40647e-09 | 2.10 | 0 | yes |
| plm | left-going fast magnetosonic | 64 | 8.882641e-08 | 1.43 | 0 | yes |
| plm | right-going fast magnetosonic | 64 | 8.882653e-08 | 1.43 | 0 | yes |
| teno | left-going fast magnetosonic | 64 | 1.410097e-09 | 2.19 | 0 | yes |
| teno | right-going fast magnetosonic | 64 | 1.410097e-09 | 2.19 | 0 | yes |
| wenoz | left-going fast magnetosonic | 64 | 1.410101e-09 | 2.19 | 0 | yes |
| wenoz | right-going fast magnetosonic | 64 | 1.410101e-09 | 2.19 | 0 | yes |

### Shocks

![shock_bw_dens](figures/shock_bw_dens.png)

![shock_rj2a_dens](figures/shock_rj2a_dens.png)

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| bw | 1d | fv4_fb | 256 | 0 | 15620 | 5.455307722091675 |
| bw | 1d | fv4_fb | 512 | 0 | 46976 | 6.407660722732544 |
| bw | 1d | fv4_fb | 1024 | 0 | 139286 | 8.186083555221558 |
| bw | 1d | fv6_fb | 256 | 0 | 20716 | 1.0440690517425537 |
| bw | 1d | fv6_fb | 512 | 0 | 64711 | 2.2514889240264893 |
| bw | 1d | fv6_fb | 1024 | 0 | 202510 | 5.148961067199707 |
| bw | 1d | fv8_fb | 256 | 0 | 21156 | 1.0830469131469727 |
| bw | 1d | fv8_fb | 512 | 0 | 67262 | 2.3356029987335205 |
| bw | 1d | fv8_fb | 1024 | 0 | 213808 | 5.349076986312866 |
| bw | 1d | plm | 256 | 0 | 0 | 4.831498384475708 |
| bw | 1d | plm | 512 | 0 | 0 | 5.09998893737793 |
| bw | 1d | plm | 1024 | 0 | 0 | 5.740323543548584 |
| bw | 1d | teno | 256 | 0 | 0 | 4.895114898681641 |
| bw | 1d | teno | 512 | 0 | 0 | 5.3945393562316895 |
| bw | 1d | teno | 1024 | 0 | 0 | 6.336426734924316 |
| bw | 1d | wenoz | 256 | 0 | 0 | 5.056381940841675 |
| bw | 1d | wenoz | 512 | 0 | 0 | 5.467697620391846 |
| bw | 1d | wenoz | 1024 | 0 | 0 | 6.279923915863037 |
| rj2a | 1d | fv4_fb | 256 | 0 | 44934 | 6.194762706756592 |
| rj2a | 1d | fv4_fb | 512 | 0 | 160045 | 7.311729192733765 |
| rj2a | 1d | fv4_fb | 1024 | 0 | 530082 | 10.259937524795532 |
| rj2a | 1d | fv6_fb | 256 | 0 | 68344 | 1.6532340049743652 |
| rj2a | 1d | fv6_fb | 512 | 0 | 213337 | 3.3467071056365967 |
| rj2a | 1d | fv6_fb | 1024 | 0 | 649935 | 8.165112018585205 |
| rj2a | 1d | fv8_fb | 256 | 0 | 71530 | 1.574977159500122 |
| rj2a | 1d | fv8_fb | 512 | 0 | 232156 | 3.6153247356414795 |
| rj2a | 1d | fv8_fb | 1024 | 0 | 699868 | 7.934898853302002 |
| rj2a | 1d | plm | 256 | 0 | 0 | 4.898343086242676 |
| rj2a | 1d | plm | 512 | 0 | 0 | 5.323077917098999 |
| rj2a | 1d | plm | 1024 | 0 | 0 | 6.295633792877197 |
| rj2a | 1d | teno | 256 | 0 | 0 | 5.098629713058472 |
| rj2a | 1d | teno | 512 | 0 | 0 | 5.725203275680542 |
| rj2a | 1d | teno | 1024 | 0 | 0 | 7.114076614379883 |
| rj2a | 1d | wenoz | 256 | 0 | 0 | 5.156071424484253 |
| rj2a | 1d | wenoz | 512 | 0 | 0 | 5.787648439407349 |
| rj2a | 1d | wenoz | 1024 | 0 | 0 | 7.087417125701904 |

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| jet_n600 | 2d | fv4_fb | **floored** | 550 | 425945941 | 96.45877623558044 |
| blast_mhd | 2d | wenoz | clean | 0 | 0 | 4.424925327301025 |
| jet | 2d | teno | **floored** | 2002 | 0 | 17.087567806243896 |
| current_sheet_n1024 | 2d | fv6_fb | clean | 0 | 2980295958 | 26354.41700410843 |
| rotor | 2d | fv8_fb | **floored** | 32 | 19992510 | 259.824697971344 |
| current_sheet_n512 | 2d | fv8_fb | **floored** | 316 | 552930752 | 3765.5608942508698 |
| orszag_tang | 2d | fv6_fb | **floored** | 69 | 91084393 | 1487.0636160373688 |
| blast_mhd | 2d | plm | clean | 0 | 0 | 4.108715295791626 |
| rotor | 2d | teno | clean | 0 | 0 | 4.676168203353882 |
| jet | 2d | fv6_fb | **non-finite** | 3009 | 75406908 | 641.204922914505 |
| kh_rr22_n256 | 2d | fv8_fb | clean | 0 | 558562997 | 7630.115083932877 |
| rotor | 2d | plm | clean | 0 | 0 | 4.2501161098480225 |
| current_sheet | 2d | fv6_fb | **floored** | 1190 | 112173446 | 566.8557329177856 |
| blast_mhd | 2d | teno | clean | 0 | 0 | 4.338674068450928 |
| blast_mhd | 2d | fv8_fb | clean | 0 | 102503498 | 979.7400200366974 |
| jet | 2d | plm | **floored** | 28186 | 0 | 15.856359481811523 |
| kh_rr22 | 2d | teno | clean | 0 | 0 | 27.219162940979004 |
| kh_rr22 | 2d | wenoz | clean | 0 | 0 | 27.560834646224976 |
| blast_mhd | 2d | fv4_fb | clean | 0 | 2865612 | 5.676389217376709 |
| kh_rr22 | 2d | fv6_fb | clean | 0 | 89137357 | 1126.128788948059 |
| current_sheet | 2d | plm | clean | 0 | 0 | 7.914682388305664 |
| blast_mhd_lowbeta | 2d | fv6_fb | **floored** | 112117 | 90283532 | 377.54402208328247 |
| rotor | 2d | wenoz | clean | 0 | 0 | 4.7452497482299805 |
| mhd_jet | 2d | fv8_fb | **non-finite** | 5279 | 34489270 | 24.737060070037842 |
| jet_n600 | 2d | fv8_fb | **floored** | 403 | 192374395 | 6125.73609995842 |
| current_sheet | 2d | teno | **dt collapse** | 57853 | 0 | 434.2576529979706 |
| current_sheet_n1024 | 2d | wenoz | clean | 0 | 0 | 80.21520495414734 |
| mhd_jet_revs4 | 2d | fv6_fb | **non-finite** | 97393 | 40646287 | 41.8792622089386 |
| rotor | 2d | fv4_fb | clean | 0 | 18877768 | 6.933132171630859 |
| current_sheet_n512 | 2d | fv4_fb | clean | 0 | 445302218 | 45.406437397003174 |
| current_sheet | 2d | fv4_fb | **floored** | 10 | 93507990 | 22.261358976364136 |
| current_sheet_n1024 | 2d | teno | **dt collapse** | 1111726 | 0 | 63.33787560462952 |
| jet | 2d | fv4_fb | clean | 0 | 59372029 | 31.7730495929718 |
| kh_rr22 | 2d | fv8_fb | clean | 0 | 89986183 | 1160.5801091194153 |
| current_sheet | 2d | wenoz | **floored** | 642 | 0 | 10.626406908035278 |
| blast_mhd_lowbeta | 2d | fv8_fb | **floored** | 152708 | 86170694 | 382.24168610572815 |
| jet_n600 | 2d | plm | clean | 0 | 0 | 26.293771743774414 |
| jet_n600 | 2d | wenoz | **dt collapse** | 98098 | 0 | 46.97315859794617 |
| jet_n600 | 2d | teno | **floored** | 1560 | 0 | 42.42583513259888 |
| mhd_jet | 2d | fv6_fb | **non-finite** | 97393 | 40646287 | 44.80704689025879 |
| current_sheet_n512 | 2d | plm | clean | 0 | 0 | 14.827360153198242 |
| jet_n600 | 2d | fv6_fb | **floored** | 3355 | 190672457 | 6664.968143224716 |
| current_sheet_n1024 | 2d | fv4_fb | clean | 0 | 2153430653 | 172.02649068832397 |
| current_sheet_n1024 | 2d | plm | clean | 0 | 0 | 46.333914279937744 |
| mhd_jet_revs4 | 2d | fv8_fb | **non-finite** | 5279 | 34489270 | 23.332226753234863 |
| jet | 2d | wenoz | **floored** | 5466 | 0 | 17.281517028808594 |
| orszag_tang | 2d | fv8_fb | **floored** | 54 | 93285611 | 1425.3590211868286 |
| kh_rr22 | 2d | plm | clean | 0 | 0 | 19.056479692459106 |
| current_sheet_n512 | 2d | fv6_fb | **floored** | 6 | 537693519 | 3399.263725042343 |
| rotor | 2d | fv6_fb | **floored** | 148 | 20056797 | 273.52879524230957 |
| current_sheet_n1024 | 2d | fv8_fb | clean | 0 | 3015150945 | 27923.13966679573 |
| kh_rr22 | 2d | fv4_fb | clean | 0 | 73353124 | 57.68581295013428 |
| kh_rr22_n256 | 2d | fv6_fb | clean | 0 | 542358161 | 7427.124178886414 |
| current_sheet_n512 | 2d | wenoz | clean | 0 | 0 | 22.701947450637817 |
| jet | 2d | fv8_fb | **non-finite** | 33600 | 62831203 | 607.3037672042847 |
| current_sheet_n512 | 2d | teno | **dt collapse** | 79200 | 0 | 21.408766984939575 |
| current_sheet | 2d | fv8_fb | **floored** | 254 | 117941562 | 591.8266229629517 |
| blast_mhd | 2d | fv6_fb | clean | 0 | 102062017 | 946.5443339347839 |

![blast_mhd stress mosaic](figures/stress_blast_mhd_mosaic.png)

![blast_mhd |B| mosaic](figures/stress_blast_mhd_bmag_mosaic.png)

![blast_mhd_lowbeta stress mosaic](figures/stress_blast_mhd_lowbeta_mosaic.png)

![blast_mhd_lowbeta |B| mosaic](figures/stress_blast_mhd_lowbeta_bmag_mosaic.png)

The double Harris sheet is seeded with the pgen tearing perturbation and run to $t=5$, a few Alfv\'en times, so plasmoid chains have formed on both sheets.

**This case previously reported only $t=1$.** The reason every high-order scheme collapsed before plasmoid time was an initial-condition defect, not a limiter failure: `current_sheet.cpp` wrote *primitive* variables into the *conserved* array -- momenta received $v$ rather than $\rho v$, and the total energy omitted both $\rho v^2/2$ and $B^2/2$. In the stock deck that makes $P=(\gamma-1)(17.86-50)=-12.9$ throughout the background, so **95.8\% of cells sat on the pressure floor at $t=0$** (measured: $e_{\rm int}$ floor count 62816/65536, median $\beta=5.9\times10^{-39}$). The initial state was a cold, zero-pressure, magnetically dominated medium, not a Harris equilibrium.

The pgen now sets $\rho v$ and $E=p/(\gamma-1)+\rho v^2/2+B^2/2$, with the pressure taken from transverse force balance, $p(x)=p_0 n_g + (b_0^2/2)(\mathrm{sech}^2\frac{x+x_{01}}{a_0}+\mathrm{sech}^2\frac{x-x_{01}}{a_0})$, which is an exact equilibrium for any $d_0$ and $\gamma$ and reduces to the uniform-temperature form when $d_0=\gamma b_0^2/2$. Two gates confirm it: zero floored cells at $t=0$ with $p+B^2/2$ constant to output precision, and a static run ($\epsilon_b=\epsilon_v=0$, ideal MHD) whose residual $|v|_{\max}$ converges $8.1\times10^{-2}\to2.5\times10^{-2}\to8.0\times10^{-3}$ over $N=128/256/512$.

With the corrected IC every scheme reaches $t=5$ **when given its own a-posteriori protection** -- MOOD for unlimited FV4/FV6/FV8, FOFC for the others (the two are mutually exclusive, and unlimited FV schemes require `mood=true`). The difference is cost, not survival: FV4 reaches $t=5$ with no EOS floor events at all, while TENO needs 2173 FOFC demotions, $\sim10^5$ floor hits, 48\% more cycles and half the timestep. FOFC never fires for PLM or WENO-Z.

![current_sheet stress mosaic](figures/stress_current_sheet_mosaic.png)

![current_sheet |B| mosaic](figures/stress_current_sheet_bmag_mosaic.png)

![current_sheet_n1024 stress mosaic](figures/stress_current_sheet_n1024_mosaic.png)

![current_sheet_n1024 |B| mosaic](figures/stress_current_sheet_n1024_bmag_mosaic.png)

![current_sheet_n512 stress mosaic](figures/stress_current_sheet_n512_mosaic.png)

![current_sheet_n512 |B| mosaic](figures/stress_current_sheet_n512_bmag_mosaic.png)

![jet stress mosaic](figures/stress_jet_mosaic.png)

![jet |B| mosaic](figures/stress_jet_bmag_mosaic.png)

![jet_n600 stress mosaic](figures/stress_jet_n600_mosaic.png)

![jet_n600 |B| mosaic](figures/stress_jet_n600_bmag_mosaic.png)

Magnetized Kelvin--Helmholtz of Rueda-Ram\'irez, Hindenlang, Chan \& Gassner (2022, arXiv:2203.06062) section 5.2, replacing the Lecoanet MHD KH: the perturbation is a single deterministic mode (no random seed to reproduce across schemes or decompositions) and the shear layer $y_0=1/20$ is resolved. The field is tilted in the $xz$ plane, so the toroidal $B_z$ is nonzero and the run is genuinely pseudo-2D.

![kh_rr22 stress mosaic](figures/stress_kh_rr22_mosaic.png)

![kh_rr22 |B| mosaic](figures/stress_kh_rr22_bmag_mosaic.png)

![kh_rr22_n256 stress mosaic](figures/stress_kh_rr22_n256_mosaic.png)

![kh_rr22_n256 |B| mosaic](figures/stress_kh_rr22_n256_bmag_mosaic.png)

![mhd_jet stress mosaic](figures/stress_mhd_jet_mosaic.png)

![mhd_jet |B| mosaic](figures/stress_mhd_jet_bmag_mosaic.png)

![mhd_jet_revs4 stress mosaic](figures/stress_mhd_jet_revs4_mosaic.png)

![mhd_jet_revs4 |B| mosaic](figures/stress_mhd_jet_revs4_bmag_mosaic.png)

The Orszag--Tang vortex is shown at $t=1$ on the classic $400^2$ Athena mesh.

![orszag_tang stress mosaic](figures/stress_orszag_tang_mosaic.png)

![orszag_tang |B| mosaic](figures/stress_orszag_tang_bmag_mosaic.png)

![rotor stress mosaic](figures/stress_rotor_mosaic.png)

![rotor |B| mosaic](figures/stress_rotor_bmag_mosaic.png)

### 3D ringing (Apollo)

Latest UCT-HLLD ringing mosaic from Apollo `feature/fallback` @ `d60ab73` (256³).

## SR hydrodynamics

### 1D linear waves

_No results yet for this dimension._

### 2D linear waves

_No results yet for this dimension._

### 3D linear waves

_No results yet for this dimension._

### Shocks

_No shock cases for this physics._

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| kh_srhydro | 2d | fv8_fb | clean | 0 | 7709131 | 113.93709182739258 |
| blast_srhydro | 2d | fv8_fb | clean | 0 | 473592 | 9.710411071777344 |
| blast_srhydro | 2d | fv6_fb | clean | 0 | 423096 | 9.177867889404297 |
| kh_srhydro | 2d | fv6_fb | clean | 0 | 7448713 | 112.9156928062439 |

![blast_srhydro stress mosaic](figures/stress_blast_srhydro_mosaic.png)

![kh_srhydro stress mosaic](figures/stress_kh_srhydro_mosaic.png)

## SR MHD

### 1D linear waves

_No results yet for this dimension._

### 2D linear waves

_No results yet for this dimension._

### 3D linear waves

_No results yet for this dimension._

### Shocks

_No shock cases for this physics._

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| blast_srmhd | 2d | fv6_fb | **floored** | 798 | 1396089 | 10.567103862762451 |
| blast_srmhd_uct | 2d | fv6_fb | **floored** | 1844 | 1291018 | 13.308178901672363 |
| blast_srmhd_uct | 2d | fv8_fb | **floored** | 2072 | 1291034 | 13.894895076751709 |
| blast_srmhd | 2d | fv8_fb | **floored** | 689 | 1409532 | 10.62426495552063 |

![blast_srmhd stress mosaic](figures/stress_blast_srmhd_mosaic.png)

![blast_srmhd |B| mosaic](figures/stress_blast_srmhd_bmag_mosaic.png)

![blast_srmhd_uct stress mosaic](figures/stress_blast_srmhd_uct_mosaic.png)

![blast_srmhd_uct |B| mosaic](figures/stress_blast_srmhd_uct_bmag_mosaic.png)

## GR hydrodynamics (Minkowski)

### 1D linear waves

![Linear-wave mosaic grhydro 1d](figures/lwave_grhydro_1d_mosaic.png)

![FV4 demotions grhydro](figures/lwave_nmood_grhydro.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going acoustic | 256 | 1.26141e-11 | 0.16 | 0 | yes |
| fv4_fb | transverse shear z | 256 | 3.321304e-12 | 1.10 | 0 | yes |
| fv4_fb | right-going acoustic | 256 | 9.194108e-12 | -0.20 | 0 | yes |
| fv6_fb | left-going acoustic | 256 | 1.251686e-11 | -0.00 | 0 | yes |
| fv6_fb | transverse shear z | 256 | 3.294177e-12 | -0.99 | 0 | yes |
| fv6_fb | right-going acoustic | 256 | 9.270931e-12 | -0.00 | 0 | yes |
| fv8_fb | left-going acoustic | 256 | 1.251801e-11 | -0.00 | 0 | yes |
| fv8_fb | transverse shear z | 256 | 3.296206e-12 | -0.99 | 0 | yes |
| fv8_fb | right-going acoustic | 256 | 9.268668e-12 | -0.00 | 0 | yes |
| plm | left-going acoustic | 256 | 1.099402e-09 | 2.05 | 0 | yes |
| plm | transverse shear z | 256 | 9.014784e-09 | 2.09 | 0 | yes |
| plm | right-going acoustic | 256 | 8.684197e-10 | 2.03 | 0 | yes |
| teno | left-going acoustic | 256 | 1.251668e-11 | -0.00 | 0 | yes |
| teno | transverse shear z | 256 | 3.298609e-12 | -0.88 | 0 | yes |
| teno | right-going acoustic | 256 | 9.272032e-12 | -0.00 | 0 | yes |
| wenoz | left-going acoustic | 256 | 1.251813e-11 | -0.00 | 0 | yes |
| wenoz | transverse shear z | 256 | 3.297545e-12 | -0.88 | 0 | yes |
| wenoz | right-going acoustic | 256 | 9.271985e-12 | -0.00 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic grhydro 2d](figures/lwave_grhydro_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going acoustic | 128 | 1.528824e-11 | 4.05 | 0 | yes |
| fv4_fb | right-going acoustic | 128 | 1.426472e-11 | 4.64 | 0 | yes |
| fv6_fb | left-going acoustic | 128 | 1.480681e-12 | 2.78 | 0 | yes |
| fv6_fb | right-going acoustic | 128 | 8.57987e-12 | -0.50 | 0 | yes |
| fv8_fb | left-going acoustic | 128 | 1.499175e-12 | 2.77 | 0 | yes |
| fv8_fb | right-going acoustic | 128 | 8.627286e-12 | 0.05 | 0 | yes |
| plm | left-going acoustic | 128 | 1.351972e-08 | 1.86 | 0 | yes |
| plm | right-going acoustic | 128 | 1.623479e-08 | 1.84 | 0 | yes |
| teno | left-going acoustic | 128 | 2.215331e-12 | 4.05 | 0 | yes |
| teno | right-going acoustic | 128 | 8.710016e-12 | 2.25 | 0 | yes |
| wenoz | left-going acoustic | 128 | 2.215358e-12 | 4.05 | 0 | yes |
| wenoz | right-going acoustic | 128 | 8.71163e-12 | 2.25 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic grhydro 3d](figures/lwave_grhydro_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going acoustic | 64 | 3.043753e-10 | 3.99 | 0 | yes |
| fv4_fb | right-going acoustic | 64 | 3.683656e-10 | 3.94 | 0 | yes |
| fv6_fb | left-going acoustic | 64 | 2.452066e-11 | 3.29 | 0 | yes |
| fv6_fb | right-going acoustic | 64 | 1.394979e-11 | 3.84 | 0 | yes |
| fv8_fb | left-going acoustic | 64 | 2.478509e-11 | 2.98 | 0 | yes |
| fv8_fb | right-going acoustic | 64 | 1.126728e-11 | 1.95 | 0 | yes |
| plm | left-going acoustic | 64 | 6.227724e-08 | 1.47 | 0 | yes |
| plm | right-going acoustic | 64 | 6.453648e-08 | 1.55 | 0 | yes |
| teno | left-going acoustic | 64 | 6.745318e-11 | 4.50 | 0 | yes |
| teno | right-going acoustic | 64 | 5.922496e-11 | 4.83 | 0 | yes |
| wenoz | left-going acoustic | 64 | 6.745222e-11 | 4.53 | 0 | yes |
| wenoz | right-going acoustic | 64 | 5.922592e-11 | 4.86 | 0 | yes |

### Shocks

![shock_mb2_gr_dens](figures/shock_mb2_gr_dens.png)

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| mb2_gr | 1d | fv4_fb | 256 | 0 | 13397 | 5.236825466156006 |
| mb2_gr | 1d | fv4_fb | 512 | 0 | 25142 | 6.132622480392456 |
| mb2_gr | 1d | fv4_fb | 1024 | 0 | 44057 | 7.560010671615601 |
| mb2_gr | 1d | fv6_fb | 256 | 0 | 10970 | 1.4876806735992432 |
| mb2_gr | 1d | fv6_fb | 512 | 0 | 18842 | 3.189343214035034 |
| mb2_gr | 1d | fv6_fb | 1024 | 0 | 33205 | 8.998086929321289 |
| mb2_gr | 1d | fv8_fb | 256 | 0 | 9913 | 1.3129429817199707 |
| mb2_gr | 1d | fv8_fb | 512 | 0 | 16357 | 3.2180240154266357 |
| mb2_gr | 1d | fv8_fb | 1024 | 0 | 27336 | 9.4079749584198 |
| mb2_gr | 1d | plm | 256 | 0 | 0 | 5.976938486099243 |
| mb2_gr | 1d | plm | 512 | 0 | 0 | 5.3372015953063965 |
| mb2_gr | 1d | plm | 1024 | 0 | 0 | 6.497895002365112 |
| mb2_gr | 1d | teno | 256 | 0 | 0 | 5.054182767868042 |
| mb2_gr | 1d | teno | 512 | 0 | 0 | 5.706254482269287 |
| mb2_gr | 1d | teno | 1024 | 0 | 0 | 6.717432022094727 |
| mb2_gr | 1d | wenoz | 256 | 0 | 0 | 5.083803415298462 |
| mb2_gr | 1d | wenoz | 512 | 0 | 0 | 5.588774919509888 |
| mb2_gr | 1d | wenoz | 1024 | 0 | 0 | 6.645209789276123 |

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| kh_grhydro | 2d | fv8_fb | clean | 0 | 9992357 | 197.65495610237122 |
| blast_grhydro_3d | 3d | fv6_fb | clean | 0 | 1624576 | 181.8482151031494 |
| blast_grhydro | 2d | fv8_fb | clean | 0 | 519772 | 16.928126335144043 |
| blast_grhydro | 2d | fv6_fb | clean | 0 | 461764 | 16.649996995925903 |
| kh_grhydro | 2d | fv6_fb | clean | 0 | 9648967 | 193.98315811157227 |
| blast_grhydro_3d | 3d | fv8_fb | clean | 0 | 1636240 | 193.8033049106598 |

![blast_grhydro stress mosaic](figures/stress_blast_grhydro_mosaic.png)

![blast_grhydro_3d stress mosaic](figures/stress_blast_grhydro_3d_mosaic.png)

![kh_grhydro stress mosaic](figures/stress_kh_grhydro_mosaic.png)

## GRMHD (Minkowski)

### 1D linear waves

![Linear-wave mosaic grmhd 1d](figures/lwave_grmhd_1d_mosaic.png)

![FV4 demotions grmhd](figures/lwave_nmood_grmhd.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going fast magnetosonic | 256 | 5.183723e-11 | 1.22 | 0 | yes |
| fv4_fb | left-going Alfvén | 256 | 8.10974e-11 | 1.79 | 0 | yes |
| fv4_fb | left-going slow magnetosonic | 256 | 7.7931e-12 | 0.86 | 0 | yes |
| fv4_fb | entropy/contact | 256 | 6.377738e-12 | -0.99 | 0 | yes |
| fv4_fb | right-going slow magnetosonic | 256 | 8.398216e-12 | 1.78 | 0 | yes |
| fv4_fb | right-going Alfvén | 256 | 4.455176e-11 | 2.16 | 0 | yes |
| fv4_fb | right-going fast magnetosonic | 256 | 3.284181e-11 | 1.72 | 0 | yes |
| fv6_fb | left-going fast magnetosonic | 256 | 5.207392e-11 | 1.23 | 0 | yes |
| fv6_fb | left-going Alfvén | 256 | 8.121116e-11 | 1.80 | 0 | yes |
| fv6_fb | left-going slow magnetosonic | 256 | 7.738174e-12 | 0.81 | 0 | yes |
| fv6_fb | entropy/contact | 256 | 6.381126e-12 | -0.99 | 0 | yes |
| fv6_fb | right-going slow magnetosonic | 256 | 8.447548e-12 | 1.80 | 0 | yes |
| fv6_fb | right-going Alfvén | 256 | 4.453584e-11 | 2.17 | 0 | yes |
| fv6_fb | right-going fast magnetosonic | 256 | 3.290413e-11 | 1.72 | 0 | yes |
| fv8_fb | left-going fast magnetosonic | 256 | 5.207274e-11 | 1.23 | 0 | yes |
| fv8_fb | left-going Alfvén | 256 | 8.121179e-11 | 1.80 | 0 | yes |
| fv8_fb | left-going slow magnetosonic | 256 | 7.742714e-12 | 0.81 | 0 | yes |
| fv8_fb | entropy/contact | 256 | 6.375589e-12 | -0.99 | 0 | yes |
| fv8_fb | right-going slow magnetosonic | 256 | 8.451996e-12 | 1.80 | 0 | yes |
| fv8_fb | right-going Alfvén | 256 | 4.454045e-11 | 2.17 | 0 | yes |
| fv8_fb | right-going fast magnetosonic | 256 | 3.290873e-11 | 1.72 | 0 | yes |
| plm | left-going fast magnetosonic | 256 | 3.423237e-09 | 2.01 | 0 | yes |
| plm | left-going Alfvén | 256 | 2.616158e-09 | 2.02 | 0 | yes |
| plm | left-going slow magnetosonic | 256 | 9.296147e-10 | 2.05 | 0 | yes |
| plm | entropy/contact | 256 | 1.747406e-09 | 2.11 | 0 | yes |
| plm | right-going slow magnetosonic | 256 | 2.244178e-09 | 2.09 | 0 | yes |
| plm | right-going Alfvén | 256 | 3.450083e-09 | 2.05 | 0 | yes |
| plm | right-going fast magnetosonic | 256 | 1.325634e-09 | 2.01 | 0 | yes |
| teno | left-going fast magnetosonic | 256 | 5.206991e-11 | 1.23 | 0 | yes |
| teno | left-going Alfvén | 256 | 8.121246e-11 | 1.80 | 0 | yes |
| teno | left-going slow magnetosonic | 256 | 7.744603e-12 | 0.81 | 0 | yes |
| teno | entropy/contact | 256 | 6.327116e-12 | -1.00 | 0 | yes |
| teno | right-going slow magnetosonic | 256 | 8.443481e-12 | 1.81 | 0 | yes |
| teno | right-going Alfvén | 256 | 4.454427e-11 | 2.17 | 0 | yes |
| teno | right-going fast magnetosonic | 256 | 3.2907e-11 | 1.72 | 0 | yes |
| wenoz | left-going fast magnetosonic | 256 | 5.207114e-11 | 1.23 | 0 | yes |
| wenoz | left-going Alfvén | 256 | 8.12167e-11 | 1.80 | 0 | yes |
| wenoz | left-going slow magnetosonic | 256 | 7.743481e-12 | 0.81 | 0 | yes |
| wenoz | entropy/contact | 256 | 6.329077e-12 | -1.00 | 0 | yes |
| wenoz | right-going slow magnetosonic | 256 | 8.443103e-12 | 1.81 | 0 | yes |
| wenoz | right-going Alfvén | 256 | 4.454208e-11 | 2.17 | 0 | yes |
| wenoz | right-going fast magnetosonic | 256 | 3.290584e-11 | 1.72 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic grmhd 2d](figures/lwave_grmhd_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going fast magnetosonic | 128 | 1.369417e-09 | 2.12 | 0 | yes |
| fv4_fb | right-going fast magnetosonic | 128 | 1.320045e-09 | 2.08 | 0 | yes |
| fv6_fb | left-going fast magnetosonic | 128 | 1.335162e-09 | 2.00 | 0 | yes |
| fv6_fb | right-going fast magnetosonic | 128 | 1.29571e-09 | 2.00 | 0 | yes |
| fv8_fb | left-going fast magnetosonic | 128 | 1.335089e-09 | 2.00 | 0 | yes |
| fv8_fb | right-going fast magnetosonic | 128 | 1.295726e-09 | 2.00 | 0 | yes |
| plm | left-going fast magnetosonic | 128 | 4.470755e-08 | 1.84 | 0 | yes |
| plm | right-going fast magnetosonic | 128 | 1.988208e-08 | 1.85 | 0 | yes |
| teno | left-going fast magnetosonic | 128 | 1.334265e-09 | 2.00 | 0 | yes |
| teno | right-going fast magnetosonic | 128 | 1.295627e-09 | 2.00 | 0 | yes |
| wenoz | left-going fast magnetosonic | 128 | 1.334264e-09 | 2.00 | 0 | yes |
| wenoz | right-going fast magnetosonic | 128 | 1.295627e-09 | 2.00 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic grmhd 3d](figures/lwave_grmhd_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| fv4_fb | left-going fast magnetosonic | 64 | 2.408413e-08 | 2.22 | 0 | yes |
| fv4_fb | right-going fast magnetosonic | 64 | 1.075696e-08 | 2.21 | 0 | yes |
| fv6_fb | left-going fast magnetosonic | 64 | 2.296624e-08 | 2.05 | 0 | yes |
| fv6_fb | right-going fast magnetosonic | 64 | 1.022898e-08 | 2.03 | 0 | yes |
| fv8_fb | left-going fast magnetosonic | 64 | 2.295714e-08 | 2.05 | 0 | yes |
| fv8_fb | right-going fast magnetosonic | 64 | 1.022468e-08 | 2.02 | 0 | yes |
| plm | left-going fast magnetosonic | 64 | 1.948717e-07 | 1.46 | 0 | yes |
| plm | right-going fast magnetosonic | 64 | 8.528868e-08 | 1.46 | 0 | yes |
| teno | left-going fast magnetosonic | 64 | 2.293793e-08 | 2.04 | 0 | yes |
| teno | right-going fast magnetosonic | 64 | 1.022882e-08 | 2.03 | 0 | yes |
| wenoz | left-going fast magnetosonic | 64 | 2.293787e-08 | 2.04 | 0 | yes |
| wenoz | right-going fast magnetosonic | 64 | 1.022883e-08 | 2.03 | 0 | yes |

### Shocks

![shock_mub1_gr_dens](figures/shock_mub1_gr_dens.png)

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| mub1_gr | 1d | fv4_fb | 256 | 0 | 18875 | 5.615639686584473 |
| mub1_gr | 1d | fv4_fb | 512 | 0 | 49559 | 6.514868259429932 |
| mub1_gr | 1d | fv4_fb | 1024 | 0 | 137084 | 8.86355972290039 |
| mub1_gr | 1d | fv6_fb | 256 | 0 | 86662 | 1.3191649913787842 |
| mub1_gr | 1d | fv6_fb | 512 | 0 | 293940 | 3.160266876220703 |
| mub1_gr | 1d | fv6_fb | 1024 | 0 | 1021282 | 8.895801067352295 |
| mub1_gr | 1d | fv8_fb | 256 | 0 | 87413 | 1.6509120464324951 |
| mub1_gr | 1d | fv8_fb | 512 | 0 | 297610 | 3.24454402923584 |
| mub1_gr | 1d | fv8_fb | 1024 | 0 | 1039918 | 9.393831968307495 |
| mub1_gr | 1d | plm | 256 | 0 | 0 | 5.131254196166992 |
| mub1_gr | 1d | plm | 512 | 0 | 0 | 5.699239015579224 |
| mub1_gr | 1d | plm | 1024 | 0 | 0 | 7.1900458335876465 |
| mub1_gr | 1d | teno | 256 | 0 | 0 | 5.354448318481445 |
| mub1_gr | 1d | teno | 512 | 0 | 0 | 6.118960380554199 |
| mub1_gr | 1d | teno | 1024 | 0 | 0 | 7.702042102813721 |
| mub1_gr | 1d | wenoz | 256 | 0 | 0 | 5.251308441162109 |
| mub1_gr | 1d | wenoz | 512 | 0 | 0 | 6.196975946426392 |
| mub1_gr | 1d | wenoz | 1024 | 0 | 0 | 7.92611837387085 |

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| blast_grmhd | 2d | fv6_fb | **floored** | 2441 | 12473625 | 64.05350708961487 |
| blast_grmhd_uct | 2d | fv6_fb | **floored** | 3563 | 11885516 | 75.10579800605774 |
| blast_grmhd | 2d | wenoz | **rc!=0** | 0 | 0 | 3.345273733139038 |
| blast_grmhd | 2d | teno | **rc!=0** | 0 | 0 | 3.432358503341675 |
| blast_grmhd | 2d | fv4_fb | **rc!=0** | 0 | 0 | 3.4680516719818115 |
| blast_grmhd | 2d | fv8_fb | **floored** | 2512 | 12600010 | 65.99214911460876 |
| blast_grmhd_uct | 2d | fv8_fb | **floored** | 19681 | 11942899 | 76.55016112327576 |
| blast_grmhd | 2d | plm | **rc!=0** | 0 | 0 | 0.5711870193481445 |

![blast_grmhd stress mosaic](figures/stress_blast_grmhd_mosaic.png)

![blast_grmhd |B| mosaic](figures/stress_blast_grmhd_bmag_mosaic.png)

![blast_grmhd_uct stress mosaic](figures/stress_blast_grmhd_uct_mosaic.png)

![blast_grmhd_uct |B| mosaic](figures/stress_blast_grmhd_uct_bmag_mosaic.png)

### Multi-D blast (Apollo SRMHD)

Apollo `~/srmhd_blast` scheme comparison with FV4 (~/srmhd_blast on Apollo (fv4_fb vs plm/wenoz/ppmx/dc)). The mosaic uses the exact final frame ($t=4$) of each movie; full movies are linked below.

| Movie | Path |
|---|---|

## Limitations

- No MOOD in dynamical GRMHD (`dyn_grmhd`).
- `mood=true` incompatible with BH excision and with `fofc=true`; the non-MOOD schemes therefore run with `fofc=true` as their counterpart a-posteriori protection.
- MHD linear waves use `mood_nad_v=off`; velocity NAD at amp=1e-6 falsely demotes Alfvén/entropy families.
- NR MHD uses `emf=uct_hlld` + `rsolver=hlld` for all schemes; GR MHD defaults to `ct_contact`; `uct_hll` is available since the relativistic solvers were fixed to pass the transport velocity to the EMF composition.
- 2D/3D linear waves use diagonal wavevectors and cover both left- and right-going acoustic (hydro) or fast-magnetosonic (MHD) modes.
- Local 2D stress (paper-typical grids on Apollo A100s): hydro slotted cylinder (200², one rotation) / Liska–Wendroff implode (200²); MHD Orszag–Tang (400², $t=1$) / current sheet (256², $t=1$ tearing seed) / Lecoanet KH (256×512). Apollo multi-D stress: MHD ringing (`figures/apollo_ringing/`) and SRMHD blast with FV4 (`figures/apollo_grmhd/`, from `~/srmhd_blast`).
- The stock current-sheet input is a resistive-diffusion test (`epsb=0`, `epsv=0.001`). The suite now enables the pgen tearing seed (`epsb=0.05`, `epsv=0.01`). High-order schemes still collapse near $t=1$; the mosaic uses the last common finite time ($t=1$). TENO has no finite evolved dump.

## Reproducibility

```bash
cmake -B build -DAthena_ENABLE_MPI=OFF -DPROBLEM=built_in_pgens
cmake --build build --target athena -j8
validation/fallback/scripts/build_2d_problem_binaries.sh
python3 validation/fallback/scripts/run_suite.py --athena build/src/athena --suite all
python3 validation/fallback/scripts/analyze.py --write
python3 validation/fallback/scripts/plot_results.py
python3 validation/fallback/scripts/plot_2d_stress.py
python3 validation/fallback/scripts/generate_report.py --pdf
```
