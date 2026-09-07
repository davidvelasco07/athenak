# MOOD Fallback Validation Report

_Generated 2026-09-07 20:48 UTC_
_Commit `unknown`_

## Summary

- Records analyzed: **762**
- PPM+MOOD linear-wave zero demotions: **True**
- All runs completed (rc=0): **True**

## Method

We compare unlimited PPM with a-posteriori MOOD fallback (RK3) against PLM+RK2, WENOZ+RK3, and TENO-5+RK3. Detection uses NAD (`mood_nad_scale=gcfl`, `mood_rtol=1e-5`; `mood_nad_v=off` for MHD *linear waves* only -- see the note below) plus PAD/NaN. Passive-scalar concentrations are included in the relaxed-DMP test, so scalar discontinuities can trigger fallback even on a uniform-density kinematic background. NAD demotions are floored at PLM.

**Each scheme carries its own a-posteriori protection.** MOOD and FOFC are mutually exclusive in the code, and unlimited PPM *requires* `mood=true`, so the like-for-like comparison gives PPM the MOOD cascade and gives PLM / WENO-Z / TENO first-order flux correction (`fofc=true`). Benchmarking MOOD against unprotected competitors would overstate it: on the current sheet, TENO fails outright without FOFC and survives with it. `fofc` is a per-scheme property of the manifest so it cannot silently differ per case. MOOD is unsupported with BH excision. Newtonian MHD uses `emf=uct_hlld` for all schemes. Relativistic MHD defaults to `ct_contact` and may opt into `uct_hll` per case (`uct_hlld` needs `rsolver=hlld`, which SR/GR reject); the `blast_grmhd` / `blast_grmhd_uct` pair runs both.

**`returncode == 0` is not evidence of success.** A run whose state has been wiped still prints "Terminating on time limit": on the current sheet, unprotected TENO reported rc=0 with an all-NaN final dump. Every stress run is therefore health-gated on the EOS floor/failure counters and the timestep history (`dt_collapse` when $\Delta t_{\min} \le 0$ or $< 10^{-6}\Delta t_{\max}$).

The report is organized by physics. Each section covers **1D / 2D / 3D** linear waves (convergence mosaics with shared axes), plus 1D shocks and available multi-D stress cases.

## Hydrodynamics

### 1D linear waves

![Linear-wave mosaic hydro 1d](figures/lwave_hydro_1d_mosaic.png)

![PPM+MOOD demotions hydro](figures/lwave_nmood_hydro.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going acoustic | 256 | 1.228225e-09 | 2.02 | 0 | yes |
| plm | transverse shear z | 256 | 5.517613e-10 | 2.03 | 0 | yes |
| plm | right-going acoustic | 256 | 1.228227e-09 | 2.02 | 0 | yes |
| ppm_fb | left-going acoustic | 256 | 5.497869e-12 | 0.04 | 0 | yes |
| ppm_fb | transverse shear z | 256 | 1.808209e-13 | 2.11 | 0 | yes |
| ppm_fb | right-going acoustic | 256 | 5.498177e-12 | 0.04 | 0 | yes |
| teno | left-going acoustic | 256 | 5.497859e-12 | 0.02 | 0 | yes |
| teno | transverse shear z | 256 | 1.79895e-13 | -0.56 | 0 | yes |
| teno | right-going acoustic | 256 | 5.497667e-12 | 0.02 | 0 | yes |
| wenoz | left-going acoustic | 256 | 5.497803e-12 | 0.02 | 0 | yes |
| wenoz | transverse shear z | 256 | 1.849777e-13 | -0.59 | 0 | yes |
| wenoz | right-going acoustic | 256 | 5.497693e-12 | 0.02 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic hydro 2d](figures/lwave_hydro_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going acoustic | 128 | 1.802754e-08 | 1.84 | 0 | yes |
| plm | right-going acoustic | 128 | 1.802754e-08 | 1.84 | 0 | yes |
| ppm_fb | left-going acoustic | 128 | 2.838798e-11 | 3.60 | 0 | yes |
| ppm_fb | right-going acoustic | 128 | 2.838823e-11 | 3.60 | 0 | yes |
| teno | left-going acoustic | 128 | 1.462483e-11 | 3.19 | 0 | yes |
| teno | right-going acoustic | 128 | 1.462488e-11 | 3.19 | 0 | yes |
| wenoz | left-going acoustic | 128 | 1.462479e-11 | 3.19 | 0 | yes |
| wenoz | right-going acoustic | 128 | 1.462486e-11 | 3.19 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic hydro 3d](figures/lwave_hydro_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going acoustic | 64 | 7.883148e-08 | 1.49 | 0 | yes |
| plm | right-going acoustic | 64 | 7.883148e-08 | 1.49 | 0 | yes |
| ppm_fb | left-going acoustic | 64 | 4.202752e-10 | 3.78 | 0 | yes |
| ppm_fb | right-going acoustic | 64 | 4.202752e-10 | 3.78 | 0 | yes |
| teno | left-going acoustic | 64 | 2.894962e-10 | 3.60 | 0 | yes |
| teno | right-going acoustic | 64 | 2.894963e-10 | 3.60 | 0 | yes |
| wenoz | left-going acoustic | 64 | 2.894976e-10 | 3.62 | 0 | yes |
| wenoz | right-going acoustic | 64 | 2.894978e-10 | 3.62 | 0 | yes |

### Shocks

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| shu_osher | 1d | plm | 256 | 0 | 0 | 4.85801362991333 |
| shu_osher | 1d | plm | 512 | 0 | 0 | 5.158158779144287 |
| shu_osher | 1d | plm | 1024 | 0 | 0 | 6.00898551940918 |
| shu_osher | 1d | ppm_fb | 256 | 0 | 27576 | 6.583750009536743 |
| shu_osher | 1d | ppm_fb | 512 | 0 | 67195 | 8.699394702911377 |
| shu_osher | 1d | ppm_fb | 1024 | 0 | 150202 | 12.121439456939697 |
| shu_osher | 1d | teno | 256 | 0 | 0 | 5.015064477920532 |
| shu_osher | 1d | teno | 512 | 0 | 0 | 5.576136589050293 |
| shu_osher | 1d | teno | 1024 | 0 | 0 | 6.819214582443237 |
| shu_osher | 1d | wenoz | 256 | 0 | 0 | 4.977221250534058 |
| shu_osher | 1d | wenoz | 512 | 0 | 0 | 5.588316202163696 |
| shu_osher | 1d | wenoz | 1024 | 0 | 0 | 6.855767488479614 |
| sod | 1d | plm | 256 | 0 | 0 | 4.611562728881836 |
| sod | 1d | plm | 512 | 0 | 0 | 4.9116575717926025 |
| sod | 1d | plm | 1024 | 0 | 0 | 5.254113435745239 |
| sod | 1d | ppm_fb | 256 | 0 | 22045 | 6.614264249801636 |
| sod | 1d | ppm_fb | 512 | 0 | 48894 | 6.407215118408203 |
| sod | 1d | ppm_fb | 1024 | 0 | 105257 | 8.302880048751831 |
| sod | 1d | teno | 256 | 0 | 0 | 4.681830883026123 |
| sod | 1d | teno | 512 | 0 | 0 | 4.949263095855713 |
| sod | 1d | teno | 1024 | 0 | 0 | 5.4772748947143555 |
| sod | 1d | wenoz | 256 | 0 | 0 | 4.689580202102661 |
| sod | 1d | wenoz | 512 | 0 | 0 | 4.921968936920166 |
| sod | 1d | wenoz | 1024 | 0 | 0 | 5.66727352142334 |

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| blast_hydro | 2d | plm_bare | clean | 0 | 0 | 5.4710853099823 |
| blast_hydro | 2d | plm | clean | 0 | 0 | 6.202395915985107 |
| blast_hydro | 2d | ppm_fb | clean | 0 | 22573532 | 20.653642892837524 |
| blast_hydro | 2d | teno_bare | clean | 0 | 0 | 6.968309640884399 |
| blast_hydro | 2d | teno | clean | 0 | 0 | 7.053554058074951 |
| blast_hydro | 2d | wenoz_bare | clean | 0 | 0 | 6.358892917633057 |
| blast_hydro | 2d | wenoz | clean | 0 | 0 | 7.719371795654297 |
| fu_jet_m80 | 2d | plm_bare | clean | 0 | 0 | 4.022455453872681 |
| fu_jet_m80 | 2d | plm | clean | 0 | 0 | 5.153618574142456 |
| fu_jet_m80 | 2d | ppm_fb | **floored** | 1 | 8001988 | 10.29435920715332 |
| fu_jet_m80 | 2d | teno_bare | **non-finite** | 42 | 0 | 3.842344045639038 |
| fu_jet_m80 | 2d | teno | **non-finite** | 165 | 0 | 163.77474737167358 |
| fu_jet_m80 | 2d | wenoz_bare | clean | 0 | 0 | 4.33439826965332 |
| fu_jet_m80 | 2d | wenoz | clean | 0 | 0 | 5.095012187957764 |
| ha_jet | 2d | plm_bare | clean | 0 | 0 | 4.523331165313721 |
| ha_jet | 2d | plm | clean | 0 | 0 | 5.7679603099823 |
| ha_jet | 2d | ppm_fb | **floored** | 6 | 8495421 | 10.488210439682007 |
| ha_jet | 2d | teno_bare | **non-finite** | 34 | 0 | 4.037832021713257 |
| ha_jet | 2d | teno | **non-finite** | 38 | 0 | 5.158279895782471 |
| ha_jet | 2d | wenoz_bare | clean | 0 | 0 | 5.599099159240723 |
| ha_jet | 2d | wenoz | clean | 0 | 0 | 6.697075605392456 |
| implode_hydro | 2d | plm_bare | clean | 0 | 0 | 7.516884088516235 |
| implode_hydro | 2d | plm | clean | 0 | 0 | 9.043246746063232 |
| implode_hydro | 2d | ppm_fb | clean | 0 | 175861689 | 31.760163068771362 |
| implode_hydro | 2d | teno_bare | clean | 0 | 0 | 9.638213634490967 |
| implode_hydro | 2d | teno | clean | 0 | 0 | 10.908190488815308 |
| implode_hydro | 2d | wenoz_bare | clean | 0 | 0 | 10.161576509475708 |
| implode_hydro | 2d | wenoz | clean | 0 | 0 | 11.987500667572021 |
| slotted_cyl | 2d | plm_bare | clean | 0 | 0 | 4.174513816833496 |
| slotted_cyl | 2d | plm | clean | 0 | 0 | 4.381020545959473 |
| slotted_cyl | 2d | ppm_fb | clean | 0 | 2440854 | 11.137042045593262 |
| slotted_cyl | 2d | teno_bare | clean | 0 | 0 | 4.457943916320801 |
| slotted_cyl | 2d | teno | clean | 0 | 0 | 5.551639795303345 |
| slotted_cyl | 2d | wenoz_bare | clean | 0 | 0 | 6.035802125930786 |
| slotted_cyl | 2d | wenoz | clean | 0 | 0 | 4.628018140792847 |

![blast_hydro stress mosaic](figures/stress_blast_hydro_mosaic.png)

![fu_jet_m80 stress mosaic](figures/stress_fu_jet_m80_mosaic.png)

![ha_jet stress mosaic](figures/stress_ha_jet_mosaic.png)

![implode_hydro stress mosaic](figures/stress_implode_hydro_mosaic.png)

The slotted-cylinder test advects a bounded passive scalar through one full rotation on a $200^2$ mesh. Scalar-aware NAD confines fallback to the discontinuity and suppresses PPM ringing.

![slotted_cyl stress mosaic](figures/stress_slotted_cyl_mosaic.png)

### NAD tolerance study

Relaxed-DMP tolerance sweeps keep the base scheme, grid, Riemann solver, and integrator fixed. The Liska--Wendroff implosion probes interacting shocks and contacts; the slotted disk isolates the passive-scalar detector.

#### Relativistic magnetized blast (Minkowski)

![Relativistic magnetized blast (Minkowski) snapshot mosaic](figures/tolerance_blast_grmhd_mosaic.png)

![Relativistic magnetized blast (Minkowski) demotion curve](figures/tolerance_blast_grmhd_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 256 | 0 | 9932282 | 5.566143989562988 |
| 1e-03 | 256 | 0 | 10647492 | 5.20865797996521 |
| 1e-04 | 256 | 0 | 12020211 | 4.732102870941162 |
| 1e-05 | 256 | 0 | 12660580 | 5.900129079818726 |
| 1e-06 | 256 | 0 | 13523525 | 4.647405385971069 |

#### Hydrodynamic blast

![Hydrodynamic blast snapshot mosaic](figures/tolerance_blast_hydro_mosaic.png)

![Hydrodynamic blast demotion curve](figures/tolerance_blast_hydro_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 7647640 | 16.249534606933594 |
| 1e-03 | 200 | 0 | 16134404 | 17.240208387374878 |
| 1e-04 | 200 | 0 | 21485584 | 14.453444242477417 |
| 1e-05 | 200 | 0 | 22573532 | 14.552955150604248 |
| 1e-06 | 200 | 0 | 22947472 | 15.483547925949097 |

#### Magnetized blast

![Magnetized blast snapshot mosaic](figures/tolerance_blast_mhd_mosaic.png)

![Magnetized blast demotion curve](figures/tolerance_blast_mhd_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 41460902 | 59.82132530212402 |
| 1e-03 | 200 | 0 | 80533475 | 55.520235776901245 |
| 1e-04 | 200 | 0 | 92965009 | 57.27581071853638 |
| 1e-05 | 200 | 0 | 95416781 | 59.123695611953735 |
| 1e-06 | 200 | 0 | 96365791 | 56.92161178588867 |

#### blast_mhd_lowbeta

![blast_mhd_lowbeta snapshot mosaic](figures/tolerance_blast_mhd_lowbeta_mosaic.png)

![blast_mhd_lowbeta demotion curve](figures/tolerance_blast_mhd_lowbeta_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 12307882 | 25.77090311050415 |
| 1e-03 | 200 | 0 | 64423083 | 20.240992307662964 |
| 1e-04 | 200 | 0 | 101089350 | 20.909144401550293 |
| 1e-05 | 200 | 0 | 107223730 | 21.437143325805664 |
| 1e-06 | 200 | 0 | 109024155 | 21.419799089431763 |

#### Double Harris current sheet ($t=5$, tearing seed)

![Double Harris current sheet ($t=5$, tearing seed) snapshot mosaic](figures/tolerance_current_sheet_mosaic.png)

![Double Harris current sheet ($t=5$, tearing seed) demotion curve](figures/tolerance_current_sheet_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 256 | 0 | 44046017915 | 785.6429033279419 |
| 1e-03 | 256 | 0 | 35391283 | 19.716842889785767 |
| 1e-04 | 256 | 0 | 72434550 | 19.29183053970337 |
| 1e-05 | 256 | 0 | 105619933 | 19.570242881774902 |
| 1e-06 | 256 | 0 | 111274060 | 19.358506441116333 |

#### fu_jet_m80

![fu_jet_m80 snapshot mosaic](figures/tolerance_fu_jet_m80_mosaic.png)

![fu_jet_m80 demotion curve](figures/tolerance_fu_jet_m80_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 448 | 0 | 335018083 | 95.79762291908264 |
| 1e-03 | 448 | 0 | 220484524 | 11.463873147964478 |
| 1e-04 | 448 | 0 | 4394821 | 6.799262046813965 |
| 1e-05 | 448 | 0 | 8001988 | 7.364190101623535 |
| 1e-06 | 448 | 0 | 9063204 | 6.811092376708984 |

#### ha_jet

![ha_jet snapshot mosaic](figures/tolerance_ha_jet_mosaic.png)

![ha_jet demotion curve](figures/tolerance_ha_jet_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 512 | 0 | 546496233 | 36.643460750579834 |
| 1e-03 | 512 | 0 | 555526671 | 16.449207544326782 |
| 1e-04 | 512 | 0 | 4762453 | 7.930532217025757 |
| 1e-05 | 512 | 0 | 8495421 | 8.638962268829346 |
| 1e-06 | 512 | 0 | 10291043 | 7.9528727531433105 |

#### Liska--Wendroff implosion

![Liska--Wendroff implosion snapshot mosaic](figures/tolerance_implode_hydro_mosaic.png)

![Liska--Wendroff implosion demotion curve](figures/tolerance_implode_hydro_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 62205225 | 34.349257469177246 |
| 1e-03 | 200 | 0 | 159450725 | 33.01205110549927 |
| 1e-04 | 200 | 0 | 174105603 | 34.09156346321106 |
| 1e-05 | 200 | 0 | 175861689 | 30.102568864822388 |
| 1e-06 | 200 | 0 | 176301945 | 30.58934187889099 |

#### Underdense Mach-10 MHD jet

![Underdense Mach-10 MHD jet snapshot mosaic](figures/tolerance_jet_mosaic.png)

![Underdense Mach-10 MHD jet demotion curve](figures/tolerance_jet_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 300 | 0 | 45108683 | 12.802889108657837 |
| 1e-03 | 300 | 0 | 30322796 | 42.533358573913574 |
| 1e-04 | 300 | 0 | 23932782 | 22.538772583007812 |
| 1e-05 | 300 | 0 | 31582594 | 22.658840894699097 |
| 1e-06 | 300 | 0 | 35280279 | 22.491897106170654 |

#### Magnetized Kelvin--Helmholtz (Rueda-Ram\'irez+ 2022)

![Magnetized Kelvin--Helmholtz (Rueda-Ram\'irez+ 2022) snapshot mosaic](figures/tolerance_kh_rr22_mosaic.png)

![Magnetized Kelvin--Helmholtz (Rueda-Ram\'irez+ 2022) demotion curve](figures/tolerance_kh_rr22_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 128 | 0 | 30040095 | 65.42592692375183 |
| 1e-03 | 128 | 0 | 63342024 | 61.91054105758667 |
| 1e-04 | 128 | 0 | 82938962 | 59.110140562057495 |
| 1e-05 | 128 | 0 | 91187458 | 59.15301704406738 |
| 1e-06 | 128 | 0 | 96318115 | 57.75718927383423 |

#### mhd_jet

![mhd_jet demotion curve](figures/tolerance_mhd_jet_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 78186664 | 16.2451274394989 |
| 1e-03 | 200 | 0 | 127717420380 | 769.9116103649139 |
| 1e-04 | 200 | 0 | 86188129480 | 881.3024089336395 |
| 1e-05 | 200 | 0 | 80757710 | 6.728329181671143 |
| 1e-06 | 200 | 0 | 80709679 | 7.2623443603515625 |

#### Orszag--Tang vortex ($t=1$)

![Orszag--Tang vortex ($t=1$) snapshot mosaic](figures/tolerance_orszag_tang_mosaic.png)

![Orszag--Tang vortex ($t=1$) demotion curve](figures/tolerance_orszag_tang_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 400 | 0 | 35082072 | 52.33875870704651 |
| 1e-03 | 400 | 0 | 71550770 | 69.57169914245605 |
| 1e-04 | 400 | 0 | 90149247 | 70.223468542099 |
| 1e-05 | 400 | 0 | 92770841 | 67.65663146972656 |
| 1e-06 | 400 | 0 | 93295438 | 75.0028989315033 |

#### MHD rotor (Balsara \& Spicer)

![MHD rotor (Balsara \& Spicer) snapshot mosaic](figures/tolerance_rotor_mosaic.png)

![MHD rotor (Balsara \& Spicer) demotion curve](figures/tolerance_rotor_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 400 | 0 | 2578886 | 11.084071636199951 |
| 1e-03 | 400 | 0 | 16793705 | 7.936384201049805 |
| 1e-04 | 400 | 0 | 18826349 | 7.726302623748779 |
| 1e-05 | 400 | 0 | 19741585 | 7.76148247718811 |
| 1e-06 | 400 | 0 | 20492178 | 8.162254095077515 |

#### Slotted disk

![Slotted disk snapshot mosaic](figures/tolerance_slotted_cyl_mosaic.png)

![Slotted disk demotion curve](figures/tolerance_slotted_cyl_demotions.png)

| rtol | N | rc | nmood | wall [s] |
|---|---|---|---|---|
| 1e-02 | 200 | 0 | 61122 | 7.023053169250488 |
| 1e-03 | 200 | 0 | 650314 | 7.488023281097412 |
| 1e-04 | 200 | 0 | 1410463 | 7.728758811950684 |
| 1e-05 | 200 | 0 | 2440854 | 8.23508334159851 |
| 1e-06 | 200 | 0 | 3129071 | 7.633999824523926 |

## Magnetohydrodynamics

### 1D linear waves

![Linear-wave mosaic mhd 1d](figures/lwave_mhd_1d_mosaic.png)

![PPM+MOOD demotions mhd](figures/lwave_nmood_mhd.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going fast magnetosonic | 256 | 1.465093e-09 | 2.02 | 0 | yes |
| plm | left-going Alfvén | 256 | 7.800619e-10 | 2.03 | 0 | yes |
| plm | left-going slow magnetosonic | 256 | 9.08071e-10 | 2.05 | 0 | yes |
| plm | entropy/contact | 256 | 8.15686e-10 | 2.04 | 0 | yes |
| plm | right-going slow magnetosonic | 256 | 9.080688e-10 | 2.05 | 0 | yes |
| plm | right-going Alfvén | 256 | 7.800627e-10 | 2.03 | 0 | yes |
| plm | right-going fast magnetosonic | 256 | 1.465099e-09 | 2.02 | 0 | yes |
| ppm_fb | left-going fast magnetosonic | 256 | 1.009797e-11 | 2.01 | 0 | yes |
| ppm_fb | left-going Alfvén | 256 | 3.122887e-13 | 1.83 | 0 | yes |
| ppm_fb | left-going slow magnetosonic | 256 | 3.846367e-12 | 0.04 | 0 | yes |
| ppm_fb | entropy/contact | 256 | 5.821247e-13 | 1.02 | 0 | yes |
| ppm_fb | right-going slow magnetosonic | 256 | 3.846872e-12 | 0.03 | 0 | yes |
| ppm_fb | right-going Alfvén | 256 | 3.100634e-13 | 1.84 | 0 | yes |
| ppm_fb | right-going fast magnetosonic | 256 | 1.009795e-11 | 2.01 | 0 | yes |
| teno | left-going fast magnetosonic | 256 | 1.009767e-11 | 2.00 | 0 | yes |
| teno | left-going Alfvén | 256 | 3.092695e-13 | -0.67 | 0 | yes |
| teno | left-going slow magnetosonic | 256 | 3.846018e-12 | -0.01 | 0 | yes |
| teno | entropy/contact | 256 | 5.548657e-13 | -1.00 | 0 | yes |
| teno | right-going slow magnetosonic | 256 | 3.845343e-12 | -0.01 | 0 | yes |
| teno | right-going Alfvén | 256 | 3.092114e-13 | -0.66 | 0 | yes |
| teno | right-going fast magnetosonic | 256 | 1.009797e-11 | 2.00 | 0 | yes |
| wenoz | left-going fast magnetosonic | 256 | 1.009762e-11 | 2.00 | 0 | yes |
| wenoz | left-going Alfvén | 256 | 3.089875e-13 | -0.66 | 0 | yes |
| wenoz | left-going slow magnetosonic | 256 | 3.844766e-12 | -0.01 | 0 | yes |
| wenoz | entropy/contact | 256 | 5.552948e-13 | -1.00 | 0 | yes |
| wenoz | right-going slow magnetosonic | 256 | 3.845515e-12 | -0.01 | 0 | yes |
| wenoz | right-going Alfvén | 256 | 3.096604e-13 | -0.66 | 0 | yes |
| wenoz | right-going fast magnetosonic | 256 | 1.009759e-11 | 2.00 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic mhd 2d](figures/lwave_mhd_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going fast magnetosonic | 128 | 2.15449e-08 | 1.81 | 0 | yes |
| plm | right-going fast magnetosonic | 128 | 2.154531e-08 | 1.81 | 0 | yes |
| ppm_fb | left-going fast magnetosonic | 128 | 4.075089e-10 | 2.18 | 0 | yes |
| ppm_fb | right-going fast magnetosonic | 128 | 4.075083e-10 | 2.18 | 0 | yes |
| teno | left-going fast magnetosonic | 128 | 3.659035e-10 | 2.01 | 0 | yes |
| teno | right-going fast magnetosonic | 128 | 3.659036e-10 | 2.01 | 0 | yes |
| wenoz | left-going fast magnetosonic | 128 | 3.659036e-10 | 2.01 | 0 | yes |
| wenoz | right-going fast magnetosonic | 128 | 3.659035e-10 | 2.01 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic mhd 3d](figures/lwave_mhd_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going fast magnetosonic | 64 | 8.882641e-08 | 1.43 | 0 | yes |
| plm | right-going fast magnetosonic | 64 | 8.882653e-08 | 1.43 | 0 | yes |
| ppm_fb | left-going fast magnetosonic | 64 | 1.811124e-09 | 2.70 | 0 | yes |
| ppm_fb | right-going fast magnetosonic | 64 | 1.811123e-09 | 2.70 | 0 | yes |
| teno | left-going fast magnetosonic | 64 | 1.410097e-09 | 2.19 | 0 | yes |
| teno | right-going fast magnetosonic | 64 | 1.410097e-09 | 2.19 | 0 | yes |
| wenoz | left-going fast magnetosonic | 64 | 1.410101e-09 | 2.19 | 0 | yes |
| wenoz | right-going fast magnetosonic | 64 | 1.410101e-09 | 2.19 | 0 | yes |

### Shocks

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| bw | 1d | plm | 256 | 0 | 0 | 4.831498384475708 |
| bw | 1d | plm | 512 | 0 | 0 | 5.09998893737793 |
| bw | 1d | plm | 1024 | 0 | 0 | 5.740323543548584 |
| bw | 1d | ppm_fb | 256 | 0 | 15620 | 5.455307722091675 |
| bw | 1d | ppm_fb | 512 | 0 | 46976 | 6.407660722732544 |
| bw | 1d | ppm_fb | 1024 | 0 | 139286 | 8.186083555221558 |
| bw | 1d | teno | 256 | 0 | 0 | 4.895114898681641 |
| bw | 1d | teno | 512 | 0 | 0 | 5.3945393562316895 |
| bw | 1d | teno | 1024 | 0 | 0 | 6.336426734924316 |
| bw | 1d | wenoz | 256 | 0 | 0 | 5.056381940841675 |
| bw | 1d | wenoz | 512 | 0 | 0 | 5.467697620391846 |
| bw | 1d | wenoz | 1024 | 0 | 0 | 6.279923915863037 |
| rj2a | 1d | plm | 256 | 0 | 0 | 4.898343086242676 |
| rj2a | 1d | plm | 512 | 0 | 0 | 5.323077917098999 |
| rj2a | 1d | plm | 1024 | 0 | 0 | 6.295633792877197 |
| rj2a | 1d | ppm_fb | 256 | 0 | 44934 | 6.194762706756592 |
| rj2a | 1d | ppm_fb | 512 | 0 | 160045 | 7.311729192733765 |
| rj2a | 1d | ppm_fb | 1024 | 0 | 530082 | 10.259937524795532 |
| rj2a | 1d | teno | 256 | 0 | 0 | 5.098629713058472 |
| rj2a | 1d | teno | 512 | 0 | 0 | 5.725203275680542 |
| rj2a | 1d | teno | 1024 | 0 | 0 | 7.114076614379883 |
| rj2a | 1d | wenoz | 256 | 0 | 0 | 5.156071424484253 |
| rj2a | 1d | wenoz | 512 | 0 | 0 | 5.787648439407349 |
| rj2a | 1d | wenoz | 1024 | 0 | 0 | 7.087417125701904 |

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| blast_mhd_lowbeta | 2d | plm_bare | **floored** | 21837 | 0 | 9.197941303253174 |
| blast_mhd_lowbeta | 2d | plm | **floored** | 9244 | 0 | 8.389042139053345 |
| blast_mhd_lowbeta | 2d | ppm_fb | **floored** | 10956 | 107223730 | 25.202003955841064 |
| blast_mhd_lowbeta | 2d | teno_bare | **non-finite** | 318248 | 0 | 586.9671671390533 |
| blast_mhd_lowbeta | 2d | teno | **floored** | 747673 | 0 | 15.562714338302612 |
| blast_mhd_lowbeta | 2d | wenoz_bare | **floored** | 987790 | 0 | 14.398079633712769 |
| blast_mhd_lowbeta | 2d | wenoz | **floored** | 831319 | 0 | 16.307144165039062 |
| blast_mhd | 2d | plm_bare | clean | 0 | 0 | 11.2309250831604 |
| blast_mhd | 2d | plm | clean | 0 | 0 | 12.885554313659668 |
| blast_mhd | 2d | ppm_fb | clean | 0 | 95416781 | 58.25843954086304 |
| blast_mhd | 2d | teno_bare | clean | 0 | 0 | 15.08973503112793 |
| blast_mhd | 2d | teno | clean | 0 | 0 | 17.54408073425293 |
| blast_mhd | 2d | wenoz_bare | clean | 0 | 0 | 15.569182634353638 |
| blast_mhd | 2d | wenoz | clean | 0 | 0 | 17.858931064605713 |
| current_sheet_n1024 | 2d | plm_bare | clean | 0 | 0 | 40.5736882686615 |
| current_sheet_n1024 | 2d | plm | clean | 0 | 0 | 46.522233963012695 |
| current_sheet_n1024 | 2d | ppm_fb | clean | 0 | 2957138485 | 195.962327003479 |
| current_sheet_n1024 | 2d | teno_bare | **non-finite** | 3720 | 0 | 24.36186718940735 |
| current_sheet_n1024 | 2d | teno | **non-finite** | 1111726 | 0 | 67.9896035194397 |
| current_sheet_n1024 | 2d | wenoz_bare | clean | 0 | 0 | 71.32430768013 |
| current_sheet_n1024 | 2d | wenoz | clean | 0 | 0 | 80.9039614200592 |
| current_sheet_n512 | 2d | plm_bare | clean | 0 | 0 | 13.270320415496826 |
| current_sheet_n512 | 2d | plm | clean | 0 | 0 | 15.058834075927734 |
| current_sheet_n512 | 2d | ppm_fb | clean | 0 | 530494387 | 47.54116702079773 |
| current_sheet_n512 | 2d | teno_bare | **non-finite** | 7684 | 0 | 11.721870183944702 |
| current_sheet_n512 | 2d | teno | **non-finite** | 79200 | 0 | 21.670106410980225 |
| current_sheet_n512 | 2d | wenoz_bare | **non-finite** | 747 | 0 | 15.769938707351685 |
| current_sheet_n512 | 2d | wenoz | clean | 0 | 0 | 22.84101390838623 |
| current_sheet | 2d | plm_bare | clean | 0 | 0 | 7.15811824798584 |
| current_sheet | 2d | plm | clean | 0 | 0 | 9.257392644882202 |
| current_sheet | 2d | ppm_fb | clean | 0 | 105619933 | 22.879512548446655 |
| current_sheet | 2d | teno_bare | **non-finite** | 5784 | 0 | 6.885414123535156 |
| current_sheet | 2d | teno | **non-finite** | 57853 | 0 | 425.61244440078735 |
| current_sheet | 2d | wenoz_bare | **floored** | 642 | 0 | 9.1358163356781 |
| current_sheet | 2d | wenoz | **floored** | 642 | 0 | 11.465545654296875 |
| jet_n600 | 2d | plm_bare | clean | 0 | 0 | 14.938365697860718 |
| jet_n600 | 2d | plm | clean | 0 | 0 | 17.870752334594727 |
| jet_n600 | 2d | ppm_fb | clean | 0 | 174181722 | 72.20312070846558 |
| jet_n600 | 2d | teno_bare | **non-finite** | 346 | 0 | 18.94264030456543 |
| jet_n600 | 2d | teno | **floored** | 1559 | 0 | 27.83327007293701 |
| jet_n600 | 2d | wenoz_bare | **non-finite** | 513 | 0 | 17.767769813537598 |
| jet_n600 | 2d | wenoz | **floored** | 1085 | 0 | 29.2946674823761 |
| jet | 2d | plm_bare | **floored** | 18217 | 0 | 10.734564304351807 |
| jet | 2d | plm | **floored** | 15207 | 0 | 12.166862487792969 |
| jet | 2d | ppm_fb | clean | 0 | 31582594 | 25.100942850112915 |
| jet | 2d | teno_bare | **floored** | 1562 | 0 | 10.917438268661499 |
| jet | 2d | teno | **floored** | 1018 | 0 | 12.244779109954834 |
| jet | 2d | wenoz_bare | **floored** | 892 | 0 | 10.928439617156982 |
| jet | 2d | wenoz | **floored** | 892 | 0 | 12.993719577789307 |
| kh_rr22_n256 | 2d | plm_bare | clean | 0 | 0 | 55.350093841552734 |
| kh_rr22_n256 | 2d | plm | clean | 0 | 0 | 40.601829051971436 |
| kh_rr22_n256 | 2d | ppm_fb | clean | 0 | 565704741 | 148.76915740966797 |
| kh_rr22_n256 | 2d | teno_bare | clean | 0 | 0 | 56.66392111778259 |
| kh_rr22_n256 | 2d | teno | clean | 0 | 0 | 65.65192127227783 |
| kh_rr22_n256 | 2d | wenoz_bare | clean | 0 | 0 | 57.97780466079712 |
| kh_rr22_n256 | 2d | wenoz | clean | 0 | 0 | 64.64863204956055 |
| kh_rr22 | 2d | plm_bare | clean | 0 | 0 | 16.769900798797607 |
| kh_rr22 | 2d | plm | clean | 0 | 0 | 19.217286586761475 |
| kh_rr22 | 2d | ppm_fb | clean | 0 | 91187458 | 73.33564639091492 |
| kh_rr22 | 2d | teno_bare | clean | 0 | 0 | 23.64289689064026 |
| kh_rr22 | 2d | teno | clean | 0 | 0 | 28.076272010803223 |
| kh_rr22 | 2d | wenoz_bare | clean | 0 | 0 | 23.739564895629883 |
| kh_rr22 | 2d | wenoz | clean | 0 | 0 | 28.012728691101074 |
| mhd_jet | 2d | plm_bare | **floored** | 11902795 | 0 | 10.679098844528198 |
| mhd_jet | 2d | plm | **floored** | 7129272 | 0 | 15.476367235183716 |
| mhd_jet | 2d | ppm_fb | **non-finite** | 241541 | 80757710 | 15.537095069885254 |
| mhd_jet_revs4 | 2d | plm_bare | **floored** | 11902795 | 0 | 9.94114351272583 |
| mhd_jet_revs4 | 2d | plm | **floored** | 7129272 | 0 | 11.332451343536377 |
| mhd_jet_revs4 | 2d | ppm_fb | **floored** | 64076053 | 164123198 | 26.95353078842163 |
| mhd_jet_revs4 | 2d | teno_bare | **non-finite** | 9484175 | 0 | 8.749964952468872 |
| mhd_jet_revs4 | 2d | teno | **floored** | 10538414 | 0 | 12.801611423492432 |
| mhd_jet_revs4 | 2d | wenoz_bare | **non-finite** | 50526442 | 0 | 25.062954902648926 |
| mhd_jet_revs4 | 2d | wenoz | **floored** | 11532463 | 0 | 13.831437110900879 |
| mhd_jet | 2d | teno_bare | **non-finite** | 9484175 | 0 | 9.069286108016968 |
| mhd_jet | 2d | teno | **floored** | 10538414 | 0 | 18.397409439086914 |
| mhd_jet | 2d | wenoz_bare | **non-finite** | 50526442 | 0 | 33.40688419342041 |
| mhd_jet | 2d | wenoz | **floored** | 11532463 | 0 | 14.960851192474365 |
| orszag_tang | 2d | plm_bare | **floored** | 1 | 0 | 10.16148591041565 |
| orszag_tang | 2d | plm | clean | 0 | 0 | 11.835667848587036 |
| orszag_tang | 2d | ppm_fb | clean | 0 | 92770841 | 27.372113943099976 |
| orszag_tang | 2d | teno_bare | **floored** | 3 | 0 | 17.624778270721436 |
| orszag_tang | 2d | teno | clean | 0 | 0 | 16.059730052947998 |
| orszag_tang | 2d | wenoz_bare | clean | 0 | 0 | 14.3320951461792 |
| orszag_tang | 2d | wenoz | clean | 0 | 0 | 15.153056859970093 |
| rotor | 2d | plm_bare | clean | 0 | 0 | 4.272541284561157 |
| rotor | 2d | plm | clean | 0 | 0 | 4.542306423187256 |
| rotor | 2d | ppm_fb | clean | 0 | 19741585 | 7.5611584186553955 |
| rotor | 2d | teno_bare | clean | 0 | 0 | 4.730059385299683 |
| rotor | 2d | teno | clean | 0 | 0 | 5.228531360626221 |
| rotor | 2d | wenoz_bare | clean | 0 | 0 | 4.712350130081177 |
| rotor | 2d | wenoz | clean | 0 | 0 | 5.013249397277832 |

![blast_mhd stress mosaic](figures/stress_blast_mhd_mosaic.png)

![blast_mhd |B| mosaic](figures/stress_blast_mhd_bmag_mosaic.png)

![blast_mhd_lowbeta stress mosaic](figures/stress_blast_mhd_lowbeta_mosaic.png)

![blast_mhd_lowbeta |B| mosaic](figures/stress_blast_mhd_lowbeta_bmag_mosaic.png)

The double Harris sheet is seeded with the pgen tearing perturbation and run to $t=5$, a few Alfv\'en times, so plasmoid chains have formed on both sheets.

**This case previously reported only $t=1$.** The reason every high-order scheme collapsed before plasmoid time was an initial-condition defect, not a limiter failure: `current_sheet.cpp` wrote *primitive* variables into the *conserved* array -- momenta received $v$ rather than $\rho v$, and the total energy omitted both $\rho v^2/2$ and $B^2/2$. In the stock deck that makes $P=(\gamma-1)(17.86-50)=-12.9$ throughout the background, so **95.8\% of cells sat on the pressure floor at $t=0$** (measured: $e_{\rm int}$ floor count 62816/65536, median $\beta=5.9\times10^{-39}$). The initial state was a cold, zero-pressure, magnetically dominated medium, not a Harris equilibrium.

The pgen now sets $\rho v$ and $E=p/(\gamma-1)+\rho v^2/2+B^2/2$, with the pressure taken from transverse force balance, $p(x)=p_0 n_g + (b_0^2/2)(\mathrm{sech}^2\frac{x+x_{01}}{a_0}+\mathrm{sech}^2\frac{x-x_{01}}{a_0})$, which is an exact equilibrium for any $d_0$ and $\gamma$ and reduces to the uniform-temperature form when $d_0=\gamma b_0^2/2$. Two gates confirm it: zero floored cells at $t=0$ with $p+B^2/2$ constant to output precision, and a static run ($\epsilon_b=\epsilon_v=0$, ideal MHD) whose residual $|v|_{\max}$ converges $8.1\times10^{-2}\to2.5\times10^{-2}\to8.0\times10^{-3}$ over $N=128/256/512$.

With the corrected IC every scheme reaches $t=5$ **when given its own a-posteriori protection** -- MOOD for unlimited PPM, FOFC for the others (the two are mutually exclusive, and unlimited PPM requires `mood=true`). The difference is cost, not survival: PPM+MOOD reaches $t=5$ with no EOS floor events at all, while TENO needs 2173 FOFC demotions, $\sim10^5$ floor hits, 48\% more cycles and half the timestep. FOFC never fires for PLM or WENO-Z.

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

## GR hydrodynamics (Minkowski)

### 1D linear waves

![Linear-wave mosaic grhydro 1d](figures/lwave_grhydro_1d_mosaic.png)

![PPM+MOOD demotions grhydro](figures/lwave_nmood_grhydro.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going acoustic | 256 | 1.099402e-09 | 2.05 | 0 | yes |
| plm | transverse shear z | 256 | 9.014784e-09 | 2.09 | 0 | yes |
| plm | right-going acoustic | 256 | 8.684197e-10 | 2.03 | 0 | yes |
| ppm_fb | left-going acoustic | 256 | 1.26141e-11 | 0.16 | 0 | yes |
| ppm_fb | transverse shear z | 256 | 3.325364e-12 | 1.10 | 0 | yes |
| ppm_fb | right-going acoustic | 256 | 9.192933e-12 | -0.20 | 0 | yes |
| teno | left-going acoustic | 256 | 1.251677e-11 | -0.00 | 0 | yes |
| teno | transverse shear z | 256 | 3.298609e-12 | -0.88 | 0 | yes |
| teno | right-going acoustic | 256 | 9.272032e-12 | -0.00 | 0 | yes |
| wenoz | left-going acoustic | 256 | 1.251813e-11 | -0.00 | 0 | yes |
| wenoz | transverse shear z | 256 | 3.302021e-12 | -0.89 | 0 | yes |
| wenoz | right-going acoustic | 256 | 9.271985e-12 | -0.00 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic grhydro 2d](figures/lwave_grhydro_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going acoustic | 128 | 1.351972e-08 | 1.86 | 0 | yes |
| plm | right-going acoustic | 128 | 1.623479e-08 | 1.84 | 0 | yes |
| ppm_fb | left-going acoustic | 128 | 1.528824e-11 | 4.05 | 0 | yes |
| ppm_fb | right-going acoustic | 128 | 1.426472e-11 | 4.64 | 0 | yes |
| teno | left-going acoustic | 128 | 2.215331e-12 | 4.05 | 0 | yes |
| teno | right-going acoustic | 128 | 8.710016e-12 | 2.25 | 0 | yes |
| wenoz | left-going acoustic | 128 | 2.215358e-12 | 4.05 | 0 | yes |
| wenoz | right-going acoustic | 128 | 8.71163e-12 | 2.25 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic grhydro 3d](figures/lwave_grhydro_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going acoustic | 64 | 6.227724e-08 | 1.47 | 0 | yes |
| plm | right-going acoustic | 64 | 6.453648e-08 | 1.55 | 0 | yes |
| ppm_fb | left-going acoustic | 64 | 3.043753e-10 | 3.99 | 0 | yes |
| ppm_fb | right-going acoustic | 64 | 3.683656e-10 | 3.94 | 0 | yes |
| teno | left-going acoustic | 64 | 6.745318e-11 | 4.50 | 0 | yes |
| teno | right-going acoustic | 64 | 5.922496e-11 | 4.83 | 0 | yes |
| wenoz | left-going acoustic | 64 | 6.745222e-11 | 4.53 | 0 | yes |
| wenoz | right-going acoustic | 64 | 5.922592e-11 | 4.86 | 0 | yes |

### Shocks

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| mb2_gr | 1d | plm | 256 | 0 | 0 | 5.976938486099243 |
| mb2_gr | 1d | plm | 512 | 0 | 0 | 5.3372015953063965 |
| mb2_gr | 1d | plm | 1024 | 0 | 0 | 6.497895002365112 |
| mb2_gr | 1d | ppm_fb | 256 | 0 | 13397 | 5.236825466156006 |
| mb2_gr | 1d | ppm_fb | 512 | 0 | 25142 | 6.132622480392456 |
| mb2_gr | 1d | ppm_fb | 1024 | 0 | 44057 | 7.560010671615601 |
| mb2_gr | 1d | teno | 256 | 0 | 0 | 5.054182767868042 |
| mb2_gr | 1d | teno | 512 | 0 | 0 | 5.706254482269287 |
| mb2_gr | 1d | teno | 1024 | 0 | 0 | 6.717432022094727 |
| mb2_gr | 1d | wenoz | 256 | 0 | 0 | 5.083803415298462 |
| mb2_gr | 1d | wenoz | 512 | 0 | 0 | 5.588774919509888 |
| mb2_gr | 1d | wenoz | 1024 | 0 | 0 | 6.645209789276123 |

### Multi-D stress

_No local stress smokes for this physics._

## GRMHD (Minkowski)

### 1D linear waves

![Linear-wave mosaic grmhd 1d](figures/lwave_grmhd_1d_mosaic.png)

![PPM+MOOD demotions grmhd](figures/lwave_nmood_grmhd.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going fast magnetosonic | 256 | 3.423236e-09 | 2.01 | 0 | yes |
| plm | left-going Alfvén | 256 | 2.616156e-09 | 2.02 | 0 | yes |
| plm | left-going slow magnetosonic | 256 | 9.296152e-10 | 2.05 | 0 | yes |
| plm | entropy/contact | 256 | 1.747433e-09 | 2.11 | 0 | yes |
| plm | right-going slow magnetosonic | 256 | 2.24418e-09 | 2.09 | 0 | yes |
| plm | right-going Alfvén | 256 | 3.450078e-09 | 2.05 | 0 | yes |
| plm | right-going fast magnetosonic | 256 | 1.325635e-09 | 2.01 | 0 | yes |
| ppm_fb | left-going fast magnetosonic | 256 | 5.183825e-11 | 1.22 | 0 | yes |
| ppm_fb | left-going Alfvén | 256 | 8.109603e-11 | 1.79 | 0 | yes |
| ppm_fb | left-going slow magnetosonic | 256 | 7.79768e-12 | 0.86 | 0 | yes |
| ppm_fb | entropy/contact | 256 | 6.375936e-12 | -0.99 | 0 | yes |
| ppm_fb | right-going slow magnetosonic | 256 | 8.389354e-12 | 1.78 | 0 | yes |
| ppm_fb | right-going Alfvén | 256 | 4.455398e-11 | 2.16 | 0 | yes |
| ppm_fb | right-going fast magnetosonic | 256 | 3.284096e-11 | 1.72 | 0 | yes |
| teno | left-going fast magnetosonic | 256 | 5.207129e-11 | 1.23 | 0 | yes |
| teno | left-going Alfvén | 256 | 8.121331e-11 | 1.80 | 0 | yes |
| teno | left-going slow magnetosonic | 256 | 7.743036e-12 | 0.81 | 0 | yes |
| teno | entropy/contact | 256 | 6.335483e-12 | -1.00 | 0 | yes |
| teno | right-going slow magnetosonic | 256 | 8.438445e-12 | 1.81 | 0 | yes |
| teno | right-going Alfvén | 256 | 4.454113e-11 | 2.17 | 0 | yes |
| teno | right-going fast magnetosonic | 256 | 3.290605e-11 | 1.72 | 0 | yes |
| wenoz | left-going fast magnetosonic | 256 | 5.20704e-11 | 1.23 | 0 | yes |
| wenoz | left-going Alfvén | 256 | 8.121487e-11 | 1.80 | 0 | yes |
| wenoz | left-going slow magnetosonic | 256 | 7.740979e-12 | 0.81 | 0 | yes |
| wenoz | entropy/contact | 256 | 6.319949e-12 | -1.00 | 0 | yes |
| wenoz | right-going slow magnetosonic | 256 | 8.433607e-12 | 1.81 | 0 | yes |
| wenoz | right-going Alfvén | 256 | 4.454403e-11 | 2.17 | 0 | yes |
| wenoz | right-going fast magnetosonic | 256 | 3.290819e-11 | 1.72 | 0 | yes |

### 2D linear waves

![Linear-wave mosaic grmhd 2d](figures/lwave_grmhd_2d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going fast magnetosonic | 128 | 4.470755e-08 | 1.84 | 0 | yes |
| plm | right-going fast magnetosonic | 128 | 1.988208e-08 | 1.85 | 0 | yes |
| ppm_fb | left-going fast magnetosonic | 128 | 1.369417e-09 | 2.12 | 0 | yes |
| ppm_fb | right-going fast magnetosonic | 128 | 1.320045e-09 | 2.08 | 0 | yes |
| teno | left-going fast magnetosonic | 128 | 1.334266e-09 | 2.00 | 0 | yes |
| teno | right-going fast magnetosonic | 128 | 1.295627e-09 | 2.00 | 0 | yes |
| wenoz | left-going fast magnetosonic | 128 | 1.334265e-09 | 2.00 | 0 | yes |
| wenoz | right-going fast magnetosonic | 128 | 1.295628e-09 | 2.00 | 0 | yes |

### 3D linear waves

![Linear-wave mosaic grmhd 3d](figures/lwave_grmhd_3d_mosaic.png)

| Scheme | Wave | N | L1 | Order | nmood | ok |
|---|---|---|---|---|---|---|
| plm | left-going fast magnetosonic | 64 | 1.948717e-07 | 1.46 | 0 | yes |
| plm | right-going fast magnetosonic | 64 | 8.528868e-08 | 1.46 | 0 | yes |
| ppm_fb | left-going fast magnetosonic | 64 | 2.408413e-08 | 2.22 | 0 | yes |
| ppm_fb | right-going fast magnetosonic | 64 | 1.075696e-08 | 2.21 | 0 | yes |
| teno | left-going fast magnetosonic | 64 | 2.293793e-08 | 2.04 | 0 | yes |
| teno | right-going fast magnetosonic | 64 | 1.022882e-08 | 2.03 | 0 | yes |
| wenoz | left-going fast magnetosonic | 64 | 2.293787e-08 | 2.04 | 0 | yes |
| wenoz | right-going fast magnetosonic | 64 | 1.022883e-08 | 2.03 | 0 | yes |

### Shocks

| Case | Dim | Scheme | N | rc | nmood | wall [s] |
|---|---|---|---|---|---|---|
| mub1_gr | 1d | plm | 256 | 0 | 0 | 5.131254196166992 |
| mub1_gr | 1d | plm | 512 | 0 | 0 | 5.699239015579224 |
| mub1_gr | 1d | plm | 1024 | 0 | 0 | 7.1900458335876465 |
| mub1_gr | 1d | ppm_fb | 256 | 0 | 18875 | 5.615639686584473 |
| mub1_gr | 1d | ppm_fb | 512 | 0 | 49559 | 6.514868259429932 |
| mub1_gr | 1d | ppm_fb | 1024 | 0 | 137084 | 8.86355972290039 |
| mub1_gr | 1d | teno | 256 | 0 | 0 | 5.354448318481445 |
| mub1_gr | 1d | teno | 512 | 0 | 0 | 6.118960380554199 |
| mub1_gr | 1d | teno | 1024 | 0 | 0 | 7.702042102813721 |
| mub1_gr | 1d | wenoz | 256 | 0 | 0 | 5.251308441162109 |
| mub1_gr | 1d | wenoz | 512 | 0 | 0 | 6.196975946426392 |
| mub1_gr | 1d | wenoz | 1024 | 0 | 0 | 7.92611837387085 |

### Multi-D stress

| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |
|---|---|---|---|---|---|---|
| blast_grmhd | 2d | plm_bare | **non-finite** | 16821395 | 0 | 4.175511837005615 |
| blast_grmhd | 2d | plm | **floored** | 3352 | 0 | 4.899155616760254 |
| blast_grmhd | 2d | ppm_fb | **floored** | 1994 | 12660580 | 4.9667067527771 |
| blast_grmhd | 2d | teno_bare | **non-finite** | 29086371 | 0 | 4.27195143699646 |
| blast_grmhd | 2d | teno | **floored** | 3469 | 0 | 4.7388060092926025 |
| blast_grmhd_uct | 2d | plm_bare | **floored** | 10671 | 0 | 5.430624723434448 |
| blast_grmhd_uct | 2d | plm | **floored** | 12069 | 0 | 4.817351579666138 |
| blast_grmhd_uct | 2d | ppm_fb | **floored** | 25241 | 11926124 | 10.237368106842041 |
| blast_grmhd_uct | 2d | teno_bare | **non-finite** | 41912775 | 0 | 5.322360277175903 |
| blast_grmhd_uct | 2d | teno | **non-finite** | 5805290 | 0 | 4.985810995101929 |
| blast_grmhd_uct | 2d | wenoz_bare | **non-finite** | 38741085 | 0 | 5.430042505264282 |
| blast_grmhd_uct | 2d | wenoz | **floored** | 17339 | 0 | 5.521485805511475 |
| blast_grmhd | 2d | wenoz_bare | **floored** | 97 | 0 | 5.316668272018433 |
| blast_grmhd | 2d | wenoz | **floored** | 97 | 0 | 4.378544807434082 |

![blast_grmhd stress mosaic](figures/stress_blast_grmhd_mosaic.png)

![blast_grmhd_uct stress mosaic](figures/stress_blast_grmhd_uct_mosaic.png)

### Multi-D blast (Apollo SRMHD)

Apollo `~/srmhd_blast` scheme comparison with PPM+MOOD (~/srmhd_blast on Apollo (ppm_fb vs plm/wenoz/ppmx/dc)). The mosaic uses the exact final frame ($t=4$) of each movie; full movies are linked below.

| Movie | Path |
|---|---|

## Limitations

- No MOOD in dynamical GRMHD (`dyn_grmhd`).
- `mood=true` incompatible with BH excision and with `fofc=true`; the non-MOOD schemes therefore run with `fofc=true` as their counterpart a-posteriori protection.
- MHD linear waves use `mood_nad_v=off`; velocity NAD at amp=1e-6 falsely demotes Alfvén/entropy families.
- NR MHD uses `emf=uct_hlld` + `rsolver=hlld` for all schemes; GR MHD defaults to `ct_contact`; `uct_hll` is available since the relativistic solvers were fixed to pass the transport velocity to the EMF composition.
- 2D/3D linear waves use diagonal wavevectors and cover both left- and right-going acoustic (hydro) or fast-magnetosonic (MHD) modes.
- Local 2D stress (paper-typical grids on Apollo A100s): hydro slotted cylinder (200², one rotation) / Liska–Wendroff implode (200²); MHD Orszag–Tang (400², $t=1$) / current sheet (256², $t=1$ tearing seed) / Lecoanet KH (256×512). Apollo multi-D stress: MHD ringing (`figures/apollo_ringing/`) and SRMHD blast with PPM+MOOD (`figures/apollo_grmhd/`, from `~/srmhd_blast`).
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
