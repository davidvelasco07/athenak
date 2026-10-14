# Well-balanced MHD on `feature/fallback-validation`

Status after the 2026-10 port and extension (commits from `0e5feb62b` to the tip). The hydro
scheme is the port of the spd / low-Mach-paper method; the MHD scheme is AthenaK's own
extension to static magnetohydrostatic equilibria, `grad p = rho g + J x B`, `v = 0`.

## Options

| block / key | default | meaning |
|---|---|---|
| `<mhd>/well_balanced` | false | evolve `u0 = U - U_eq` (energy includes B^2/2); `b0` stays the full face field, updated by CT in perturbation form; the equilibrium comes from the pgen's `user_eq_func` |
| `<mhd>/wb_diagnostics` | false | equilibrium arrays and `mhd_wpert` output without the scheme (the negative control) |
| `<mhd>/mood_eps0_eq` | 0 | round-off floor of the NAD tolerance relative to the equilibrium, `eps >= mood_eps0_eq * |q_eq|` (as `<hydro>/mood_eps0_eq`) |
| `<mhd>/rsolver = lhlld` | | L-HLLD (Minoshima & Miyoshi 2021): low-Mach star pressure and carbuncle-controlled contact speed; UCT coefficients, MOOD re-solve, WB path and an isothermal branch |
| `<mhd>/emf = uct` | `ct_contact` | UCT with the coefficients that match the Riemann solver: HLLD-type (five-wave fan) for `hlld`/`lhlld`, HLL-type (outer speeds) for every other solver. `uct_hll` / `uct_hlld` still pin a variant explicitly |
| `<mhd>/mood_touched_only` | false | MOOD revisions >= 2 re-test only the cells a revised face can reach (bitwise identical, cheaper) |
| `<hydro>/ppm_boundary_closure` | true | one-sided fv4 closure at `inflow`/`user` boundaries (hydro only) |

## What is supported with `well_balanced = true` (MHD)

| feature | status | gate |
|---|---|---|
| base scheme plm, Riemann solvers llf / hlle / hlld / lhlld | exact | `mhd_gates.sh` "base scheme" |
| fv6 / fv8 + MOOD (hydro and MHD, ct_contact and uct) | exact; the face equilibrium (and the UCT equilibrium edge value) is a single-valued central interpolant at the scheme's order, added to both sides. Since 2026-10-14 the fvN base reconstruction is the two-sided cell polynomial (fv.hpp), so the equilibrium is no longer the base scheme applied to `w_eq`; the gate stays exact at rest because the perturbation states vanish there | "fv6 / fv8", "hydro", `wb_onoff_1d.sh` |
| `emf = ct_contact`, `uct_hll`, `uct_hlld` | exact (the UCT edge reconstruction acts on `b0 - b_eq` plus a single-valued equilibrium edge value) | all rows |
| MOOD fallback (`fv4 + mood`), `mood_edge = flag` and `blend`, `mood_nad_b = comps` and `mag` | exact, 0 demotions at rest | "fv4 + MOOD", `nmood_check.sh` |
| FOFC (`plm + fofc`) | exact; replacement fluxes through the face equilibrium | "plm + FOFC" |
| SMR, AMR (regrids and load balancing), MPI | exact | "SMR", "AMR", `rank_gates.sh` (an MPI build: `ranks=4` in the gate line) |
| physical boundaries (reflect / outflow / diode), user boundaries | exact; during the boundary phase `b0` holds `B' = b0 - b_eq` (see `mhd.hpp`, `wb_b0_is_pert`), so user boundary functions must write `B'` (0 for an equilibrium ghost) in that state | "physical walls" |
| ideal gas only, Newtonian only, `const_accel` sources only, no conduction / resistivity / shearing box / prolong_prims | refused at startup | |

Ghost zones with WB: the face equilibrium exists on faces `r .. n-r` with `r` = 3 (fv6), 4
(fv8), 2 otherwise, so `nghost >= r + halo` (halo = `mood_halo0` for MHD MOOD, `mood_max_revs`
for hydro MOOD, 1 for FOFC); the MOOD requirements already imply it, and `WBSetEquilibrium`
refuses anything smaller.

Face equilibrium at the scheme's order (`results/wb_onoff_1d.txt`): on a 1D hydro pulse with
MOOD detection off and `dt ~ h^2`, `max|w'_WBon - w'_WBoff|` for fv6 falls from 5.8e-11
(4th-order face equilibrium inside fv6) to 8.4e-13, for fv8 from 2.9e-11 to 5.7e-14, at
N = 32; fv4 is bitwise unchanged. The pulse response itself converges at 2nd order for
every fvN on the stratified atmosphere (cell averages are used as point values in the
nonlinear background), identically for both face-equilibrium orders at 4 digits.

Decomposition invariance of a perturbed run (1 vs 4x1 / 1x4 / 2x2 / 4x4 blocks, reassembled on
the global grid with `cmp_global.py`): bitwise for plm, plm+FOFC and fv4+MOOD with both EMF
methods, WB on and off, hlld and lhlld. 1 vs 4 MPI ranks (same MPI binary): bitwise for
fv4/fv6/fv8 + MOOD and hydro fv6 (`results/rank_gates_cpu.txt`). Note: the first 4-rank rows
of this document (2026-10-13 morning) were run with a build that had `Athena_ENABLE_MPI=OFF`,
so `mpirun -np 4` ran four serial copies and the rows were vacuous; gate.sh now prints the
rank count the code reports. For fv4+MOOD under UCT this required composing the
staggered candidate field over the whole light-cone halo and growing that halo by the edge
reach per revision (`mood_halo0 = reach * mood_max_revs`). With the central single-valued edge
interpolant that meant `nghost >= 7` for fv4+MOOD with `uct_*`; since the two-sided cell-polynomial
reconstruction of 2026-10-14 the edge reach is radius+1 (3, 4, 5 for fv4, fv6, fv8) and the
requirement with the default two revisions is `nghost >= 9 / 12 / 15` (even counts with
SMR/AMR); ct_contact is unchanged.

## Pulse sweep (`scripts/mhd/pulse_sweep.sh`, `results/mhd_pulse_sweep.{txt,png}`)

Pressure pulse of amplitude `eta` over the two magnetized atmospheres at 64^2, `max|dp|/eta`
at `t = 0.25`:

| atmosphere | scheme | eta = 1e-2 | 1e-4 | 1e-8 | 1e-12 |
|---|---|---|---|---|---|
| uniform oblique B (iprob 3) | fv4+MOOD uct_hlld, WB on | 0.241 | 0.253 | 0.253 | 0.251 |
| | WB off | 0.239 | 0.265 | 3.2e2 | 3.2e6 |
| | PLM+FOFC ct_contact, WB on | 0.229 | 0.228 | 0.228 | 0.227 |
| | WB off | 0.237 | 0.264 | 3.2e2 | 3.2e6 |
| Parker constant-beta (iprob 4) | fv4+MOOD uct_hlld, WB on | 0.362 | 0.397 | 0.343 | 0.343 |
| | WB off | 30 | 3.0e3 | 3.0e7 | 3.0e11 |
| | PLM+FOFC ct_contact, WB on | 0.354 | 0.356 | 0.358 | 0.355 |
| | WB off | 28 | 2.8e3 | 2.8e7 | 2.8e11 |

With the scheme the response is scale-invariant down to `eta = 1e-12`; without it the
truncation error of the equilibrium (Mach 0.29 on the Parker atmosphere) swamps every pulse
below `eta ~ 1e-4` on the uniform-field case and every pulse on the Parker case. MOOD
demotions with WB: 3.9e3 (iprob 3) and 7.1e3 (iprob 4) cell-stages at `eta = 1e-2`, 0 at
every smaller amplitude, against 0.9 to 1.1e6 without WB (`results/wb_nad_demotions.txt`).
The remaining amplitude dependence of the fv4+MOOD `uct_hlld` rows (0.397 at 1e-4 vs 0.343
at 1e-8 on Parker; identical with MOOD detection off) is the HLLD-UCT coefficient block of
`hlld_mhd.hpp`: its `nu*` weight is forced to 0 when the Alfvén-speed split is below
`1e-9` of the fast fan, so below `eta ~ 1e-7` the UCT weights take the degenerate branch;
`ct_contact` (0.409 / 0.409) and `uct_hll` (0.397 / 0.397) are flat. Not changed here (it
would alter round-off-level `B_x ~ 0` faces of every non-WB run); use `uct_hll` or
`ct_contact` for perturbations that small, or lower that threshold deliberately.

## MOOD detector tolerance with WB (fixed 2026-10-13)

The relaxed DMP acts on the perturbation (`rho'`, `e'`, and in MHD `v`, `B'`), but its
TOLERANCE is the one the same run would use without WB: relative bands and global scales
from the full state `w_eq + w'`, the rcfl/gcfl Courant factor from the signal speed
`|v| + c_s` (`|v| + c_f` in MHD) instead of the advective one, the MHD velocity band
relative to `|v| + c_f` and the B-component band to the full-state bounds plus the local
`|B_eq|` (components with a vanishing equilibrium value have no scale of their own). Before
this, every one of those scales was the perturbation's own (advective Courant number =
the pulse's Mach number, `|v| ~ |B'_2| ~ eta`), so the band shrank with the amplitude and a
fixed fraction of cell-steps was demoted at ANY amplitude: 1D hydro pulse of `1e-3` at
256 cells, 2975 demotions (WB off: 0); now 0. A strong pulse (amp 1) still demotes (2875,
WB off 941), the `-0.95` MHD pulse 5.6e4 (WB off 5.4e5). WB-off paths are bitwise
unchanged; the exact gates, 0 demotions at rest, 1 vs 4x4 blocks and 1 vs 4 ranks hold
(`wb_nad_pulse.sh`, `wb_nad_knobs_*.sh`).

## Running the gates

```bash
export ATH=/path/to/build/src/athena
export RUNS=/some/persistent/dir   # copy scripts/mhd/{mkdeck.py,stats.py,cmp_global.py} there
validation/wellbalance/scripts/mhd/mhd_gates.sh     # exact-gate matrix (wb=true rows: all 0.000e+00)
validation/wellbalance/scripts/mhd/nmood_check.sh   # 0 demotions at rest (event log has no rows)
validation/wellbalance/scripts/mhd/pulse_sweep.sh   # the table above
RUNS=... validation/wellbalance/scripts/mhd/rank_gates.sh $ATH_MPI   # 4-rank exact gates + 1 vs 4 bitwise
RUNS=... ATHENAK_VIS=vis/python validation/wellbalance/scripts/mhd/wb_onoff_1d.sh $REF $NEW
```

`mkdeck.py` refuses unknown keys (free-form lines go through `extra=` / `tail=`); the
`pulse_x1/pulse_x2`, `mood_eps0_eq` and `mood_detect` keys exist for the checks above.

`stats.py` prints `max|q|` of every variable of a dump; the exact gate reads the `mhd_wpert`
dump (W - W_eq and B - B_eq in double precision, since the dumps are single precision).
Raw `.bin` files embed the input deck and differ across block decompositions, so compare
payloads with `cmp.py` (same decomposition) or reassembled grids with `cmp_global.py`.

## Not done

Moving equilibria, isothermal WB; the fv6/fv8 rows and `rank_gates.sh` on GPU (queued).

## GPU (Vista GH200, CUDA build, `results/mhd_gates_gpu.txt`, `results/mhd_gpu_dec_sweep.txt`)

The 2026-10-13 matrix (plm / fv4+MOOD / plm+FOFC x ct_contact / uct_hll / uct_hlld x hlld /
lhlld / hlle / llf, SMR, walls, AMR band, 4 ranks under `ibrun`) is exact on the GPU: every
`wb=true` row 0.000e+00 in all perturbation columns, controls as on CPU, so FMA contraction
does not break the `f - f_eq` cancellation. 0 demotions at rest (6.98e5 without WB on the
arcade), 1 vs 4x4 blocks bitwise for WB on and off, and the pulse sweep reproduces the CPU
table (`max|dp|/eta` 0.236-0.238 for fv4+MOOD, 0.227-0.229 for PLM+FOFC on iprob 3). Vista's
`python3` has no h5py: `vis/python/bin_convert.py` now imports it lazily.

The over-demotion of WB + MOOD on smooth pulses reported earlier in this file is fixed (see
"MOOD detector tolerance with WB"); the `uct_hlld` small-amplitude branch above is open.
