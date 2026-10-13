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
| `<mhd>/mood_touched_only` | false | MOOD revisions >= 2 re-test only the cells a revised face can reach (bitwise identical, cheaper) |
| `<hydro>/ppm_boundary_closure` | true | one-sided fv4 closure at `inflow`/`user` boundaries (hydro only) |

## What is supported with `well_balanced = true` (MHD)

| feature | status | gate |
|---|---|---|
| base scheme plm, Riemann solvers llf / hlle / hlld / lhlld | exact | `mhd_gates.sh` "base scheme" |
| `emf = ct_contact`, `uct_hll`, `uct_hlld` | exact (the UCT edge reconstruction acts on `b0 - b_eq` plus a single-valued equilibrium edge value) | all rows |
| MOOD fallback (`fv4 + mood`), `mood_edge = flag` and `blend`, `mood_nad_b = comps` and `mag` | exact, 0 demotions at rest | "fv4 + MOOD", `nmood_check.sh` |
| FOFC (`plm + fofc`) | exact; replacement fluxes through the face equilibrium | "plm + FOFC" |
| SMR, AMR (regrids and load balancing), MPI | exact | "SMR", "AMR", `4` rank rows |
| physical boundaries (reflect / outflow / diode), user boundaries | exact; during the boundary phase `b0` holds `B' = b0 - b_eq` (see `mhd.hpp`, `wb_b0_is_pert`), so user boundary functions must write `B'` (0 for an equilibrium ghost) in that state | "physical walls" |
| ideal gas only, Newtonian only, `const_accel` sources only, no conduction / resistivity / shearing box / prolong_prims | refused at startup | |

Decomposition invariance of a perturbed run (1 vs 4x1 / 1x4 / 2x2 / 4x4 blocks, reassembled on
the global grid with `cmp_global.py`): bitwise for plm, plm+FOFC and fv4+MOOD with both EMF
methods, WB on and off, hlld and lhlld. For fv4+MOOD under UCT this required composing the
staggered candidate field over the whole light-cone halo and growing that halo by the edge
reach per revision (`mood_halo0 = reach * mood_max_revs`), so fv4+MOOD with `uct_*` now needs
`nghost >= 7` (8 with SMR/AMR, which need an even count); ct_contact is unchanged.

## Pulse sweep (`scripts/mhd/pulse_sweep.sh`, `results/mhd_pulse_sweep.{txt,png}`)

Pressure pulse of amplitude `eta` over the two magnetized atmospheres at 64^2, `max|dp|/eta`
at `t = 0.25`:

| atmosphere | scheme | eta = 1e-2 | 1e-4 | 1e-8 | 1e-12 |
|---|---|---|---|---|---|
| uniform oblique B (iprob 3) | fv4+MOOD uct_hlld, WB on | 0.236 | 0.237 | 0.237 | 0.237 |
| | WB off | 0.239 | 0.265 | 3.2e2 | 3.2e6 |
| | PLM+FOFC ct_contact, WB on | 0.229 | 0.228 | 0.228 | 0.227 |
| | WB off | 0.237 | 0.264 | 3.2e2 | 3.2e6 |
| Parker constant-beta (iprob 4) | fv4+MOOD uct_hlld, WB on | 0.361 | 0.361 | 0.222 | 0.221 |
| | WB off | 30 | 3.0e3 | 3.0e7 | 3.0e11 |
| | PLM+FOFC ct_contact, WB on | 0.354 | 0.356 | 0.358 | 0.355 |
| | WB off | 28 | 2.8e3 | 2.8e7 | 2.8e11 |

With the scheme the response is scale-invariant down to `eta = 1e-12`; without it the
truncation error of the equilibrium (Mach 0.29 on the Parker atmosphere) swamps every pulse
below `eta ~ 1e-4` on the uniform-field case and every pulse on the Parker case. MOOD
demotions with WB grow from 1.1e5 to 3.0e5 cell-stages as `eta` falls (round-off fence
sitting at `mood_eps0_eq = 0`, as in hydro) against 0.9 to 1.1e6 without.

## Running the gates

```bash
export ATH=/path/to/build/src/athena
export RUNS=/some/persistent/dir   # copy scripts/mhd/{mkdeck.py,stats.py,cmp_global.py} there
validation/wellbalance/scripts/mhd/mhd_gates.sh     # exact-gate matrix (wb=true rows: all 0.000e+00)
validation/wellbalance/scripts/mhd/nmood_check.sh   # 0 demotions at rest (event log has no rows)
validation/wellbalance/scripts/mhd/pulse_sweep.sh   # the table above
```

`stats.py` prints `max|q|` of every variable of a dump; the exact gate reads the `mhd_wpert`
dump (W - W_eq and B - B_eq in double precision, since the dumps are single precision).
Raw `.bin` files embed the input deck and differ across block decompositions, so compare
payloads with `cmp.py` (same decomposition) or reassembled grids with `cmp_global.py`.

## Not done

GPU runs of the MHD gates (FMA may break `f - f_eq` cancellation; the hydro scheme was
checked on GH200 earlier), fv6/fv8 with WB (the face equilibrium is the 4th-order
interpolant; the gates above are fv4 and plm), moving equilibria, isothermal WB.
