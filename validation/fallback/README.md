# MOOD fallback validation

Reproducible matrix comparing **PPM+MOOD+RK3** against **PLM+RK2**, **WENOZ+RK3**, and **TENO-5+RK3** for Newtonian and fixed-background GR hydro/MHD.

## Build

```bash
cmake -B build -DAthena_ENABLE_MPI=OFF -DPROBLEM=built_in_pgens
cmake --build build --target athena -j8
```

2D stress PROBLEM binaries (slotted cylinder, KH, current sheet):

```bash
validation/fallback/scripts/build_2d_problem_binaries.sh
```

For the turbulence ringing case (cluster):

```bash
cmake -B build_turb -DAthena_ENABLE_MPI=ON -DPROBLEM=fluids/turb
cmake --build build_turb --target athena -j8
```

## Run

```bash
pip install pyyaml numpy matplotlib
python3 validation/fallback/scripts/run_suite.py \
  --athena build/src/athena --suite all
python3 validation/fallback/scripts/analyze.py --write
python3 validation/fallback/scripts/plot_results.py
python3 validation/fallback/scripts/generate_report.py --pdf
```

Suites: `linear`, `shocks`, `stress`, `all`.

## Outputs

| Path | Content | Tracked |
|------|---------|---------|
| `results/summary_*.json`, `results/analysis.json` | Aggregated results the report is built from | yes |
| `results/*/` | Per-run directories (inputs, provenance, dumps) | no |
| `figures/` | Convergence / profile / demotion plots | no |
| `FALLBACK_VALIDATION.md` | Markdown report | yes |
| `FALLBACK_VALIDATION.pdf` | PDF (via tectonic) | no |
| `slurm/` | Cluster launchers | yes |

## Regenerating the figures

`figures/` is **not tracked** -- 91 plots, ~25 MB, and every one of them is reproducible
from the tracked `results/*.json` plus the per-run dumps.  The report's image links are
therefore dead in a fresh clone until you rebuild them.

With `results/` already populated (i.e. the suites have been run), no simulation is
needed -- the plotting scripts read the dumps and the summaries:

```bash
cd validation/fallback
python3 scripts/plot_results.py                      # linear waves, shocks, tolerance sweeps
for c in $(python3 - <<'EOF'
import yaml
m = yaml.safe_load(open("manifest.yaml"))
print(" ".join(c["id"] for c in m["stress"]["local_smoke"]))
EOF
); do python3 scripts/plot_2d_stress.py --case "$c"; done
python3 scripts/plot_time_series.py --case blast_hydro --times 0.2,1,2,3 --suffix _evo
python3 scripts/plot_time_series.py --case blast_mhd   --times 0.2,1,2,3 --suffix _evo
python3 scripts/plot_time_series.py --case kh_rr22     --times 5,8,12,20
python3 scripts/generate_report.py
```

From nothing, run the suites first (see above); on a cluster use `slurm/`.

Raw bulky dumps under `results/`, the vendored python tree and the built PDF are
intentionally untracked.
