#!/usr/bin/env python3
"""Generate Markdown + LaTeX/PDF reports organized by physics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

VAL = Path(__file__).resolve().parents[1]
RESULTS = VAL / "results"
FIGURES = VAL / "figures"
REPORT = VAL / "report"
MANIFEST = VAL / "manifest.yaml"
VENDOR = VAL / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

PHYSICS = [
    ("hydro", "Hydrodynamics"),
    ("mhd", "Magnetohydrodynamics"),
    ("grhydro", "GR hydrodynamics (Minkowski)"),
    ("grmhd", "GRMHD (Minkowski)"),
]
DIMS = [("1d", "1D"), ("2d", "2D"), ("3d", "3D")]
CASE_TITLE = {
    "slotted_cyl": "Slotted disk",
    "implode_hydro": "Liska--Wendroff implosion",
    "orszag_tang": r"Orszag--Tang vortex ($t=1$)",
    "current_sheet": r"Double Harris current sheet ($t=5$, tearing seed)",
    "current_sheet_n512": r"Current sheet, $512^2$",
    "current_sheet_n1024": r"Current sheet, $1024^2$",
    "kh_rr22": r"Magnetized Kelvin--Helmholtz (Rueda-Ram\'irez+ 2022)",
    "jet": r"Underdense Mach-10 MHD jet",
    "jet_n600": r"MHD jet, $600\times500$",
    "rotor": r"MHD rotor (Balsara \& Spicer)",
    "blast_hydro": "Hydrodynamic blast",
    "blast_mhd": "Magnetized blast",
    "blast_grmhd": "Relativistic magnetized blast (Minkowski)",
    "sod": "Sod",
    "shu_osher": "Shu--Osher",
    "rj2a": "Ryu--Jones 2a",
    "bw": "Brio--Wu",
    "mb2_gr": "Marti--Muller blast 2",
    "mub1_gr": "Mignone--Ugliano--Bodo 1",
}


def load_manifest():
    try:
        import yaml
    except ImportError:
        return {}
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST) as f:
        return yaml.safe_load(f) or {}


def md_escape(s):
    return str(s).replace("|", "\\|")


def _by_physics(items, key="physics"):
    out = defaultdict(list)
    for it in items:
        out[it.get(key) or "unknown"].append(it)
    return out


def write_markdown(analysis, commit, manif):
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# MOOD Fallback Validation Report")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append(f"_Commit `{commit}`_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    gates = analysis.get("gates") or {}
    lines.append(f"- Records analyzed: **{analysis.get('n_records', 0)}**")
    lines.append(
        f"- PPM+MOOD linear-wave zero demotions: **{gates.get('ppm_fb_linear_zero_demotions')}**"
    )
    lines.append(f"- All runs completed (rc=0): **{gates.get('all_completed')}**")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "We compare unlimited PPM with a-posteriori MOOD fallback (RK3) against "
        "PLM+RK2, WENOZ+RK3, and TENO-5+RK3. Detection uses NAD "
        "(`mood_nad_scale=gcfl`, `mood_rtol=1e-5`; `mood_nad_v=off` for MHD *linear waves* "
        "only -- see the note below) plus "
        "PAD/NaN. Passive-scalar concentrations are included in the relaxed-DMP test, "
        "so scalar discontinuities can trigger fallback even on a uniform-density "
        "kinematic background. NAD demotions are floored at PLM.\n\n"
        "**Each scheme carries its own a-posteriori protection.** MOOD and FOFC are "
        "mutually exclusive in the code, and unlimited PPM *requires* `mood=true`, so "
        "the like-for-like comparison gives PPM the MOOD cascade and gives "
        "PLM / WENO-Z / TENO first-order flux correction (`fofc=true`). Benchmarking "
        "MOOD against unprotected competitors would overstate it: on the current "
        "sheet, TENO fails outright without FOFC and survives with it. `fofc` is a "
        "per-scheme property of the manifest so it cannot silently differ per case. "
        "MOOD is unsupported with BH excision. Newtonian MHD uses "
        "`emf=uct_hlld` for all schemes. Relativistic MHD defaults to `ct_contact` and may "
        "opt into `uct_hll` per case (`uct_hlld` needs `rsolver=hlld`, which SR/GR "
        "reject); the `blast_grmhd` / `blast_grmhd_uct` pair runs both.\n\n"
        "**`returncode == 0` is not evidence of success.** A run whose state has been "
        "wiped still prints \"Terminating on time limit\": on the current sheet, "
        "unprotected TENO reported rc=0 with an all-NaN final dump. Every stress run "
        "is therefore health-gated on the EOS floor/failure counters and the timestep "
        "history (`dt_collapse` when $\\Delta t_{\\min} \\le 0$ or "
        "$< 10^{-6}\\Delta t_{\\max}$)."
    )
    lines.append("")
    lines.append(
        "The report is organized by physics. Each section covers **1D / 2D / 3D** "
        "linear waves (convergence mosaics with shared axes), plus 1D shocks and "
        "available multi-D stress cases."
    )
    lines.append("")

    lin_by = _by_physics(analysis.get("linear", []))
    shock_by = _by_physics(analysis.get("shocks", []))
    stress_by = _by_physics(analysis.get("stress", []))

    for phys_id, phys_title in PHYSICS:
        lines.append(f"## {phys_title}")
        lines.append("")

        # --- Linear waves by dimension ---
        for dim_id, dim_title in DIMS:
            items = [
                f
                for f in lin_by.get(phys_id, [])
                if (f.get("dim") or "1d") == dim_id
            ]
            lines.append(f"### {dim_title} linear waves")
            lines.append("")
            if not items:
                lines.append("_No results yet for this dimension._")
                lines.append("")
                continue
            incomplete = [f for f in items if not f.get("completed")]
            if incomplete and phys_id == "grmhd" and dim_id in ("2d", "3d"):
                lines.append(
                    "_Note: some local CPU GRMHD multi-D linear-wave exits were "
                    "incomplete; Apollo SRMHD blast movies below are the multi-D "
                    "stress reference for PPM+MOOD._"
                )
                lines.append("")
            demoting = []
            for f in items:
                if f.get("scheme") != "ppm_fb":
                    continue
                if any((m or 0) > 0 for m in (f.get("nmood") or [])):
                    demoting.append(f.get("wave_name", f.get("wave")))
            if demoting:
                lines.append(
                    "_Zero-demotion gate fails for PPM+MOOD in this panel: "
                    + ", ".join(sorted(set(demoting)))
                    + "._"
                )
                lines.append("")
            mosaic = FIGURES / f"lwave_{phys_id}_{dim_id}_mosaic.png"
            if mosaic.exists():
                lines.append(
                    f"![Linear-wave mosaic {phys_id} {dim_id}]"
                    f"({mosaic.relative_to(VAL)})"
                )
                lines.append("")
            nmood = FIGURES / f"lwave_nmood_{phys_id}.png"
            if dim_id == "1d" and nmood.exists():
                lines.append(
                    f"![PPM+MOOD demotions {phys_id}]({nmood.relative_to(VAL)})"
                )
                lines.append("")
            lines.append("| Scheme | Wave | N | L1 | Order | nmood | ok |")
            lines.append("|---|---|---|---|---|---|---|")
            for f in items:
                orders = f.get("orders") or []
                order = f"{orders[-1]:.2f}" if orders else "—"
                # last non-null L1 if highest-N failed to write
                l1 = None
                for v in reversed(f.get("l1") or []):
                    if v is not None:
                        l1 = v
                        break
                nm = None
                for v in reversed(f.get("nmood") or []):
                    if v is not None:
                        nm = v
                        break
                n = f["resolutions"][-1] if f.get("resolutions") else "—"
                ok = "yes" if f.get("completed") else ("partial" if f.get("usable") else "no")
                lines.append(
                    f"| {f['scheme']} | {f.get('wave_name', f['wave'])} | {n} | "
                    f"{l1 if l1 is not None else '—'} | {order} | "
                    f"{nm if nm is not None else '—'} | {ok} |"
                )
            lines.append("")

        # --- Shocks (usually 1D) ---
        shocks = shock_by.get(phys_id, [])
        lines.append("### Shocks")
        lines.append("")
        if shocks:
            for fig in sorted(FIGURES.glob("shock_*.png")):
                # include only shocks belonging to this physics
                case = fig.stem.replace("shock_", "").replace("_dens", "")
                if any(s["case"] == case for s in shocks):
                    lines.append(f"![{fig.stem}]({fig.relative_to(VAL)})")
                    lines.append("")
            lines.append("| Case | Dim | Scheme | N | rc | nmood | wall [s] |")
            lines.append("|---|---|---|---|---|---|---|")
            for s in shocks:
                lines.append(
                    f"| {s['case']} | {s.get('dim','1d')} | {s['scheme']} | "
                    f"{s['nx1']} | {s['rc']} | {s.get('nmood')} | {s.get('wall_s')} |"
                )
            lines.append("")
        else:
            lines.append("_No shock cases for this physics._")
            lines.append("")

        # --- Multi-D stress ---
        stress = stress_by.get(phys_id, [])
        lines.append("### Multi-D stress")
        lines.append("")
        if stress:
            lines.append("| Case | Dim | Scheme | health | EOS floors | nmood | wall [s] |")
            lines.append("|---|---|---|---|---|---|---|")
            for s in stress:
                v = s.get("health", "—")
                lines.append(
                    f"| {s.get('case')} | {s.get('dim','—')} | {s.get('scheme')} | "
                    f"{'**' + v + '**' if v not in ('clean', '—') else v} | "
                    f"{s.get('floors') if s.get('floors') is not None else '—'} | "
                    f"{s.get('nmood_total')} | {s.get('wall_s')} |"
                )
            lines.append("")
            # 2D snapshot mosaics for this physics
            case_ids = sorted({s.get("case") for s in stress if s.get("case")})
            for cid in case_ids:
                if cid == "slotted_cyl":
                    lines.append(
                        "The slotted-cylinder test advects a bounded passive scalar "
                        "through one full rotation on a $200^2$ mesh. Scalar-aware NAD "
                        "confines fallback to the discontinuity and suppresses PPM ringing."
                    )
                    lines.append("")
                elif cid == "orszag_tang":
                    lines.append(
                        "The Orszag--Tang vortex is shown at $t=1$ on the classic "
                        "$400^2$ Athena mesh."
                    )
                    lines.append("")
                elif cid == "current_sheet":
                    lines.append(
                        "The double Harris sheet is seeded with the pgen tearing "
                        "perturbation and run to $t=5$, a few Alfv\\'en times, so "
                        "plasmoid chains have formed on both sheets.\n\n"
                        "**This case previously reported only $t=1$.** The reason "
                        "every high-order scheme collapsed before plasmoid time was "
                        "an initial-condition defect, not a limiter failure: "
                        "`current_sheet.cpp` wrote *primitive* variables into the "
                        "*conserved* array -- momenta received $v$ rather than "
                        "$\\rho v$, and the total energy omitted both "
                        "$\\rho v^2/2$ and $B^2/2$. In the stock deck that makes "
                        "$P=(\\gamma-1)(17.86-50)=-12.9$ throughout the background, "
                        "so **95.8\\% of cells sat on the pressure floor at $t=0$** "
                        "(measured: $e_{\\rm int}$ floor count 62816/65536, median "
                        "$\\beta=5.9\\times10^{-39}$). The initial state was a cold, "
                        "zero-pressure, magnetically dominated medium, not a Harris "
                        "equilibrium.\n\n"
                        "The pgen now sets $\\rho v$ and "
                        "$E=p/(\\gamma-1)+\\rho v^2/2+B^2/2$, with the pressure "
                        "taken from transverse force balance, "
                        "$p(x)=p_0 n_g + (b_0^2/2)("
                        "\\mathrm{sech}^2\\frac{x+x_{01}}{a_0}"
                        "+\\mathrm{sech}^2\\frac{x-x_{01}}{a_0})$, which is an "
                        "exact equilibrium for any $d_0$ and $\\gamma$ and reduces "
                        "to the uniform-temperature form when "
                        "$d_0=\\gamma b_0^2/2$. Two gates confirm it: zero floored "
                        "cells at $t=0$ with $p+B^2/2$ constant to output precision, "
                        "and a static run ($\\epsilon_b=\\epsilon_v=0$, ideal MHD) "
                        "whose residual $|v|_{\\max}$ converges "
                        "$8.1\\times10^{-2}\\to2.5\\times10^{-2}"
                        "\\to8.0\\times10^{-3}$ over $N=128/256/512$.\n\n"
                        "With the corrected IC every scheme reaches $t=5$ **when "
                        "given its own a-posteriori protection** -- MOOD for "
                        "unlimited PPM, FOFC for the others (the two are mutually "
                        "exclusive, and unlimited PPM requires `mood=true`). The "
                        "difference is cost, not survival: PPM+MOOD reaches $t=5$ "
                        "with no EOS floor events at all, while TENO needs 2173 FOFC "
                        "demotions, $\\sim10^5$ floor hits, 48\\% more cycles and "
                        "half the timestep. FOFC never fires for PLM or WENO-Z."
                    )
                    lines.append("")
                elif cid == "kh_rr22":
                    lines.append(
                        "Magnetized Kelvin--Helmholtz of Rueda-Ram\\'irez, "
                        "Hindenlang, Chan \\& Gassner (2022, arXiv:2203.06062) "
                        "section 5.2, replacing the Lecoanet MHD KH: the "
                        "perturbation is a single deterministic mode (no random seed "
                        "to reproduce across schemes or decompositions) and the shear "
                        "layer $y_0=1/20$ is resolved. The field is tilted in the "
                        "$xz$ plane, so the toroidal $B_z$ is nonzero and the run is "
                        "genuinely pseudo-2D."
                    )
                    lines.append("")
                mosaic = FIGURES / f"stress_{cid}_mosaic.png"
                if mosaic.exists():
                    rel = f"figures/stress_{cid}_mosaic.png"
                    lines.append(f"![{cid} stress mosaic]({rel})")
                    lines.append("")
                bmos = FIGURES / f"stress_{cid}_bmag_mosaic.png"
                if bmos.exists():
                    rel = f"figures/stress_{cid}_bmag_mosaic.png"
                    lines.append(f"![{cid} |B| mosaic]({rel})")
                    lines.append("")
        else:
            lines.append("_No local stress smokes for this physics._")
            lines.append("")

        if phys_id == "hydro":
            tolerance = analysis.get("tolerance") or []
            lines.append("### NAD tolerance study")
            lines.append("")
            if tolerance:
                lines.append(
                    "Relaxed-DMP tolerance sweeps keep the base scheme, grid, Riemann "
                    "solver, and integrator fixed. The Liska--Wendroff implosion "
                    "probes interacting shocks and contacts; the slotted disk isolates "
                    "the passive-scalar detector."
                )
                lines.append("")
                by_case = defaultdict(list)
                for row in tolerance:
                    by_case[row.get("case") or "unknown"].append(row)
                for cid, case_rows in by_case.items():
                    title = CASE_TITLE.get(cid, cid)
                    lines.append(f"#### {title}")
                    lines.append("")
                    stem = (
                        "tolerance_implode_hydro"
                        if cid == "implode_hydro"
                        else f"tolerance_{cid}"
                    )
                    for name, alt in (
                        (f"{stem}_mosaic.png", f"{title} snapshot mosaic"),
                        (f"{stem}_demotions.png", f"{title} demotion curve"),
                    ):
                        fig = FIGURES / name
                        if fig.exists():
                            lines.append(f"![{alt}](figures/{name})")
                            lines.append("")
                    lines.append("| rtol | N | rc | nmood | wall [s] |")
                    lines.append("|---|---|---|---|---|")
                    for row in case_rows:
                        lines.append(
                            f"| {row.get('rtol'):.0e} | {row.get('nx1')} | "
                            f"{row.get('returncode')} | {row.get('nmood_total')} | "
                            f"{row.get('wall_s')} |"
                        )
                    lines.append("")
            else:
                lines.append("_Tolerance sweep pending._")
                lines.append("")

        # MHD 3D ringing: single Apollo mosaic
        if phys_id == "mhd":
            ring = manif.get("apollo_ringing_figure") or (
                "figures/apollo_ringing/mosaic_uct_hlld_256_t1p5.png"
            )
            ring_path = VAL / ring
            lines.append("### 3D ringing (Apollo)")
            lines.append("")
            lines.append(
                "Latest UCT-HLLD ringing mosaic from Apollo "
                "`feature/fallback` @ `d60ab73` (256³)."
            )
            lines.append("")
            if ring_path.exists():
                lines.append(f"![Ringing UCT-HLLD 256]({ring})")
                lines.append("")

        # GRMHD multi-D blast from Apollo ~/srmhd_blast
        if phys_id == "grmhd":
            gcfg = manif.get("apollo_grmhd_blast") or {}
            gmos = VAL / (
                gcfg.get("mosaic")
                or "figures/apollo_grmhd/srmhd_blast_final_mosaic.png"
            )
            lines.append("### Multi-D blast (Apollo SRMHD)")
            lines.append("")
            lines.append(
                "Apollo `~/srmhd_blast` scheme comparison with PPM+MOOD "
                f"({gcfg.get('source', 'ppm_fb vs PLM/WENOZ/PPMX/DC')}). "
                "The mosaic uses the exact final frame ($t=4$) of each movie; full movies "
                "are linked below."
            )
            lines.append("")
            if gmos.exists():
                lines.append(
                    f"![SRMHD blast PPM+MOOD mosaic]"
                    f"({gmos.relative_to(VAL)})"
                )
                lines.append("")
            movies = gcfg.get("movies") or []
            if movies:
                lines.append("| Movie | Path |")
                lines.append("|---|---|")
                for m in movies:
                    p = VAL / m
                    if p.exists():
                        lines.append(f"| `{p.name}` | [{m}]({m}) |")
                lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- No MOOD in dynamical GRMHD (`dyn_grmhd`).")
    lines.append(
        "- `mood=true` incompatible with BH excision and with `fofc=true`; the "
        "non-MOOD schemes therefore run with `fofc=true` as their counterpart "
        "a-posteriori protection."
    )
    lines.append(
        "- MHD linear waves use `mood_nad_v=off`; velocity NAD at amp=1e-6 "
        "falsely demotes Alfvén/entropy families."
    )
    lines.append(
        "- NR MHD uses `emf=uct_hlld` + `rsolver=hlld` for all schemes; "
        "GR MHD defaults to `ct_contact`; `uct_hll` is available since the relativistic "
        "solvers were fixed to pass the transport velocity to the EMF composition."
    )
    lines.append(
        "- 2D/3D linear waves use diagonal wavevectors and cover both left- and "
        "right-going acoustic (hydro) or fast-magnetosonic (MHD) modes."
    )
    lines.append(
        "- Local 2D stress (paper-typical grids on Apollo A100s): hydro slotted cylinder "
        "(200², one rotation) / Liska–Wendroff implode (200²); "
        "MHD Orszag–Tang (400², $t=1$) / current sheet (256², $t=1$ tearing seed) / "
        "Lecoanet KH (256×512). "
        "Apollo multi-D stress: MHD ringing (`figures/apollo_ringing/`) and "
        "SRMHD blast with PPM+MOOD (`figures/apollo_grmhd/`, from `~/srmhd_blast`)."
    )
    lines.append(
        "- The stock current-sheet input is a resistive-diffusion test "
        "(`epsb=0`, `epsv=0.001`). The suite now enables the pgen tearing seed "
        "(`epsb=0.05`, `epsv=0.01`). High-order schemes still collapse near "
        "$t=1$; the mosaic uses the last common finite time ($t=1$). TENO has "
        "no finite evolved dump."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("cmake -B build -DAthena_ENABLE_MPI=OFF -DPROBLEM=built_in_pgens")
    lines.append("cmake --build build --target athena -j8")
    lines.append("validation/fallback/scripts/build_2d_problem_binaries.sh")
    lines.append(
        "python3 validation/fallback/scripts/run_suite.py "
        "--athena build/src/athena --suite all"
    )
    lines.append("python3 validation/fallback/scripts/analyze.py --write")
    lines.append("python3 validation/fallback/scripts/plot_results.py")
    lines.append("python3 validation/fallback/scripts/plot_2d_stress.py")
    lines.append("python3 validation/fallback/scripts/generate_report.py --pdf")
    lines.append("```")
    lines.append("")
    out = VAL / "FALLBACK_VALIDATION.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return out


def write_latex(analysis, commit, manif):
    REPORT.mkdir(parents=True, exist_ok=True)
    body = []
    body.append(r"\documentclass[11pt]{article}")
    body.append(r"\usepackage{graphicx,booktabs,hyperref,geometry,lmodern,microtype,float,xcolor}")
    body.append(r"\usepackage[font=small,labelfont=bf]{caption}")
    body.append(r"\geometry{margin=0.8in}")
    body.append(r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue!50!black}")
    body.append(r"\title{MOOD Fallback Validation Report}")
    body.append(r"\author{AthenaK feature/fallback-validation}")
    body.append(r"\date{" + datetime.now(timezone.utc).strftime("%Y-%m-%d") + r"}")
    body.append(r"\begin{document}")
    body.append(r"\maketitle")
    body.append(r"\noindent Commit \texttt{" + commit + r"}.")
    gates = analysis.get("gates") or {}
    body.append(r"\section{Summary}")
    body.append(
        f"Records: {analysis.get('n_records', 0)}. "
        f"PPM+MOOD linear zero demotions: {gates.get('ppm_fb_linear_zero_demotions')}. "
        f"All completed: {gates.get('all_completed')}."
    )
    body.append(r"\section{Method}")
    body.append(
        r"Unlimited PPM with RK3 and a-posteriori MOOD fallback is compared with "
        r"PLM+RK2, WENOZ+RK3, and TENO-5+RK3. The relaxed-DMP detector includes "
        r"passive-scalar concentrations; scalar discontinuities therefore trigger "
        r"fallback even for uniform-density kinematic advection."
    )
    for phys_id, phys_title in PHYSICS:
        body.append(r"\clearpage")
        body.append(rf"\section{{{phys_title}}}")
        for dim_id, dim_title in DIMS:
            mosaic = FIGURES / f"lwave_{phys_id}_{dim_id}_mosaic.png"
            if mosaic.exists():
                body.append(rf"\subsection{{{dim_title} linear waves}}")
                body.append(r"\begin{figure}[H]\centering")
                body.append(
                    rf"\includegraphics[width=0.95\linewidth,height=0.78\textheight,"
                    rf"keepaspectratio]{{{mosaic.resolve()}}}"
                )
                body.append(rf"\caption{{{phys_title} {dim_title} linear-wave mosaic}}")
                body.append(r"\end{figure}")
        shock_cases = sorted(
            {
                s.get("case")
                for s in (analysis.get("shocks") or [])
                if s.get("physics") == phys_id and s.get("case")
            }
        )
        shock_figs = [
            (case, FIGURES / f"shock_{case}_dens.png")
            for case in shock_cases
            if (FIGURES / f"shock_{case}_dens.png").exists()
        ]
        if shock_figs:
            body.append(r"\subsection{Shock tubes}")
            for case, shock in shock_figs:
                body.append(r"\begin{figure}[H]\centering")
                body.append(
                    rf"\includegraphics[width=0.95\linewidth,height=0.78\textheight,"
                    rf"keepaspectratio]{{{shock.resolve()}}}"
                )
                body.append(
                    rf"\caption{{{CASE_TITLE.get(case, case)} density profile}}"
                )
                body.append(r"\end{figure}")
        # 2D stress mosaics for this physics
        stress_cases = [
            c
            for c in ((manif.get("stress") or {}).get("local_smoke") or [])
            if c.get("physics") == phys_id
        ]
        if stress_cases:
            body.append(r"\subsection{Multi-D stress}")
            for c in stress_cases:
                cid = c["id"]
                if cid == "slotted_cyl":
                    body.append(
                        r"The bounded passive scalar is advected through one complete "
                        r"rotation on a $200^2$ mesh. Scalar-aware NAD confines fallback "
                        r"to the discontinuity and suppresses unlimited-PPM ringing."
                    )
                elif cid == "kh_mhd":
                    body.append(
                        r"The Lecoanet MHD Kelvin--Helmholtz test includes a passive "
                        r"scalar and uses the same scalar-aware NAD criterion."
                    )
                for suffix, cap in (
                    ("_mosaic.png", "density/scalar"),
                    ("_bmag_mosaic.png", r"$|B|$"),
                ):
                    mosaic = FIGURES / f"stress_{cid}{suffix}"
                    if mosaic.exists():
                        body.append(r"\begin{figure}[H]\centering")
                        body.append(
                            rf"\includegraphics[width=0.95\linewidth,height=0.78\textheight,"
                            rf"keepaspectratio]{{{mosaic.resolve()}}}"
                        )
                        body.append(
                            rf"\caption{{{CASE_TITLE.get(cid, cid)} {cap} mosaic}}"
                        )
                        body.append(r"\end{figure}")
        if phys_id == "hydro" and (analysis.get("tolerance") or []):
            body.append(r"\subsection{NAD tolerance study}")
            body.append(
                r"Relaxed-DMP tolerance sweeps keep the base scheme, grid, Riemann "
                r"solver, and integrator fixed. The implosion probes interacting shocks; "
                r"the slotted disk isolates the passive-scalar detector."
            )
            by_case = defaultdict(list)
            for row in analysis.get("tolerance") or []:
                by_case[row.get("case") or "unknown"].append(row)
            for cid in by_case:
                title = CASE_TITLE.get(cid, cid)
                stem = (
                    "tolerance_implode_hydro"
                    if cid == "implode_hydro"
                    else f"tolerance_{cid}"
                )
                captions = (
                    (
                        f"{stem}_mosaic.png",
                        rf"{title} final field across NAD tolerances",
                    ),
                    (
                        f"{stem}_demotions.png",
                        rf"{title}: cumulative MOOD demotions versus NAD tolerance",
                    ),
                )
                for name, caption in captions:
                    fig = FIGURES / name
                    if fig.exists():
                        body.append(r"\begin{figure}[H]\centering")
                        body.append(
                            rf"\includegraphics[width=0.95\linewidth,height=0.78\textheight,"
                            rf"keepaspectratio]{{{fig.resolve()}}}"
                        )
                        body.append(rf"\caption{{{caption}}}")
                        body.append(r"\end{figure}")
        if phys_id == "mhd":
            ring = VAL / (
                manif.get("apollo_ringing_figure")
                or "figures/apollo_ringing/mosaic_uct_hlld_256_t1p5.png"
            )
            if ring.exists():
                body.append(r"\subsection{3D ringing (Apollo)}")
                body.append(r"\begin{figure}[H]\centering")
                body.append(
                    rf"\includegraphics[width=0.95\linewidth,height=0.78\textheight,"
                    rf"keepaspectratio]{{{ring.resolve()}}}"
                )
                body.append(r"\caption{Latest Apollo UCT-HLLD ringing mosaic (256$^3$)}")
                body.append(r"\end{figure}")
        if phys_id == "grmhd":
            gcfg = manif.get("apollo_grmhd_blast") or {}
            gmos = VAL / (
                gcfg.get("mosaic")
                or "figures/apollo_grmhd/srmhd_blast_final_mosaic.png"
            )
            if gmos.exists():
                body.append(r"\subsection{Multi-D blast (Apollo SRMHD)}")
                body.append(r"\begin{figure}[H]\centering")
                body.append(
                    rf"\includegraphics[width=0.95\linewidth,height=0.78\textheight,"
                    rf"keepaspectratio]{{{gmos.resolve()}}}"
                )
                body.append(
                    r"\caption{Apollo SRMHD blast final frames at $t=4$: "
                    r"PLM / WENOZ / PPM+MOOD density and PPM+MOOD fields}"
                )
                body.append(r"\end{figure}")
    body.append(r"\section{Limitations}")
    body.append(
        r"No dyn-GRMHD MOOD; \texttt{uct\_hlld} Newtonian-only (it needs \texttt{hlld}); "
        r"2D/3D linear waves use diagonal wavevectors and test left/right "
        r"acoustic or fast-magnetosonic propagation. Apollo SRMHD blast movies "
        r"are the multi-D PPM+MOOD stress reference. "
        r"The Orszag--Tang vortex is shown at $t=1$. The current sheet uses the "
        r"tearing seed and is shown at the last common finite time ($t=1$)."
    )
    body.append(r"\end{document}")
    tex = REPORT / "FALLBACK_VALIDATION.tex"
    tex.write_text("\n".join(body) + "\n")
    print(f"Wrote {tex}")
    return tex


def compile_pdf_matplotlib(analysis, commit, manif):
    import os as _os

    _os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    pdf_dst = VAL / "FALLBACK_VALIDATION.pdf"
    figs = []
    for phys_id, _ in PHYSICS:
        for dim_id, _ in DIMS:
            p = FIGURES / f"lwave_{phys_id}_{dim_id}_mosaic.png"
            if p.exists():
                figs.append(p)
        nm = FIGURES / f"lwave_nmood_{phys_id}.png"
        if nm.exists():
            figs.append(nm)
    for p in sorted(FIGURES.glob("shock_*.png")):
        figs.append(p)
    for p in sorted(FIGURES.glob("stress_*_mosaic.png")):
        figs.append(p)
    ring = VAL / (
        manif.get("apollo_ringing_figure")
        or "figures/apollo_ringing/mosaic_uct_hlld_256_t1p5.png"
    )
    if ring.exists():
        figs.append(ring)
    gcfg = manif.get("apollo_grmhd_blast") or {}
    gmos = VAL / (
        gcfg.get("mosaic") or "figures/apollo_grmhd/srmhd_blast_final_mosaic.png"
    )
    if gmos.exists():
        figs.append(gmos)

    with PdfPages(pdf_dst) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.78, "MOOD Fallback Validation Report", fontsize=18, weight="bold")
        fig.text(0.1, 0.72, f"Commit {commit}", fontsize=11)
        gates = analysis.get("gates") or {}
        fig.text(
            0.1,
            0.55,
            f"Records: {analysis.get('n_records', 0)}\n"
            f"PPM+MOOD linear zero demotions: {gates.get('ppm_fb_linear_zero_demotions')}\n"
            f"All completed: {gates.get('all_completed')}\n\n"
            "Organized by physics: hydro / mhd / grhydro / grmhd\n"
            "with 1D, 2D, and 3D linear-wave mosaics.",
            fontsize=11,
            va="top",
            family="monospace",
        )
        pdf.savefig(fig)
        plt.close(fig)
        for f in figs:
            img = imread(f)
            fig = plt.figure(figsize=(8.5, 11.0))
            ax = fig.add_axes([0.05, 0.08, 0.9, 0.85])
            ax.imshow(img)
            ax.set_axis_off()
            fig.text(0.5, 0.02, f.stem.replace("_", " "), ha="center", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {pdf_dst} (matplotlib PdfPages)")
    return pdf_dst


def compile_pdf(tex: Path):
    try:
        subprocess.run(
            ["tectonic", str(tex), "--outdir", str(REPORT)],
            check=True,
            cwd=REPORT,
        )
        src = REPORT / "FALLBACK_VALIDATION.pdf"
        if src.exists():
            src.replace(VAL / "FALLBACK_VALIDATION.pdf")
            print(f"Wrote {VAL / 'FALLBACK_VALIDATION.pdf'}")
            return VAL / "FALLBACK_VALIDATION.pdf"
    except Exception as e:
        print(f"PDF compile failed: {e}; trying matplotlib fallback")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    analysis_path = RESULTS / "analysis.json"
    if not analysis_path.exists():
        analysis = {"n_records": 0, "linear": [], "shocks": [], "stress": [], "gates": {}}
    else:
        analysis = json.loads(analysis_path.read_text())
    manif = load_manifest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    write_markdown(analysis, commit, manif)
    tex = write_latex(analysis, commit, manif)
    if args.pdf:
        out = compile_pdf(tex)
        if out is None:
            compile_pdf_matplotlib(analysis, commit, manif)


if __name__ == "__main__":
    main()
