#!/usr/bin/env python3
"""Generate comparison figures for the fallback validation report.

Linear-wave convergence is drawn as shared-axis mosaics (one panel per wave
family), grouped by physics × dimension.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

VAL = Path(__file__).resolve().parents[1]
RESULTS = VAL / "results"
FIGURES = VAL / "figures"
ROOT = VAL.parents[1]
VENDOR = VAL / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
try:
    import cmasher as cmr
    PHYS_CMAP = cmr.chroma
except ImportError:
    PHYS_CMAP = "inferno"
import numpy as np
from plot_style import apply_publication_style
import yaml


def load_manifest():
    return yaml.safe_load((Path(__file__).resolve().parents[1] /
                           "manifest.yaml").read_text()) or {}

apply_publication_style()

sys.path.insert(0, str(ROOT / "vis" / "python"))

SCHEME_STYLE = {
    "ppm_fb": {"label": "PPM + MOOD + RK3", "ls": "-", "color": "C0"},
    "plm": {"label": "PLM + RK2", "ls": "--", "color": "C1"},
    "wenoz": {"label": "WENO-Z + RK3", "ls": "-.", "color": "C2"},
    "teno": {"label": "TENO + RK3", "ls": ":", "color": "C3"},
}

PHYSICS_ORDER = ["hydro", "mhd", "grhydro", "grmhd"]
DIM_ORDER = ["1d", "2d", "3d"]
PHYSICS_LABEL = {
    "hydro": "Hydrodynamics",
    "mhd": "Magnetohydrodynamics",
    "grhydro": "GR hydrodynamics",
    "grmhd": "GR magnetohydrodynamics",
}
SHOCK_LABEL = {
    "sod": "Sod",
    "shu_osher": "Shu--Osher",
    "rj2a": "Ryu--Jones 2a",
    "bw": "Brio--Wu",
    "mb2_gr": "Martí--Müller blast 2",
    "mub1_gr": "Mignone--Ugliano--Bodo 1",
}
HYDRO_WAVE_LABEL = {
    "0": r"Acoustic $-$",
    "1": "Entropy/contact",
    "2": r"Shear $y$",
    "3": r"Shear $z$",
    "4": r"Acoustic $+$",
}
MHD_WAVE_LABEL = {
    "0": r"Fast $-$",
    "1": r"Alfvén $-$",
    "2": r"Slow $-$",
    "3": "Entropy/contact",
    "4": r"Slow $+$",
    "5": r"Alfvén $+$",
    "6": r"Fast $+$",
}


def load_analysis():
    p = RESULTS / "analysis.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _clear_old_lwave_singles():
    """Remove legacy per-wave PNGs once mosaics replace them."""
    for p in FIGURES.glob("lwave_*_w*.png"):
        if "_mosaic_" in p.name:
            continue
        p.unlink(missing_ok=True)


def plot_linear_mosaics(analysis):
    """One shared-axis mosaic per (physics, dim)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    # group: (physics, dim) -> wave -> scheme -> finding
    groups = defaultdict(lambda: defaultdict(list))
    for f in analysis.get("linear", []):
        phys = f.get("physics") or f["case"]
        dim = f.get("dim") or "1d"
        groups[(phys, dim)][str(f["wave"])].append(f)

    for (phys, dim), by_wave in sorted(
        groups.items(), key=lambda kv: (PHYSICS_ORDER.index(kv[0][0]) if kv[0][0] in PHYSICS_ORDER else 99,
                                        DIM_ORDER.index(kv[0][1]) if kv[0][1] in DIM_ORDER else 99)
    ):
        waves = sorted(by_wave.keys(), key=lambda w: int(w) if str(w).isdigit() else w)
        n = len(waves)
        if n == 0:
            continue
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.2, 2.15 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        # Collect global ranges for consistent reference slopes
        all_ns, all_l1 = [], []
        for wave in waves:
            for f in by_wave[wave]:
                all_ns.extend(f.get("resolutions") or [])
                all_l1.extend([v for v in (f.get("l1") or []) if v is not None and v > 0])

        handles, labels = [], []
        for idx, wave in enumerate(waves):
            ax = axes[idx // ncols][idx % ncols]
            for f in by_wave[wave]:
                st = SCHEME_STYLE.get(
                    f["scheme"], {"label": f["scheme"], "ls": "-", "color": "k"}
                )
                ns = np.array(f["resolutions"], dtype=float)
                l1 = np.array(
                    [np.nan if v is None else v for v in f["l1"]], dtype=float
                )
                (line,) = ax.loglog(
                    ns, l1, st["ls"], color=st["color"], marker="o", ms=4, label=st["label"]
                )
                if st["label"] not in labels:
                    handles.append(line)
                    labels.append(st["label"])
            if all_ns and all_l1:
                n0, n1 = min(all_ns), max(all_ns)
                y0 = np.median(all_l1)
                ax.loglog(
                    [n0, n1],
                    [y0, y0 * (n0 / n1) ** 2],
                    "k-",
                    alpha=0.25,
                    lw=1,
                )
                ax.loglog(
                    [n0, n1],
                    [y0, y0 * (n0 / n1) ** 4],
                    "k--",
                    alpha=0.25,
                    lw=1,
                )
            wave_labels = (
                MHD_WAVE_LABEL if phys in ("mhd", "grmhd") else HYDRO_WAVE_LABEL
            )
            ax.set_title(wave_labels.get(str(wave), f"Wave {wave}"))
            ax.grid(True, which="both", color="0.88", lw=0.45)
            if idx // ncols == nrows - 1:
                ax.set_xlabel(r"$N_{x1}$")
            if idx % ncols == 0:
                ax.set_ylabel(r"RMS $L_1$")

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"{PHYSICS_LABEL.get(phys, phys)} linear waves ({dim.upper()})")
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=min(4, len(labels)),
            frameon=False,
        )
        out = FIGURES / f"lwave_{phys}_{dim}_mosaic.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")

    _clear_old_lwave_singles()


def plot_nmood_linear(analysis):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ppm = [f for f in analysis.get("linear", []) if f["scheme"] == "ppm_fb"]
    if not ppm:
        return
    # One bar chart per physics
    by_phys = defaultdict(list)
    for f in ppm:
        by_phys[f.get("physics") or f["case"]].append(f)
    for phys, items in by_phys.items():
        wave_labels = MHD_WAVE_LABEL if phys in ("mhd", "grmhd") else HYDRO_WAVE_LABEL
        labels = [
            f"{f.get('dim','1d').upper()}: "
            f"{wave_labels.get(str(f['wave']), f['wave'])}"
            for f in items
        ]
        vals = [max([0 if m is None else m for m in f["nmood"]] + [0]) for f in items]
        fig, ax = plt.subplots(
            figsize=(7.2, 2.6), layout="constrained"
        )
        ax.bar(range(len(labels)), vals, color="C0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Total MOOD demotions")
        ax.set_title(
            f"PPM + MOOD demotions: {PHYSICS_LABEL.get(phys, phys)} "
            "(expected: zero)"
        )
        ax.axhline(0, color="k", lw=0.5)
        out = FIGURES / f"lwave_nmood_{phys}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")
    # Remove old combined nmood figure if present
    old = FIGURES / "lwave_nmood_ppm_fb.png"
    if old.exists():
        old.unlink()


def plot_shock_profiles():
    """Overlay density from tab outputs when present."""
    try:
        import athena_read
    except ImportError:
        print("athena_read unavailable; skip shock profiles")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict

    groups = defaultdict(dict)
    for d in (RESULTS / "shocks").glob("*_N512"):
        if not d.is_dir():
            continue
        name = d.name
        scheme = None
        for s in ("ppm_fb", "wenoz", "teno", "plm"):
            suffix = f"_{s}_N512"
            if name.endswith(suffix):
                scheme = s
                case = name[: -len(suffix)]
                break
        if scheme is None:
            continue
        tabs = list(d.glob("tab/*.tab")) + list(d.glob("*.tab"))
        if not tabs:
            tabs = list(d.rglob("*.tab"))
        if not tabs:
            continue
        # Prefer the last non-IC dump (highest index), which should be t ≈ tlim
        tabs = sorted(tabs)
        tab_path = tabs[-1]
        try:
            data = athena_read.tab(str(tab_path))
        except Exception as e:
            print(f"skip {d}: {e}")
            continue
        # Guard: if only IC exists, or final dens is absurdly flat, skip
        dens = data.get("dens")
        if dens is not None and float(np.nanmax(dens) - np.nanmin(dens)) < 1e-6:
            print(f"skip {d}: flat density in {tab_path.name}")
            continue
        groups[case][scheme] = data
    for case, schemes in groups.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
        for scheme, data in schemes.items():
            st = SCHEME_STYLE.get(scheme, {"label": scheme, "ls": "-", "color": "k"})
            x = data["x1v"] if "x1v" in data else data.get("x1")
            dens = data["dens"] if "dens" in data else data.get("d")
            if x is None or dens is None:
                continue
            ax.plot(x, dens, st["ls"], color=st["color"], label=st["label"], lw=1.4)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"{SHOCK_LABEL.get(case, case)} shock ($N=512$)")
        ax.legend()
        ax.grid(True, color="0.88", lw=0.45)
        out = FIGURES / f"shock_{case}_dens.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")


TOLERANCE_TITLE = {
    "implode_hydro": "Liska--Wendroff implosion: NAD tolerance",
    "slotted_cyl": "Slotted disk: NAD tolerance",
}
TOLERANCE_FIELD_LABEL = {
    "dens": r"$\rho$",
    "s_00": r"$s$",
}


def _tolerance_fig_stem(case_id):
    if case_id == "implode_hydro":
        return "tolerance_implode_hydro"
    return f"tolerance_{case_id}"


def plot_tolerance_study(analysis):
    """Plot snapshots and demotions across NAD tolerances for each case."""
    rows = analysis.get("tolerance") or []
    if not rows:
        return
    from plot_2d_stress import read_vtk_structured

    # `plot_log` is a per-case manifest key shared with the stress mosaics: a jet spans
    # four decades in density, and a linear norm renders the cocoon and the ambient as
    # one flat colour.  Read it from both the tolerance cases and the stress cases,
    # since `include_stress` pulls the latter's config into this suite.
    manif = load_manifest()
    log_by_case = {}
    for _grp in ((manif.get("tolerance") or {}).get("cases") or [],
                 (manif.get("stress") or {}).get("local_smoke") or []):
        for _c in _grp:
            if _c.get("plot_log"):
                log_by_case[_c["id"]] = True

    by_case = defaultdict(list)
    for row in rows:
        by_case[row.get("case") or "unknown"].append(row)

    for case_id, case_rows in by_case.items():
        samples = []
        for row in sorted(case_rows, key=lambda r: float(r["rtol"]), reverse=True):
            outdir = RESULTS / "tolerance" / (
                f"{row['case']}_ppm_fb_rtol_{float(row['rtol']):.0e}"
            )
            vtks = sorted(outdir.rglob("*.vtk"))
            if not vtks:
                continue
            fields, _ = read_vtk_structured(vtks[-1])
            field = row.get("plot_field") or "dens"
            arr = fields.get(field)
            if arr is None and field != "dens":
                arr = fields.get("dens")
                field = "dens"
            # A tolerance that destroys the run must still occupy a panel.  Dropping it
            # silently turns a five-tolerance sweep into a three-tolerance figure and
            # hides the threshold the sweep exists to find -- on both hydro jets
            # PPM+MOOD is entirely NaN at rtol >= 1e-3 and clean at 1e-4.
            if arr is not None and np.isfinite(arr).all():
                samples.append((row, arr, field))
            else:
                samples.append((row, None, field))

        stem = _tolerance_fig_stem(case_id)
        if samples:
            live = [(r, a, f) for r, a, f in samples if a is not None]
            if not live:
                # every tolerance destroyed the run: no colour scale exists, so there is
                # nothing to draw.  The demotion plot below still carries the sweep.
                print(f"skip {stem}: no finite dump at any tolerance")
                plt.close("all")
                samples = []
            if live:
                values = np.concatenate([a[np.isfinite(a)] for _, a, _ in live])
                field = live[0][2]
                vmin, vmax = float(values.min()), float(values.max())
                norm = None
                if log_by_case.get(case_id):
                    pos = values[values > 0]
                    if pos.size:
                        norm = LogNorm(vmin=max(float(pos.min()), vmax * 1.0e-4),
                                       vmax=vmax)
                fig, axes = plt.subplots(
                    1,
                    len(samples),
                    figsize=(7.2, 1.95),
                    squeeze=False,
                    layout="constrained",
                )
                im = None
                # the placeholder panels carry no image, so nothing sets their aspect;
                # borrow it from a live panel or they render taller than their neighbours
                box_aspect = live[0][1].shape[0] / live[0][1].shape[1]
                for ax, (row, arr, _) in zip(axes[0], samples):
                    exponent_ = int(np.floor(np.log10(float(row["rtol"]))))
                    if arr is None:
                        ax.set_box_aspect(box_aspect)
                        ax.set_facecolor("#f2f2f2")
                        ax.text(0.5, 0.5, "no finite dump", transform=ax.transAxes,
                                ha="center", va="center", fontsize=6.5, color="#a82a3a")
                        ax.set_title(rf"$r_{{\rm tol}}=10^{{{exponent_}}}$", color="#a82a3a")
                        for sp in ax.spines.values():
                            sp.set_edgecolor("#a82a3a"); sp.set_linewidth(1.2)
                        ax.set_xticks([]); ax.set_yticks([])
                        continue
                    if norm is not None:
                        # clip rather than let LogNorm mask non-positive cells, which
                        # would punch blank holes through an evacuated cocoon
                        arr = np.clip(arr, norm.vmin, norm.vmax)
                    im = ax.imshow(
                        arr,
                        origin="lower",
                        cmap=PHYS_CMAP,
                        **({"norm": norm} if norm is not None
                           else {"vmin": vmin, "vmax": vmax}),
                        interpolation="nearest",
                        aspect="equal",
                        rasterized=True,
                    )
                    exponent = int(np.floor(np.log10(float(row["rtol"]))))
                    ax.set_title(rf"$r_{{\rm tol}}=10^{{{exponent}}}$")
                    ax.set_xticks([])
                    ax.set_yticks([])
                if im is not None:
                    cbar = fig.colorbar(
                        im,
                        ax=axes[0].tolist(),
                        location="right",
                        shrink=0.88,
                        aspect=28,
                        pad=0.018,
                    )
                    cbar.ax.set_title(TOLERANCE_FIELD_LABEL.get(field, field), pad=4)
                    cbar.outline.set_linewidth(0.6)
                fig.suptitle(TOLERANCE_TITLE.get(case_id, f"{case_id}: NAD tolerance"))
                out = FIGURES / f"{stem}_mosaic.png"
                fig.savefig(out)
                plt.close(fig)
                print(f"Wrote {out}")

        good = [r for r in case_rows if r.get("nmood_total") is not None]
        if good:
            good = sorted(good, key=lambda r: float(r["rtol"]))
            fig, ax = plt.subplots(figsize=(4.5, 2.8), layout="constrained")
            ax.loglog(
                [float(r["rtol"]) for r in good],
                [int(r["nmood_total"]) for r in good],
                "o-",
                color="C0",
            )
            ax.set_xlabel(r"Relaxed-DMP tolerance $r_{\rm tol}$")
            ax.set_ylabel("Cumulative MOOD demotions")
            ax.grid(True, which="both", color="0.88", lw=0.45)
            out = FIGURES / f"{stem}_demotions.png"
            fig.savefig(out)
            plt.close(fig)
            print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-profiles", action="store_true")
    ap.add_argument("--skip-stress", action="store_true")
    args = ap.parse_args()
    analysis = load_analysis()
    if analysis is None:
        print("No analysis.json; run analyze.py --write first")
        return
    plot_linear_mosaics(analysis)
    plot_nmood_linear(analysis)
    if not args.skip_profiles:
        plot_shock_profiles()
    plot_tolerance_study(analysis)
    if not args.skip_stress:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_2d_stress import main as plot_stress

        plot_stress()


if __name__ == "__main__":
    main()
