#!/usr/bin/env python3
"""Scheme x time mosaics for cases whose reference is a time series.

The RR22 magnetized Kelvin--Helmholtz (arXiv:2203.06062 sec 5.2) is compared in
the paper at t = 5, 8, 12, 20, not at a single final frame, so a one-column
"final state" mosaic throws away the comparison the paper actually makes.  This
draws schemes down the rows and the reference times across the columns, on one
shared colour scale so panels are comparable both ways.

Requires the case to carry `output_dt` in the manifest so intermediate dumps
exist.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
RESULTS = HERE.parent / "results" / "stress"
FIGURES = HERE.parent / "figures"
VENDOR = HERE / "_vendor"

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cmasher as cmr
    PHYS_CMAP = cmr.chroma
except ImportError:
    PHYS_CMAP = "inferno"

sys.path.insert(0, str(HERE))
from plot_2d_stress import (  # noqa: E402
    read_vtk_structured, _field_array, SCHEME_ORDER, SCHEME_LABEL, FIELD_LABEL,
)
from plot_style import apply_publication_style  # noqa: E402

apply_publication_style()


def _dump_time(path: Path) -> float | None:
    """VTK header carries `Athena++ data at time= <t>`."""
    try:
        with open(path, "rb") as f:
            head = f.read(400).decode("latin-1")
        m = re.search(r"time=\s*([0-9eE.+-]+)", head)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def collect(outdir: Path, field: str, want_times):
    """Pick the dump closest to each requested time; report what was found."""
    got = {}
    for p in sorted(outdir.rglob("*.vtk")):
        t = _dump_time(p)
        if t is None:
            continue
        got[t] = p
    if not got:
        return {}
    out = {}
    for tw in want_times:
        t_near = min(got, key=lambda t: abs(t - tw))
        # only accept if the dump really is near the requested time
        if abs(t_near - tw) > 0.6:
            continue
        try:
            fields, _ = read_vtk_structured(got[t_near])
        except Exception:
            continue
        arr = _field_array(fields, field)
        if arr is None or arr.ndim != 2:
            continue
        out[tw] = (t_near, arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="kh_rr22")
    ap.add_argument("--field", default="dens")
    ap.add_argument("--times", default="5,8,12,20")
    ap.add_argument("--results", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    results_dir = Path(args.results).resolve() if args.results else RESULTS
    want = [float(x) for x in args.times.split(",")]

    per_scheme = {}
    health = {}
    for scheme in SCHEME_ORDER:
        d = results_dir / f"{args.case}_{scheme}_smoke"
        if not d.is_dir():
            continue
        got = collect(d, args.field, want)
        if got:
            per_scheme[scheme] = got
        try:
            health[scheme] = (
                json.loads((d / "summary.json").read_text()).get("health") or {}
            )
        except Exception:
            health[scheme] = {}

    if not per_scheme:
        print(f"no data for {args.case} in {results_dir}")
        return

    schemes = [s for s in SCHEME_ORDER if s in per_scheme]
    ncol = len(want)
    nrow = len(schemes)

    vals = [
        a[np.isfinite(a)]
        for sc in schemes
        for (_t, a) in per_scheme[sc].values()
    ]
    vals = [v for v in vals if v.size]
    vmin = min(float(v.min()) for v in vals)
    vmax = max(float(v.max()) for v in vals)
    if abs(vmax - vmin) < 1e-30:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(1.75 * ncol + 1.1, 1.95 * nrow),
        squeeze=False, layout="constrained",
    )
    im = None
    for r, sc in enumerate(schemes):
        dead = bool((health.get(sc) or {}).get("dt_collapse"))
        for c, tw in enumerate(want):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            entry = per_scheme[sc].get(tw)
            if entry is None:
                ax.text(0.5, 0.5, "no dump", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="#a82a3a")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#a82a3a")
                continue
            _t, arr = entry
            im = ax.imshow(arr, origin="lower", cmap=PHYS_CMAP, vmin=vmin, vmax=vmax,
                           aspect="equal", interpolation="nearest", rasterized=True)
            if r == 0:
                ax.set_title(f"$t={tw:g}$", fontsize=9)
            if c == 0:
                ax.set_ylabel(SCHEME_LABEL.get(sc, sc), fontsize=8,
                              color=("#a82a3a" if dead else "black"))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location="right",
                            shrink=0.9, aspect=30, pad=0.015)
        cbar.ax.set_title(FIELD_LABEL.get(args.field, args.field), pad=4)
    fig.suptitle(
        "Magnetized Kelvin\u2013Helmholtz (Rueda-Ram\u00edrez+ 2022), reference times"
        if args.case == "kh_rr22" else args.case.replace("_", " "),
        y=1.02,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"timeseries_{args.case}{args.suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
