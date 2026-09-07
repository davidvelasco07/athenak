#!/usr/bin/env python3
"""Plot 2D stress-smoke VTK mosaics (one panel per scheme)."""

from __future__ import annotations

import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parents[1]
RESULTS = VAL / "results" / "stress"
FIGURES = VAL / "figures"
MANIFEST = VAL / "manifest.yaml"
VENDOR = VAL / "vendor"

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
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
from plot_style import apply_publication_style

apply_publication_style()

SCHEME_ORDER = ["ppm_fb", "plm", "wenoz", "teno"]
SCHEME_LABEL = {
    "ppm_fb": r"PPM + MOOD + RK3",
    "plm": r"PLM + RK2",
    "wenoz": r"WENO-Z + RK3",
    "teno": r"TENO + RK3",
}

CASE_LABEL = {
    "slotted_cyl": "Slotted disk",
    "implode_hydro": "Liska\u2013Wendroff implosion",
    "orszag_tang": r"Orszag\u2013Tang vortex",
    "current_sheet": r"Double Harris current sheet, $256^2$",
    "current_sheet_n512": r"Double Harris current sheet, $512^2$",
    "current_sheet_n1024": r"Double Harris current sheet, $1024^2$",
    "kh_rr22": r"Magnetized Kelvin\u2013Helmholtz (RR+22, $t=20$)",
    "jet": r"Underdense Mach-10 MHD jet",
    "jet_n600": r"Underdense Mach-10 MHD jet ($600\times500$)",
    "rotor": r"MHD rotor",
    "blast_hydro": r"Hydrodynamic blast",
    "blast_mhd": r"Magnetized blast",
    "blast_grmhd": r"Relativistic magnetized blast, Minkowski",
    "blast_mhd_lowbeta": r"Magnetized blast, $\beta=0.02$ (Stone+08 fig 30)",
    "mhd_jet": r"MHD jet, $\beta=5\times10^{-11}$ (Wu+ Ex 6.3)",
    "mhd_jet_revs4": r"MHD jet, $\beta=5\times10^{-11}$, MOOD depth 4",
    "ha_jet": r"Mach-2000 hydrodynamic jet (Wu+ Ex 4.9)",
    "kh_rr22_n256": r"Magnetized Kelvin--Helmholtz (RR+22), $256\times512$",
}

FIELD_LABEL = {
    "dens": r"$\rho$",
    "s_00": r"$s$",
    "bmag": r"$|\mathbf{B}|$",
    "|B|": r"$|\mathbf{B}|$",
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


def read_vtk_structured(path: Path):
    """Read AthenaK binary VTK STRUCTURED_POINTS (big-endian floats)."""
    raw = path.read_bytes()
    text_end = raw.find(b"LOOKUP_TABLE default\n")
    if text_end < 0:
        raise ValueError(f"no LOOKUP_TABLE in {path}")
    header = raw[:text_end].decode("ascii", "replace")
    dims_m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", header)
    ncell_m = re.search(r"CELL_DATA\s+(\d+)", header)
    if not dims_m or not ncell_m:
        raise ValueError(f"missing DIMENSIONS/CELL_DATA in {path}")
    nx, ny, nz = (int(dims_m.group(i)) for i in (1, 2, 3))
    # STRUCTURED_POINTS dimensions are node counts; cell array is (nx-1)*(ny-1)*(nz-1)
    ncx, ncy, ncz = max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)
    ncell = int(ncell_m.group(1))
    if ncx * ncy * ncz != ncell:
        # fall back to treating DIMENSIONS as cell counts (rare)
        ncx, ncy, ncz = nx, ny, nz

    fields = {}
    # Walk field markers in file order
    pattern = re.compile(rb"(SCALARS|VECTORS)\s+(\S+)\s+float")
    matches = list(pattern.finditer(raw))
    for i, m in enumerate(matches):
        kind = m.group(1).decode()
        name = m.group(2).decode()
        # data starts after "LOOKUP_TABLE default\n" for SCALARS, or after the
        # VECTORS header line for VECTORS
        if kind == "SCALARS":
            start = raw.find(b"LOOKUP_TABLE default\n", m.end())
            if start < 0:
                continue
            start += len(b"LOOKUP_TABLE default\n")
            nbytes = 4 * ncell
            blob = raw[start : start + nbytes]
            arr = np.array(struct.unpack(">" + "f" * ncell, blob), dtype=np.float64)
            fields[name] = arr.reshape(ncz, ncy, ncx)[0]
        else:
            # VECTORS name float\n then 3*ncell floats
            line_end = raw.find(b"\n", m.end())
            start = line_end + 1
            nbytes = 12 * ncell
            blob = raw[start : start + nbytes]
            arr = np.array(struct.unpack(">" + "f" * (3 * ncell), blob), dtype=np.float64)
            fields[name] = arr.reshape(ncz, ncy, ncx, 3)[0]

    origin_m = re.search(
        r"ORIGIN\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", header
    )
    spacing_m = re.search(
        r"SPACING\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", header
    )
    meta = {
        "nx": ncx,
        "ny": ncy,
        "origin": tuple(float(origin_m.group(i)) for i in (1, 2, 3)) if origin_m else (0, 0, 0),
        "spacing": tuple(float(spacing_m.group(i)) for i in (1, 2, 3))
        if spacing_m
        else (1, 1, 1),
    }
    return fields, meta


def _field_array(fields, name: str):
    if name == "bmag":
        if not all(k in fields for k in ("bcc1", "bcc2", "bcc3")):
            return None
        return np.sqrt(fields["bcc1"] ** 2 + fields["bcc2"] ** 2 + fields["bcc3"] ** 2)
    return fields.get(name)


def _latest_finite_vtk(outdir: Path, field: str, *, allow_ic: bool = False) -> Path | None:
    """Prefer the last VTK whose requested field is finite (skip NaN dumps)."""
    vtks = sorted(outdir.rglob("*.vtk"))
    chosen = None
    for p in vtks:
        try:
            fields, _ = read_vtk_structured(p)
            arr = _field_array(fields, field)
            if arr is None:
                arr = _field_array(fields, "dens")
            if arr is not None and np.isfinite(arr).all():
                chosen = p
        except Exception:
            continue
    if chosen is None:
        return None
    # Skip pure ICs unless explicitly allowed (evolved dump preferred)
    if not allow_ic and chosen.name.endswith(".00000.vtk"):
        later = [p for p in vtks if p != chosen]
        if later:
            return None
    return chosen


def _health(outdir: Path) -> dict:
    """Read the health record written by run_suite so a dead run cannot be drawn
    as if it were healthy: _latest_finite_vtk falls back to the last FINITE dump,
    which for a collapsed run is an earlier time than the panel title claims."""
    try:
        import json
        return (json.loads((outdir / "summary.json").read_text()).get("health") or {})
    except Exception:
        return {}


def _vtk_time(path):
    """Time stamped in the VTK header, so titles come from the data."""
    try:
        with open(path, "rb") as f:
            head = f.read(400).decode("latin-1")
        m = re.search(r"time=\s*([0-9eE.+-]+)", head)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def plot_case_mosaic(case_id: str, field: str, schemes_data: dict, out: Path,
                     health: dict | None = None, tstamp: float | None = None,
                     log: bool = False, unplottable: set | None = None):
    health = health or {}
    # A scheme that died before its first output has no finite dump to draw, but
    # dropping its panel would leave a three-panel mosaic that reads as though the
    # scheme was never run -- i.e. the figure would hide the failure it is meant to
    # show.  Those schemes get an empty panel, labelled.
    unplottable = {s for s in (unplottable or set()) if s not in schemes_data}
    schemes = [s for s in SCHEME_ORDER if s in schemes_data or s in unplottable]
    if not schemes or not schemes_data:
        return
    n = len(schemes)
    # 7.2 inches fits a typical two-column journal text width.  Constrained layout
    # reserves a separate colorbar gutter, preventing it from covering the last panel.
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(7.2, 2.15),
        squeeze=False,
        layout="constrained",
    )
    finite_vals = [
        schemes_data[s][np.isfinite(schemes_data[s])]
        for s in schemes if s in schemes_data
    ]
    finite_vals = [v for v in finite_vals if v.size]
    if not finite_vals:
        plt.close(fig)
        print(f"skip {case_id}: all-NaN field {field}")
        return
    vmin = min(float(v.min()) for v in finite_vals)
    vmax = max(float(v.max()) for v in finite_vals)
    if abs(vmax - vmin) < 1e-30:
        vmax = vmin + 1.0
    # A jet spans four decades in density (beam ~30, evacuated cocoon ~4e-3) with the
    # ambient near 0.5, so a linear norm crushes everything but the working surface to
    # the bottom of the colormap and the mosaic compares four black squares.  Cases
    # opt in with `plot_log: true`; the floor is clamped four decades below the peak so
    # a single near-zero cell cannot set the scale.
    norm = None
    if log:
        pos = [float(v[v > 0].min()) for v in finite_vals if (v > 0).any()]
        if pos:
            lo = max(min(pos), vmax * 1.0e-4)
            norm = LogNorm(vmin=lo, vmax=vmax)
    im = None
    _live = next(iter(schemes_data.values()))
    _box_aspect = _live.shape[0] / _live.shape[1]
    for ax, scheme in zip(axes[0], schemes):
        if scheme not in schemes_data:
            # no image means no aspect: match the live panels explicitly
            ax.set_box_aspect(_box_aspect)
            ax.set_facecolor("#f2f2f2")
            ax.text(0.5, 0.5, "no finite dump", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#a82a3a")
            ax.set_title(SCHEME_LABEL.get(scheme, scheme), color="#a82a3a")
            ax.text(0.5, -0.09, "failed before first output", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5, color="#a82a3a")
            for sp in ax.spines.values():
                sp.set_edgecolor("#a82a3a"); sp.set_linewidth(1.2)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        arr = schemes_data[scheme]
        if norm is not None:
            # LogNorm masks zero and negative samples, which would punch blank holes
            # through an evacuated cocoon or a |B| null.  Clip into the plotted range
            # so those cells render at the bottom of the colormap instead.
            arr = np.clip(arr, norm.vmin, norm.vmax)
        im = ax.imshow(
            arr,
            origin="lower",
            cmap=PHYS_CMAP,
            **({"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}),
            aspect="equal",
            interpolation="nearest",
            rasterized=True,
        )
        h = health.get(scheme) or {}
        dead = bool(h.get("dt_collapse")) or (h.get("healthy") is False)
        ax.set_title(SCHEME_LABEL.get(scheme, scheme),
                     color=("#a82a3a" if dead else "black"))
        if dead:
            # Say so on the panel: this is the last finite dump, not the final time.
            ax.text(0.5, -0.09, "timestep collapse", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5, color="#a82a3a")
            for sp in ax.spines.values():
                sp.set_edgecolor("#a82a3a"); sp.set_linewidth(1.2)
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
        cbar.ax.set_title(FIELD_LABEL.get(field, field), pad=4)
        cbar.outline.set_linewidth(0.6)
    ttl = CASE_LABEL.get(case_id, case_id.replace("_", " "))
    if tstamp is not None:
        ttl += f"  ($t={tstamp:g}$)"
    fig.suptitle(ttl, y=1.03)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="Only plot this case id")
    ap.add_argument("--results", default=None, help="Directory of *_smoke run dirs")
    ap.add_argument("--suffix", default="", help="Appended to output figure names")
    args = ap.parse_args()
    results_dir = Path(args.results).resolve() if args.results else RESULTS
    manif = load_manifest()
    cases = (manif.get("stress") or {}).get("local_smoke") or []
    field_by_case = {c["id"]: c.get("plot_field", "dens") for c in cases}
    log_by_case = {c["id"]: bool(c.get("plot_log", False)) for c in cases}
    physics_by_case = {c["id"]: c.get("physics", "unknown") for c in cases}
    active_cases = set(field_by_case)

    by_case = defaultdict(dict)
    by_case_health = defaultdict(dict)
    by_case_time = {}
    by_case_unplottable = defaultdict(set)
    for d in sorted(results_dir.glob("*_smoke")):
        if not d.is_dir():
            continue
        name = d.name
        scheme = None
        case_id = None
        for s in SCHEME_ORDER:
            if name.endswith(f"_{s}_smoke"):
                scheme = s
                case_id = name[: -len(f"_{s}_smoke")]
                break
        if scheme is None:
            continue
        if case_id not in active_cases:
            continue
        if args.case and case_id != args.case:
            continue
        field = field_by_case.get(case_id, "dens")
        vtk = _latest_finite_vtk(d, field)
        if vtk is None:
            print(f"skip {d.name}: no finite VTK for {field}")
            by_case_unplottable[case_id].add(scheme)
            continue
        try:
            fields, _meta = read_vtk_structured(vtk)
        except Exception as e:
            print(f"skip {d.name}: {e}")
            continue
        by_case[case_id][scheme] = fields
        by_case_health[case_id][scheme] = _health(d)
        by_case_time.setdefault(case_id, _vtk_time(vtk))

    for case_id, scheme_fields in sorted(by_case.items()):
        field = field_by_case.get(case_id, "dens")
        schemes_data = {}
        field_label = field
        for scheme, fields in scheme_fields.items():
            arr = _field_array(fields, field)
            if arr is None and field != "dens":
                arr = _field_array(fields, "dens")
                field_label = "dens"
            if arr is None:
                continue
            if arr.ndim != 2 or min(arr.shape) < 2:
                print(f"skip {case_id}/{scheme}: not 2D ({arr.shape})")
                continue
            if not np.isfinite(arr).any():
                print(f"skip {case_id}/{scheme}: all-NaN")
                continue
            schemes_data[scheme] = arr
        if not schemes_data:
            continue
        out = FIGURES / f"stress_{case_id}{args.suffix}_mosaic.png"
        plot_case_mosaic(case_id, field_label, schemes_data, out,
                         health=by_case_health.get(case_id),
                         tstamp=by_case_time.get(case_id),
                         log=log_by_case.get(case_id, False),
                         unplottable=by_case_unplottable.get(case_id))

        phys = physics_by_case.get(case_id, "")
        if phys == "mhd":
            bdata = {}
            for scheme, fields in scheme_fields.items():
                arr = _field_array(fields, "bmag")
                if arr is not None and arr.ndim == 2 and min(arr.shape) >= 2:
                    if np.isfinite(arr).any():
                        bdata[scheme] = arr
            if bdata:
                plot_case_mosaic(
                    case_id,
                    "|B|",
                    bdata,
                    FIGURES / f"stress_{case_id}{args.suffix}_bmag_mosaic.png",
                    health=by_case_health.get(case_id),
                    tstamp=by_case_time.get(case_id),
                    log=log_by_case.get(case_id, False),
                    unplottable=by_case_unplottable.get(case_id),
                )


if __name__ == "__main__":
    main()
