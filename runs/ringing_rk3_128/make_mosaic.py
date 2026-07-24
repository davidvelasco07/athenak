#!/usr/bin/env python3
"""Mosaic of the 128^3 PPM+FB mood_rtol sweep vs PLM at t=1.5 (RK3, HLLD)."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../vis/python"))
import bin_convert as bc

RUNROOT = os.path.dirname(os.path.abspath(__file__))
OUTIDX = "00006"
BASENAME = "turb_ringing_ppm_fb"
CASES = [
    ("plm", "PLM"),
    ("default", "PPM+FB rtol=1e-5"),
    ("rtol6", "PPM+FB rtol=1e-6"),
    ("rtol7", "PPM+FB rtol=1e-7"),
]
SLICES = [
    ("slice_x1_0", "x2", "x3", r"$x_1 = 0$ slice"),
    ("slice_x2_001953125", "x1", "x3", r"$x_2 \approx 0$ slice"),
]
VARS = [
    ("bmag", "bmag", r"$|B|$"),
    ("curvature", "curv_alt", r"curvature"),
    ("current2", "j2", r"$|J|^2$"),
]


def load_slice(case_dir, slice_id, file_suffix, var_key, hname, vname):
    path = os.path.join(
        RUNROOT,
        case_dir,
        "bin",
        "rank_00000000",
        f"{BASENAME}.{slice_id}_{file_suffix}.{OUTIDX}.bin",
    )
    data = bc.read_binary_as_athdf(path)
    field = np.squeeze(data[var_key])
    horiz = data[f"{hname}v"]
    vert = data[f"{vname}v"]
    return field, horiz, vert


def main():
    cache = {}
    for case_dir, _ in CASES:
        for slice_id, hname, vname, _ in SLICES:
            for file_suffix, var_key, _ in VARS:
                cache[(case_dir, slice_id, file_suffix)] = load_slice(
                    case_dir, slice_id, file_suffix, var_key, hname, vname
                )

    clim = {}
    for file_suffix, var_key, _ in VARS:
        vals = np.concatenate(
            [cache[(c, s, file_suffix)][0].ravel()
             for c, _ in CASES
             for s, _, _, _ in SLICES]
        )
        if var_key == "j2":
            clim[var_key] = (
                max(np.percentile(vals, 1), 1e-12),
                np.percentile(vals, 99.5),
                "log",
            )
        else:
            clim[var_key] = (
                np.percentile(vals, 0.5),
                np.percentile(vals, 99.5),
                "linear",
            )

    nrows = len(SLICES) * len(VARS)
    ncols = len(CASES)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False,
    )
    fig.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.04,
                        hspace=0.28, wspace=0.08)

    for irow, (slice_id, hname, vname, slice_label) in enumerate(SLICES):
        for jvar, (file_suffix, var_key, var_label) in enumerate(VARS):
            row = irow * len(VARS) + jvar
            vmin, vmax, scale = clim[var_key]
            norm = (LogNorm(vmin=vmin, vmax=vmax) if scale == "log"
                    else Normalize(vmin=vmin, vmax=vmax))
            im = None
            for jcol, (case_dir, case_label) in enumerate(CASES):
                ax = axes[row, jcol]
                field, horiz, vert = cache[(case_dir, slice_id, file_suffix)]
                im = ax.imshow(
                    field,
                    origin="lower",
                    aspect="equal",
                    extent=[horiz[0], horiz[-1], vert[0], vert[-1]],
                    norm=norm,
                    cmap="inferno",
                    interpolation="nearest",
                )
                if row == 0:
                    ax.set_title(case_label, fontsize=11, fontweight="bold")
                if jcol == 0:
                    ax.set_ylabel(f"{slice_label}\n{var_label}", fontsize=10)
                ax.tick_params(labelsize=7)
                if row == nrows - 1:
                    ax.set_xlabel(f"${hname}$")
            cax = fig.add_axes([
                0.915,
                axes[row, 0].get_position().y0,
                0.012,
                axes[row, 0].get_position().height,
            ])
            fig.colorbar(im, cax=cax)

    fig.suptitle(
        "PPM+FB mood_rtol sweep at t = 1.5 (RK3, HLLD, 128$^3$)",
        fontsize=15,
    )
    out_png = os.path.join(RUNROOT, "mosaic_rtol_sweep_t1p5.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
