#!/usr/bin/env python3
"""Mosaic of MHD ringing reproducer results at t=1.5 (RK3)."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../vis/python"))
import bin_convert as bc

RUNROOT = os.path.dirname(os.path.abspath(__file__))
OUTIDX = "00006"
CASES = [
    ("plm", "PLM", "turb_ringing_plm"),
    ("ppmx", "PPMX", "turb_ringing_ppmx"),
    ("wenoz", "WENO-Z", "turb_ringing_wenoz"),
    ("ppm_fb", "PPM+FB", "turb_ringing_ppm_fb"),
]
SLICES = [
    ("slice_x1_0", "x2", "x3", r"$x_1 = 0$ slice"),
    ("slice_x2_001953125", "x1", "x3", r"$x_2 \approx 0.02$ slice"),
]
VARS = [
    ("bmag", "bmag", r"$|B|$"),
    ("curvature", "curv_alt", r"curvature"),
    ("current2", "j2", r"$|J|$"),
]


def load_slice(basename, slice_id, file_suffix, var_key, hname, vname):
    case_dir = basename.replace("turb_ringing_", "")
    path = os.path.join(
        RUNROOT,
        case_dir,
        "bin",
        "rank_00000000",
        f"{basename}.{slice_id}_{file_suffix}.{OUTIDX}.bin",
    )
    data = bc.read_binary_as_athdf(path)
    field = np.squeeze(data[var_key])
    horiz = data[f"{hname}v"]
    vert = data[f"{vname}v"]
    return field, horiz, vert


def main():
    cache = {}
    for case_dir, _, basename in CASES:
        for slice_id, hname, vname, _ in SLICES:
            for file_suffix, var_key, _ in VARS:
                key = (case_dir, slice_id, file_suffix)
                cache[key] = load_slice(
                    basename, slice_id, file_suffix, var_key, hname, vname
                )

    clim = {}
    for _, var_key, _ in VARS:
        vals = np.concatenate(
            [cache[(c, s, fs)][0].ravel()
             for c, _, _ in CASES
             for s, _, _, _ in SLICES
             for fs, vk, _ in VARS if vk == var_key]
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
        nrows,
        ncols,
        figsize=(3.4 * ncols, 2.8 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.08, right=0.9, top=0.96, bottom=0.04, hspace=0.28, wspace=0.08)

    for irow, (slice_id, hname, vname, slice_label) in enumerate(SLICES):
        for jvar, (file_suffix, var_key, var_label) in enumerate(VARS):
            row = irow * len(VARS) + jvar
            vmin, vmax, scale = clim[var_key]
            norm = LogNorm(vmin=vmin, vmax=vmax) if scale == "log" else Normalize(vmin=vmin, vmax=vmax)
            im = None
            for jcol, (case_dir, case_label, _) in enumerate(CASES):
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
                    ax.set_title(case_label, fontsize=13, fontweight="bold")
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
        "MHD ringing reproducer at t = 1.5 (RK3, HLLD, 256$^3$)",
        fontsize=15,
    )
    out_png = os.path.join(RUNROOT, "mosaic_t1p5.png")
    out_pdf = os.path.join(RUNROOT, "mosaic_t1p5.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
