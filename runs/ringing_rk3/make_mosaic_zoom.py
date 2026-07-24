#!/usr/bin/env python3
"""Zoom mosaic on documented ringing discriminator regions at t=1.5."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../vis/python"))
import bin_convert as bc

from make_mosaic import CASES, OUTIDX, RUNROOT, SLICES, VARS, load_slice

# Regions from docs/mhd_ringing_reproducer.md (approx. cell-centered coords)
ZOOM = {
    "slice_x1_0": dict(x2=(-0.18, -0.02), x3=(0.18, 0.42), h="x2", v="x3"),
    "slice_x2_001953125": dict(x1=(-0.18, -0.02), x3=(0.18, 0.42), h="x1", v="x3"),
}


def crop(field, horiz, vert, slice_id):
    z = ZOOM[slice_id]
    hmask = (horiz >= z[z["h"]][0]) & (horiz <= z[z["h"]][1])
    vmask = (vert >= z[z["v"]][0]) & (vert <= z[z["v"]][1])
    return field[np.ix_(vmask, hmask)], horiz[hmask], vert[vmask]


def main():
    cache = {}
    for case_dir, _, basename in CASES:
        for slice_id, hname, vname, _ in SLICES:
            for file_suffix, var_key, _ in VARS:
                key = (case_dir, slice_id, file_suffix)
                f, h, v = load_slice(basename, slice_id, file_suffix, var_key, hname, vname)
                cache[key] = crop(f, h, v, slice_id)

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
                max(np.percentile(vals, 2), 1e-12),
                np.percentile(vals, 98),
                "log",
            )
        else:
            clim[var_key] = (np.percentile(vals, 2), np.percentile(vals, 98), "linear")

    nrows = len(SLICES) * len(VARS)
    ncols = len(CASES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False)
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
        "Ringing discriminator zoom at t = 1.5 (exact cell values, no smoothing)",
        fontsize=15,
    )
    out_png = os.path.join(RUNROOT, "mosaic_zoom_t1p5.png")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
