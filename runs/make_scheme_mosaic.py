#!/usr/bin/env python3
"""Scheme mosaic: PLM / PPMX / WENO-Z / PPM+FB for one EMF method and resolution."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../fallback/vis/python"))
# allow both layouts
_VIS = [
    os.path.join(os.path.dirname(__file__), "../fallback/vis/python"),
    os.path.join(os.path.dirname(__file__), "../../vis/python"),
    "/home/velasco/athenak/fallback/vis/python",
]
for p in _VIS:
    if os.path.isdir(p):
        sys.path.insert(0, p)
        break
import bin_convert as bc

BASENAMES = {
    "plm": "turb_ringing_plm",
    "ppmx": "turb_ringing_ppmx",
    "wenoz": "turb_ringing_wenoz",
    "ppm_fb": "turb_ringing_ppm_fb",
}
CASES = [
    ("plm", "PLM"),
    ("ppmx", "PPMX"),
    ("wenoz", "WENO-Z"),
    ("ppm_fb", "PPM+FB"),
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
OUTIDX = "00006"


def load_slice(runroot, case_dir, basename, slice_id, file_suffix, var_key, hname, vname):
    path = os.path.join(
        runroot, case_dir, "bin", "rank_00000000",
        f"{basename}.{slice_id}_{file_suffix}.{OUTIDX}.bin",
    )
    if not os.path.exists(path):
        # try any rank_* directory
        bin_root = os.path.join(runroot, case_dir, "bin")
        for d in sorted(os.listdir(bin_root)):
            cand = os.path.join(bin_root, d,
                                f"{basename}.{slice_id}_{file_suffix}.{OUTIDX}.bin")
            if os.path.exists(cand):
                path = cand
                break
    data = bc.read_binary_as_athdf(path)
    return np.squeeze(data[var_key]), data[f"{hname}v"], data[f"{vname}v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True, choices=[128, 256])
    ap.add_argument("--emf", required=True, choices=["uct_hlld", "uct_hll"])
    args = ap.parse_args()

    runroot = f"/home/velasco/athenak/fallback/runs/ringing_rk3_{args.nx}/{args.emf}"
    cache = {}
    for case_dir, _ in CASES:
        for slice_id, hname, vname, _ in SLICES:
            for file_suffix, var_key, _ in VARS:
                cache[(case_dir, slice_id, file_suffix)] = load_slice(
                    runroot, case_dir, BASENAMES[case_dir],
                    slice_id, file_suffix, var_key, hname, vname,
                )

    clim = {}
    for file_suffix, var_key, _ in VARS:
        vals = np.concatenate(
            [cache[(c, s, file_suffix)][0].ravel()
             for c, _ in CASES for s, _, _, _ in SLICES]
        )
        if var_key == "j2":
            clim[var_key] = (max(np.percentile(vals, 1), 1e-12),
                             np.percentile(vals, 99.5), "log")
        else:
            clim[var_key] = (np.percentile(vals, 0.5),
                             np.percentile(vals, 99.5), "linear")

    nrows = len(SLICES) * len(VARS)
    ncols = len(CASES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows),
                             squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.94, bottom=0.04,
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
                im = ax.imshow(field, origin="lower", aspect="equal",
                               extent=[horiz[0], horiz[-1], vert[0], vert[-1]],
                               norm=norm, cmap="inferno", interpolation="nearest")
                if row == 0:
                    ax.set_title(case_label, fontsize=12, fontweight="bold")
                if jcol == 0:
                    ax.set_ylabel(f"{slice_label}\n{var_label}", fontsize=10)
                ax.tick_params(labelsize=7)
                if row == nrows - 1:
                    ax.set_xlabel(f"${hname}$")
            cax = fig.add_axes([0.915, axes[row, 0].get_position().y0,
                                0.012, axes[row, 0].get_position().height])
            fig.colorbar(im, cax=cax)

    fig.suptitle(
        f"Ringing reproducer at t = 1.5 (RK3, HLLD, {args.emf}, {args.nx}$^3$)",
        fontsize=15,
    )
    out_png = os.path.join(runroot, f"mosaic_{args.emf}_{args.nx}_t1p5.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
