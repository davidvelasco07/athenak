#!/usr/bin/env python3
"""
3D inclined Gresho vortex mosaic: 3 methods x 3 Mach numbers x 3 planes.

Runs the 3D inclined Gresho vortex (problem=4) at a single resolution for three
reconstruction methods (PLM, WENOZ, WENOZ+Mignone) and three Mach numbers.
Produces one mosaic PNG per coordinate midplane (xy, xz, yz) of |v_phi|/cs at t=1.

Usage:
  python3 tst/gresho_3d_mosaic.py [path/to/athena]
      --machs 0.1 0.01 0.001
      --nx 32
      --skip-runs
"""

import argparse
import os
import re
import subprocess
import sys
import numpy as np

# ── method definitions ──────────────────────────────────────────────
METHODS = [
    {
        "key": "plm",
        "label": "PLM (2nd order)",
        "overrides": {
            "hydro/reconstruct": "plm",
            "mesh/nghost": "2",
            "time/integrator": "rk2",
        },
    },
    {
        "key": "wenoz",
        "label": "WENOZ",
        "overrides": {
            "hydro/reconstruct": "wenoz",
            "mesh/nghost": "3",
            "time/integrator": "rk3",
        },
    },
    {
        "key": "mignone",
        "label": "WENOZ + Mignone",
        "overrides": {
            "hydro/reconstruct": "wenoz",
            "hydro/mignone": "true",
            "mesh/nghost": "5",
            "time/integrator": "rk3",
        },
    },
]


# ── helpers ─────────────────────────────────────────────────────────
def find_root_dir():
    """Return the AthenaK root directory (parent of tst/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)  # tst/ -> root
    if os.path.isdir(os.path.join(root, "src")) and os.path.isdir(
        os.path.join(root, "vis")
    ):
        return root
    return os.getcwd()


def resolve(path, root):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def prepare_input_file(base_athinput, rundir):
    """Copy base input and inject parameters that might be missing."""
    dst = os.path.join(rundir, "gresho_vortex_3d.athinput")
    with open(base_athinput, "r") as f:
        text = f.read()

    # Inject missing <hydro> parameters
    hydro_extra = (
        "\n# --- injected by gresho_3d_mosaic.py ---\n"
        "fourth_order_diff = false\n"
        "mignone           = false\n"
    )
    if "mignone" not in text:
        text = text.replace("<problem>", hydro_extra + "\n<problem>")

    # Inject pgen_name if missing
    if "pgen_name" not in text:
        text = text.replace("<problem>", "<problem>\npgen_name = gresho_vortex\n")

    # Remove slice parameters (they clip bin output to 1D)
    text = re.sub(r'(?m)^slice_x[123]\s*=.*\n', '', text)

    with open(dst, "w") as f:
        f.write(text)
    return dst


def run_simulation(exe, athinput, nx, method, mach, rundir):
    """Run a single 3D AthenaK Gresho vortex simulation."""
    key = method["key"]
    mach_tag = f"{mach}".replace(".", "p")
    subdir = os.path.join(rundir, f"{key}_M{mach_tag}_nx{nx}")
    os.makedirs(subdir, exist_ok=True)
    basename = f"gresho3d_{key}_M{mach_tag}_{nx}"

    overrides = {
        # resolution — cubic domain [0,1]^3
        "mesh/nx1": str(nx),
        "mesh/nx2": str(nx),
        "mesh/nx3": str(nx),
        "meshblock/nx1": str(nx),
        "meshblock/nx2": str(nx),
        "meshblock/nx3": str(nx),
        "mesh/x1min": "0.0",
        "mesh/x1max": "1.0",
        "mesh/x2min": "0.0",
        "mesh/x2max": "1.0",
        "mesh/x3min": "0.0",
        "mesh/x3max": "1.0",
        "mesh_refinement/refinement": "none",
        # physics — problem=4 (3D inclined)
        "problem/problem": "4",
        "problem/Mach": str(mach),
        "hydro/rsolver": "hllc",
        # time
        "time/tlim": "1.0",
        "time/nlim": "-1",
        "time/cfl_number": "0.3",
        # output — bin snapshots every dt=1
        "job/basename": basename,
        "output1/file_type": "bin",
        "output1/variable": "hydro_w",
        "output1/dt": "1.0",
    }
    overrides.update(method["overrides"])

    cmd = [exe, "-i", athinput] + [f"{k}={v}" for k, v in overrides.items()]
    print(f"  {key:10s}  Mach={mach:<8s}  nx={nx}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=subdir)
    if result.returncode != 0:
        print(f"  *** FAILED: {key} Mach={mach} ***", flush=True)
        print(result.stdout[-2000:])
        print(result.stderr[-1000:])
    return subdir, basename


def find_bin_file(subdir, basename, output_num):
    """Return path to .bin file for a given output number."""
    fname = os.path.join(subdir, "bin", f"{basename}.hydro_w.{output_num:05d}.bin")
    if not os.path.isfile(fname):
        fname2 = os.path.join(subdir, f"{basename}.hydro_w.{output_num:05d}.bin")
        if os.path.isfile(fname2):
            return fname2
        raise FileNotFoundError(f"Not found: {fname}")
    return fname


def _stitch_var_3d(fd, varname):
    """Stitch a single variable from multi-MB bin data into a 3D array."""
    n_mbs = fd["n_mbs"]
    Nx1, Nx2, Nx3 = fd["Nx1"], fd["Nx2"], fd["Nx3"]
    if n_mbs == 1:
        return fd["mb_data"][varname][0]  # shape (nx3, nx2, nx1)
    arr = np.empty((Nx3, Nx2, Nx1), dtype=np.float64)
    nx1_out = fd["nx1_out_mb"]
    nx2_out = fd["nx2_out_mb"]
    nx3_out = fd["nx3_out_mb"]
    for m in range(n_mbs):
        lx1 = fd["mb_logical"][m][0]
        lx2 = fd["mb_logical"][m][1]
        lx3 = fd["mb_logical"][m][2]
        i0 = lx1 * nx1_out
        j0 = lx2 * nx2_out
        k0 = lx3 * nx3_out
        arr[k0:k0+nx3_out, j0:j0+nx2_out, i0:i0+nx1_out] = \
            fd["mb_data"][varname][m]
    return arr


def compute_rotation_angles(x1size, x2size, x3size):
    """Compute rotation angles matching the pgen (linear_wave convention)."""
    ang_3 = np.arctan(x1size / x2size)
    sin_a3, cos_a3 = np.sin(ang_3), np.cos(ang_3)
    ang_2 = np.arctan(0.5 * (x1size * cos_a3 + x2size * sin_a3) / x3size)
    sin_a2, cos_a2 = np.sin(ang_2), np.cos(ang_2)
    return sin_a2, cos_a2, sin_a3, cos_a3


def read_vphi_over_cs_3d(filename, root_dir, gamma=5.0/3.0):
    """Read |v_phi|/cs on all three midplanes from a 3D .bin file.

    Transforms velocities and coordinates to the vortex frame using the
    same rotation angles as the pgen, computes v_phi in the vortex plane,
    then returns midplane slices for xy, xz, and yz planes.

    Returns
    -------
    slices : dict
        Keys 'xy', 'xz', 'yz'.  Each value is (2D-array, extent-list).
    time : float
    """
    sys.path.insert(0, os.path.join(root_dir, "vis", "python"))
    import bin_convert

    fd = bin_convert.read_binary(filename)
    for var in ("velx", "vely", "velz", "dens", "eint"):
        if var not in fd["var_names"]:
            raise ValueError(
                f"Variable '{var}' not found. Available: {fd['var_names']}"
            )

    vx = _stitch_var_3d(fd, "velx")    # (Nx3, Nx2, Nx1)
    vy = _stitch_var_3d(fd, "vely")
    vz = _stitch_var_3d(fd, "velz")
    rho = _stitch_var_3d(fd, "dens")
    eint = _stitch_var_3d(fd, "eint")

    Nx1, Nx2, Nx3 = fd["Nx1"], fd["Nx2"], fd["Nx3"]

    # Build 3D cell-center coordinates
    dx = (fd["x1max"] - fd["x1min"]) / Nx1
    dy = (fd["x2max"] - fd["x2min"]) / Nx2
    dz = (fd["x3max"] - fd["x3min"]) / Nx3
    x1d = fd["x1min"] + (np.arange(Nx1) + 0.5) * dx
    x2d = fd["x2min"] + (np.arange(Nx2) + 0.5) * dy
    x3d = fd["x3min"] + (np.arange(Nx3) + 0.5) * dz
    X, Y, Z = np.meshgrid(x1d, x2d, x3d, indexing='ij')  # (Nx1, Nx2, Nx3)
    # Transpose to (Nx3, Nx2, Nx1) to match data layout
    X = X.T
    Y = Y.T
    Z = Z.T

    # Domain center
    xc = 0.5 * (fd["x1min"] + fd["x1max"])
    yc = 0.5 * (fd["x2min"] + fd["x2max"])
    zc = 0.5 * (fd["x3min"] + fd["x3max"])

    # Rotation angles (matching pgen)
    x1size = fd["x1max"] - fd["x1min"]
    x2size = fd["x2max"] - fd["x2min"]
    x3size = fd["x3max"] - fd["x3min"]

    a3 = np.pi / 16.0  # should match pgen; compute_rotation_angles is a sanity check
    sin_a3 = np.sin(a3)
    cos_a3 = np.cos(a3)

    # Transform coordinates to vortex frame
    Dx = X - xc
    Dy = Y - yc
    Dz = Z - zc
    Xp = Dx * cos_a3 + Dz * sin_a3
    Yp = Dy
    Zp = -Dx * sin_a3 + Dz * cos_a3
    R = np.sqrt(Xp**2 + Yp**2)

    # Transform velocity to vortex frame
    vxp =   vx * cos_a3 + vz * sin_a3
    vyp =   vy
    vzp =  -vx * sin_a3 + vz * cos_a3

    # v_phi = (-v_y' * z' + v_z' * y') / r
    vphi = np.where(R > 0, (-vxp * Yp + vyp * Xp) / R, 0.0)

    # cs = sqrt(gamma * P / rho);  P = eint * (gamma - 1)
    gm1 = gamma - 1.0
    cs = np.sqrt(gamma * gm1 * eint / rho)

    Mach = np.abs(vphi) / cs

    x1min, x1max = fd["x1min"], fd["x1max"]
    x2min, x2max = fd["x2min"], fd["x2max"]
    x3min, x3max = fd["x3min"], fd["x3max"]

    # Midplane slices — data shape is (Nx3, Nx2, Nx1)
    slices = {
        'xy': (Mach[Nx3 // 2, :, :], [x1min, x1max, x2min, x2max]),
        'xz': (Mach[:, Nx2 // 2, :], [x1min, x1max, x3min, x3max]),
        'yz': (Mach[:, :, Nx1 // 2], [x2min, x2max, x3min, x3max]),
    }
    return slices, fd["time"]


def make_mosaic(data_dict, machs, methods, nx, outfile,
                ref_dict=None, suptitle=None, domain_extent=None,
                xlabel="$x$", ylabel="$y$"):
    """Create a 3-row x 3-col mosaic PNG (methods x Mach numbers).

    Each column (Mach number) is independently normalized 0..vmax where
    vmax is the peak from the reference (t=0) data if ref_dict is provided,
    otherwise the max across methods at that Mach.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    nrows = len(methods)
    ncols = len(machs)
    panel_size = 3.0
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols + 1.0, panel_size * nrows + 1.2),
        squeeze=False,
    )
    cmap = "plasma"

    # Compute per-column vmax from reference (t=0) or from data
    col_vmax = {}
    for mach in machs:
        if ref_dict and mach in ref_dict:
            col_vmax[mach] = ref_dict[mach]
        else:
            vals = [data_dict.get(m["key"], {}).get(mach, None) for m in methods]
            vals = [v.max() for v in vals if v is not None]
            col_vmax[mach] = max(vals) if vals else 1.0

    ims = {}  # one im per column for colorbars
    for row, meth in enumerate(methods):
        for col, mach in enumerate(machs):
            ax = axes[row, col]
            arr = data_dict.get(meth["key"], {}).get(mach, None)
            norm = Normalize(vmin=0.0, vmax=col_vmax[mach])
            if arr is not None:
                im = ax.imshow(
                    arr, origin="lower",
                    extent=domain_extent or [0.0, 1.0, 0.0, 1.0],
                    cmap=cmap, norm=norm,
                    aspect="equal", interpolation="none",
                )
                ims[col] = im
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#999999")
            if row == 0:
                ax.set_title(rf"$\mathrm{{Ma}} = {mach}$", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{meth['label']}\n{ylabel}", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=10)
            else:
                ax.set_xticklabels([])

    # One colorbar per column
    fig.subplots_adjust(hspace=0.08, wspace=0.12)
    for col, mach in enumerate(machs):
        if col not in ims:
            continue
        bbox = axes[-1, col].get_position()
        cbar_ax = fig.add_axes([bbox.x0, bbox.y0 - 0.05, bbox.width, 0.015])
        cbar = fig.colorbar(ims[col], cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(r"$|v_\phi|/c_s$", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.98)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="3D inclined Gresho vortex mosaic: compare methods at "
                    "multiple Mach numbers"
    )
    parser.add_argument(
        "exe", nargs="?", default="build/src/athena",
        help="Path to AthenaK executable",
    )
    parser.add_argument(
        "--input", default=None,
        help="Base athinput file (default: tst/inputs/gresho_vortex.athinput)",
    )
    parser.add_argument(
        "--machs", nargs="+", type=float,
        default=[0.1, 0.01, 0.001],
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--rundir", default="gresho_3d_runs")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--prefix", default="gresho_3d_mosaic")
    parser.add_argument(
        "--skip-runs", action="store_true",
        help="Skip simulations; only generate plots from existing data",
    )
    args = parser.parse_args()

    root = find_root_dir()
    exe = resolve(args.exe, root)
    base_athinput = (
        resolve(args.input, root)
        if args.input
        else os.path.join(root, "tst", "inputs", "gresho_vortex.athinput")
    )
    rundir = resolve(args.rundir, root)
    outdir = resolve(args.outdir, root)
    os.makedirs(rundir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # Prepare input file with injected parameters
    athinput = prepare_input_file(base_athinput, rundir)

    # Format Mach strings for file naming
    mach_strs = [f"{m:g}" for m in args.machs]

    # ── run simulations ──
    run_info = {}  # run_info[method_key][mach_str] = (subdir, basename)
    for meth in METHODS:
        run_info[meth["key"]] = {}
        for mach, mstr in zip(args.machs, mach_strs):
            if not args.skip_runs:
                subdir, bn = run_simulation(
                    exe, athinput, args.nx, meth, mstr, rundir
                )
            else:
                mach_tag = mstr.replace(".", "p")
                subdir = os.path.join(
                    rundir, f"{meth['key']}_M{mach_tag}_nx{args.nx}"
                )
                bn = f"gresho3d_{meth['key']}_M{mach_tag}_{args.nx}"
            run_info[meth["key"]][mstr] = (subdir, bn)

    # Plane definitions: name -> (xlabel, ylabel)
    PLANES = {
        'xy': ('$x$', '$y$'),
        'xz': ('$x$', '$z$'),
        'yz': ('$y$', '$z$'),
    }

    # ── read t=0 reference to get per-Mach vmax (per plane) ──
    print("\n--- Reading t=0 reference for normalization ---")
    ref_dict = {plane: {} for plane in PLANES}  # ref_dict[plane][mstr] = vmax
    ref_meth = METHODS[0]  # any method; t=0 is identical
    for mstr in mach_strs:
        subdir, bn = run_info[ref_meth["key"]][mstr]
        try:
            binf = find_bin_file(subdir, bn, 0)
            slices, _ = read_vphi_over_cs_3d(binf, root)
            for plane in PLANES:
                arr0, _ = slices[plane]
                ref_dict[plane][mstr] = arr0.max()
            print(f"    Mach={mstr:<8s}  vmax(t=0): "
                  + "  ".join(f"{p}={ref_dict[p][mstr]:.6f}" for p in PLANES))
        except Exception as e:
            print(f"    Mach={mstr:<8s}  ref MISSING ({e})")

    # ── build mosaics at t=1 (output 1) — one per plane ──
    output_num = 1
    print(f"\n--- Mosaics for t = 1  (output {output_num:05d}) ---")

    # Read all data once (keyed by method, mach)
    all_slices = {}  # all_slices[method_key][mstr] = {plane: (arr, extent)}
    for meth in METHODS:
        all_slices[meth["key"]] = {}
        for mstr in mach_strs:
            subdir, bn = run_info[meth["key"]][mstr]
            try:
                binf = find_bin_file(subdir, bn, output_num)
                slices, ft = read_vphi_over_cs_3d(binf, root)
                all_slices[meth["key"]][mstr] = slices
                arr_xy, _ = slices['xy']
                print(f"    {meth['key']:10s}  Mach={mstr:<8s}  t={ft:.3f}  "
                      f"|vphi/cs|_xy range=[{arr_xy.min():.4f}, {arr_xy.max():.4f}]")
            except (FileNotFoundError, Exception) as e:
                print(f"    {meth['key']:10s}  Mach={mstr:<8s}  MISSING ({e})")

    # Generate one mosaic per plane
    for plane, (xlab, ylab) in PLANES.items():
        data_dict = {}
        ext = None
        for meth in METHODS:
            data_dict[meth["key"]] = {}
            for mstr in mach_strs:
                sl = all_slices.get(meth["key"], {}).get(mstr, None)
                if sl is not None and plane in sl:
                    arr, ext_p = sl[plane]
                    data_dict[meth["key"]][mstr] = arr
                    ext = ext_p

        title = (rf"3D Inclined Gresho Vortex ($\mathit{{{plane}}}$ midplane)  "
                 rf"$|v_\phi|/c_s$    $N_x = {args.nx}$    $t = 1$")
        outfile = os.path.join(outdir, f"{args.prefix}_{plane}_nx{args.nx}.png")
        make_mosaic(data_dict, mach_strs, METHODS, args.nx, outfile,
                    ref_dict=ref_dict[plane], suptitle=title,
                    domain_extent=ext, xlabel=xlab, ylabel=ylab)

    print("\nDone.")


if __name__ == "__main__":
    main()
