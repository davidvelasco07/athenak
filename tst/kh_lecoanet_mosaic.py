#!/usr/bin/env python3
"""
Lecoanet KHI mosaic comparison: 4 methods × 4 resolutions.

Runs the Lecoanet et al. (2016) Kelvin-Helmholtz instability test
(iprob=4 in kh.cpp) for the unstratified case (drho/rho=0) with
viscosity nu=1e-5 (Re=10^5).

Produces mosaic PNGs (4 cols = resolutions, 4 rows = methods) of the
passive-scalar dye concentration C at requested output times.

Usage:
  python3 tst/kh_lecoanet_mosaic.py [path/to/athena]
      --resolutions 16 32 64 128
      --times 2 4 8
      --rundir kh_lecoanet_runs
      --skip-runs
"""

import argparse
import os
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
        "key": "wenoz4",
        "label": "WENOZ + 4th-diff",
        "overrides": {
            "hydro/reconstruct": "wenoz",
            "hydro/fourth_order_diff": "true",
            "mesh/nghost": "3",
            "time/integrator": "rk3",
        },
    },
    {
        "key": "mignone",
        "label": "WENOZ + 4th-diff + Mignone",
        "overrides": {
            "hydro/reconstruct": "wenoz",
            "hydro/fourth_order_diff": "true",
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
    """Copy base input and inject parameters that might be missing.

    AthenaK CLI overrides can only modify parameters already present in
    the input file.  The base KH input has no viscosity/mignone/fourth_order_diff
    entries, so we append them here with dummy defaults that will be
    overridden on the command line.
    """
    dst = os.path.join(rundir, "kh_lecoanet.athinput")
    with open(base_athinput, "r") as f:
        text = f.read()

    # Inject missing <hydro> parameters (viscosity, 4th-order flags)
    hydro_extra = (
        "\n# --- injected by kh_lecoanet_mosaic.py ---\n"
        "viscosity         = 0.0\n"
        "fourth_order_diff = false\n"
        "mignone           = false\n"
    )
    if "viscosity" not in text:
        text = text.replace("<problem>", hydro_extra + "\n<problem>")

    # Inject pgen_name if missing (needed for built_in_pgens builds)
    if "pgen_name" not in text:
        text = text.replace("<problem>", "<problem>\npgen_name = kh\n")

    with open(dst, "w") as f:
        f.write(text)
    return dst


MAX_MB_SIZE = 256   # max meshblock cells per direction


def choose_meshblock(nx, max_mb=MAX_MB_SIZE):
    """Choose meshblock size: largest divisor of nx that is <= max_mb."""
    if nx <= max_mb:
        return nx
    # try descending from max_mb
    for mb in range(max_mb, 0, -1):
        if nx % mb == 0:
            return mb
    return nx  # fallback (shouldn't happen for power-of-2)


def run_simulation(exe, athinput, nx1, method, rundir,
                    viscosity="1e-5", drho_rho0="0.0"):
    """Run a single AthenaK simulation and return (subdir, basename)."""
    key = method["key"]
    subdir = os.path.join(rundir, f"{key}_nx{nx1}")
    os.makedirs(subdir, exist_ok=True)
    basename = f"KH_{key}_{nx1}"
    nx2 = 2 * nx1
    mb_nx1 = choose_meshblock(nx1)
    mb_nx2 = choose_meshblock(nx2)

    overrides = {
        # resolution
        "mesh/nx1": str(nx1),
        "mesh/nx2": str(nx2),
        "meshblock/nx1": str(mb_nx1),
        "meshblock/nx2": str(mb_nx2),
        # physics
        "problem/drho_rho0": drho_rho0,
        "hydro/viscosity": viscosity,
        "hydro/rsolver": "hllc",
        # time
        "time/tlim": "8.0",
        "time/cfl_number": "0.4",
        # output — binary snapshots every dt=2
        "job/basename": basename,
        "output1/dt": "8.0",  # history only at end (keep it quiet)
        "output2/file_type": "bin",
        "output2/variable": "hydro_w",
        "output2/dt": "2.0",
    }
    overrides.update(method["overrides"])

    cmd = [exe, "-i", athinput] + [f"{k}={v}" for k, v in overrides.items()]
    n_mbs = (nx1 // mb_nx1) * (nx2 // mb_nx2)
    print(f"  CMD: {os.path.basename(exe)} ... {key} nx1={nx1} "
          f"(mb={mb_nx1}x{mb_nx2}, {n_mbs} blocks)")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=subdir)
    if result.returncode != 0:
        print(f"  *** FAILED: {key} nx={nx1} ***")
        print(result.stdout[-2000:])
        print(result.stderr[-1000:])
        return subdir, basename  # continue; mosaic will skip missing data
    return subdir, basename


def find_bin_file(subdir, basename, output_num):
    """Return path to .bin file for a given output number."""
    # AthenaK places bin outputs in a bin/ subdirectory
    fname = os.path.join(subdir, "bin", f"{basename}.hydro_w.{output_num:05d}.bin")
    if not os.path.isfile(fname):
        # fallback: check directly in subdir
        fname2 = os.path.join(subdir, f"{basename}.hydro_w.{output_num:05d}.bin")
        if os.path.isfile(fname2):
            return fname2
        raise FileNotFoundError(f"Not found: {fname}")
    return fname


def read_scalar(filename, root_dir):
    """Read passive scalar s_00 from a .bin file and return (2D-array, time).

    Handles both single and multiple meshblocks by stitching them
    into the full domain array using mb_logical indices.
    """
    sys.path.insert(0, os.path.join(root_dir, "vis", "python"))
    import bin_convert

    fd = bin_convert.read_binary(filename)
    var = "s_00"
    if var not in fd["var_names"]:
        raise ValueError(
            f"Variable '{var}' not found. Available: {fd['var_names']}"
        )

    n_mbs = fd["n_mbs"]
    if n_mbs == 1:
        # single meshblock — fast path
        data_2d = fd["mb_data"][var][0][0, :, :]  # [ny, nx]
    else:
        # multiple meshblocks — stitch using mb_logical for global position
        Nx1 = fd["Nx1"]
        Nx2 = fd["Nx2"]
        nx1_out = fd["nx1_out_mb"]
        nx2_out = fd["nx2_out_mb"]
        data_2d = np.empty((Nx2, Nx1), dtype=np.float64)
        for m in range(n_mbs):
            # mb_logical[m] = [lx1, lx2, lx3, level]
            lx1, lx2 = fd["mb_logical"][m][0], fd["mb_logical"][m][1]
            block = fd["mb_data"][var][m][0, :, :]  # [ny_mb, nx_mb]
            i0 = lx1 * nx1_out
            j0 = lx2 * nx2_out
            data_2d[j0:j0 + nx2_out, i0:i0 + nx1_out] = block

    return data_2d, fd["time"]


def make_mosaic(data_dict, time, resolutions, methods, outfile,
                suptitle=None):
    """Create a 4-row × N-col mosaic PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    nrows = len(methods)
    ncols = len(resolutions)
    panel_w, panel_h = 2.2, 4.4  # each panel ~1:2 aspect
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_w * ncols + 1.0, panel_h * nrows + 1.2),
        squeeze=False,
    )
    norm = Normalize(vmin=1.0, vmax=2.0)
    cmap = "RdBu_r"

    im = None
    for row, meth in enumerate(methods):
        for col, nx1 in enumerate(resolutions):
            ax = axes[row, col]
            arr = data_dict.get(meth["key"], {}).get(nx1, None)
            if arr is not None:
                im = ax.imshow(
                    arr,
                    origin="lower",
                    extent=[-0.5, 0.5, -1.0, 1.0],
                    cmap=cmap,
                    norm=norm,
                    aspect="equal",
                    interpolation="none",
                )
            else:
                # Blank panel for missing data
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#999999")
            if row == 0:
                ax.set_title(rf"$N_x = {nx1}$", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{meth['label']}\n$y$", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == nrows - 1:
                ax.set_xlabel("$x$", fontsize=10)
            else:
                ax.set_xticklabels([])

    fig.subplots_adjust(right=0.88, hspace=0.08, wspace=0.05)
    if im is not None:
        cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.80])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Dye concentration $C$", fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.98)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Lecoanet KHI mosaic: compare methods at multiple resolutions"
    )
    parser.add_argument(
        "exe",
        nargs="?",
        default="build/src/athena",
        help="Path to AthenaK executable",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Base athinput file (default: inputs/hydro/kh2d-lecoanet.athinput)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=[2.0, 4.0, 8.0],
    )
    parser.add_argument("--rundir", default="kh_lecoanet_runs")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--viscosity", default="1e-5",
                        help="Kinematic viscosity (default: 1e-5 → Re=10^5)")
    parser.add_argument("--drho_rho0", default="0.0",
                        help="Density contrast (0=unstratified, 1=stratified)")
    parser.add_argument("--prefix", default="kh_lecoanet",
                        help="Output filename prefix")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Run only these methods (e.g. --methods wenoz mignone). "
             "Default: all four (plm wenoz wenoz4 mignone)",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip simulations; only generate plots from existing data",
    )
    args = parser.parse_args()

    root = find_root_dir()
    exe = resolve(args.exe, root)
    base_athinput = (
        resolve(args.input, root)
        if args.input
        else os.path.join(root, "inputs", "hydro", "kh2d-lecoanet.athinput")
    )
    rundir = resolve(args.rundir, root)
    outdir = resolve(args.outdir, root)
    os.makedirs(rundir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # Filter methods if requested
    if args.methods:
        methods = [m for m in METHODS if m["key"] in args.methods]
        if not methods:
            print(f"ERROR: no matching methods for {args.methods}")
            print(f"  Available: {[m['key'] for m in METHODS]}")
            sys.exit(1)
    else:
        methods = METHODS

    # Prepare an input file with all needed parameters injected
    athinput = prepare_input_file(base_athinput, rundir)

    dt_out = 2.0  # output cadence in the simulations

    # ── run simulations ──
    run_info = {}  # run_info[method_key][nx1] = (subdir, basename)
    if not args.skip_runs:
        for meth in methods:
            run_info[meth["key"]] = {}
            for nx1 in args.resolutions:
                print(f"\n{'='*60}")
                print(f"  {meth['label']}   nx1 = {nx1}")
                print(f"{'='*60}")
                subdir, bn = run_simulation(exe, athinput, nx1, meth, rundir,
                                            viscosity=args.viscosity,
                                            drho_rho0=args.drho_rho0)
                run_info[meth["key"]][nx1] = (subdir, bn)
    else:
        for meth in methods:
            run_info[meth["key"]] = {}
            for nx1 in args.resolutions:
                subdir = os.path.join(rundir, f"{meth['key']}_nx{nx1}")
                bn = f"KH_{meth['key']}_{nx1}"
                run_info[meth["key"]][nx1] = (subdir, bn)

    # ── build suptitle components ──
    nu_f = float(args.viscosity)
    Re = 1.0 / nu_f if nu_f > 0 else float("inf")
    drho_f = float(args.drho_rho0)
    # Format Re as 10^n if it's a power of 10
    import math
    if Re > 0 and Re != float("inf") and abs(math.log10(Re) - round(math.log10(Re))) < 1e-6:
        re_str = rf"$Re = 10^{{{int(round(math.log10(Re)))}}}$"
    else:
        re_str = rf"$Re = {Re:.0e}$"
    drho_str = rf"$\Delta\rho/\rho = {drho_f:g}$"
    case_tag = f"Re{Re:.0e}_drho{drho_f:g}".replace("+", "")

    # ── make mosaics ──
    for t in args.times:
        output_num = int(round(t / dt_out))
        print(f"\n--- Mosaic for t = {t:.0f}  (output {output_num:05d}) ---")
        data_dict = {}
        available_res = []
        for meth in methods:
            data_dict[meth["key"]] = {}
        for nx1 in args.resolutions:
            any_ok = False
            for meth in methods:
                subdir, bn = run_info[meth["key"]][nx1]
                try:
                    binf = find_bin_file(subdir, bn, output_num)
                    arr, ft = read_scalar(binf, root)
                    data_dict[meth["key"]][nx1] = arr
                    print(f"    {meth['key']:10s} nx={nx1:4d}  t_file={ft:.3f}")
                    any_ok = True
                except (FileNotFoundError, Exception) as e:
                    print(f"    {meth['key']:10s} nx={nx1:4d}  MISSING ({e})")
            if any_ok:
                available_res.append(nx1)
        if not available_res:
            print("  No data available for any resolution, skipping.")
            continue

        title = rf"Lecoanet KHI    {drho_str}    {re_str}    $t = {t:.0f}$"
        outfile = os.path.join(outdir, f"{args.prefix}_{case_tag}_t{t:.0f}.png")
        make_mosaic(data_dict, t, available_res, methods, outfile,
                    suptitle=title)

    print("\nDone.")


if __name__ == "__main__":
    main()
