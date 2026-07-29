#!/usr/bin/env python3
"""Render a sink-particle run, showing the MeshBlock grid coloured by owning MPI rank.

Usage:
  render_orbit_mpi.py <run_dir> <out_basename> [out_dir] [options]

Options:
  --var=NAME       field to show (default grav_phi; hydro_w = gas density)
  --norm=log       logarithmic colour scale (a collapse spans >4 decades, where a linear
                   scale renders as an all-black frame)
  --planes=LIST    comma-separated slice planes from {xy,yz,xz} (default xy). Give all
                   three to see every rank: one slice only intersects the blocks that
                   straddle its plane, so with a 3-D decomposition a single plane shows
                   only some of the ranks.
  --coord=VALUE    slice coordinate used for every plane (default 0.0)
  --extent=a,b,c,d override the plot extent (default: the mesh bounds from the file)
  --label=TEXT     title text
  --fps=N          video frame rate (default 6; these runs have few dumps, so the old 12
                   made a 36-dump run play in 3 seconds)
  --hold=N         write each frame N times (default 2). Slows playback without dropping
                   the frame rate, which some players stutter on below ~5 fps.

Relies on the run having been written with `single_file_per_rank = true`, which puts each
rank's MeshBlocks in its own file under bin/rank_<8-digit>/ -- that directory name IS the
owning rank, so no code change was needed in AthenaK to expose the domain decomposition.

Draws, per output time and per plane:
  * the field in that plane (per-block imshow at the block's own extent, coarse->fine)
  * every MeshBlock's outline coloured by the rank that owns it (= the MPI domains)
  * sink markers from the prtcl_d NGP deposit (one nonzero cell per sink), coloured by the
    rank whose file they appear in. Sinks are PROJECTED onto each panel, so a sink off the
    slice plane is still drawn rather than vanishing from that panel.
"""
import sys, os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bin_convert sits alongside
import bin_convert

# distinct, high-contrast rank colours (works on the dark field background)
RANK_COLORS = ["#00e5ff", "#ffe000", "#ff2fb0", "#7cff00",
               "#ff7a00", "#b47cff", "#00ff9d", "#ff5555"]
# cmr.chroma is the house colormap for AthenaK physics figures
try:
    import cmasher as cmr
    CMAP = cmr.chroma
except Exception:
    CMAP = "magma"

USE_LOG = False

# plane -> (horizontal axis, vertical axis, normal axis); axes numbered 1,2,3 as in AthenaK
PLANE_AXES = {"xy": (1, 2, 3), "yz": (2, 3, 1), "xz": (1, 3, 2)}
AXIS_NAME = {1: "x", 2: "y", 3: "z"}


def make_norm(vmin, vmax):
    """LogNorm for fields spanning decades (density in a collapse), else linear."""
    if USE_LOG:
        lo = max(vmin, vmax*1.0e-6) if vmax > 0 else 1.0e-30
        return LogNorm(vmin=max(lo, 1.0e-30), vmax=max(vmax, lo*10))
    return Normalize(vmin=vmin, vmax=vmax)


def rank_dirs(run_dir):
    """Return {rank_int: path} for the per-rank bin subdirectories."""
    out = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "bin", "rank_*"))):
        m = re.search(r"rank_(\d+)$", d)
        if m:
            out[int(m.group(1))] = d
    return out


def dump_indices(rdirs, var):
    """Sorted list of dump index strings present for `var` in every rank dir."""
    sets = []
    for d in rdirs.values():
        idx = set()
        for f in glob.glob(os.path.join(d, f"*.{var}.*.bin")):
            idx.add(os.path.basename(f).split(".")[-2])
        sets.append(idx)
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def plane_blocks(fd, var, plane, coord=0.0):
    """(h_min,h_max,v_min,v_max, 2D slice, level) for blocks straddling the slice plane.

    mb_geometry columns are block MIN/MAX bounds (the bin_convert docstring is wrong).
    The stored array is indexed [k][j][i] = (x3, x2, x1), so which array axis is held fixed
    and which two survive depends on the plane.
    """
    ha, va, na = PLANE_AXES[plane]
    out = []
    data = fd["mb_data"][var]
    geo = fd["mb_geometry"]
    logi = fd["mb_logical"]
    for m in range(fd["n_mbs"]):
        b = geo[m]                       # [x1min,x1max,x2min,x2max,x3min,x3max]
        lo, hi = b[2*(na-1)], b[2*(na-1)+1]
        if not (lo <= coord < hi):
            continue
        d = np.asarray(data[m])          # (nk, nj, ni)
        nk, nj, ni = d.shape
        n_along = {1: ni, 2: nj, 3: nk}[na]
        idx = int((coord - lo)/((hi - lo)/n_along))
        idx = max(0, min(n_along - 1, idx))
        if na == 3:                      # fix k -> (j,i) = (x2,x1): horiz x1, vert x2
            sl = d[idx, :, :]
        elif na == 1:                    # fix i -> (k,j) = (x3,x2): horiz x2, vert x3
            sl = d[:, :, idx]
        else:                            # fix j -> (k,i) = (x3,x1): horiz x1, vert x3
            sl = d[:, idx, :]
        out.append((b[2*(ha-1)], b[2*(ha-1)+1], b[2*(va-1)], b[2*(va-1)+1],
                    sl, int(logi[m][3])))
    return out


def load_time(rdirs, var, idx, plane, coord=0.0):
    """Read every rank's file for one dump: returns (time, {rank: [blocks]})."""
    per_rank, t = {}, None
    for r, d in rdirs.items():
        f = glob.glob(os.path.join(d, f"*.{var}.{idx}.bin"))
        if not f:
            continue
        fd = bin_convert.read_binary(f[0])
        v = fd["var_names"][0]
        per_rank[r] = plane_blocks(fd, v, plane, coord)
        t = fd["time"]
    return t, per_rank


def sinks_at(rdirs, idx, tol=1e-6, min_sep=0.004):
    """Sink (x, y, z, owning_rank) from the prtcl_d NGP deposit: each nonzero cell is one
    sink. Searched in 3-D, since the deposit cell can sit just off any slice plane."""
    found = []
    for r, d in rdirs.items():
        f = glob.glob(os.path.join(d, f"*.prtcl_d.{idx}.bin"))
        if not f:
            continue
        fd = bin_convert.read_binary(f[0])
        v = fd["var_names"][0]
        data, geo = fd["mb_data"][v], fd["mb_geometry"]
        for m in range(fd["n_mbs"]):
            dd = np.asarray(data[m])
            x1min, x1max, x2min, x2max, x3min, x3max = geo[m]
            nk, nj, ni = dd.shape
            for (k, j, i) in zip(*np.where(dd > tol)):
                x = x1min + (i + 0.5)*(x1max - x1min)/ni
                y = x2min + (j + 0.5)*(x2max - x2min)/nj
                z = x3min + (k + 0.5)*(x3max - x3min)/nk
                found.append((float(dd[k, j, i]), x, y, z, r))
    found.sort(reverse=True)
    pts = []
    for (_, x, y, z, r) in found:
        if all((x-px)**2 + (y-py)**2 + (z-pz)**2 > min_sep**2
               for (px, py, pz, _) in pts):
            pts.append((x, y, z, r))
    return pts


def draw_frame(ax, per_rank, pts, vmin, vmax, extent, plane, lw=0.7, ms=12):
    ha, va, _ = PLANE_AXES[plane]
    flat = [(b, r) for r, bl in per_rank.items() for b in bl]
    # field, coarse blocks first so finer ones land on top
    for (b, r) in sorted(flat, key=lambda br: br[0][5]):
        hmin, hmax, vlo, vhi, sl, lev = b
        ax.imshow(sl, origin="lower", extent=[hmin, hmax, vlo, vhi],
                  norm=make_norm(vmin, vmax), cmap=CMAP,
                  interpolation="nearest", aspect="auto", zorder=lev)
    # MeshBlock outlines, coloured by owning rank == the MPI domains
    for (b, r) in flat:
        hmin, hmax, vlo, vhi, _, _ = b
        ax.add_patch(Rectangle((hmin, vlo), hmax - hmin, vhi - vlo, fill=False,
                               edgecolor=RANK_COLORS[r % len(RANK_COLORS)],
                               linewidth=lw, alpha=0.95, zorder=40))
    # sinks projected onto this plane, ringed in their owner rank's colour
    halo = [pe.withStroke(linewidth=2.6, foreground="black")]
    for p in pts:
        coords = {1: p[0], 2: p[1], 3: p[2]}
        h, v, r = coords[ha], coords[va], p[3]
        ax.plot(h, v, marker="o", mfc="none", ms=ms, mew=2.0,
                mec=RANK_COLORS[r % len(RANK_COLORS)], zorder=50, path_effects=halo)
        ax.plot(h, v, marker=".", color="white", ms=ms*0.30, zorder=51,
                path_effects=halo)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    global USE_LOG
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                if a.startswith("--") and "=" in a)
    run_dir, out_base = pos[0], pos[1]
    out_dir = pos[2] if len(pos) > 2 else run_dir
    VAR = opts.get("var", "grav_phi")
    USE_LOG = (opts.get("norm", "linear") == "log")
    PLANES = [p.strip() for p in opts.get("planes", "xy").split(",") if p.strip()]
    for p in PLANES:
        if p not in PLANE_AXES:
            print(f"unknown plane {p!r}; use xy, yz or xz")
            return
    COORD = float(opts.get("coord", 0.0))
    LABEL = opts.get("label", "binary sink orbit")
    FPS = float(opts.get("fps", 6))
    HOLD = max(1, int(opts.get("hold", 2)))
    CBLAB = (r"$\phi$  (gravitational potential)" if VAR == "grav_phi"
             else r"gas density $\rho$")

    rdirs = rank_dirs(run_dir)
    if not rdirs:
        print("no bin/rank_* dirs -- was single_file_per_rank=true set?")
        return
    idxs = dump_indices(rdirs, VAR)
    nr = len(rdirs)
    print(f"{nr} rank dir(s), {len(idxs)} dumps, planes {PLANES}")

    # Drop leading dumps where the field is identically zero: phi is written once before
    # the first gravity solve and renders as a flat frame that also skews the colour range.
    while len(idxs) > 1:
        _, pr0 = load_time(rdirs, VAR, idxs[0], PLANES[0], COORD)
        v0 = np.concatenate([b[4].ravel() for bl in pr0.values() for b in bl])
        if np.nanmax(np.abs(v0)) > 0.0:
            break
        idxs = idxs[1:]

    # Colour range sampled ACROSS dumps and planes, not from the last dump alone: the peak
    # can occur mid-run, and a last-dump range saturates those frames.
    vmin, vmax = np.inf, -np.inf
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        ix = idxs[int(round(f*(len(idxs)-1)))]
        for pl in PLANES:
            _, prs = load_time(rdirs, VAR, ix, pl, COORD)
            vv = np.concatenate([b[4].ravel() for bl in prs.values() for b in bl])
            vmin = min(vmin, float(np.nanmin(vv)))
            vmax = max(vmax, float(np.nanmax(vv)))
    print(f"{VAR} range [{vmin:.6f}, {vmax:.6f}]")

    # per-plane extent from the MESH bounds in the file (never hardcode: the orbit box is
    # +-0.5, the Boss-Bodenheimer box +-2, and matplotlib autoscales to a block subset)
    d0 = sorted(rdirs.values())[0]
    md = bin_convert.read_binary(
        sorted(glob.glob(os.path.join(d0, f"*.{VAR}.{idxs[0]}.bin")))[0])
    bounds = {1: (md["x1min"], md["x1max"]), 2: (md["x2min"], md["x2max"]),
              3: (md["x3min"], md["x3max"])}
    EXT = {}
    for pl in PLANES:
        ha, va, _ = PLANE_AXES[pl]
        EXT[pl] = ([float(v) for v in opts["extent"].split(",")] if "extent" in opts
                   else [bounds[ha][0], bounds[ha][1], bounds[va][0], bounds[va][1]])
    print(f"extents {EXT}")

    handles = [Line2D([0], [0], color=RANK_COLORS[r % len(RANK_COLORS)], lw=2.2,
                      label=f"rank {r}") for r in sorted(rdirs)]

    def plane_title(pl):
        ha, va, na = PLANE_AXES[pl]
        return f"{AXIS_NAME[ha]}-{AXIS_NAME[va]}  ({AXIS_NAME[na]} = {COORD:g})"

    # ---- mosaic: planes down the rows, 4 evenly spaced times across the columns ----
    sel = [idxs[int(round(f*(len(idxs)-1)))] for f in (0.0, 0.33, 0.66, 1.0)]
    nrow = len(PLANES)
    fig, axes = plt.subplots(nrow, 4, figsize=(19, 5.0*nrow), constrained_layout=True,
                             squeeze=False)
    for irow, pl in enumerate(PLANES):
        for icol, ix in enumerate(sel):
            ax = axes[irow][icol]
            t, pr = load_time(rdirs, VAR, ix, pl, COORD)
            pts = sinks_at(rdirs, ix)
            draw_frame(ax, pr, pts, vmin, vmax, EXT[pl], pl)
            if irow == 0:
                ax.set_title(f"t = {t:.3f}   ({len(pts)} sinks)", fontsize=12)
            if icol == 0:
                ax.set_ylabel(plane_title(pl), fontsize=11)
    axes[0][0].legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.65,
                      facecolor="black", labelcolor="white", handlelength=1.4)
    sm = plt.cm.ScalarMappable(norm=make_norm(vmin, vmax), cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.013, pad=0.01)
    cb.set_label(CBLAB, fontsize=11)
    fig.suptitle(f"{LABEL} on {nr} MPI rank(s) — MeshBlocks coloured by owning rank "
                 f"(MPI domains); sinks ringed in their owner's colour, projected onto "
                 f"each plane", fontsize=13)
    out = os.path.join(out_dir, f"{out_base}_mosaic.png")
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # ---- video: one row of panels, one per plane ----
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa
        vout = os.path.join(out_dir, f"{out_base}.mp4")
        w = imageio.get_writer(vout, fps=FPS, codec="libx264", quality=8,
                               macro_block_size=1)
        print(f"video: {len(idxs)} dumps x {HOLD} @ {FPS:g} fps "
              f"= {len(idxs)*HOLD/FPS:.1f} s")
        for ix in idxs:
            figv, axv = plt.subplots(1, len(PLANES), figsize=(6.0*len(PLANES), 6.6),
                                     squeeze=False, constrained_layout=True)
            t = None
            pts = sinks_at(rdirs, ix)
            for icol, pl in enumerate(PLANES):
                t, pr = load_time(rdirs, VAR, ix, pl, COORD)
                draw_frame(axv[0][icol], pr, pts, vmin, vmax, EXT[pl], pl, ms=13)
                axv[0][icol].set_title(plane_title(pl), fontsize=11)
            axv[0][0].legend(handles=handles, loc="upper left", fontsize=8,
                             framealpha=0.6, facecolor="black", labelcolor="white",
                             handlelength=1.2)
            figv.suptitle(f"{LABEL} — {nr} MPI rank(s)   t = {t:.3f}\n"
                          f"MeshBlocks coloured by owning rank", fontsize=12)
            figv.canvas.draw()
            ww, hh = figv.canvas.get_width_height()
            buf = np.frombuffer(figv.canvas.buffer_rgba(),
                                dtype=np.uint8).reshape(hh, ww, 4)
            frame = buf[..., :3].copy()
            for _ in range(HOLD):      # repeat rather than re-render: the file reads dominate
                w.append_data(frame)
            plt.close(figv)
        w.close()
        print("wrote", vout)
    except Exception as e:
        print("video skipped:", e)


if __name__ == "__main__":
    main()
