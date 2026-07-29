#!/usr/bin/env python3
"""Render Boss-Bodenheimer collapse: AMR-aware density-slice mosaic + evolution video
with EXACT sink-particle markers.

Usage:
  render_bb.py <run_dir> <t_ff> [out_basename] [out_dir]

Sink markers come from the deposited-particle-density output (prtcl_d / "pdens"):
the sink deposit is nearest-grid-point, so every sink shows up as exactly ONE cell of
value 1.0 in the 3-D pdens field.  Counting nonzero pdens cells in 3-D therefore gives
the exact number and (cell-centre) position of every sink -- this matches n_sink from
the .user.hst at every dump, and unlike the hst it is not limited to the first 2 tags,
so ALL sinks are marked (the BB Fig-8 run has up to 4).
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
import cmasher as cmr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bin_convert sits alongside
import bin_convert

CMAP = cmr.chroma

# --------------------------------------------------------------------------------------
def zslice_blocks(fd, var, z0=0.0):
    """Return list of (x1min,x1max,x2min,x2max, 2D array, level) for blocks straddling z0.
    NOTE: bin_convert docstring is WRONG; mb_geometry columns are block MIN/MAX bounds."""
    out = []
    data = fd["mb_data"][var]
    geo = fd["mb_geometry"]
    logi = fd["mb_logical"]
    nx3 = fd["nx3_out_mb"]
    for m in range(fd["n_mbs"]):
        x1min, x1max, x2min, x2max, x3min, x3max = geo[m]
        if not (x3min <= z0 < x3max):
            continue
        dx3 = (x3max - x3min)/nx3
        k = int((z0 - x3min)/dx3)
        k = max(0, min(nx3-1, k))
        sl = data[m][k, :, :]
        out.append((x1min, x1max, x2min, x2max, sl, int(logi[m][3])))
    return out

# distinct meshblock-edge colours per AMR level (chosen to contrast with cmr.chroma)
LEVEL_COLORS = ["#ffffff", "#16f0ff", "#ffe000", "#ff35d0", "#ff7a00", "#8cff00"]

def draw_slice(ax, blocks, vmin, vmax, extent):
    blocks = sorted(blocks, key=lambda b: b[5])   # coarse first, fine on top
    for (x1min, x1max, x2min, x2max, sl, lev) in blocks:
        ax.imshow(sl, origin="lower", extent=[x1min, x1max, x2min, x2max],
                  norm=LogNorm(vmin=vmin, vmax=vmax), cmap=CMAP,
                  interpolation="nearest", aspect="auto", zorder=lev)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

def draw_meshblocks(ax, blocks, lw=0.5):
    """Overlay meshblock boundaries, edge-coloured by AMR level (block size already
    encodes level, so this makes the refinement hierarchy explicit)."""
    from matplotlib.patches import Rectangle
    for (x1min, x1max, x2min, x2max, sl, lev) in blocks:
        c = LEVEL_COLORS[lev % len(LEVEL_COLORS)]
        ax.add_patch(Rectangle((x1min, x2min), x1max-x1min, x2max-x2min,
                               fill=False, edgecolor=c, linewidth=lw,
                               alpha=0.9, zorder=40))

# --------------------------------------------------------------------------------------
def sink_positions_3d(prtcl_fd, tol=1e-6, min_sep=0.008):
    """Exact sink positions from the deposited particle-density field.
    Every nonzero pdens cell (value ~1.0, NGP deposit) is one sink.  Returns a list of
    (x, y) projected to the z=0 plane, de-duplicated within min_sep in case a sink ever
    straddles two cells."""
    var = prtcl_fd["var_names"][0]
    data = prtcl_fd["mb_data"][var]
    geo = prtcl_fd["mb_geometry"]
    cand = []  # (value, x, y)
    for m in range(prtcl_fd["n_mbs"]):
        d = data[m]
        x1min, x1max, x2min, x2max, x3min, x3max = geo[m]
        nk, nj, ni = d.shape
        for (k, j, i) in zip(*np.where(d > tol)):
            x = x1min + (i+0.5)*(x1max-x1min)/ni
            y = x2min + (j+0.5)*(x2max-x2min)/nj
            cand.append((d[k, j, i], x, y))
    cand.sort(reverse=True)
    pts = []
    for (_, x, y) in cand:
        if all((x-px)**2 + (y-py)**2 > min_sep*min_sep for (px, py) in pts):
            pts.append((x, y))
    return pts

def load_hst(run_dir):
    import re
    f = glob.glob(os.path.join(run_dir, "*.user.hst"))
    if not f:
        return None
    names = None
    with open(f[0]) as fh:
        for line in fh:
            if "[1]=" in line:
                names = re.findall(r"\[\d+\]=(\S+)", line)
                break
    arr = np.loadtxt(f[0], comments="#")
    if names is None or arr.ndim != 2:
        return None
    return {nm: arr[:, i] for i, nm in enumerate(names) if i < arr.shape[1]}

def mark_sinks(ax, pts, ms=13):
    """Draw a highly visible sink marker: an open circle + centre dot in cyan, each with
    a black stroke halo so it stands out on both the dark ambient and the bright (green /
    white) filament of cmr.chroma."""
    halo = [pe.withStroke(linewidth=3.0, foreground="black")]
    for (x, y) in pts:
        ax.plot(x, y, marker="o", mfc="none", mec="#16f0ff", ms=ms, mew=2.0,
                zorder=50, path_effects=halo)
        ax.plot(x, y, marker=".", color="#16f0ff", ms=ms*0.28,
                zorder=51, path_effects=halo)

# --------------------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    show_mb = "mb" in args                 # pass "mb" anywhere to overlay meshblocks
    args = [a for a in args if a != "mb"]
    run_dir = args[0]
    tff = float(args[1]) if len(args) > 1 else 0.351241
    out_base = args[2] if len(args) > 2 else "bb"
    out_dir = args[3] if len(args) > 3 else run_dir
    if show_mb:
        out_base += "_mb"

    bindir = os.path.join(run_dir, "bin")
    dens_files = sorted(glob.glob(os.path.join(bindir, "*.hydro_w.*.bin")))
    prtcl_files = sorted(glob.glob(os.path.join(bindir, "*.prtcl_d.*.bin")))
    print(f"{len(dens_files)} density dumps, {len(prtcl_files)} prtcl dumps")
    prtcl_by_idx = {os.path.basename(p).split('.')[-2]: p for p in prtcl_files}
    hst = load_hst(run_dir)

    fd_last = bin_convert.read_binary(dens_files[-1])
    dvar = "dens" if "dens" in fd_last["var_names"] else fd_last["var_names"][0]
    all_last = np.concatenate([b[4].ravel() for b in zslice_blocks(fd_last, dvar)])
    vmax = np.nanpercentile(all_last, 100)
    vmin = max(np.nanmin(all_last), vmax/1e6)
    print(f"dens var={dvar} vmin={vmin:.3e} vmax={vmax:.3e}")

    ZOOM = 0.6

    _cache = {}
    def sinks_for(key):
        """Exact sink (x,y) list for the dump with catalog index `key`, using prtcl_d;
        falls back to the (<=2 tag) hst positions if no prtcl dump matches."""
        if key in _cache:
            return _cache[key]
        pts = []
        if key in prtcl_by_idx:
            pts = sink_positions_3d(bin_convert.read_binary(prtcl_by_idx[key]))
        elif hst is not None:
            # fallback only
            pass
        _cache[key] = pts
        return pts

    def key_of(path):
        return os.path.basename(path).split('.')[-2]

    # ---- mosaic: 4 evenly spaced late-time frames + 1 zoomed final panel ----
    n = len(dens_files)
    idxs = [int(round(x)) for x in np.linspace(n-1-3*max(1, (n-1)//8), n-1, 4)]
    idxs = sorted(set(max(0, i) for i in idxs))
    npanel = len(idxs) + 1
    fig, axes = plt.subplots(1, npanel, figsize=(4*npanel, 4.5), constrained_layout=True)
    levels_seen = set()
    for ax, i in zip(axes[:-1], idxs):
        fd = bin_convert.read_binary(dens_files[i])
        blocks = zslice_blocks(fd, dvar)
        draw_slice(ax, blocks, vmin, vmax, [-ZOOM, ZOOM, -ZOOM, ZOOM])
        if show_mb:
            draw_meshblocks(ax, blocks)
            levels_seen.update(b[5] for b in blocks)
        pts = sinks_for(key_of(dens_files[i]))
        mark_sinks(ax, pts)
        ax.set_title(f"t = {fd['time']/tff:.2f} $t_{{ff}}$   "
                     f"($N_{{\\rm sink}}$={len(pts)})", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    # zoomed final panel framed on the central (smallest-radius) pair of sinks
    fd = bin_convert.read_binary(dens_files[idxs[-1]])
    fpts = sinks_for(key_of(dens_files[idxs[-1]]))
    inner = sorted(fpts, key=lambda p: p[0]**2 + p[1]**2)[:2]
    if len(inner) >= 2:
        cx = 0.5*(inner[0][0] + inner[1][0]); cy = 0.5*(inner[0][1] + inner[1][1])
        sep = np.hypot(inner[0][0]-inner[1][0], inner[0][1]-inner[1][1])
        # frame the pair, but never wider than half the main view or the "zoom" panel
        # would duplicate it at ~1x (happens once the pair has separated substantially)
        Z2 = min(0.5*ZOOM, max(0.12, 0.9*sep))
    elif len(inner) == 1:
        cx, cy, Z2 = inner[0][0], inner[0][1], 0.15
    else:
        cx = cy = 0.0; Z2 = 0.15
    zoom_fac = ZOOM/Z2
    zblocks = zslice_blocks(fd, dvar)
    draw_slice(axes[-1], zblocks, vmin, vmax, [cx-Z2, cx+Z2, cy-Z2, cy+Z2])
    if show_mb:
        draw_meshblocks(axes[-1], zblocks, lw=0.8)
        levels_seen.update(b[5] for b in zblocks)
    mark_sinks(axes[-1], fpts, ms=20)
    axes[-1].set_title(f"t = {fd['time']/tff:.2f} $t_{{ff}}$  (zoom ×{zoom_fac:.0f})",
                       fontsize=12)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    for sp in axes[-1].spines.values():
        sp.set_color("white"); sp.set_linewidth(2)
    sm = plt.cm.ScalarMappable(norm=LogNorm(vmin=vmin, vmax=vmax), cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(r"$\rho$  (code units)", fontsize=11)
    if show_mb:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=LEVEL_COLORS[l % len(LEVEL_COLORS)], lw=2,
                          label=f"level {l}") for l in sorted(levels_seen)]
        axes[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.6,
                       facecolor="black", labelcolor="white", handlelength=1.4)
        fig.suptitle("Boss-Bodenheimer collapse — density slice (z=0);  "
                     "sinks = cyan circles,  meshblocks = coloured by AMR level", fontsize=13)
    else:
        fig.suptitle("Boss-Bodenheimer collapse — density slice (z=0);  sinks = cyan circles",
                     fontsize=13)
    out = os.path.join(out_dir, f"{out_base}_mosaic.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # ---- video ----
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa
        vout = os.path.join(out_dir, f"{out_base}.mp4")
        writer = imageio.get_writer(vout, fps=12, codec="libx264",
                                    quality=8, macro_block_size=1)
        for df in dens_files:
            fd = bin_convert.read_binary(df)
            figv, axv = plt.subplots(figsize=(5.2, 5))
            blocks = zslice_blocks(fd, dvar)
            draw_slice(axv, blocks, vmin, vmax, [-ZOOM, ZOOM, -ZOOM, ZOOM])
            if show_mb:
                draw_meshblocks(axv, blocks)
            pts = sinks_for(key_of(df))
            mark_sinks(axv, pts, ms=12)
            axv.set_title(f"BB collapse   t = {fd['time']/tff:.3f} $t_{{ff}}$   "
                          f"$N_{{\\rm sink}}$={len(pts)}")
            axv.set_xticks([]); axv.set_yticks([])
            figv.canvas.draw()
            w, h = figv.canvas.get_width_height()
            buf = np.frombuffer(figv.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            writer.append_data(buf[..., :3].copy())
            plt.close(figv)
        writer.close()
        print("wrote", vout)
    except Exception as e:
        print("video skipped:", e)

if __name__ == "__main__":
    main()
