#!/usr/bin/env python3
"""Three-panel current-sheet movie: density, |B|, and MOOD cascade level.

Cascade levels retained after the last RK stage:
  0 = base reconstruction (PPM)
  1 = PLM
  2 = DC (first-order)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from plot_2d_stress import _field_array, read_vtk_structured
from plot_style import apply_publication_style

apply_publication_style()

VAL = Path(__file__).resolve().parents[1]
FIGURES = VAL / "figures"
FB_LABELS = {0: "PPM", 1: "PLM", 2: "DC"}
FB_CMAP = ListedColormap(["#4c78a8", "#f2c45a", "#c44e52"])
FB_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], FB_CMAP.N)


def _index(path: Path) -> int | None:
    m = re.search(r"\.(\d{5})\.vtk$", path.name)
    return int(m.group(1)) if m else None


def _pair_frames(rundir: Path):
    w_by = {}
    fb_by = {}
    for p in rundir.rglob("*.vtk"):
        idx = _index(p)
        if idx is None:
            continue
        if "fb_level" in p.name:
            fb_by[idx] = p
        elif "mhd_w_bcc" in p.name or "mhd_w." in p.name:
            w_by[idx] = p
    for idx in sorted(set(w_by) & set(fb_by)):
        yield idx, w_by[idx], fb_by[idx]


def _load_panel(w_vtk: Path, fb_vtk: Path):
    wfields, _ = read_vtk_structured(w_vtk)
    dens = _field_array(wfields, "dens")
    bmag = _field_array(wfields, "bmag")
    fbfields, _ = read_vtk_structured(fb_vtk)
    level = fbfields.get("fb_level")
    if dens is None or bmag is None or level is None:
        return None
    if not (np.isfinite(dens).all() and np.isfinite(bmag).all() and np.isfinite(level).all()):
        return None
    return dens, bmag, np.rint(level).astype(int)


def write_frames(rundir: Path, framedir: Path, dt: float):
    framedir.mkdir(parents=True, exist_ok=True)
    samples = []
    for idx, w_vtk, fb_vtk in _pair_frames(rundir):
        loaded = _load_panel(w_vtk, fb_vtk)
        if loaded is None:
            print(f"skip frame {idx}: missing/NaN")
            continue
        samples.append((idx, *loaded))
    if not samples:
        raise SystemExit(f"no finite (density, |B|, fb_level) triples in {rundir}")

    dens_vals = np.concatenate([d[np.isfinite(d)] for _, d, _, _ in samples])
    b_vals = np.concatenate([b[np.isfinite(b)] for _, _, b, _ in samples])
    dmin, dmax = float(dens_vals.min()), float(dens_vals.max())
    bmin, bmax = float(b_vals.min()), float(b_vals.max())
    if abs(dmax - dmin) < 1e-30:
        dmax = dmin + 1.0
    if abs(bmax - bmin) < 1e-30:
        bmax = bmin + 1.0

    paths = []
    for i, (idx, dens, bmag, level) in enumerate(samples):
        fig, axes = plt.subplots(
            1, 3, figsize=(7.4, 2.35), layout="constrained", squeeze=False
        )
        im0 = axes[0, 0].imshow(
            dens, origin="lower", cmap="cividis", vmin=dmin, vmax=dmax,
            interpolation="nearest", aspect="equal", rasterized=True,
        )
        im1 = axes[0, 1].imshow(
            bmag, origin="lower", cmap="cividis", vmin=bmin, vmax=bmax,
            interpolation="nearest", aspect="equal", rasterized=True,
        )
        im2 = axes[0, 2].imshow(
            level, origin="lower", cmap=FB_CMAP, norm=FB_NORM,
            interpolation="nearest", aspect="equal", rasterized=True,
        )
        axes[0, 0].set_title(r"$\rho$")
        axes[0, 1].set_title(r"$|\mathbf{B}|$")
        axes[0, 2].set_title("MOOD cascade")
        for ax in axes[0]:
            ax.set_xticks([])
            ax.set_yticks([])
        cb0 = fig.colorbar(im0, ax=axes[0, 0], location="right", fraction=0.046, pad=0.04)
        cb1 = fig.colorbar(im1, ax=axes[0, 1], location="right", fraction=0.046, pad=0.04)
        cb2 = fig.colorbar(im2, ax=axes[0, 2], location="right", fraction=0.046, pad=0.04,
                           ticks=[0, 1, 2])
        cb2.ax.set_yticklabels([r"PPM", r"PLM", r"DC"])
        for cb in (cb0, cb1, cb2):
            cb.outline.set_linewidth(0.6)
        t = idx * dt
        fig.suptitle(rf"Current sheet, PPM + MOOD + RK3 ($t={t:.2f}$)", y=1.04)
        out = framedir / f"frame_{i:05d}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        paths.append(out)
        counts = ", ".join(
            f"{FB_LABELS[k]}={int(np.sum(level == k))}" for k in (0, 1, 2)
        )
        print(f"frame {i:03d} t={t:.2f}  {counts}")
    return paths


def _ffmpeg_bin() -> str:
    from shutil import which
    found = which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise SystemExit("ffmpeg not found (also no imageio_ffmpeg)") from exc


def encode_mp4(framedir: Path, mp4: Path, fps: int):
    mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_bin(), "-y", "-framerate", str(fps),
        "-i", str(framedir / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p", "-crf", "18", str(mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {mp4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True, help="Athena run directory with VTK dumps")
    ap.add_argument("--dt", type=float, default=0.05, help="Output cadence used in the run")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument(
        "--out",
        default=str(FIGURES / "apollo_current_sheet" / "current_sheet_ppm_fb_cascade.mp4"),
    )
    args = ap.parse_args()
    rundir = Path(args.rundir).resolve()
    framedir = Path(args.out).resolve().with_suffix("") / "frames"
    write_frames(rundir, framedir, args.dt)
    encode_mp4(framedir, Path(args.out).resolve(), args.fps)


if __name__ == "__main__":
    main()
