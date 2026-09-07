#!/usr/bin/env python3
"""Extract exact final SRMHD-blast movie frames and build a report mosaic."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import apply_publication_style

apply_publication_style()

VAL = Path(__file__).resolve().parents[1]
MEDIA = VAL / "figures" / "apollo_grmhd"

PANELS = [
    ("plm", "PLM"),
    ("wenoz", "WENO-Z"),
    ("ppm_fb", "PPM + MOOD"),
    ("ppm_fb_fields", "PPM + MOOD fields"),
]


def final_frame(movie: Path):
    reader = imageio.get_reader(movie)
    try:
        count = reader.count_frames()
        return reader.get_data(count - 1), count
    finally:
        reader.close()


def main() -> None:
    frames = []
    for stem, title in PANELS:
        movie = MEDIA / f"{stem}.mp4"
        frame, count = final_frame(movie)
        out = MEDIA / f"{stem}_final.png"
        imageio.imwrite(out, frame)
        frames.append((title, frame))
        print(f"Wrote {out} (frame {count - 1}/{count - 1})")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), layout="constrained")
    for ax, (title, frame) in zip(axes.flat, frames):
        # Remove the movie's outer scheme title; retain physical panel labels,
        # axes, and colorbars embedded by the original renderer.
        crop = frame[42:, :, :]
        ax.imshow(crop)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("SRMHD blast: final frames")
    out = MEDIA / "srmhd_blast_final_mosaic.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
