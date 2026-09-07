"""Shared publication-quality Matplotlib style for validation figures."""

from __future__ import annotations

import matplotlib as mpl


def apply_publication_style() -> None:
    """Use a compact LaTeX-like serif style without requiring a TeX runtime."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.65,
            "axes.titlepad": 5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
