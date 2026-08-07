"""IEEE-ish matplotlib/seaborn style for CuKD manuscript figures.

Usage:
    from ieee_style import apply_ieee_style, save_fig, FIG_SINGLE, FIG_DOUBLE
    apply_ieee_style()
    ...
    save_fig(fig, "fig_name")  # writes PDF + PNG under manuscript/figures/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Column widths (inches) for IEEEtran-ish layout
FIG_SINGLE = 3.45
FIG_DOUBLE = 7.05

# Colorblind-safe pair (still readable in grayscale)
C_BLUE = "#0071BC"
C_ORANGE = "#D95218"
C_GRAY = "#333333"
C_LIGHT = "#888888"

ROOT = Path(__file__).resolve().parents[1]  # manuscript/figures


def apply_ieee_style() -> None:
    """Apply a clean, print-friendly style (no chartjunk)."""
    try:
        import seaborn as sns

        sns.set_theme(
            style="ticks",
            context="paper",
            font="serif",
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
            },
        )
    except ImportError:
        pass

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.9,
            "lines.markersize": 5,
            "grid.linewidth": 0.4,
            "grid.linestyle": ":",
            "grid.color": "0.75",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_fig(fig: plt.Figure, stem: str, out_dir: Path | None = None) -> Path:
    """Save PDF (vector) and PNG preview. Returns PDF path."""
    out_dir = out_dir or ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    return pdf
