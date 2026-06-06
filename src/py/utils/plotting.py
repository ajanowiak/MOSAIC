# plotting.py
"""Shared plotting helpers for MOSAIC pipeline scripts."""

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def set_style() -> None:
    """Apply project-wide matplotlib style settings."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
        }
    )
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def get_cluster_palette(n_clusters: int, palette_name: str = "tab10") -> dict:
    """Return a colour mapping for cluster labels 0 through n_clusters - 1.

    Args:
        n_clusters: Number of clusters.
        palette_name: Matplotlib colour palette name.

    Returns:
        Dictionary mapping integer cluster labels to hex colours.
    """
    cmap = plt.get_cmap(palette_name)
    return {label: mcolors.to_hex(cmap(label)) for label in range(n_clusters)}


def savefig(fig, path: str, dpi: int = 300) -> None:
    """Save a figure to disk and close it.

    Args:
        fig: Matplotlib figure object.
        path: Output file path.
        dpi: Resolution in dots per inch.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", output_path)
