# 06.1_clustering_plot.py
"""
Final clustering visualisation for the MOSAIC pipeline.

Pipeline stage 06.1. Projects cluster assignments from Stage 06 onto the PCA
and UMAP embeddings produced by Stage 05. Contains no clustering logic — only
loads cluster_assignments.csv and embedding CSVs to produce two publication-
quality scatter plots.

Inputs:
  results/clustering/cluster_assignments.csv
  results/clustering/cluster_sizes.csv
  results/delta/delta_magnitude.csv
  results/delta/repr_comparison/embeddings/<viz_repr>_pca.csv
  results/delta/repr_comparison/embeddings/<viz_repr>_umap.csv
  config/config.yml

Outputs:
  figures/clustering/cluster_umap.pdf
  figures/clustering/cluster_pca.pdf
"""

# Stage 06.1 — Final Clustering Visualisation
#
# This script projects the cluster assignments produced by Stage 06 onto the
# PCA and UMAP embedding spaces produced by Stage 05.
#
# It contains no clustering logic. It reads cluster_assignments.csv and the
# appropriate embedding CSVs and produces publication-quality figures.
#
# Prerequisites: Stage 06 must have completed with final parameters set in
# config.yml under clustering.manual (or clustering.automatic if applicable).
#
# To run manually:
#   python src/py/06.1_clustering_plot.py --config config/config.yml

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, load_csv
from utils.plotting import get_cluster_palette, savefig, set_style

VALID_REPRESENTATIONS = {"raw_all", "raw_nof1", "norm_all", "norm_nof1"}

REPR_TO_FILE = {
    "raw_all":   "results/delta/delta_z.csv",
    "raw_nof1":  "results/delta/delta_z_no_f1.csv",
    "norm_all":  "results/delta/delta_z_normalized.csv",
    "norm_nof1": "results/delta/delta_z_normalized_no_f1.csv",
}

REPR_TO_EMBEDDINGS = {
    "raw_all": {
        "pca":  "results/delta/repr_comparison/embeddings/raw_all_pca.csv",
        "umap": "results/delta/repr_comparison/embeddings/raw_all_umap.csv",
    },
    "raw_nof1": {
        "pca":  "results/delta/repr_comparison/embeddings/raw_nof1_pca.csv",
        "umap": "results/delta/repr_comparison/embeddings/raw_nof1_umap.csv",
    },
    "norm_all": {
        "pca":  "results/delta/repr_comparison/embeddings/norm_all_pca.csv",
        "umap": "results/delta/repr_comparison/embeddings/norm_all_umap.csv",
    },
    "norm_nof1": {
        "pca":  "results/delta/repr_comparison/embeddings/norm_nof1_pca.csv",
        "umap": "results/delta/repr_comparison/embeddings/norm_nof1_umap.csv",
    },
}

REPR_TITLES = {
    "raw_all": {
        "pca":  "Raw ΔZ | PCA | All factors",
        "umap": "Raw ΔZ | UMAP | All factors",
    },
    "raw_nof1": {
        "pca":  "Raw ΔZ | PCA | Factor1 excluded",
        "umap": "Raw ΔZ | UMAP | Factor1 excluded",
    },
    "norm_all": {
        "pca":  "L2-normalised ΔZ | PCA | All factors",
        "umap": "L2-normalised ΔZ | UMAP | All factors",
    },
    "norm_nof1": {
        "pca":  "L2-normalised ΔZ | PCA | Factor1 excluded",
        "umap": "L2-normalised ΔZ | UMAP | Factor1 excluded",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 06.1 — Final clustering visualisation"
    )
    parser.add_argument("--config", required=True, help="Path to config/config.yml")
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to stdout and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mosaic.clustering_plot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def validate_file(path: Path) -> None:
    """Raise FileNotFoundError if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))


def pca_labels(evr: np.ndarray) -> tuple[str, str]:
    """Return axis labels with explained variance percentages."""
    return (
        f"PC1 ({evr[0] * 100:.1f}%)",
        f"PC2 ({evr[1] * 100:.1f}%)",
    )


def plot_cluster_embedding(
    ax,
    embedding_df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    cluster_palette: dict,
    cluster_sizes: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    """Scatter plot of one embedding coloured by cluster assignment.

    Visual style matches Stage 05 plot_embedding (figsize, marker size, legend).
    """
    valid = embedding_df.dropna(subset=[x_col, y_col])
    for cluster_id in sorted(cluster_palette.keys()):
        colour = cluster_palette[cluster_id]
        n = int(cluster_sizes.loc[cluster_id, "n_patients"])
        patients = cluster_assignments.index[cluster_assignments["cluster"] == cluster_id]
        valid_ids = [p for p in patients if p in valid.index]
        if not valid_ids:
            continue
        coords = valid.loc[valid_ids]
        ax.scatter(
            coords[x_col],
            coords[y_col],
            c=[colour],
            s=25,
            alpha=0.85,
            label=f"Cluster {cluster_id} (n={n})",
            edgecolors="none",
        )
    ax.legend(title="Cluster", loc="best", fontsize=7)


def main() -> None:
    """Run the Stage 06.1 final clustering visualisation pipeline."""
    args = parse_args()
    config = load_config(args.config)
    set_style()

    logger = setup_logging(Path("log") / "06.1_clustering_plot.log")

    # Step A — Load cluster assignments
    assignments_path = Path("results/clustering/cluster_assignments.csv")
    validate_file(assignments_path)
    cluster_assignments = load_csv(str(assignments_path))

    if list(cluster_assignments.columns) != ["cluster"]:
        raise ValueError(
            f"cluster_assignments.csv must have exactly one column named 'cluster', "
            f"got: {cluster_assignments.columns.tolist()}"
        )

    n_clusters = int(cluster_assignments["cluster"].nunique())
    logger.info(
        "Loaded cluster assignments: %d patients, k=%d clusters.",
        len(cluster_assignments), n_clusters,
    )

    cl_cfg = config["clustering"]
    viz_repr = cl_cfg["visualization_representation"]
    if viz_repr not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"visualization_representation '{viz_repr}' is not valid. "
            f"Valid options: {sorted(VALID_REPRESENTATIONS)}"
        )

    logger.info("Visualization representation: %s", viz_repr)

    # Load embeddings
    pca_path = Path(REPR_TO_EMBEDDINGS[viz_repr]["pca"])
    umap_path = Path(REPR_TO_EMBEDDINGS[viz_repr]["umap"])
    validate_file(pca_path)
    validate_file(umap_path)

    pca_emb = load_csv(str(pca_path))
    umap_emb = load_csv(str(umap_path))

    for emb_name, emb_df in [("PCA", pca_emb), ("UMAP", umap_emb)]:
        missing = [p for p in cluster_assignments.index if p not in emb_df.index]
        if missing:
            logger.warning(
                "%s embedding missing %d patients from cluster_assignments (will be skipped): %s",
                emb_name, len(missing), missing,
            )

    # Load supporting data
    sizes_path = Path("results/clustering/cluster_sizes.csv")
    mag_path = Path("results/delta/delta_magnitude.csv")
    validate_file(sizes_path)
    validate_file(mag_path)
    cluster_sizes = load_csv(str(sizes_path))

    # Step B — Build cluster palette (identical to Stage 06 via same config key)
    k_final = n_clusters
    cluster_palette = get_cluster_palette(k_final, config["figures"]["cluster_palette"])

    # Recover PCA explained variance ratios by refitting on the visualisation matrix.
    # Stage 05 does not save EVR separately; this refit is purely for axis labels
    # and does NOT overwrite any Stage 05 outputs.
    viz_matrix_path = Path(REPR_TO_FILE[viz_repr])
    validate_file(viz_matrix_path)
    viz_matrix = load_csv(str(viz_matrix_path)).dropna()
    pca_seed = int(config["dimensionality_reduction"]["pca"]["random_state"])
    pca_for_evr = PCA(n_components=2, random_state=pca_seed)
    pca_for_evr.fit(viz_matrix.values)
    evr = pca_for_evr.explained_variance_ratio_
    logger.info(
        "PCA EVR for %s: PC1=%.1f%%, PC2=%.1f%%",
        viz_repr, evr[0] * 100, evr[1] * 100,
    )

    dpi = int(config["figures"]["dpi"])
    fig_dir = Path("figures/clustering")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Step C — Figures

    umap_subtitle = f"Coloured by transformation cluster (k={k_final})"
    pca_subtitle = (
        f"Coloured by transformation cluster (k={k_final}) | "
        "clustering in full-dimensional ΔZ space"
    )

    # Figure 1: UMAP scatter — layout matches Stage 05 plot_embedding
    fig_umap, ax_umap = plt.subplots(figsize=(6, 5))
    plot_cluster_embedding(
        ax=ax_umap,
        embedding_df=umap_emb,
        cluster_assignments=cluster_assignments,
        cluster_palette=cluster_palette,
        cluster_sizes=cluster_sizes,
        x_col="umap1",
        y_col="umap2",
    )
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.set_title(REPR_TITLES[viz_repr]["umap"])
    fig_umap.text(0.5, 0.01, umap_subtitle, ha="center", fontsize=7, style="italic")
    savefig(fig_umap, str(fig_dir / "cluster_umap.pdf"), dpi=dpi)

    # Figure 2: PCA scatter — layout matches Stage 05 plot_embedding
    pca_xlabel, pca_ylabel = pca_labels(evr)
    fig_pca, ax_pca = plt.subplots(figsize=(6, 5))
    plot_cluster_embedding(
        ax=ax_pca,
        embedding_df=pca_emb,
        cluster_assignments=cluster_assignments,
        cluster_palette=cluster_palette,
        cluster_sizes=cluster_sizes,
        x_col="pc1",
        y_col="pc2",
    )
    ax_pca.set_xlabel(pca_xlabel)
    ax_pca.set_ylabel(pca_ylabel)
    ax_pca.set_title(REPR_TITLES[viz_repr]["pca"])
    fig_pca.text(0.5, 0.01, pca_subtitle, ha="center", fontsize=7, style="italic")
    savefig(fig_pca, str(fig_dir / "cluster_pca.pdf"), dpi=dpi)

    logger.info("Stage 06.1 complete. Figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()
