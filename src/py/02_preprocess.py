# 02_preprocess.py
"""
Preprocess multi-omics ingestion outputs by MAD-based feature selection and z-scoring.

Pipeline stage 2 (preprocessing). Reads rna, proteomics, phospho, and metadata from
results/ingestion/. Writes z-scored, feature-filtered matrices to results/preprocessing/
and diagnostic figures to figures/preprocessing/.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_csv, save_csv
from utils.plotting import savefig, set_style

VIEWS = ("rna", "proteomics", "phospho")


def parse_arguments():
    """Parse CLI arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess multi-omics data for the MOSAIC pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config/config.yml.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to stdout and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mosaic.preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_input_paths(config: dict) -> dict[str, Path]:
    """Build absolute paths to ingestion inputs."""
    ingestion_dir = Path(config["dataset"]["ingestion_dir"])
    return {
        "rna": ingestion_dir / "rna.csv",
        "proteomics": ingestion_dir / "proteomics.csv",
        "phospho": ingestion_dir / "phospho.csv",
        "metadata": ingestion_dir / "metadata.csv",
    }


def load_and_validate(input_paths: dict[str, Path], logger: logging.Logger) -> tuple:
    """Load omics matrices and metadata; validate sample alignment."""
    data = {}
    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(str(path.resolve()))
        data[name] = load_csv(str(path))

    for view in VIEWS:
        if not data[view].index.equals(data["metadata"].index):
            raise ValueError(
                f"Sample index mismatch between {view} and metadata."
            )

    for i, view_a in enumerate(VIEWS):
        for view_b in VIEWS[i + 1 :]:
            if not data[view_a].index.equals(data[view_b].index):
                raise ValueError(
                    f"Sample index mismatch between {view_a} and {view_b}."
                )

    n_samples = len(data["metadata"])
    tissue_counts = data["metadata"]["tissue_type"].value_counts().to_dict()
    logger.info("Loaded %d samples", n_samples)
    logger.info("Tissue type breakdown: %s", tissue_counts)
    return data["rna"], data["proteomics"], data["phospho"], data["metadata"]


def select_top_mad_features(
    df: pd.DataFrame, n_features: int
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Select top-N features by median absolute deviation.

    Args:
        df: Samples × features DataFrame.
        n_features: Number of features to retain.

    Returns:
        Filtered DataFrame, MAD values for all features, MAD cutoff of last selected feature.
    """
    mad_values = median_abs_deviation(df.values, axis=0, scale=1.0)
    feature_names = df.columns.to_numpy()
    order = np.argsort(mad_values)[::-1]
    selected_idx = order[:n_features]
    selected_names = feature_names[selected_idx]
    cutoff = mad_values[selected_idx[-1]]
    return df.loc[:, selected_names], mad_values, cutoff


def zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each feature across all samples using a single StandardScaler."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def plot_mad_distributions(
    mad_by_view: dict[str, np.ndarray],
    cutoff_by_view: dict[str, float],
    n_retained_by_view: dict[str, int],
    output_path: Path,
) -> None:
    """Plot MAD histograms with selection cutoffs for all three views."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, view in zip(axes, VIEWS):
        mad_values = mad_by_view[view]
        cutoff = cutoff_by_view[view]
        n_retained = n_retained_by_view[view]

        ax.hist(mad_values, bins=100, color="steelblue", alpha=0.7)
        ax.axvline(cutoff, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("MAD")
        ax.set_ylabel("Feature count")
        ax.set_title(view)
        ax.text(
            0.98,
            0.98,
            f"Retained: {n_retained}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    fig.tight_layout()
    savefig(fig, str(output_path))


def plot_sample_correlation(
    rna_preprocessed: pd.DataFrame,
    metadata: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot Pearson sample correlation heatmap sorted by tissue type."""
    set_style()
    tissue_order = metadata.sort_values("tissue_type").index
    sorted_tissue = metadata.loc[tissue_order, "tissue_type"]
    sorted_df = rna_preprocessed.loc[tissue_order]
    corr = np.corrcoef(sorted_df.values)

    n_normal = (sorted_tissue == "normal").sum()
    n_tumor = len(sorted_df) - n_normal
    n_total = len(sorted_df)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="RdBu_r",
        xticklabels=False,
        yticklabels=False,
        square=True,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.axhline(n_normal, color="black", linewidth=1.5)
    ax.axvline(n_normal, color="black", linewidth=1.5)

    ax.set_title(
        "Pearson correlation coefficient between samples (transcriptomics only)",
        fontsize=9,
    )

    label_kwargs = {"ha": "center", "va": "center", "fontsize": 7, "clip_on": False}
    ax.text(
        n_normal / 2,
        n_total + 4,
        f"Normal\n(n={n_normal})",
        **label_kwargs,
    )
    ax.text(
        n_normal + n_tumor / 2,
        n_total + 4,
        f"Tumor\n(n={n_tumor})",
        **label_kwargs,
    )
    ax.text(
        -4,
        n_normal / 2,
        f"Normal\n(n={n_normal})",
        rotation=90,
        ha="right",
        va="center",
        fontsize=7,
        clip_on=False,
    )
    ax.text(
        -4,
        n_normal + n_tumor / 2,
        f"Tumor\n(n={n_tumor})",
        rotation=90,
        ha="right",
        va="center",
        fontsize=7,
        clip_on=False,
    )

    fig.subplots_adjust(left=0.12, bottom=0.10, top=0.92)
    savefig(fig, str(output_path))


def main():
    """Run preprocessing: MAD selection, z-scoring, validation, and figure generation."""
    args = parse_arguments()
    config = load_config(args.config)
    logger = setup_logging(Path("log/02_preprocess.log"))

    input_paths = resolve_input_paths(config)
    rna, proteomics, phospho, metadata = load_and_validate(input_paths, logger)

    view_dfs = {"rna": rna, "proteomics": proteomics, "phospho": phospho}
    top_n = config["preprocessing"]["top_mad_features"]

    preprocessed = {}
    feature_counts = {}
    mad_by_view = {}
    cutoff_by_view = {}
    n_retained_by_view = {}

    for view in VIEWS:
        df = view_dfs[view]
        n_original = df.shape[1]
        selected_df, mad_values, cutoff = select_top_mad_features(df, top_n[view])
        mad_by_view[view] = mad_values
        cutoff_by_view[view] = cutoff
        n_retained_by_view[view] = top_n[view]

        if config["preprocessing"]["scale"]:
            selected_df = zscore_features(selected_df)

        preprocessed[view] = selected_df
        feature_counts[view] = {"before": n_original, "after": top_n[view]}
        logger.info(
            "View %s: %d → %d features",
            view,
            n_original,
            top_n[view],
        )

    for view in VIEWS:
        if preprocessed[view].isnull().any().any():
            raise ValueError(f"NaN values found in preprocessed {view} matrix.")

    results_dir = Path(config["paths"]["results"]) / "preprocessing"
    figures_dir = Path(config["paths"]["figures"]) / "preprocessing"

    save_csv(preprocessed["rna"], str(results_dir / "rna_preprocessed.csv"))
    save_csv(preprocessed["proteomics"], str(results_dir / "proteomics_preprocessed.csv"))
    save_csv(preprocessed["phospho"], str(results_dir / "phospho_preprocessed.csv"))

    counts_path = results_dir / "feature_counts.json"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(counts_path, "w", encoding="utf-8") as handle:
        json.dump(feature_counts, handle, indent=2)
    logger.info("Saved %s", counts_path)

    plot_mad_distributions(
        mad_by_view,
        cutoff_by_view,
        n_retained_by_view,
        figures_dir / "mad_distributions.pdf",
    )
    plot_sample_correlation(
        preprocessed["rna"],
        metadata,
        figures_dir / "sample_correlation_matrix.pdf",
    )

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
