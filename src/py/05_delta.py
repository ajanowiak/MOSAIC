# 05_delta.py
"""
ΔZ representation analysis for the MOSAIC pipeline.

Pipeline stage 5. Reads per-patient ΔZ from Stage 04 and systematically compares
eight embedding variants (raw vs L2-normalised × all factors vs Factor1 excluded ×
PCA vs UMAP). Computes silhouette sweeps over four input matrix combinations and
produces diagnostic figures for representation selection before Stage 06 clustering.

Inputs: results/delta/delta_z.csv, results/ingestion/metadata.csv, config/config.yml
Outputs: normalised ΔZ matrices, embeddings, silhouette tables, and comparison figures
under results/delta/ and figures/delta/.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import kruskal, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_csv, save_csv
from utils.plotting import savefig, set_style

TUMOUR_TISSUE = {"tumour", "tumor"}
STAGE_ORDER = {"Stage I": 0, "Stage II": 1, "Stage III": 2}
PASTEL_VIOLIN_COLORS = [
    "#B8D4E3",
    "#F4C2C2",
    "#C9E4CA",
    "#E8D5B7",
    "#D4C4E9",
    "#FDEBD0",
]


def parse_arguments():
    """Parse CLI arguments for the delta representation script."""
    parser = argparse.ArgumentParser(
        description="ΔZ representation comparison for the MOSAIC pipeline."
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
    logger = logging.getLogger("mosaic.delta")
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


def validate_file(path: Path) -> None:
    """Raise FileNotFoundError if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))


def stage_column_name(metadata: pd.DataFrame) -> str:
    """Return the tumour-stage column name present in metadata."""
    if "tumour_stage" in metadata.columns:
        return "tumour_stage"
    if "tumor_stage" in metadata.columns:
        return "tumor_stage"
    raise KeyError("No tumour_stage or tumor_stage column found in metadata.")


def sort_stages(stages) -> list:
    """Sort stage labels in natural clinical order (I, II, III)."""
    return sorted(stages, key=lambda s: STAGE_ORDER.get(s, 99))


def build_patient_meta(
    metadata: pd.DataFrame, patient_ids: pd.Index, logger: logging.Logger
) -> pd.DataFrame:
    """Build patient-level metadata from tumour-side rows joined on patient_id."""
    tumour_mask = metadata["tissue_type"].astype(str).str.lower().isin(TUMOUR_TISSUE)
    tumour_meta = metadata.loc[tumour_mask].drop_duplicates(subset="patient_id")
    patient_meta = tumour_meta.set_index("patient_id").loc[patient_ids]
    logger.info("patient_meta columns: %s", patient_meta.columns.tolist())
    return patient_meta


def l2_normalize_matrix(
    matrix_df: pd.DataFrame, variant_name: str, logger: logging.Logger
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """L2-normalise rows; near-zero rows become NaN in the output."""
    magnitudes = np.linalg.norm(matrix_df.values, axis=1)
    near_zero_mask = magnitudes < 1e-10
    if near_zero_mask.any():
        patient_ids = matrix_df.index[near_zero_mask].tolist()
        logger.warning("Near-zero ΔZ magnitude patients in %s: %s", variant_name, patient_ids)
        logger.warning("Retained in raw pipeline. EXCLUDED from normalised pipeline.")
        logger.warning(
            "This may indicate molecularly quiescent tumours — report as a finding."
        )

    safe_norms = np.where(near_zero_mask, 1.0, magnitudes)
    normed = matrix_df.values / safe_norms[:, np.newaxis]
    normed[near_zero_mask] = np.nan
    result = pd.DataFrame(normed, index=matrix_df.index, columns=matrix_df.columns)

    valid = ~near_zero_mask
    if valid.any():
        row_norms = np.linalg.norm(normed[valid], axis=1)
        assert np.allclose(row_norms, 1.0, atol=1e-6)

    return result, near_zero_mask, magnitudes


def compute_summary_stats(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Compute per-factor summary statistics across all patients."""
    records = []
    for factor in raw_all.columns:
        values = raw_all[factor].values
        records.append(
            {
                "factor": factor,
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "median": np.median(values),
                "skewness": skew(values, nan_policy="omit"),
            }
        )
    return pd.DataFrame(records).set_index("factor")


def run_kruskal_magnitude_stage(
    delta_magnitude: pd.DataFrame,
    patient_meta: pd.DataFrame,
    stage_col: str,
    output_path: Path,
    logger: logging.Logger,
) -> tuple[float, float]:
    """Test whether ΔZ magnitude differs across tumour stages."""
    merged = delta_magnitude.join(patient_meta[[stage_col]], how="left")
    merged = merged.dropna(subset=[stage_col, "magnitude"])
    stages = sort_stages(merged[stage_col].unique())
    groups = [merged.loc[merged[stage_col] == stage, "magnitude"].values for stage in stages]

    if len(groups) < 2:
        logger.warning("Fewer than two stage groups with data; skipping Kruskal-Wallis.")
        h_stat, p_val = np.nan, np.nan
    else:
        h_stat, p_val = kruskal(*groups)
        logger.info(
            "Kruskal-Wallis magnitude vs tumour stage: H=%.4f, p=%.4f", h_stat, p_val
        )
        if p_val < 0.05:
            logger.info(
                "Magnitude significantly associates with stage — raw ΔZ carries "
                "clinical information. Consider retaining magnitude as a covariate in Stage 08."
            )
        else:
            logger.info(
                "No significant stage association. Normalisation unlikely to "
                "discard clinically relevant magnitude information."
            )

    rows = []
    for stage in stages:
        stage_vals = merged.loc[merged[stage_col] == stage, "magnitude"]
        rows.append(
            {
                "stage": stage,
                "n_patients": len(stage_vals),
                "median_magnitude": stage_vals.median(),
                "H_statistic": h_stat,
                "p_value": p_val,
            }
        )
    save_csv(pd.DataFrame(rows), str(output_path), index=False)
    return h_stat, p_val


def fit_pca(matrix_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit 2D PCA on valid rows and reindex to the full patient list."""
    valid = matrix_df.dropna()
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(valid.values)
    df = pd.DataFrame(coords, index=valid.index, columns=["pc1", "pc2"])
    return df.reindex(matrix_df.index), pca.explained_variance_ratio_


def fit_umap(
    matrix_df: pd.DataFrame, n_neighbors: int, min_dist: float, seed: int
) -> pd.DataFrame:
    """Fit 2D UMAP on valid rows and reindex to the full patient list."""
    valid = matrix_df.dropna()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    coords = reducer.fit_transform(valid.values)
    df = pd.DataFrame(coords, index=valid.index, columns=["umap1", "umap2"])
    return df.reindex(matrix_df.index)


def run_silhouette_sweep(
    matrix_df: pd.DataFrame,
    k_range: list,
    random_state: int,
    variant_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Sweep k and compute silhouette scores in the full-dimensional space."""
    valid_matrix = matrix_df.dropna().values
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        labels = km.fit_predict(valid_matrix)
        score = silhouette_score(valid_matrix, labels)
        records.append({"k": k, "silhouette": score})

    result = pd.DataFrame(records)
    best = result.loc[result["silhouette"].idxmax()]
    logger.info(
        "%s: best k=%s, silhouette=%.4f",
        variant_name,
        int(best["k"]),
        best["silhouette"],
    )
    return result


def pca_labels(evr: np.ndarray) -> tuple[str, str]:
    """Return axis labels with explained variance percentages."""
    return (
        f"PC1 ({evr[0] * 100:.1f}%)",
        f"PC2 ({evr[1] * 100:.1f}%)",
    )


def umap_labels() -> tuple[str, str]:
    """Return standard UMAP axis labels."""
    return ("UMAP 1", "UMAP 2")


def build_stage_palette(stages: list, palette_name: str) -> dict:
    """Map sorted stage labels to colours from a matplotlib palette."""
    cmap = plt.get_cmap(palette_name)
    return {stage: cmap(i) for i, stage in enumerate(stages)}


def plot_embedding(
    coords: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
    colour_mode: str,
    magnitudes: Optional[pd.Series],
    stages: Optional[pd.Series],
    stage_palette: Optional[dict],
    near_zero_mask: Optional[np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot a single embedding figure with consistent colour conventions."""
    fig, ax = plt.subplots(figsize=(6, 5))
    valid = coords.dropna(subset=[x_col, y_col])
    valid_ids = valid.index

    if colour_mode == "magnitude":
        sc = ax.scatter(
            valid[x_col],
            valid[y_col],
            c=magnitudes.loc[valid_ids],
            cmap="viridis",
            s=25,
            alpha=0.85,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Magnitude")
    else:
        for stage, colour in stage_palette.items():
            mask = stages.loc[valid_ids] == stage
            stage_ids = mask.index[mask]
            if len(stage_ids) == 0:
                continue
            ax.scatter(
                valid.loc[stage_ids, x_col],
                valid.loc[stage_ids, y_col],
                c=[colour],
                s=25,
                alpha=0.85,
                label=stage,
                edgecolors="none",
            )
        ax.legend(title="Stage", loc="best", fontsize=7)

    if near_zero_mask is not None and near_zero_mask.any():
        nz_ids = coords.index[near_zero_mask]
        nz_valid = valid.index.intersection(nz_ids)
        if len(nz_valid) > 0:
            ax.scatter(
                valid.loc[nz_valid, x_col],
                valid.loc[nz_valid, y_col],
                marker="x",
                c="red",
                s=60,
                zorder=5,
                linewidths=1.2,
                label="Near-zero ΔZ",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=7, style="italic")
    savefig(fig, str(output_path), dpi=dpi)


def plot_magnitude_histogram(
    magnitudes: pd.Series,
    near_zero_mask: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    """Histogram of ΔZ magnitude with near-zero and median markers."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = magnitudes.values
    sns.histplot(
        vals,
        bins=30,
        color="#6BAED6",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        ax=ax,
    )

    median_val = float(np.median(vals))
    ax.axvline(
        median_val,
        color="#C44E52",
        linestyle="--",
        linewidth=1.2,
        label=f"Median = {median_val:.2f}",
        zorder=4,
    )

    if near_zero_mask.any():
        nz_vals = magnitudes.loc[near_zero_mask].values
        ax.scatter(
            nz_vals,
            np.zeros_like(nz_vals),
            marker="|",
            s=120,
            color="#C44E52",
            linewidths=1.5,
            label=f"Near-zero (n={len(nz_vals)})",
            zorder=5,
        )

    ax.set_xlabel("L2 norm of ΔZ (all factors)")
    ax.set_ylabel("Patients")
    ax.set_title("Transformation magnitude distribution")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    savefig(fig, str(output_path), dpi=dpi)


def plot_magnitude_by_stage(
    magnitudes: pd.Series,
    stages: pd.Series,
    h_stat: float,
    p_val: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Box plot of magnitude stratified by tumour stage."""
    fig, ax = plt.subplots(figsize=(6, 4))
    stage_labels = sort_stages(stages.dropna().unique())
    data = [magnitudes.loc[stages == stage].dropna().values for stage in stage_labels]

    bp = ax.boxplot(data, tick_labels=stage_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
        patch.set_alpha(0.7)

    rng = np.random.default_rng(42)
    for i, stage in enumerate(stage_labels, start=1):
        vals = magnitudes.loc[stages == stage].dropna().values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.6, s=15, color="steelblue")

    ax.set_xlabel("Tumour stage")
    ax.set_ylabel("Magnitude (L2 norm)")
    ax.set_title(f"Magnitude vs tumour stage (H={h_stat:.2f}, p={p_val:.3f})")
    savefig(fig, str(output_path), dpi=dpi)


def plot_delta_distributions(raw_all: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Violin plots of ΔZ per factor with jittered patient points."""
    factors = raw_all.columns.tolist()
    n_factors = len(factors)
    n_cols = 4
    n_rows = math.ceil(n_factors / n_cols)
    n_patients = len(raw_all)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle(
        f"Per-factor ΔZ distributions across patients (n={n_patients})",
        fontsize=18,
        y=1.02,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for i, factor in enumerate(factors):
        ax = axes_flat[i]
        color = PASTEL_VIOLIN_COLORS[i % len(PASTEL_VIOLIN_COLORS)]
        plot_df = pd.DataFrame({"delta_z": raw_all[factor].values})
        sns.violinplot(
            data=plot_df,
            y="delta_z",
            color=color,
            cut=0,
            inner="quartile",
            linewidth=0.8,
            alpha=0.85,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            y="delta_z",
            color="#5a6470",
            alpha=0.35,
            size=3,
            jitter=0.25,
            ax=ax,
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(factor, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("ΔZ" if i % n_cols == 0 else "")
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    for j in range(n_factors, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, str(output_path), dpi=dpi)


def plot_silhouette_comparison(
    silhouette_dfs: dict[str, pd.DataFrame],
    output_path: Path,
    dpi: int,
) -> None:
    """Overlay silhouette curves for all four representation variants."""
    styles = {
        "raw_all": {"color": "blue", "linestyle": "-", "label": "Raw ΔZ | all factors"},
        "raw_nof1": {
            "color": "blue",
            "linestyle": "--",
            "label": "Raw ΔZ | no Factor1",
        },
        "norm_all": {
            "color": "orange",
            "linestyle": "-",
            "label": "Normalised ΔZ | all factors",
        },
        "norm_nof1": {
            "color": "orange",
            "linestyle": "--",
            "label": "Normalised ΔZ | no Factor1",
        },
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, df in silhouette_dfs.items():
        style = styles[name]
        ax.plot(df["k"], df["silhouette"], color=style["color"], linestyle=style["linestyle"])
        best = df.loc[df["silhouette"].idxmax()]
        ax.scatter(
            best["k"],
            best["silhouette"],
            color=style["color"],
            s=40,
            zorder=5,
        )
        ax.plot([], [], color=style["color"], linestyle=style["linestyle"], label=style["label"])

    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score vs k — all representation variants")
    ax.legend(loc="best", fontsize=7)
    savefig(fig, str(output_path), dpi=dpi)


def _scatter_panel(
    ax,
    coords: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    colour_mode: str,
    magnitudes: Optional[pd.Series],
    stages: Optional[pd.Series],
    stage_palette: Optional[dict],
    near_zero_mask: Optional[np.ndarray],
    cax=None,
):
    """Draw one panel for the 2×2 summary figure."""
    valid = coords.dropna(subset=[x_col, y_col])
    valid_ids = valid.index

    if colour_mode == "magnitude":
        sc = ax.scatter(
            valid[x_col],
            valid[y_col],
            c=magnitudes.loc[valid_ids],
            cmap="viridis",
            s=20,
            alpha=0.85,
            edgecolors="none",
        )
        if cax is not None:
            plt.colorbar(sc, cax=cax, label="Magnitude")
    else:
        for stage, colour in stage_palette.items():
            mask = stages.loc[valid_ids] == stage
            stage_ids = mask.index[mask]
            if len(stage_ids) == 0:
                continue
            ax.scatter(
                valid.loc[stage_ids, x_col],
                valid.loc[stage_ids, y_col],
                c=[colour],
                s=20,
                alpha=0.85,
                label=stage,
                edgecolors="none",
            )

    if near_zero_mask is not None and near_zero_mask.any():
        nz_ids = coords.index[near_zero_mask]
        nz_valid = valid.index.intersection(nz_ids)
        if len(nz_valid) > 0:
            ax.scatter(
                valid.loc[nz_valid, x_col],
                valid.loc[nz_valid, y_col],
                marker="x",
                c="red",
                s=50,
                zorder=5,
                linewidths=1.0,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=8)


def plot_summary_panel(
    panels: list[dict],
    supertitle: str,
    stage_palette: dict,
    output_path: Path,
    dpi: int,
) -> None:
    """Render a 2×2 summary panel with shared colorbar and stage legend."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(supertitle, fontsize=10, y=1.01)

    cax = fig.add_axes([0.90, 0.55, 0.015, 0.35])
    legend_ax = fig.add_axes([0.90, 0.12, 0.08, 0.25])
    legend_ax.axis("off")

    stage_handles = []
    for stage, colour in stage_palette.items():
        stage_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=colour, markersize=6, label=stage
            )
        )
    legend_ax.legend(handles=stage_handles, title="Stage", loc="center", fontsize=7)

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, 2)
        use_cax = cax if idx == 0 else None
        _scatter_panel(
            axes[row, col],
            panel["coords"],
            panel["x_col"],
            panel["y_col"],
            panel["x_label"],
            panel["y_label"],
            panel["title"],
            panel["colour_mode"],
            panel.get("magnitudes"),
            panel.get("stages"),
            stage_palette,
            panel.get("near_zero_mask"),
            cax=use_cax,
        )

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    savefig(fig, str(output_path), dpi=dpi)


def main():
    """Run the Stage 05 ΔZ representation comparison pipeline."""
    args = parse_arguments()
    config = load_config(args.config)
    set_style()

    results_dir = Path(config["paths"]["results"]) / "delta"
    figures_dir = Path(config["paths"]["figures"]) / "delta"
    emb_dir = results_dir / "repr_comparison" / "embeddings"
    sil_dir = results_dir / "repr_comparison" / "silhouette"
    repr_fig_dir = figures_dir / "repr_comparison"
    dpi = config["figures"]["dpi"]
    palette_name = config["figures"]["cluster_palette"]

    logger = setup_logging(Path("log") / "05_delta.log")

    delta_z_path = results_dir / "delta_z.csv"
    metadata_path = Path(config["dataset"]["ingestion_dir"]) / "metadata.csv"
    validate_file(delta_z_path)
    validate_file(metadata_path)

    # Step A — load inputs
    delta_z = load_csv(str(delta_z_path))
    logger.info("Loaded delta_z: shape %s, factors: %s", delta_z.shape, delta_z.columns.tolist())

    metadata = load_csv(str(metadata_path))
    patient_meta = build_patient_meta(metadata, delta_z.index, logger)
    stage_col = stage_column_name(patient_meta)
    stages_series = patient_meta[stage_col]
    stage_palette = build_stage_palette(sort_stages(stages_series.dropna().unique()), palette_name)

    # Step B — build four input matrices
    if "Factor1" not in delta_z.columns:
        raise ValueError("Factor1 not found in delta_z column headers.")

    raw_all = delta_z.copy()
    raw_nof1 = delta_z.drop(columns=["Factor1"])

    norm_all, near_zero_all, magnitudes_all = l2_normalize_matrix(raw_all, "raw_all", logger)
    norm_nof1, near_zero_nof1, _ = l2_normalize_matrix(raw_nof1, "raw_nof1", logger)
    logger.info(
        "Near-zero counts: raw_all=%s, raw_nof1=%s",
        int(near_zero_all.sum()),
        int(near_zero_nof1.sum()),
    )

    delta_magnitude = pd.DataFrame({"magnitude": magnitudes_all}, index=delta_z.index)
    delta_summary_stats = compute_summary_stats(raw_all)

    save_csv(raw_nof1, str(results_dir / "delta_z_no_f1.csv"))
    save_csv(norm_all, str(results_dir / "delta_z_normalized.csv"))
    save_csv(norm_nof1, str(results_dir / "delta_z_normalized_no_f1.csv"))
    save_csv(delta_magnitude, str(results_dir / "delta_magnitude.csv"))
    save_csv(delta_summary_stats, str(results_dir / "delta_summary_stats.csv"))

    # Step C — Kruskal-Wallis magnitude vs stage
    h_stat, p_val = run_kruskal_magnitude_stage(
        delta_magnitude,
        patient_meta,
        stage_col,
        sil_dir / "kruskal_magnitude_stage.csv",
        logger,
    )

    # Step D — compute all 8 embeddings
    dr_cfg = config["dimensionality_reduction"]
    umap_neighbors = dr_cfg["umap"]["n_neighbors"]
    umap_min_dist = dr_cfg["umap"]["min_dist"]
    umap_seed = dr_cfg["umap"]["random_state"]
    pca_seed = dr_cfg["pca"]["random_state"]

    emb_dir.mkdir(parents=True, exist_ok=True)

    raw_all_pca, evr_raw_all = fit_pca(raw_all, pca_seed)
    raw_nof1_pca, evr_raw_nof1 = fit_pca(raw_nof1, pca_seed)
    norm_all_pca, evr_norm_all = fit_pca(norm_all, pca_seed)
    norm_nof1_pca, evr_norm_nof1 = fit_pca(norm_nof1, pca_seed)

    raw_all_umap = fit_umap(raw_all, umap_neighbors, umap_min_dist, umap_seed)
    raw_nof1_umap = fit_umap(raw_nof1, umap_neighbors, umap_min_dist, umap_seed)
    norm_all_umap = fit_umap(norm_all, umap_neighbors, umap_min_dist, umap_seed)
    norm_nof1_umap = fit_umap(norm_nof1, umap_neighbors, umap_min_dist, umap_seed)

    embeddings = {
        "raw_all_pca": raw_all_pca,
        "raw_all_umap": raw_all_umap,
        "raw_nof1_pca": raw_nof1_pca,
        "raw_nof1_umap": raw_nof1_umap,
        "norm_all_pca": norm_all_pca,
        "norm_all_umap": norm_all_umap,
        "norm_nof1_pca": norm_nof1_pca,
        "norm_nof1_umap": norm_nof1_umap,
    }
    for name, df in embeddings.items():
        save_csv(df, str(emb_dir / f"{name}.csv"))

    # Step E — silhouette sweeps
    k_range = config["clustering"]["k_range"]
    cluster_seed = config["clustering"]["random_state"]
    matrices = {
        "raw_all": raw_all,
        "raw_nof1": raw_nof1,
        "norm_all": norm_all,
        "norm_nof1": norm_nof1,
    }
    silhouette_dfs = {}
    sil_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        silhouette_dfs[name] = run_silhouette_sweep(
            matrix, k_range, cluster_seed, name, logger
        )
        save_csv(silhouette_dfs[name], str(sil_dir / f"silhouette_{name}.csv"), index=False)

    # Step F — figures
    repr_fig_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_magnitude_histogram(
        delta_magnitude["magnitude"],
        near_zero_all,
        figures_dir / "magnitude_histogram.pdf",
        dpi,
    )
    plot_magnitude_by_stage(
        delta_magnitude["magnitude"],
        stages_series,
        h_stat,
        p_val,
        figures_dir / "magnitude_by_stage.pdf",
        dpi,
    )
    plot_delta_distributions(raw_all, figures_dir / "delta_distributions.pdf", dpi)

    mag_series = delta_magnitude["magnitude"]
    embedding_specs = [
        {
            "name": "raw_all_pca",
            "coords": raw_all_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_raw_all)[0],
            "y_label": pca_labels(evr_raw_all)[1],
            "title": "Raw ΔZ | PCA | All factors",
            "subtitle": "Coloured by transformation magnitude",
            "colour_mode": "magnitude",
            "near_zero_mask": near_zero_all,
        },
        {
            "name": "raw_all_umap",
            "coords": raw_all_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "Raw ΔZ | UMAP | All factors",
            "subtitle": "Coloured by transformation magnitude",
            "colour_mode": "magnitude",
            "near_zero_mask": near_zero_all,
        },
        {
            "name": "raw_nof1_pca",
            "coords": raw_nof1_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_raw_nof1)[0],
            "y_label": pca_labels(evr_raw_nof1)[1],
            "title": "Raw ΔZ | PCA | Factor1 excluded",
            "subtitle": "Coloured by transformation magnitude | Factor1 (tumour axis) removed",
            "colour_mode": "magnitude",
            "near_zero_mask": near_zero_nof1,
        },
        {
            "name": "raw_nof1_umap",
            "coords": raw_nof1_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "Raw ΔZ | UMAP | Factor1 excluded",
            "subtitle": "Coloured by transformation magnitude | Factor1 (tumour axis) removed",
            "colour_mode": "magnitude",
            "near_zero_mask": near_zero_nof1,
        },
        {
            "name": "norm_all_pca",
            "coords": norm_all_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_norm_all)[0],
            "y_label": pca_labels(evr_norm_all)[1],
            "title": "L2-normalised ΔZ | PCA | All factors",
            "subtitle": "Coloured by tumour stage",
            "colour_mode": "stage",
            "near_zero_mask": None,
        },
        {
            "name": "norm_all_umap",
            "coords": norm_all_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "L2-normalised ΔZ | UMAP | All factors",
            "subtitle": "Coloured by tumour stage",
            "colour_mode": "stage",
            "near_zero_mask": None,
        },
        {
            "name": "norm_nof1_pca",
            "coords": norm_nof1_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_norm_nof1)[0],
            "y_label": pca_labels(evr_norm_nof1)[1],
            "title": "L2-normalised ΔZ | PCA | Factor1 excluded",
            "subtitle": "Coloured by tumour stage | Factor1 (tumour axis) removed",
            "colour_mode": "stage",
            "near_zero_mask": None,
        },
        {
            "name": "norm_nof1_umap",
            "coords": norm_nof1_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "L2-normalised ΔZ | UMAP | Factor1 excluded",
            "subtitle": "Coloured by tumour stage | Factor1 (tumour axis) removed",
            "colour_mode": "stage",
            "near_zero_mask": None,
        },
    ]

    for spec in embedding_specs:
        plot_embedding(
            coords=spec["coords"],
            x_col=spec["x_col"],
            y_col=spec["y_col"],
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            title=spec["title"],
            subtitle=spec["subtitle"],
            colour_mode=spec["colour_mode"],
            magnitudes=mag_series if spec["colour_mode"] == "magnitude" else None,
            stages=stages_series if spec["colour_mode"] == "stage" else None,
            stage_palette=stage_palette if spec["colour_mode"] == "stage" else None,
            near_zero_mask=spec["near_zero_mask"],
            output_path=repr_fig_dir / f"{spec['name']}.pdf",
            dpi=dpi,
        )

    plot_silhouette_comparison(
        silhouette_dfs,
        repr_fig_dir / "silhouette_comparison.pdf",
        dpi,
    )

    all_factor_panels = [
        {
            "coords": raw_all_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_raw_all)[0],
            "y_label": pca_labels(evr_raw_all)[1],
            "title": "Raw ΔZ | PCA | All factors",
            "colour_mode": "magnitude",
            "magnitudes": mag_series,
            "near_zero_mask": near_zero_all,
        },
        {
            "coords": raw_all_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "Raw ΔZ | UMAP | All factors",
            "colour_mode": "magnitude",
            "magnitudes": mag_series,
            "near_zero_mask": near_zero_all,
        },
        {
            "coords": norm_all_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_norm_all)[0],
            "y_label": pca_labels(evr_norm_all)[1],
            "title": "L2-normalised ΔZ | PCA | All factors",
            "colour_mode": "stage",
            "stages": stages_series,
            "near_zero_mask": None,
        },
        {
            "coords": norm_all_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "L2-normalised ΔZ | UMAP | All factors",
            "colour_mode": "stage",
            "stages": stages_series,
            "near_zero_mask": None,
        },
    ]
    plot_summary_panel(
        all_factor_panels,
        "ΔZ Representation Comparison — All Factors | Review before Stage 06",
        stage_palette,
        repr_fig_dir / "summary_panel_all_factors.pdf",
        dpi,
    )

    no_f1_panels = [
        {
            "coords": raw_nof1_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_raw_nof1)[0],
            "y_label": pca_labels(evr_raw_nof1)[1],
            "title": "Raw ΔZ | PCA | Factor1 excluded",
            "colour_mode": "magnitude",
            "magnitudes": mag_series,
            "near_zero_mask": near_zero_nof1,
        },
        {
            "coords": raw_nof1_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "Raw ΔZ | UMAP | Factor1 excluded",
            "colour_mode": "magnitude",
            "magnitudes": mag_series,
            "near_zero_mask": near_zero_nof1,
        },
        {
            "coords": norm_nof1_pca,
            "x_col": "pc1",
            "y_col": "pc2",
            "x_label": pca_labels(evr_norm_nof1)[0],
            "y_label": pca_labels(evr_norm_nof1)[1],
            "title": "L2-normalised ΔZ | PCA | Factor1 excluded",
            "colour_mode": "stage",
            "stages": stages_series,
            "near_zero_mask": None,
        },
        {
            "coords": norm_nof1_umap,
            "x_col": "umap1",
            "y_col": "umap2",
            "x_label": umap_labels()[0],
            "y_label": umap_labels()[1],
            "title": "L2-normalised ΔZ | UMAP | Factor1 excluded",
            "colour_mode": "stage",
            "stages": stages_series,
            "near_zero_mask": None,
        },
    ]
    plot_summary_panel(
        no_f1_panels,
        "ΔZ Representation Comparison — Factor1 Excluded | Review before Stage 06",
        stage_palette,
        repr_fig_dir / "summary_panel_no_factor1.pdf",
        dpi,
    )

    logger.info("Stage 05 complete.")


if __name__ == "__main__":
    main()
