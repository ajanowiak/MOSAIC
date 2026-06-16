# 06_clustering.py
"""
Trajectory clustering for the MOSAIC pipeline.

Pipeline stage 6. Evaluates clustering quality (silhouette, GMM AIC/BIC) across
all k in config.clustering.k_range on the primary ΔZ representation, then fits
a final model using analyst-specified or automatically selected parameters. Computes
cross-representation ARI to assess robustness, within-cluster dispersion geometry,
and cluster summaries. Produces five diagnostic figures. Does NOT produce embedding
scatter plots — those are Stage 06.1.

Inputs:
  results/delta/delta_z.csv
  results/delta/delta_z_no_f1.csv
  results/delta/delta_z_normalized.csv
  results/delta/delta_z_normalized_no_f1.csv
  results/delta/delta_magnitude.csv
  results/delta/repr_comparison/embeddings/*.csv  (8 files)
  results/ingestion/metadata.csv
  config/config.yml

Outputs:
  results/clustering/cluster_assignments.csv
  results/clustering/all_k_assignments.csv
  results/clustering/k_final.txt
  results/clustering/decision_table.csv
  results/clustering/cross_repr_ari.csv
  results/clustering/cluster_centroids_raw.csv
  results/clustering/cluster_centroids_norm.csv
  results/clustering/cluster_sizes.csv
  results/clustering/cluster_summary.csv
  results/clustering/within_cluster_dispersion.csv
  results/clustering/centroid_distances.csv
  results/clustering/gmm_soft_assignments.csv  (GMM method only)
  figures/clustering/model_selection.pdf
  figures/clustering/cross_repr_ari.pdf
  figures/clustering/delta_heatmap.pdf
  figures/clustering/cluster_centroids.pdf
  figures/clustering/dispersion_summary.pdf
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, load_csv, save_csv
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

# NEXT STEPS FOR THE ANALYST
#
# 1. Review figures/clustering/model_selection.pdf
#    Look for: silhouette peak, BIC elbow, DB minimum, CH peak.
#    These may not agree — that is expected. Use judgment.
#
# 2. Review figures/clustering/dispersion_summary.pdf
#    Check whether clusters are more separated than they are dispersed.
#    separation_ratio > 1 for all pairs is a good sign.
#
# 3. Review results/clustering/cross_repr_ari.csv
#    ARI > 0.5 across representations: structure is robust.
#    ARI for primary_vs_pca2d: tests whether UMAP is revealing real
#    non-linear structure or inventing blobs.
#
# 4. Review results/clustering/decision_table.csv
#    Full metric table for all k.
#
# 5. Set your chosen parameters in config.yml:
#    clustering:
#      clustering_parameter_selection: manual
#      manual:
#        k: <your chosen k>
#        method: <kmeans or gmm>
#        representation: <your chosen representation>
#
# 6. Run Stage 06.1:
#    python src/py/06.1_clustering_plot.py --config config/config.yml
#    or trigger snakemake to pick up the new dependency.

_NEXT_STEPS_MSG = """
================================================================================
NEXT STEPS FOR THE ANALYST

1. Review figures/clustering/model_selection.pdf
   Look for: silhouette peak, BIC elbow.
   These may not agree — that is expected. Use judgment.

2. Review figures/clustering/dispersion_summary.pdf
   Check whether clusters are more separated than they are dispersed.
   separation_ratio > 1 for all pairs is a good sign.

3. Review results/clustering/cross_repr_ari.csv
   ARI > 0.5 across representations: structure is robust.
   ARI for primary_vs_pca2d: tests whether UMAP is revealing real
   non-linear structure or inventing blobs.

4. Review results/clustering/decision_table.csv
   Full metric table for all k.

5. Set your chosen parameters in config.yml:
   clustering:
     clustering_parameter_selection: manual
     manual:
       k: <your chosen k>
       method: <kmeans or gmm>
       representation: <your chosen representation>

6. Run Stage 06.1:
   python src/py/06.1_clustering_plot.py --config config/config.yml
   or trigger snakemake to pick up the new dependency.
================================================================================
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 06 — Trajectory Clustering")
    parser.add_argument("--config", required=True, help="Path to config/config.yml")
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to stdout and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mosaic.clustering")
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


def load_matrix(repr_name: str, logger: logging.Logger) -> pd.DataFrame:
    """Load a ΔZ matrix, dropping NaN rows (present in L2-normalised variants).

    Args:
        repr_name: Key from REPR_TO_FILE.
        logger: Logger instance.

    Returns:
        DataFrame with patient_id index, factor columns, no NaN rows.
    """
    path = Path(REPR_TO_FILE[repr_name])
    validate_file(path)
    df = load_csv(str(path))
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Representation %s: dropped %d NaN row(s) (near-zero magnitude patients). "
            "%d patients retained.",
            repr_name, n_dropped, len(df),
        )
    logger.info("Loaded %s: shape %s", repr_name, df.shape)
    return df


def fit_model(
    matrix: np.ndarray,
    k: int,
    method: str,
    n_init: int,
    random_state: int,
) -> tuple:
    """Fit a KMeans or GMM clustering model.

    Args:
        matrix: (n_samples, n_features) array.
        k: Number of clusters.
        method: "kmeans" or "gmm".
        n_init: Number of initialisations (KMeans n_init; GMM uses 10).
        random_state: Random seed.

    Returns:
        Tuple of (fitted model, integer label array).
    """
    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(matrix)
    else:
        model = GaussianMixture(n_components=k, n_init=10, random_state=random_state)
        model.fit(matrix)
        labels = model.predict(matrix)
    return model, labels


def build_patient_meta(
    metadata: pd.DataFrame, patient_ids: pd.Index, logger: logging.Logger
) -> pd.DataFrame:
    """Select tumour-side patient metadata for the given patient IDs.

    Args:
        metadata: DataFrame indexed by sample_id; must have patient_id column.
        patient_ids: Patient IDs to retrieve.
        logger: Logger instance.

    Returns:
        DataFrame indexed by patient_id.
    """
    tumour_mask = metadata["tissue_type"].astype(str).str.lower().isin({"tumour", "tumor"})
    tumour_meta = metadata.loc[tumour_mask].drop_duplicates(subset="patient_id")
    available = set(tumour_meta["patient_id"].tolist())
    missing = [pid for pid in patient_ids if pid not in available]
    if missing:
        logger.warning("Patient IDs not found in tumour metadata: %s", missing)
    patient_meta = tumour_meta.set_index("patient_id").reindex(patient_ids)
    logger.info("patient_meta shape: %s", patient_meta.shape)
    return patient_meta


def _log_ari(logger: logging.Logger, comparison: str, ari: float) -> None:
    """Log ARI value with a plain-English interpretation."""
    if np.isnan(ari):
        logger.info("ARI %s: NaN (could not compute)", comparison)
        return
    if ari > 0.8:
        interp = "representations produce nearly identical groupings"
    elif ari > 0.5:
        interp = "substantial agreement — same broad structure"
    else:
        interp = "structure is sensitive to representation choice"
    logger.info("ARI %s: %.4f — %s", comparison, ari, interp)


def plot_model_selection(
    decision_df: pd.DataFrame,
    k_final: int,
    representation: str,
    method: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Two-panel silhouette + GMM AIC/BIC model-selection figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"Clustering model selection — {representation} | {method} | k_final={k_final}",
        fontsize=9,
    )

    ax1 = axes[0]
    ax1.plot(decision_df["k"], decision_df["silhouette"], "o-", color="steelblue", linewidth=1.5)
    k_sil = decision_df.loc[decision_df["k"] == k_final, "silhouette"]
    if len(k_sil):
        ax1.axvline(k_final, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.annotate(
            f"k={k_final}",
            xy=(k_final, float(k_sil.values[0])),
            xytext=(k_final + 0.15, float(decision_df["silhouette"].max()) * 0.98),
            fontsize=7,
            color="dimgrey",
        )
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette score")
    ax1.set_title("Silhouette")
    ax1.set_xticks(decision_df["k"].tolist())

    ax2 = axes[1]
    ax2.plot(decision_df["k"], decision_df["gmm_aic"], "o-",
             color="steelblue", linewidth=1.5, label="AIC")
    ax2.plot(decision_df["k"], decision_df["gmm_bic"], "s--",
             color="darkorange", linewidth=1.5, label="BIC")
    ax2.axvline(k_final, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_xlabel("k")
    ax2.set_ylabel("Score (lower = better)")
    ax2.set_title("GMM AIC / BIC (descriptive)")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.set_xticks(decision_df["k"].tolist())

    fig.tight_layout()
    savefig(fig, str(output_path), dpi=dpi)


# Soft bar colours used across clustering diagnostic bar charts.
BAR_PALETTE = ["#6B9AC4", "#88B892", "#E8B86D", "#D98880", "#9B8EC4", "#7EC8C8"]
BAR_ALPHA = 0.82
NEGATIVE_BAR_COLOR = "#d4d4d4"


def plot_cross_repr_ari(
    ari_df: pd.DataFrame,
    k_final: int,
    output_path: Path,
    dpi: int,
) -> None:
    """Horizontal bar chart of cross-representation ARI values."""
    n_comparisons = len(ari_df)
    fig, ax = plt.subplots(figsize=(7, 3 + 0.4 * n_comparisons))
    colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(n_comparisons)]
    ax.barh(
        ari_df["comparison"],
        ari_df["ari"],
        color=colors,
        height=0.55,
        alpha=BAR_ALPHA,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Adjusted rand index vs primary representation (norm_all)")
    # ax.set_ylabel("Comparison") 
    ax.set_title(f"Cross-representation cluster agreement at k={k_final}")
    fig.tight_layout()
    savefig(fig, str(output_path), dpi=dpi)


def plot_delta_heatmap(
    raw_all_matrix: pd.DataFrame,
    sorted_patients: list,
    final_labels: np.ndarray,
    primary_index: pd.Index,
    k_final: int,
    cluster_palette: dict,
    output_path: Path,
    dpi: int,
) -> None:
    """Heatmap of raw ΔZ sorted by cluster with a cluster colour annotation bar."""
    display_matrix = raw_all_matrix.reindex(sorted_patients)
    display_matrix.index.name = None
    factor_names = display_matrix.columns.tolist()
    n_patients = len(sorted_patients)

    all_vals = display_matrix.values.flatten()
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    cluster_series = pd.Series(final_labels, index=primary_index, name="cluster")

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[0.03, 0.9, 0.02],
        wspace=0.03,
        left=0.07, right=0.95, top=0.93, bottom=0.12,
    )
    ax_clust = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_cbar_heat = fig.add_subplot(gs[0, 2])

    sns.heatmap(
        display_matrix,
        ax=ax_heat,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=factor_names,
        yticklabels=False,
        cbar=False,
    )
    plt.setp(ax_heat.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax_heat.set_title(
        f"Raw ΔZ heatmap (k={k_final}, sorted by cluster)",
        fontsize=8,
    )

    # Left cluster colour bar
    rgba_clust = np.zeros((n_patients, 1, 4))
    for i, pid in enumerate(sorted_patients):
        c = int(cluster_series.loc[pid])
        rgba = mcolors.to_rgba(cluster_palette[c])
        rgba_clust[i, 0, :] = (*rgba[:3], BAR_ALPHA)
    ax_clust.imshow(rgba_clust, aspect="auto", interpolation="nearest")
    ax_clust.set_xticks([])
    ax_clust.set_yticks([])
    ax_clust.set_ylabel("")
    for c in range(k_final):
        rows = [i for i, pid in enumerate(sorted_patients)
                if int(cluster_series.loc[pid]) == c]
        if not rows:
            continue
        mid = int(np.mean(rows))
        ax_clust.text(
            0, mid, f"C{c}", ha="center", va="center",
            fontsize=6, color="white", fontweight="bold",
        )

    sm_heat = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=vmin, vmax=vmax))
    sm_heat.set_array([])
    cbar_heat = fig.colorbar(sm_heat, cax=ax_cbar_heat)
    cbar_heat.set_label("Raw ΔZ", fontsize=7)
    cbar_heat.ax.tick_params(labelsize=6)

    savefig(fig, str(output_path), dpi=dpi)


def plot_cluster_centroids(
    centroids_raw: pd.DataFrame,
    cluster_sizes: pd.DataFrame,
    k_final: int,
    cluster_palette: dict,
    output_path: Path,
    dpi: int,
) -> None:
    """One horizontal bar panel per cluster showing raw ΔZ centroid loadings."""
    n_cols = min(k_final, 3)
    n_rows = math.ceil(k_final / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for c in range(k_final):
        ax = axes_flat[c]
        centroid = centroids_raw.iloc[c]
        sorted_factors = centroid.abs().sort_values(ascending=False).index.tolist()
        sorted_vals = centroid[sorted_factors].values
        n_factors = len(sorted_factors)
        colors = [
            (*mcolors.to_rgb(cluster_palette[c]), BAR_ALPHA) if v >= 0
            else NEGATIVE_BAR_COLOR
            for v in sorted_vals
        ]
        n_patients = int(cluster_sizes.loc[c, "n_patients"])

        ax.barh(range(n_factors), sorted_vals, color=colors, height=0.65)
        ax.set_yticks(range(n_factors))
        ax.set_yticklabels(sorted_factors, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Mean raw ΔZ", fontsize=7)
        ax.set_title(f"Cluster {c} (n={n_patients})", fontsize=8)

        for i in range(min(3, n_factors)):
            v = sorted_vals[i]
            fname = sorted_factors[i]
            ax.text(
                v, i,
                f"  {fname}: {v:.3f}" if v >= 0 else f"{fname}: {v:.3f}  ",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=5,
            )

    for idx in range(k_final, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Cluster centroids in raw ΔZ space", fontsize=9)
    fig.tight_layout()
    savefig(fig, str(output_path), dpi=dpi)


# Separate palette for pair-distance bars — intentionally distinct from cluster_palette
# (tab10) so the two panels are not read as sharing the same colour encoding.
PAIR_DISTANCE_PALETTE = ["#BC6C7C", "#7A9E7E", "#D4A373", "#8E7DBE", "#5B9A8B", "#C97B63"]


def plot_dispersion_summary(
    dispersion_df: pd.DataFrame,
    centroid_dist_df: pd.DataFrame,
    k_final: int,
    cluster_palette: dict,
    output_path: Path,
    dpi: int,
) -> None:
    """Two-panel figure: within-cluster dispersion and pairwise centroid distances."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Cluster geometry summary", fontsize=9)

    ax1 = axes[0]
    clusters = dispersion_df["cluster"].tolist()
    means = dispersion_df["mean_dist_to_centroid"].tolist()
    stds = dispersion_df["std_dist_to_centroid"].tolist()
    bar_colors = [cluster_palette[c] for c in clusters]
    ax1.bar(
        clusters, means, yerr=stds, color=bar_colors,
        capsize=4, width=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "#444444", "capthick": 1.2},
    )
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Mean distance to centroid")
    ax1.set_title("Within-cluster dispersion")
    ax1.set_xticks(clusters)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax1.set_axisbelow(True)

    ax2 = axes[1]
    mean_wc_disp = float(dispersion_df["mean_dist_to_centroid"].mean())
    pair_labels = [
        f"C{int(r.cluster_i)}-C{int(r.cluster_j)}"
        for _, r in centroid_dist_df.iterrows()
    ]
    pair_dists = centroid_dist_df["euclidean_distance"].tolist()
    pair_colors = [
        PAIR_DISTANCE_PALETTE[i % len(PAIR_DISTANCE_PALETTE)]
        for i in range(len(pair_labels))
    ]
    ax2.bar(pair_labels, pair_dists, color=pair_colors, width=0.55)
    ax2.axhline(
        mean_wc_disp, color="#C44E52", linestyle="--", linewidth=1.2,
        label="mean within-cluster dispersion",
    )
    ax2.set_xlabel("Cluster pair")
    ax2.set_ylabel("Euclidean distance (normalised ΔZ space)")
    ax2.set_title("Between-cluster centroid distances")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    savefig(fig, str(output_path), dpi=dpi)


def main() -> None:
    """Run the Stage 06 clustering pipeline."""
    args = parse_args()
    config = load_config(args.config)
    set_style()

    logger = setup_logging(Path("log") / "06_clustering.log")

    cl_cfg = config["clustering"]
    k_range = cl_cfg["k_range"]
    n_init = int(cl_cfg["n_init"])
    random_state = int(cl_cfg["random_state"])
    mode = cl_cfg["clustering_parameter_selection"]
    dpi = int(config["figures"]["dpi"])

    if mode not in {"manual", "automatic"}:
        raise ValueError(
            f"clustering_parameter_selection must be 'manual' or 'automatic', got '{mode}'"
        )

    if mode == "manual":
        method = cl_cfg["manual"]["method"]
        representation = cl_cfg["manual"]["representation"]
    else:
        method = cl_cfg["automatic"]["method"]
        representation = cl_cfg["automatic"]["representation"]

    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"representation '{representation}' is not valid. "
            f"Valid options: {sorted(VALID_REPRESENTATIONS)}"
        )
    if method not in {"kmeans", "gmm"}:
        raise ValueError(f"method must be 'kmeans' or 'gmm', got '{method}'")

    logger.info(
        "Clustering mode: %s | representation: %s | method: %s",
        mode, representation, method,
    )
    if mode == "manual":
        logger.info("Analyst-specified k: %d", int(cl_cfg["manual"]["k"]))
    else:
        logger.info("k will be selected by silhouette (automatic mode)")

    # Step A — Load inputs
    primary_matrix = load_matrix(representation, logger)

    all_matrices: dict[str, pd.DataFrame] = {}
    for r in VALID_REPRESENTATIONS:
        all_matrices[r] = load_matrix(r, logger)

    mag_path = Path("results/delta/delta_magnitude.csv")
    meta_path = Path("results/ingestion/metadata.csv")
    validate_file(mag_path)
    validate_file(meta_path)
    delta_magnitude = load_csv(str(mag_path))
    metadata = load_csv(str(meta_path))
    patient_meta = build_patient_meta(metadata, primary_matrix.index, logger)  # noqa: F841

    out_dir = Path("results/clustering")
    fig_dir = Path("figures/clustering")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Step B — Silhouette sweep across k (primary representation only)
    silhouette_rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_range:
        _, labels = fit_model(primary_matrix.values, k, method, n_init, random_state)
        sil = float(silhouette_score(primary_matrix.values, labels))
        logger.info("k=%d: silhouette=%.4f", k, sil)
        silhouette_rows.append({"k": k, "silhouette": sil})
        labels_by_k[k] = labels

    # Step C — GMM AIC/BIC sweep (descriptive, always computed regardless of method)
    aic_bic_rows = []
    for k in k_range:
        gm = GaussianMixture(n_components=k, n_init=10, random_state=random_state)
        gm.fit(primary_matrix.values)
        aic = float(gm.aic(primary_matrix.values))
        bic = float(gm.bic(primary_matrix.values))
        logger.info("GMM k=%d: AIC=%.2f, BIC=%.2f", k, aic, bic)
        aic_bic_rows.append({"k": k, "gmm_aic": aic, "gmm_bic": bic})

    # Step D — Build and save decision table
    sil_df = pd.DataFrame(silhouette_rows)
    aic_bic_df = pd.DataFrame(aic_bic_rows)
    decision_df = sil_df.merge(aic_bic_df, on="k")
    # silhouette:         higher = better separation (range -1 to 1)
    # gmm_aic / gmm_bic:  lower = better model fit (BIC penalises complexity more)
    save_csv(decision_df, str(out_dir / "decision_table.csv"), index=False)
    logger.info("Decision table:\n%s", decision_df.round(4).to_string(index=False))

    # Step E — Select k_final
    if mode == "manual":
        k_final = int(cl_cfg["manual"]["k"])
        logger.info("k_final = %d (analyst-specified, manual mode)", k_final)
    else:
        best_row = decision_df.loc[decision_df["silhouette"].idxmax()]
        k_final = int(best_row["k"])
        logger.info("k_final = %d selected by silhouette (automatic mode)", k_final)
        logger.info("Winning row: %s", best_row.to_dict())

    (out_dir / "k_final.txt").write_text(str(k_final))
    logger.info("Saved k_final = %d", k_final)

    # Step F — Fit final model and compute assignments
    final_model, final_labels = fit_model(
        primary_matrix.values, k_final, method, n_init, random_state
    )

    if method == "kmeans":
        centroids_norm = pd.DataFrame(
            final_model.cluster_centers_,
            columns=primary_matrix.columns,
        )
    else:
        centroids_norm = pd.DataFrame(
            final_model.means_,
            columns=primary_matrix.columns,
        )
        soft_probs = final_model.predict_proba(primary_matrix.values)
        soft_df = pd.DataFrame(
            soft_probs,
            index=primary_matrix.index,
            columns=[f"cluster_{i}_prob" for i in range(k_final)],
        )
        soft_df.index.name = primary_matrix.index.name
        save_csv(soft_df, str(out_dir / "gmm_soft_assignments.csv"))

    centroids_norm.index.name = "cluster"
    save_csv(centroids_norm, str(out_dir / "cluster_centroids_norm.csv"))

    labels_series = pd.Series(final_labels, index=primary_matrix.index, name="cluster")
    assignments_df = labels_series.to_frame()
    save_csv(assignments_df, str(out_dir / "cluster_assignments.csv"))
    logger.info(
        "Cluster assignments: %d patients, counts: %s",
        len(assignments_df),
        labels_series.value_counts().sort_index().to_dict(),
    )

    all_k_df = pd.DataFrame(
        {f"k{k}": labels_by_k[k] for k in k_range},
        index=primary_matrix.index,
    )
    all_k_df.index.name = primary_matrix.index.name
    save_csv(all_k_df, str(out_dir / "all_k_assignments.csv"))

    # Centroids in raw ΔZ space for biological interpretability
    raw_all_aligned = all_matrices["raw_all"].reindex(primary_matrix.index)
    centroids_raw = raw_all_aligned.groupby(labels_series).mean()
    centroids_raw.index.name = "cluster"
    save_csv(centroids_raw, str(out_dir / "cluster_centroids_raw.csv"))

    # Step G — Cross-representation ARI at k_final
    ari_rows = []
    alt_reprs = [r for r in sorted(VALID_REPRESENTATIONS) if r != representation]
    for alt in alt_reprs:
        alt_matrix = all_matrices[alt]
        shared = primary_matrix.index.intersection(alt_matrix.index)
        n_shared = int(len(shared))
        if n_shared < k_final:
            logger.warning(
                "Skipping ARI %s vs %s: %d shared patients, need >= %d",
                representation, alt, n_shared, k_final,
            )
            ari = np.nan
        else:
            primary_labels_shared = labels_series.loc[shared].values
            _, alt_labels = fit_model(
                alt_matrix.loc[shared].values, k_final, method, n_init, random_state
            )
            ari = float(adjusted_rand_score(primary_labels_shared, alt_labels))
        ari_rows.append({
            "comparison": f"{representation}_vs_{alt}",
            "k": k_final,
            "ari": ari,
            "n_shared_patients": n_shared,
        })
        _log_ari(logger, f"{representation}_vs_{alt}", ari)

    # Special comparison: cluster on primary representation's 2D PCA embedding
    pca2d_path = Path(REPR_TO_EMBEDDINGS[representation]["pca"])
    if pca2d_path.exists():
        pca2d_df = load_csv(str(pca2d_path)).dropna()
        shared_2d = primary_matrix.index.intersection(pca2d_df.index)
        n_shared_2d = int(len(shared_2d))
        if n_shared_2d >= k_final:
            primary_labels_shared_2d = labels_series.loc[shared_2d].values
            km_2d = KMeans(n_clusters=k_final, n_init=n_init, random_state=random_state)
            labels_2d = km_2d.fit_predict(pca2d_df.loc[shared_2d].values)
            ari_2d = float(adjusted_rand_score(primary_labels_shared_2d, labels_2d))
        else:
            logger.warning(
                "Skipping pca2d ARI: %d shared patients, need >= %d", n_shared_2d, k_final
            )
            ari_2d = np.nan
        ari_rows.append({
            "comparison": f"{representation}_vs_{representation}_pca2d",
            "k": k_final,
            "ari": ari_2d,
            "n_shared_patients": n_shared_2d,
        })
        _log_ari(logger, f"{representation}_vs_{representation}_pca2d", ari_2d)
    else:
        logger.warning("PCA 2D embedding not found at %s; skipping pca2d comparison.", pca2d_path)

    ari_df = pd.DataFrame(ari_rows)
    save_csv(ari_df, str(out_dir / "cross_repr_ari.csv"), index=False)

    # Step H — Within-cluster dispersion and centroid distances
    dispersion_rows = []
    mean_dispersions: dict[int, float] = {}
    for c in range(k_final):
        mask = final_labels == c
        patients_c = primary_matrix.values[mask]
        centroid_c = centroids_norm.iloc[c].values
        dists = np.linalg.norm(patients_c - centroid_c, axis=1)
        mean_dist = float(dists.mean())
        std_dist = float(dists.std(ddof=1)) if len(dists) > 1 else 0.0
        mean_dispersions[c] = mean_dist
        dispersion_rows.append({
            "cluster": c,
            "n_patients": int(mask.sum()),
            "mean_dist_to_centroid": mean_dist,
            "std_dist_to_centroid": std_dist,
        })
    dispersion_df = pd.DataFrame(dispersion_rows)
    save_csv(dispersion_df, str(out_dir / "within_cluster_dispersion.csv"), index=False)

    centroid_dist_rows = []
    for i in range(k_final):
        for j in range(i + 1, k_final):
            dist = float(np.linalg.norm(
                centroids_norm.iloc[i].values - centroids_norm.iloc[j].values
            ))
            centroid_dist_rows.append({"cluster_i": i, "cluster_j": j, "euclidean_distance": dist})
            # separation_ratio > 1: clusters are more separated than they are dispersed — good
            # separation_ratio < 1: clusters overlap substantially — interpret with caution
            sep_ratio = dist / (mean_dispersions[i] + mean_dispersions[j])
            logger.info(
                "Pair (C%d, C%d): centroid_dist=%.4f, sep_ratio=%.3f (%s)",
                i, j, dist, sep_ratio,
                "good" if sep_ratio > 1 else "interpret with caution",
            )
    centroid_dist_df = pd.DataFrame(centroid_dist_rows)
    save_csv(centroid_dist_df, str(out_dir / "centroid_distances.csv"), index=False)

    # Step I — Cluster summaries
    sizes_df = pd.DataFrame(
        {"n_patients": [int(np.sum(final_labels == c)) for c in range(k_final)]}
    )
    sizes_df.index.name = "cluster"
    save_csv(sizes_df, str(out_dir / "cluster_sizes.csv"))

    for c in range(k_final):
        n_c = int(sizes_df.loc[c, "n_patients"])
        if n_c < 8:
            logger.warning(
                "Cluster %d has %d patients — fewer than the recommended minimum of 8.", c, n_c
            )

    summary_rows = []
    for c in range(k_final):
        n_pat = int(sizes_df.loc[c, "n_patients"])
        cluster_patient_ids = primary_matrix.index[final_labels == c]
        mag_vals = delta_magnitude.loc[
            delta_magnitude.index.isin(cluster_patient_ids), "magnitude"
        ]
        mean_mag = float(mag_vals.mean()) if len(mag_vals) > 0 else np.nan
        raw_centroid = centroids_raw.iloc[c]
        top3_factors = raw_centroid.abs().nlargest(3).index.tolist()
        top3_vals = raw_centroid[top3_factors].values.tolist()
        summary_rows.append({
            "cluster": c,
            "n_patients": n_pat,
            "mean_magnitude": round(mean_mag, 4),
            "top3_factors": ",".join(top3_factors),
            "top3_centroid_values": ",".join(f"{v:.4f}" for v in top3_vals),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("cluster")
    save_csv(summary_df, str(out_dir / "cluster_summary.csv"))
    logger.info("Cluster summary:\n%s", summary_df.to_string())

    # Step J — Figures
    cluster_palette = get_cluster_palette(k_final, config["figures"]["cluster_palette"])

    plot_model_selection(
        decision_df, k_final, representation, method,
        fig_dir / "model_selection.pdf", dpi,
    )

    plot_cross_repr_ari(ari_df, k_final, fig_dir / "cross_repr_ari.pdf", dpi)

    sort_df = pd.DataFrame(
        {
            "cluster": final_labels,
            "magnitude": [
                float(delta_magnitude.loc[pid, "magnitude"])
                if pid in delta_magnitude.index else np.nan
                for pid in primary_matrix.index
            ],
        },
        index=primary_matrix.index,
    )
    sort_df = sort_df.sort_values(["cluster", "magnitude"], ascending=[True, True])
    sorted_patients = sort_df.index.tolist()

    plot_delta_heatmap(
        raw_all_matrix=all_matrices["raw_all"],
        sorted_patients=sorted_patients,
        final_labels=final_labels,
        primary_index=primary_matrix.index,
        k_final=k_final,
        cluster_palette=cluster_palette,
        output_path=fig_dir / "delta_heatmap.pdf",
        dpi=dpi,
    )

    plot_cluster_centroids(
        centroids_raw, sizes_df, k_final, cluster_palette,
        fig_dir / "cluster_centroids.pdf", dpi,
    )

    plot_dispersion_summary(
        dispersion_df, centroid_dist_df, k_final, cluster_palette,
        fig_dir / "dispersion_summary.pdf", dpi,
    )

    logger.info("Stage 06 complete.")
    print(_NEXT_STEPS_MSG)


if __name__ == "__main__":
    main()
