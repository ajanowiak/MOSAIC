# 04_factor_qc.py
"""
Factor QC and characterisation for retained MOFA+ factors.

Pipeline stage 4. Reads factor scores, variance explained, and sample metadata.
Computes per-patient ΔZ, tissue-separation statistics, metadata correlations,
PCA/UMAP embeddings, and diagnostic figures. Writes delta_z.csv for Stage 05.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_csv, save_csv
from utils.plotting import savefig, set_style

TUMOUR_TISSUE = {"tumour", "tumor"}
NORMAL_TISSUE = {"normal"}
MUTATION_COLS = [
    "KRAS_mutation",
    "EGFR_mutation",
    "TP53_mutation",
    "ALK_fusion",
]
CONTINUOUS_COLS = ["age_at_diagnosis", "overall_survival_days"]

LABEL_MAP = {
    "smoking_Current reformed smoker within past 15 years": "smoking:\nreformed <15yr",
    "smoking_Current reformed smoker, more than 15 years": "smoking:\nreformed >15yr",
    "smoking_Current smoker: Includes daily and non-daily smokers": "smoking:\ncurrent",
    "smoking_Lifelong non-smoker: Less than 100 cigarettes smoked in lifetime": "smoking:\nnon-smoker",
    "smoking_Smoking history not available": "smoking:\nunknown",
    "stage_Stage I": "Stage I",
    "stage_Stage II": "Stage II",
    "stage_Stage III": "Stage III",
    "overall_survival_event": "OS event",
    "overall_survival_days": "OS days",
    "age_at_diagnosis": "age",
    "KRAS_mutation": "KRAS mut",
    "EGFR_mutation": "EGFR mut",
    "TP53_mutation": "TP53 mut",
    "ALK_fusion": "ALK fusion",
    "sex": "sex",
}


def parse_arguments():
    """Parse CLI arguments for the factor QC script."""
    parser = argparse.ArgumentParser(
        description="Factor QC and characterisation for the MOSAIC pipeline."
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
    logger = logging.getLogger("mosaic.factor_qc")
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


def validate_inputs(factor_scores: pd.DataFrame, metadata: pd.DataFrame, logger: logging.Logger):
    """Ensure every factor-score sample appears in metadata."""
    missing = factor_scores.index.difference(metadata.index)
    if len(missing) > 0:
        logger.error("Missing sample IDs in metadata: %s", missing.tolist())
        raise ValueError(
            f"{len(missing)} factor-score sample(s) missing from metadata: "
            f"{missing.tolist()[:10]}"
        )


def is_female(value) -> bool:
    """Return True if a sex label corresponds to female."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"female", "f"} or "female" in text


def is_male(value) -> bool:
    """Return True if a sex label corresponds to male."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"male", "m"} or ("male" in text and "female" not in text)


def stage_column_name(metadata: pd.DataFrame) -> str:
    """Return the tumour-stage column name present in metadata."""
    if "tumour_stage" in metadata.columns:
        return "tumour_stage"
    if "tumor_stage" in metadata.columns:
        return "tumor_stage"
    raise KeyError("No tumour_stage or tumor_stage column found in metadata.")


def encode_sex(series: pd.Series, logger: logging.Logger) -> pd.Series:
    """Binary-encode sex: female-like values → 0, male-like → 1."""
    female_vals = [v for v in series.dropna().unique() if is_female(v)]
    male_vals = [v for v in series.dropna().unique() if is_male(v)]

    if female_vals:
        female_ref = series[series.map(is_female)].value_counts().idxmax()
    else:
        female_ref = None
    if male_vals:
        male_ref = series[series.map(is_male)].value_counts().idxmax()
    else:
        male_ref = None

    logger.info(
        "Sex encoding: female-like → 0 (most common: %s), male-like → 1 (most common: %s)",
        female_ref,
        male_ref,
    )

    encoded = pd.Series(np.nan, index=series.index, dtype=float)
    for value in series.dropna().unique():
        if is_female(value):
            encoded[series == value] = 0.0
        elif is_male(value):
            encoded[series == value] = 1.0
        else:
            logger.warning("Unrecognised sex value mapped to NaN: %s", value)
    return encoded


def encode_binary_column(
    series: pd.Series, col_name: str, logger: logging.Logger
) -> pd.Series:
    """Cast a binary metadata column to int, warning on unexpected values."""
    encoded = pd.to_numeric(series, errors="coerce")
    valid_mask = series.notna()
    unexpected = valid_mask & ~encoded.isin([0, 1])
    if unexpected.any():
        bad_values = series[unexpected].unique().tolist()
        logger.warning(
            "Unexpected values in %s mapped to NaN: %s", col_name, bad_values
        )
        encoded[unexpected] = np.nan
    logger.info("%s: cast to binary int", col_name)
    return encoded.astype(float)


def one_hot_encode(series: pd.Series, prefix: str, logger: logging.Logger) -> pd.DataFrame:
    """One-hot encode a categorical column, keeping all dummy columns."""
    dummies = pd.get_dummies(series, prefix=prefix, drop_first=False, dtype=float)
    for category in series.dropna().unique():
        dummy_col = f"{prefix}_{category}"
        if dummy_col in dummies.columns:
            logger.info("One-hot mapping: %s → %s", category, dummy_col)
        else:
            logger.info("One-hot mapping: %s → (column name sanitised by pandas)", category)
    return dummies


def drop_all_nan_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Drop columns that are entirely NaN after encoding."""
    kept = df.copy()
    for col in df.columns:
        if kept[col].isna().all():
            logger.warning("Dropping column %s: entirely NaN after encoding.", col)
            kept = kept.drop(columns=[col])
    return kept


def encode_tumour_metadata(
    metadata: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """Encode tumour-side metadata for correlation analysis."""
    tumour_mask = metadata["tissue_type"].astype(str).str.lower().isin(TUMOUR_TISSUE)
    tumour_meta = metadata.loc[tumour_mask].copy()
    tumour_meta = tumour_meta.set_index("patient_id", drop=False)
    tumour_meta = tumour_meta[~tumour_meta.index.duplicated(keep="first")]

    encoded_parts = []

    if "sex" in tumour_meta.columns:
        encoded_parts.append(encode_sex(tumour_meta["sex"], logger).rename("sex"))

    if "smoking_status" in tumour_meta.columns:
        smoking_dummies = one_hot_encode(
            tumour_meta["smoking_status"], prefix="smoking", logger=logger
        )
        encoded_parts.append(smoking_dummies)

    stage_col = stage_column_name(tumour_meta)
    stage_dummies = one_hot_encode(tumour_meta[stage_col], prefix="stage", logger=logger)
    encoded_parts.append(stage_dummies)

    if "overall_survival_event" in tumour_meta.columns:
        encoded_parts.append(
            encode_binary_column(
                tumour_meta["overall_survival_event"],
                "overall_survival_event",
                logger,
            ).rename("overall_survival_event")
        )

    for col in MUTATION_COLS:
        if col in tumour_meta.columns:
            encoded_parts.append(
                encode_binary_column(tumour_meta[col], col, logger).rename(col)
            )

    for col in CONTINUOUS_COLS:
        if col in tumour_meta.columns:
            encoded_parts.append(
                pd.to_numeric(tumour_meta[col], errors="coerce").rename(col)
            )
            logger.info("%s: used as continuous (no encoding)", col)

    encoded = pd.concat(encoded_parts, axis=1)
    encoded = drop_all_nan_columns(encoded, logger)
    logger.info("Retained metadata columns for correlation: %s", encoded.columns.tolist())
    return encoded


def compute_delta_z(
    factor_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-patient ΔZ and return tumour/normal factor subsets."""
    meta_aligned = metadata.loc[factor_scores.index].copy()
    tumour_mask = meta_aligned["tissue_type"].astype(str).str.lower().isin(TUMOUR_TISSUE)
    normal_mask = meta_aligned["tissue_type"].astype(str).str.lower().isin(NORMAL_TISSUE)

    tumour_factors = factor_scores.loc[tumour_mask].copy()
    normal_factors = factor_scores.loc[normal_mask].copy()

    tumour_factors["patient_id"] = meta_aligned.loc[tumour_mask, "patient_id"].values
    normal_factors["patient_id"] = meta_aligned.loc[normal_mask, "patient_id"].values

    tumour_factors = tumour_factors.set_index("patient_id")
    normal_factors = normal_factors.set_index("patient_id")

    shared_patients = tumour_factors.index.intersection(normal_factors.index)
    logger.info("Matched tumour-normal pairs found: %s", len(shared_patients))
    if len(shared_patients) != 102:
        logger.warning(
            "Expected 102 matched pairs but found %s.", len(shared_patients)
        )

    tumour_factors = tumour_factors.loc[shared_patients]
    normal_factors = normal_factors.loc[shared_patients]

    factor_cols = factor_scores.columns.tolist()
    delta_z = tumour_factors[factor_cols] - normal_factors[factor_cols]
    return delta_z, tumour_factors[factor_cols], normal_factors[factor_cols]


def compute_tissue_stats(delta_z: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Paired Wilcoxon test per factor on ΔZ values."""
    records = []
    for factor in delta_z.columns:
        differences = delta_z[factor].values
        statistic, p_value = wilcoxon(differences, alternative="two-sided", method="approx")
        n = len(differences)
        rank_biserial_r = 1.0 - (2.0 * statistic) / (n * (n + 1) / 2.0)
        records.append(
            {
                "factor": factor,
                "wilcoxon_W": statistic,
                "p_value": p_value,
                "rank_biserial_r": rank_biserial_r,
            }
        )

    stats_df = pd.DataFrame(records).set_index("factor")
    n_factors = len(stats_df)
    stats_df["corrected_p_bonf"] = np.minimum(stats_df["p_value"] * n_factors, 1.0)
    _, pvals_bh, _, _ = multipletests(stats_df["p_value"].values, method="fdr_bh")
    stats_df["corrected_p_bh"] = pvals_bh

    best_factor = stats_df["p_value"].idxmin()
    logger.info(
        "Smallest raw p-value: %s (p=%.4g)", best_factor, stats_df.loc[best_factor, "p_value"]
    )

    sig_bonf = stats_df.index[stats_df["corrected_p_bonf"] < 0.05].tolist()
    sig_bh = stats_df.index[stats_df["corrected_p_bh"] < 0.05].tolist()
    logger.info("Factors significant under Bonferroni (p<0.05): %s", sig_bonf)
    logger.info("Factors significant under BH (p<0.05): %s", sig_bh)
    if len(sig_bh) > len(sig_bonf):
        logger.info(
            "BH detected more significant factors than Bonferroni (%s vs %s).",
            len(sig_bh),
            len(sig_bonf),
        )

    return stats_df


def apply_matrix_corrections(pval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Bonferroni and BH correction to a p-value matrix."""
    flat = pval_df.values.flatten()
    valid = ~np.isnan(flat)
    bonf = np.full_like(flat, np.nan, dtype=float)
    bh = np.full_like(flat, np.nan, dtype=float)

    if valid.any():
        n_tests = valid.sum()
        bonf[valid] = np.minimum(flat[valid] * n_tests, 1.0)
        _, bh_valid, _, _ = multipletests(flat[valid], method="fdr_bh")
        bh[valid] = bh_valid

    shape = pval_df.shape
    bonf_df = pd.DataFrame(bonf.reshape(shape), index=pval_df.index, columns=pval_df.columns)
    bh_df = pd.DataFrame(bh.reshape(shape), index=pval_df.index, columns=pval_df.columns)
    return bonf_df, bh_df


def compute_metadata_correlations(
    factor_df: pd.DataFrame,
    encoded_meta: pd.DataFrame,
    logger: logging.Logger,
    analysis_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Spearman correlations between factors and encoded metadata."""
    shared_patients = factor_df.index.intersection(encoded_meta.index)
    factor_aligned = factor_df.loc[shared_patients]
    meta_aligned = encoded_meta.loc[shared_patients]

    rho_data = {}
    pval_data = {}

    for factor in factor_aligned.columns:
        rho_row = {}
        pval_row = {}
        for meta_col in meta_aligned.columns:
            x = factor_aligned[factor].values
            y = meta_aligned[meta_col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            n_valid = mask.sum()
            if n_valid < 10:
                logger.warning(
                    "%s | %s vs %s: n_valid=%s < 10; storing NaN",
                    analysis_label,
                    factor,
                    meta_col,
                    n_valid,
                )
                rho_row[meta_col] = np.nan
                pval_row[meta_col] = np.nan
            else:
                rho, p_value = spearmanr(x[mask], y[mask], nan_policy="omit")
                rho_row[meta_col] = rho
                pval_row[meta_col] = p_value
        rho_data[factor] = rho_row
        pval_data[factor] = pval_row

    rho_df = pd.DataFrame.from_dict(rho_data, orient="index")
    pval_df = pd.DataFrame.from_dict(pval_data, orient="index")
    pval_bonf, pval_bh = apply_matrix_corrections(pval_df)
    return rho_df, pval_df, pval_bonf, pval_bh


def fit_pca(factor_scores: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """PCA embedding of all samples in factor space."""
    pca_cfg = config["dimensionality_reduction"]["pca"]
    pca = PCA(
        n_components=2,
        random_state=pca_cfg["random_state"],
    )
    coords = pca.fit_transform(factor_scores.values)
    embedding = pd.DataFrame(
        coords,
        index=factor_scores.index,
        columns=["pca1", "pca2"],
    )
    return embedding, pca.explained_variance_ratio_


def fit_umap(factor_scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    """UMAP embedding of all samples in factor space."""
    umap_cfg = config["dimensionality_reduction"]["umap"]
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        random_state=umap_cfg["random_state"],
    )
    coords = reducer.fit_transform(factor_scores.values)
    return pd.DataFrame(
        coords,
        index=factor_scores.index,
        columns=["umap1", "umap2"],
    )


def draw_trajectory_arrows(
    ax,
    embedding_df: pd.DataFrame,
    metadata: pd.DataFrame,
    patient_ids,
    coord_cols: list[str],
    colour: str,
):
    """Draw normal→tumour trajectory arrows on a 2D embedding."""
    # Arrows are steelblue placeholder. Stage 09 regenerates this figure
    # with arrows coloured by cluster assignment from Stage 06.
    meta = metadata.copy()
    if "patient_id" not in meta.columns:
        meta["patient_id"] = meta.index.to_series().str.replace(".N", "", regex=False)

    for pid in patient_ids:
        tumour_rows = meta.index[
            (meta["patient_id"] == pid)
            & meta["tissue_type"].astype(str).str.lower().isin(TUMOUR_TISSUE)
        ]
        normal_rows = meta.index[
            (meta["patient_id"] == pid)
            & meta["tissue_type"].astype(str).str.lower().isin(NORMAL_TISSUE)
        ]
        if len(tumour_rows) == 0 or len(normal_rows) == 0:
            continue
        tumour_sid = tumour_rows[0]
        normal_sid = normal_rows[0]
        x_n, y_n = embedding_df.loc[normal_sid, coord_cols]
        x_t, y_t = embedding_df.loc[tumour_sid, coord_cols]
        ax.annotate(
            "",
            xy=(x_t, y_t),
            xytext=(x_n, y_n),
            arrowprops=dict(arrowstyle="-|>", color=colour, alpha=0.45, lw=0.7),
        )

    normal_mask = metadata["tissue_type"].astype(str).str.lower().isin(NORMAL_TISSUE)
    tumour_mask = metadata["tissue_type"].astype(str).str.lower().isin(TUMOUR_TISSUE)
    normal_ids = metadata.index[normal_mask]
    tumour_ids = metadata.index[tumour_mask]

    ax.scatter(
        embedding_df.loc[normal_ids, coord_cols[0]],
        embedding_df.loc[normal_ids, coord_cols[1]],
        facecolor="none",
        edgecolor="#4C72B0",
        s=25,
        alpha=0.8,
        label="Normal",
    )
    ax.scatter(
        embedding_df.loc[tumour_ids, coord_cols[0]],
        embedding_df.loc[tumour_ids, coord_cols[1]],
        color="#DD8452",
        s=25,
        alpha=0.8,
        label="Tumour",
    )


def rename_metadata_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with metadata column names shortened for plotting."""
    renamed_cols = [LABEL_MAP.get(col, col) for col in df.columns]
    return df.copy().set_axis(renamed_cols, axis=1)


def plot_metadata_corr_heatmap(
    rho_df: pd.DataFrame,
    pval_bh: pd.DataFrame,
    output_path: Path,
    title: str,
    n_factors: int,
    dpi: int,
):
    """Heatmap of factor-metadata Spearman correlations with BH significance."""
    rho_plot = rename_metadata_labels(rho_df)
    pval_bh_plot = rename_metadata_labels(pval_bh)

    n_metadata_vars = rho_plot.shape[1]
    fig, ax = plt.subplots(figsize=(max(6, n_metadata_vars * 0.9), n_factors * 0.45 + 1))
    sns.heatmap(
        rho_plot,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )

    for i in range(rho_plot.shape[0]):
        for j in range(rho_plot.shape[1]):
            val = rho_plot.iloc[i, j]
            sig = pval_bh_plot.iloc[i, j] < 0.05
            if pd.isna(val):
                txt = "NA"
                weight = "normal"
            else:
                txt = f"{val:.2f}"
                weight = "bold" if sig else "normal"
            ax.text(
                j + 0.5,
                i + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=7,
                fontweight=weight,
                color="black",
            )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    savefig(fig, str(output_path), dpi=dpi)


def plot_trajectory_embedding(
    embedding_df: pd.DataFrame,
    metadata: pd.DataFrame,
    patient_ids,
    coord_cols: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    subtitle=None,
):
    """Trajectory arrow plot on a 2D embedding."""
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_trajectory_arrows(
        ax,
        embedding_df,
        metadata,
        patient_ids,
        coord_cols,
        colour="steelblue",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if subtitle is not None:
        fig.suptitle(subtitle, fontsize=7, y=0.98)
    ax.legend(loc="upper right")
    savefig(fig, str(output_path), dpi=dpi)


def plot_factor_violins(
    factor_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    tissue_stats: pd.DataFrame,
    output_path: Path,
    n_factors: int,
    dpi: int,
):
    """Violin plots of factor scores split by tissue type."""
    plot_df = factor_scores.copy()
    plot_df["tissue_type"] = metadata.loc[factor_scores.index, "tissue_type"].values
    plot_df["tissue_label"] = plot_df["tissue_type"].astype(str).str.lower().map(
        lambda x: "Tumour" if x in TUMOUR_TISSUE else "Normal"
    )

    n_cols = 4
    n_rows = math.ceil(n_factors / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    palette = {"Normal": "#4C72B0", "Tumour": "#DD8452"}

    for idx, factor in enumerate(factor_scores.columns):
        ax = axes_flat[idx]
        sns.violinplot(
            data=plot_df,
            x="tissue_label",
            y=factor,
            hue="tissue_label",
            order=["Normal", "Tumour"],
            palette=palette,
            cut=0,
            ax=ax,
            inner=None,
            legend=False,
        )
        sns.stripplot(
            data=plot_df,
            x="tissue_label",
            y=factor,
            order=["Normal", "Tumour"],
            color="black",
            alpha=0.35,
            size=7,
            jitter=True,
            ax=ax,
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        bonf_p = tissue_stats.loc[factor, "corrected_p_bonf"]
        bh_p = tissue_stats.loc[factor, "corrected_p_bh"]
        ax.text(
            0.5,
            0.97,
            f"Bonf p={bonf_p:.3f}  BH p={bh_p:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6,
        )
        ax.set_title(factor)
        ax.set_xlabel("")
        ax.set_ylabel("Factor score")

    for idx in range(n_factors, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    savefig(fig, str(output_path), dpi=dpi)


def main():
    """Run factor QC and characterisation."""
    args = parse_arguments()
    config = load_config(args.config)

    results_dir = Path(config["paths"]["results"])
    figures_dir = Path(config["paths"]["figures"])
    dpi = config["figures"]["dpi"]

    logger = setup_logging(Path("log/04_factor_qc.log"))
    logger.info("Starting factor QC (Stage 04)")

    factor_scores_path = results_dir / "mofa" / "factor_scores.csv"
    r2_path = results_dir / "mofa" / "r2_per_factor_per_view.csv"
    metadata_path = Path(config["dataset"]["ingestion_dir"]) / "metadata.csv"

    for path in (factor_scores_path, r2_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(str(path.resolve()))

    factor_scores = load_csv(str(factor_scores_path))
    r2_df = load_csv(str(r2_path))
    metadata = load_csv(str(metadata_path))

    factor_names = factor_scores.columns.tolist()
    logger.info("Retained factor names: %s", factor_names)
    logger.info("Retained factor count: %s", len(factor_names))

    validate_inputs(factor_scores, metadata, logger)

    delta_z, tumour_factors, _normal_factors = compute_delta_z(
        factor_scores, metadata, logger
    )
    delta_z_path = results_dir / "delta" / "delta_z.csv"
    save_csv(delta_z, str(delta_z_path), index=True)
    logger.info("ΔZ saved: shape %s", delta_z.shape)

    tissue_stats = compute_tissue_stats(delta_z, logger)
    factor_qc_dir = results_dir / "factor_qc"
    save_csv(tissue_stats, str(factor_qc_dir / "factor_tissue_stats.csv"), index=True)

    encoded_meta = encode_tumour_metadata(metadata, logger)

    rho_dz, pval_dz, pval_dz_bonf, pval_dz_bh = compute_metadata_correlations(
        delta_z, encoded_meta, logger, analysis_label="ΔZ-based"
    )
    rho_t, pval_t, pval_t_bonf, pval_t_bh = compute_metadata_correlations(
        tumour_factors, encoded_meta, logger, analysis_label="tumour-score"
    )

    save_csv(rho_dz, str(factor_qc_dir / "metadata_corr_delta_rho.csv"), index=True)
    save_csv(pval_dz, str(factor_qc_dir / "metadata_corr_delta_pval.csv"), index=True)
    save_csv(
        pval_dz_bonf,
        str(factor_qc_dir / "metadata_corr_delta_pval_bonf.csv"),
        index=True,
    )
    save_csv(
        pval_dz_bh,
        str(factor_qc_dir / "metadata_corr_delta_pval_bh.csv"),
        index=True,
    )
    save_csv(rho_t, str(factor_qc_dir / "metadata_corr_tumour_rho.csv"), index=True)
    save_csv(pval_t, str(factor_qc_dir / "metadata_corr_tumour_pval.csv"), index=True)
    save_csv(
        pval_t_bonf,
        str(factor_qc_dir / "metadata_corr_tumour_pval_bonf.csv"),
        index=True,
    )
    save_csv(
        pval_t_bh,
        str(factor_qc_dir / "metadata_corr_tumour_pval_bh.csv"),
        index=True,
    )

    pca_embedding, evr = fit_pca(factor_scores, config)
    umap_embedding = fit_umap(factor_scores, config)
    save_csv(pca_embedding, str(factor_qc_dir / "pca_all_samples.csv"), index=True)
    save_csv(umap_embedding, str(factor_qc_dir / "umap_all_samples.csv"), index=True)

    set_style()
    fig_factor_qc = figures_dir / "factor_qc"
    n_factors = len(factor_names)
    patient_ids = delta_z.index.tolist()

    plot_metadata_corr_heatmap(
        rho_dz,
        pval_dz_bh,
        fig_factor_qc / "metadata_corr_delta.pdf",
        "Factor–metadata correlations (Spearman ρ, ΔZ-based)\nbold = BH FDR < 0.05",
        n_factors,
        dpi,
    )
    plot_metadata_corr_heatmap(
        rho_t,
        pval_t_bh,
        fig_factor_qc / "metadata_corr_tumour.pdf",
        "Factor–metadata correlations (Spearman ρ, tumour scores)\nbold = BH FDR < 0.05",
        n_factors,
        dpi,
    )
    plot_trajectory_embedding(
        pca_embedding,
        metadata,
        patient_ids,
        ["pca1", "pca2"],
        f"PC1 ({evr[0] * 100:.1f}% var)",
        f"PC2 ({evr[1] * 100:.1f}% var)",
        "Tumourigenesis trajectories — PCA of MOFA factor scores Z (all factors)",
        fig_factor_qc / "trajectory_arrows_pca.pdf",
        dpi,
    )
    plot_trajectory_embedding(
        umap_embedding,
        metadata,
        patient_ids,
        ["umap1", "umap2"],
        "UMAP 1",
        "UMAP 2",
        "Tumourigenesis trajectories — UMAP of MOFA factor scores Z (all factors)",
        fig_factor_qc / "trajectory_arrows_umap.pdf",
        dpi,
    )

    # Factor1 captures the universal tumour-normal signal and dominates PC1.
    # Excluding it projects out that axis, revealing variation among transformation
    # modes in the remaining factors.
    factor1_subtitle = (
        "Factor1 (tumour-normal axis) removed to reveal transformation substructure"
    )
    factor_scores_sub = factor_scores.drop(columns=["Factor1"])
    pca_sub_embedding, evr_sub = fit_pca(factor_scores_sub, config)
    umap_sub_embedding = fit_umap(factor_scores_sub, config)
    plot_trajectory_embedding(
        pca_sub_embedding,
        metadata,
        patient_ids,
        ["pca1", "pca2"],
        f"PC1 ({evr_sub[0] * 100:.1f}% var)",
        f"PC2 ({evr_sub[1] * 100:.1f}% var)",
        "Tumourigenesis trajectories — PCA of MOFA factor scores Z (Factor1 excluded)",
        fig_factor_qc / "trajectory_arrows_pca_no_factor1.pdf",
        dpi,
        subtitle=factor1_subtitle,
    )
    plot_trajectory_embedding(
        umap_sub_embedding,
        metadata,
        patient_ids,
        ["umap1", "umap2"],
        "UMAP 1",
        "UMAP 2",
        "Tumourigenesis trajectories — UMAP of MOFA factor scores Z (Factor1 excluded)",
        fig_factor_qc / "trajectory_arrows_umap_no_factor1.pdf",
        dpi,
        subtitle=factor1_subtitle,
    )
    plot_factor_violins(
        factor_scores,
        metadata,
        tissue_stats,
        fig_factor_qc / "factor_violin_by_tissue.pdf",
        n_factors,
        dpi,
    )

    logger.info("Factor QC complete.")


if __name__ == "__main__":
    main()
