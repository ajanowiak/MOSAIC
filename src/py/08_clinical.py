# 08_clinical.py
"""
Clinical associations for transformation clusters in the MOSAIC pipeline.

Pipeline stage 8. Tests whether ΔZ-derived transformation clusters show
hypothesis-generating associations with survival, driver mutations, clinical
characteristics, and transformation magnitude. No predictive modelling is performed.

Inputs:
  results/clustering/cluster_assignments.csv
  results/delta/delta_magnitude.csv
  results/ingestion/metadata.csv
  config/config.yml

Outputs:
  results/clinical/survival_logrank.csv
  results/clinical/km_at_risk_table.csv
  results/clinical/mutation_enrichment.csv
  results/clinical/clinical_summary.csv
  results/clinical/magnitude_kruskal.csv
  figures/clinical/km_curves.pdf
  figures/clinical/mutation_enrichment_heatmap.pdf
  figures/clinical/clinical_overview_heatmap.pdf
  figures/clinical/magnitude_by_cluster.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_csv, save_csv
from utils.plotting import get_cluster_palette, savefig, set_style
from utils.stats import fisher_enrichment

TUMOUR_TISSUE = {"tumour", "tumor"}
NON_SMOKER_LABEL = (
    "Lifelong non-smoker: Less than 100 cigarettes smoked in lifetime"
)
SMOKING_UNKNOWN_LABEL = "Smoking history not available"


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the clinical associations script."""
    parser = argparse.ArgumentParser(
        description="Clinical associations for MOSAIC transformation clusters."
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
    logger = logging.getLogger("mosaic.clinical")
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


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction to a list of p-values."""
    if not p_values:
        return []
    _, corrected, _, _ = multipletests(p_values, method="fdr_bh")
    return corrected.tolist()


def stage_column_name(metadata: pd.DataFrame) -> str:
    """Return the tumour-stage column name present in metadata."""
    if "tumour_stage" in metadata.columns:
        return "tumour_stage"
    if "tumor_stage" in metadata.columns:
        return "tumor_stage"
    raise KeyError("No tumour_stage or tumor_stage column found in metadata.")


def is_smoker(smoking_status: str) -> bool:
    """Classify ever-smoker status from CPTAC smoking history labels."""
    if pd.isna(smoking_status):
        return False
    if smoking_status in {NON_SMOKER_LABEL, SMOKING_UNKNOWN_LABEL}:
        return False
    return True


def load_clinical_data(
    assignments_path: Path,
    magnitude_path: Path,
    metadata_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load inputs, filter to tumour metadata, and merge on patient_id."""
    assignments = load_csv(str(assignments_path))
    if "cluster" not in assignments.columns:
        raise KeyError("cluster_assignments.csv must contain a 'cluster' column.")
    assignments["cluster"] = assignments["cluster"].astype(int)

    magnitude = load_csv(str(magnitude_path))
    if "magnitude" not in magnitude.columns:
        raise KeyError("delta_magnitude.csv must contain a 'magnitude' column.")

    metadata = pd.read_csv(metadata_path)
    tumour_mask = metadata["tissue_type"].astype(str).str.lower().isin(TUMOUR_TISSUE)
    tumour_meta = metadata.loc[tumour_mask].drop_duplicates(subset="patient_id")
    tumour_meta = tumour_meta.set_index("patient_id")

    logger.info("Total patients in cluster assignments: %d", len(assignments))
    logger.info("Total tumour metadata rows: %d", len(tumour_meta))

    merged = assignments.join(magnitude, how="inner")
    merged = merged.join(tumour_meta, how="inner", rsuffix="_meta")

    unmatched = assignments.index.difference(merged.index)
    if len(unmatched) > 0:
        logger.warning(
            "Cluster assignments could not be matched for %d patients: %s",
            len(unmatched),
            sorted(unmatched.tolist())[:10],
        )

    logger.info("Patients retained after merging: %d", len(merged))
    cluster_sizes = merged["cluster"].value_counts().sort_index()
    for cluster, count in cluster_sizes.items():
        logger.info("Cluster %s: n = %d", cluster, count)

    return merged


def counts_at_tick(kmf: KaplanMeierFitter, tick: float) -> dict[str, int]:
    """Return at-risk, censored, and event counts at a time point (lifelines convention)."""
    event_table_slice = kmf.event_table.assign(at_risk=lambda x: x.at_risk - x.removed)
    if event_table_slice.loc[:tick].empty:
        return {"at_risk": 0, "censored": 0, "events": 0}
    aggregated = event_table_slice.loc[:tick, ["at_risk", "censored", "observed"]].agg(
        {
            "at_risk": lambda x: x.tail(1).values.item(),
            "censored": "sum",
            "observed": "sum",
        }
    )
    return {
        "at_risk": int(aggregated["at_risk"]),
        "censored": int(aggregated["censored"]),
        "events": int(aggregated["observed"]),
    }


def build_at_risk_table(
    kmfs: list[KaplanMeierFitter],
    clusters: list[int],
    time_points: list[float],
) -> pd.DataFrame:
    """Build at-risk table matching lifelines add_at_risk_counts() output."""
    rows = []
    for tick in time_points:
        for kmf, cluster in zip(kmfs, clusters):
            counts = counts_at_tick(kmf, tick)
            rows.append(
                {
                    "time_days": tick,
                    "cluster": cluster,
                    "at_risk": counts["at_risk"],
                    "censored": counts["censored"],
                    "events": counts["events"],
                }
            )
    return pd.DataFrame(rows)


def run_survival_analysis(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    cluster_palette: dict[int, str],
    output_csv: Path,
    at_risk_csv: Path,
    output_fig: Path,
    dpi: int,
    logger: logging.Logger,
) -> tuple[float, float, int]:
    """Kaplan-Meier curves, omnibus log-rank, and conditional pairwise tests."""
    surv = data[[time_col, event_col, "cluster"]].copy()
    surv = surv.dropna(subset=[time_col, event_col])
    surv[event_col] = surv[event_col].astype(bool)

    omnibus = multivariate_logrank_test(
        surv[time_col],
        surv["cluster"],
        event_observed=surv[event_col],
    )
    omnibus_stat = float(omnibus.test_statistic)
    omnibus_p = float(omnibus.p_value)
    logger.info(
        "Omnibus log-rank: statistic=%.4f, p=%.4f (hypothesis-generating associations)",
        omnibus_stat,
        omnibus_p,
    )
    if omnibus_p < 0.05:
        logger.info(
            "Significant omnibus survival difference detected; "
            "independent validation is required."
        )

    rows = [
        {
            "comparison": "all_clusters",
            "test_statistic": omnibus_stat,
            "p_value": omnibus_p,
            "corrected_p": omnibus_p,
        }
    ]

    n_pairwise = 0
    clusters = sorted(surv["cluster"].unique())
    if omnibus_p < 0.05 and len(clusters) >= 2:
        pair_rows = []
        for c1, c2 in combinations(clusters, 2):
            g1 = surv.loc[surv["cluster"] == c1]
            g2 = surv.loc[surv["cluster"] == c2]
            result = logrank_test(
                g1[time_col],
                g2[time_col],
                event_observed_A=g1[event_col],
                event_observed_B=g2[event_col],
            )
            pair_rows.append(
                {
                    "comparison": f"cluster_{c1}_vs_{c2}",
                    "test_statistic": float(result.test_statistic),
                    "p_value": float(result.p_value),
                }
            )
        n_pairwise = len(pair_rows)
        raw_ps = [row["p_value"] for row in pair_rows]
        corrected = benjamini_hochberg(raw_ps)
        for row, corr_p in zip(pair_rows, corrected):
            row["corrected_p"] = corr_p
            rows.append(row)
        logger.info("Performed %d pairwise survival log-rank tests", n_pairwise)
    else:
        logger.info(
            "Omnibus p >= 0.05; skipping pairwise survival comparisons."
        )

    save_csv(pd.DataFrame(rows), str(output_csv), index=False)

    kmfs = fit_km_models(surv, time_col, event_col)
    time_points = plot_km_curves(
        kmfs,
        clusters,
        cluster_palette,
        omnibus_p,
        output_fig,
        dpi,
    )
    at_risk_table = build_at_risk_table(kmfs, clusters, time_points)
    save_csv(at_risk_table, str(at_risk_csv), index=False)
    return omnibus_stat, omnibus_p, n_pairwise


def fit_km_models(
    surv: pd.DataFrame,
    time_col: str,
    event_col: str,
) -> list[KaplanMeierFitter]:
    """Fit Kaplan-Meier models for each cluster."""
    kmfs = []
    for cluster in sorted(surv["cluster"].unique()):
        mask = surv["cluster"] == cluster
        n_patients = int(mask.sum())
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=surv.loc[mask, time_col],
            event_observed=surv.loc[mask, event_col],
            label=f"Cluster {cluster} (n={n_patients})",
        )
        kmfs.append(kmf)
    return kmfs


def plot_km_curves(
    kmfs: list[KaplanMeierFitter],
    clusters: list[int],
    cluster_palette: dict[int, str],
    omnibus_p: float,
    output_path: Path,
    dpi: int,
) -> list[float]:
    """Plot Kaplan-Meier curves with CI and censoring marks (no at-risk table)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for kmf, cluster in zip(kmfs, clusters):
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            show_censors=True,
            color=cluster_palette[cluster],
        )

    ax.set_xlabel("Overall survival (days)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier survival by transformation cluster")
    ax.text(
        0.98,
        0.02,
        f"Log-rank p = {omnibus_p:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout()

    min_time, max_time = ax.get_xlim()
    time_points = [
        float(tick) for tick in ax.get_xticks() if min_time <= tick <= max_time
    ]
    savefig(fig, str(output_path), dpi=dpi)
    return time_points


def run_mutation_enrichment(
    data: pd.DataFrame,
    mutation_cols: list[str],
    output_csv: Path,
    output_fig: Path,
    dpi: int,
    logger: logging.Logger,
) -> int:
    """Fisher's exact tests for driver mutation enrichment per cluster."""
    clusters = sorted(data["cluster"].unique())
    rows = []

    for mutation in mutation_cols:
        if mutation not in data.columns:
            raise KeyError(f"Mutation column not found in metadata: {mutation}")
        mutated = data[mutation].fillna(0).astype(int).astype(bool)

        for cluster in clusters:
            in_cluster = (data["cluster"] == cluster).values
            odds_ratio, p_value = fisher_enrichment(in_cluster, mutated.values)
            rows.append(
                {
                    "mutation": mutation,
                    "cluster": cluster,
                    "odds_ratio": odds_ratio,
                    "p_value": p_value,
                    "n_mutated_in_cluster": int(np.sum(in_cluster & mutated.values)),
                    "n_mutated_outside": int(np.sum(~in_cluster & mutated.values)),
                }
            )

    raw_ps = [row["p_value"] for row in rows]
    corrected = benjamini_hochberg(raw_ps)
    for row, corr_p in zip(rows, corrected):
        row["corrected_p"] = corr_p

    result_df = pd.DataFrame(rows)
    save_csv(result_df, str(output_csv), index=False)

    n_sig = int((result_df["corrected_p"] < 0.05).sum())
    logger.info(
        "Mutation enrichments significant at BH-corrected p < 0.05: %d", n_sig
    )

    plot_mutation_heatmap(result_df, clusters, mutation_cols, output_fig, dpi)
    return n_sig


def format_or_annotation(or_value: float, corrected_p: float) -> str:
    """Format odds ratio with significance suffix for heatmap annotation."""
    if not np.isfinite(or_value) or or_value <= 0:
        return "NA"
    text = f"{or_value:.1f}"
    if pd.notna(corrected_p):
        if corrected_p < 0.01:
            text += "**"
        elif corrected_p < 0.05:
            text += "*"
    return text


def plot_mutation_heatmap(
    enrichment: pd.DataFrame,
    clusters: list[int],
    mutation_cols: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Heatmap of log2(odds ratio) annotated with odds ratios and significance."""
    pivot_or = enrichment.pivot(index="mutation", columns="cluster", values="odds_ratio")
    pivot_p = enrichment.pivot(index="mutation", columns="cluster", values="corrected_p")
    pivot_or = pivot_or.reindex(index=mutation_cols, columns=clusters)

    log2_or = np.log2(pivot_or.astype(float))
    log2_or.replace([np.inf, -np.inf], np.nan, inplace=True)

    annot = np.empty(log2_or.shape, dtype=object)
    for i, mutation in enumerate(mutation_cols):
        for j, cluster in enumerate(clusters):
            or_value = pivot_or.loc[mutation, cluster]
            corr_p = pivot_p.loc[mutation, cluster]
            annot[i, j] = format_or_annotation(float(or_value), corr_p)

    mutation_labels = [
        m.replace("_mutation", "").replace("_fusion", "") for m in mutation_cols
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.heatmap(
        log2_or,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "log2(odds ratio)"},
        ax=ax,
        mask=log2_or.isna(),
        yticklabels=mutation_labels,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mutation")
    ax.set_title("Driver mutation enrichment across transformation clusters")
    fig.text(
        0.5,
        -0.02,
        "Cell labels = odds ratio; * BH-adjusted p < 0.05; ** BH-adjusted p < 0.01",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout()
    savefig(fig, str(output_path), dpi=dpi)


def build_clinical_summary(
    data: pd.DataFrame,
    stage_col: str,
    mutation_cols: list[str],
) -> pd.DataFrame:
    """Descriptive clinical summary table per cluster."""
    stage_order = sorted(data[stage_col].dropna().unique())
    rows = []

    for cluster in sorted(data["cluster"].unique()):
        subset = data.loc[data["cluster"] == cluster]
        n_patients = len(subset)
        row = {
            "cluster": cluster,
            "n_patients": n_patients,
            "median_age": subset["age_at_diagnosis"].median(),
            "pct_female": 100.0 * (subset["sex"].str.lower() == "female").mean(),
            "pct_smoker": 100.0 * subset["smoking_status"].map(is_smoker).mean(),
        }
        for stage in stage_order:
            col_name = f"pct_{stage.replace(' ', '_').lower()}"
            row[col_name] = 100.0 * (subset[stage_col] == stage).mean()
        for mutation in mutation_cols:
            row[f"pct_{mutation}"] = (
                100.0 * subset[mutation].fillna(0).astype(float).mean()
            )
        rows.append(row)

    return pd.DataFrame(rows)


def cluster_column_labels(summary: pd.DataFrame) -> list[str]:
    """Return cluster column labels including sample sizes."""
    return [
        f"Cluster {int(row.cluster)} (n={int(row.n_patients)})"
        for row in summary.itertuples()
    ]


def _overview_section(
    summary: pd.DataFrame,
    row_labels: list[str],
    values: np.ndarray,
    annot: np.ndarray,
    cmap: str,
    cbar_label: str,
    ax: plt.Axes,
    show_ylabel: bool,
    cbar_ax: plt.Axes | None,
) -> None:
    """Draw one labelled section of the clinical overview heatmap."""
    data = pd.DataFrame(values, index=row_labels, columns=cluster_column_labels(summary))
    sns.heatmap(
        data,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar=cbar_ax is not None,
        cbar_ax=cbar_ax,
        cbar_kws={"label": cbar_label} if cbar_ax is not None else None,
        ax=ax,
        yticklabels=row_labels if show_ylabel else False,
    )
    if not show_ylabel:
        ax.set_ylabel("")


def plot_clinical_overview_heatmap(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    """Multi-panel heatmap of raw clinical percentages per cluster."""
    stage_cols = sorted(c for c in summary.columns if c.startswith("pct_stage"))
    mutation_cols = sorted(
        c for c in summary.columns if c.endswith("_mutation") or c.endswith("_fusion")
    )

    demo_rows = ["% Female", "% Smoker"]
    demo_values = np.array(
        [
            summary["pct_female"].values,
            summary["pct_smoker"].values,
        ]
    )
    demo_annot = np.array(
        [
            [f"{val:.0f}%" for val in summary["pct_female"].values],
            [f"{val:.0f}%" for val in summary["pct_smoker"].values],
        ],
        dtype=object,
    )

    stage_row_labels = [
        col.replace("pct_", "").replace("_", " ").title() for col in stage_cols
    ]
    stage_values = summary[stage_cols].values.T
    stage_annot = np.array(
        [[f"{val:.0f}%" for val in row] for row in stage_values], dtype=object
    )

    mutation_row_labels = [
        col.replace("pct_", "").replace("_mutation", "").replace("_fusion", "")
        for col in mutation_cols
    ]
    mutation_values = summary[mutation_cols].values.T
    mutation_annot = np.array(
        [[f"{val:.0f}%" for val in row] for row in mutation_values], dtype=object
    )

    fig = plt.figure(figsize=(8, 7.5))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[20, 1],
        height_ratios=[0.85, 1.0, 1.2],
        hspace=0.35,
        wspace=0.08,
    )
    fig.suptitle("Clinical overview of transformation clusters", fontsize=10, y=0.98)

    ax_demo = fig.add_subplot(gs[0, 0])
    cbar_demo = fig.add_subplot(gs[0, 1])
    ax_stage = fig.add_subplot(gs[1, 0])
    cbar_stage = fig.add_subplot(gs[1, 1])
    ax_mut = fig.add_subplot(gs[2, 0])
    cbar_mut = fig.add_subplot(gs[2, 1])

    _overview_section(
        summary,
        demo_rows,
        demo_values,
        demo_annot,
        cmap="Blues",
        cbar_label="% patients",
        ax=ax_demo,
        show_ylabel=True,
        cbar_ax=cbar_demo,
    )
    ax_demo.set_title("Demographics", fontsize=8, loc="left", pad=2)
    ax_demo.tick_params(axis="x", labelbottom=False)
    ax_demo.set_xlabel("")

    _overview_section(
        summary,
        stage_row_labels,
        stage_values,
        stage_annot,
        cmap="Greens",
        cbar_label="% patients",
        ax=ax_stage,
        show_ylabel=True,
        cbar_ax=cbar_stage,
    )
    ax_stage.set_title("Tumour stage", fontsize=8, loc="left", pad=2)
    ax_stage.tick_params(axis="x", labelbottom=False)
    ax_stage.set_xlabel("")

    _overview_section(
        summary,
        mutation_row_labels,
        mutation_values,
        mutation_annot,
        cmap="Purples",
        cbar_label="% patients",
        ax=ax_mut,
        show_ylabel=True,
        cbar_ax=cbar_mut,
    )
    ax_mut.set_title("Driver mutations", fontsize=8, loc="left", pad=2)
    ax_mut.set_xlabel("Cluster")

    fig.subplots_adjust(left=0.18, right=0.88, top=0.93, bottom=0.06, hspace=0.45)
    savefig(fig, str(output_path), dpi=dpi)


def run_magnitude_analysis(
    data: pd.DataFrame,
    cluster_palette: dict[int, str],
    output_csv: Path,
    output_fig: Path,
    dpi: int,
    logger: logging.Logger,
) -> tuple[float, float]:
    """Kruskal-Wallis and conditional pairwise Mann-Whitney U on ΔZ magnitude."""
    clusters = sorted(data["cluster"].unique())
    groups = [
        data.loc[data["cluster"] == cluster, "magnitude"].dropna().values
        for cluster in clusters
    ]

    if len(groups) < 2:
        logger.warning("Fewer than two clusters with magnitude data; skipping test.")
        h_stat, p_val = np.nan, np.nan
        rows = [
            {
                "comparison": "all_clusters",
                "test_statistic": h_stat,
                "p_value": p_val,
                "corrected_p": p_val,
            }
        ]
    else:
        h_stat, p_val = kruskal(*groups)
        h_stat = float(h_stat)
        p_val = float(p_val)
        logger.info("Kruskal-Wallis magnitude vs cluster: H=%.4f, p=%.4f", h_stat, p_val)

        rows = [
            {
                "comparison": "all_clusters",
                "test_statistic": h_stat,
                "p_value": p_val,
                "corrected_p": p_val,
            }
        ]

        if p_val < 0.05:
            pair_rows = []
            for c1, c2 in combinations(clusters, 2):
                g1 = data.loc[data["cluster"] == c1, "magnitude"].dropna().values
                g2 = data.loc[data["cluster"] == c2, "magnitude"].dropna().values
                stat, p_pair = mannwhitneyu(g1, g2, alternative="two-sided")
                pair_rows.append(
                    {
                        "comparison": f"cluster_{c1}_vs_{c2}",
                        "test_statistic": float(stat),
                        "p_value": float(p_pair),
                    }
                )
            corrected = benjamini_hochberg([row["p_value"] for row in pair_rows])
            for row, corr_p in zip(pair_rows, corrected):
                row["corrected_p"] = corr_p
                rows.append(row)
            logger.info("Performed %d pairwise Mann-Whitney magnitude tests", len(pair_rows))
        else:
            logger.info(
                "Omnibus Kruskal-Wallis p >= 0.05; skipping pairwise magnitude tests."
            )

    save_csv(pd.DataFrame(rows), str(output_csv), index=False)
    plot_magnitude_by_cluster(data, clusters, cluster_palette, h_stat, p_val, output_fig, dpi)
    return h_stat, p_val


def plot_magnitude_by_cluster(
    data: pd.DataFrame,
    clusters: list[int],
    cluster_palette: dict[int, str],
    h_stat: float,
    p_val: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Box plot of ΔZ magnitude by cluster, matching Stage 05 visual style."""
    fig, ax = plt.subplots(figsize=(6, 4))
    cluster_labels = [
        f"Cluster {c}\n(n={int((data['cluster'] == c).sum())})" for c in clusters
    ]
    box_data = [
        data.loc[data["cluster"] == cluster, "magnitude"].dropna().values
        for cluster in clusters
    ]

    bp = ax.boxplot(box_data, tick_labels=cluster_labels, patch_artist=True)
    for patch, cluster in zip(bp["boxes"], clusters):
        colour = cluster_palette[cluster]
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    for median_line in bp["medians"]:
        median_line.set_color("#333333")
        median_line.set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, cluster in enumerate(clusters, start=1):
        vals = data.loc[data["cluster"] == cluster, "magnitude"].dropna().values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            alpha=0.6,
            s=15,
            color=cluster_palette[cluster],
        )

    ax.set_xlabel("Transformation cluster")
    ax.set_ylabel("ΔZ magnitude")
    if np.isfinite(h_stat) and np.isfinite(p_val):
        ax.set_title(
            f"Transformation magnitude by cluster (H={h_stat:.2f}, p={p_val:.3f})"
        )
    else:
        ax.set_title("Transformation magnitude by cluster")
    savefig(fig, str(output_path), dpi=dpi)


def print_validation_summary(
    data: pd.DataFrame,
    omnibus_stat: float,
    omnibus_p: float,
    n_pairwise_survival: int,
    n_sig_mutations: int,
    kw_stat: float,
    kw_p: float,
    figure_paths: list[Path],
) -> None:
    """Print end-of-run validation summary to stdout."""
    print("\n" + "=" * 60)
    print("STAGE 08 VALIDATION SUMMARY")
    print("=" * 60)
    print("\n1. Cluster sizes used in clinical analyses:")
    for cluster, count in data["cluster"].value_counts().sort_index().items():
        print(f"   Cluster {cluster}: n = {count}")
    print(
        f"\n2. Omnibus survival log-rank: "
        f"statistic = {omnibus_stat:.4f}, p = {omnibus_p:.4f}"
    )
    print(f"\n3. Pairwise survival tests performed: {n_pairwise_survival}")
    print(
        f"\n4. Mutation enrichments significant at corrected p < 0.05: "
        f"{n_sig_mutations}"
    )
    print(
        f"\n5. Kruskal-Wallis ΔZ magnitude: "
        f"statistic = {kw_stat:.4f}, p = {kw_p:.4f}"
    )
    print("\n6. Clinical figures:")
    all_ok = True
    for path in figure_paths:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"   [{status}] {path}")
    if all_ok:
        print("\nAll four clinical figures generated successfully.")
    print("=" * 60 + "\n")


def main() -> None:
    """Run the Stage 08 clinical associations pipeline."""
    args = parse_arguments()
    config = load_config(args.config)
    set_style()

    results_dir = Path(config["paths"]["results"]) / "clinical"
    figures_dir = Path(config["paths"]["figures"]) / "clinical"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dpi = config["figures"]["dpi"]
    palette_name = config["figures"]["cluster_palette"]
    clinical_cfg = config["clinical"]

    logger = setup_logging(Path("log") / "08_clinical.log")

    assignments_path = Path(config["paths"]["results"]) / "clustering" / "cluster_assignments.csv"
    magnitude_path = Path(config["paths"]["results"]) / "delta" / "delta_magnitude.csv"
    metadata_path = Path(config["dataset"]["ingestion_dir"]) / "metadata.csv"

    validate_file(assignments_path)
    validate_file(magnitude_path)
    validate_file(metadata_path)

    data = load_clinical_data(assignments_path, magnitude_path, metadata_path, logger)

    n_clusters = data["cluster"].nunique()
    cluster_palette = get_cluster_palette(n_clusters, palette_name)

    time_col = clinical_cfg["survival_time_col"]
    event_col = clinical_cfg["survival_event_col"]
    mutation_cols = clinical_cfg["mutation_cols"]
    stage_col = stage_column_name(data)

    survival_csv = results_dir / "survival_logrank.csv"
    at_risk_csv = results_dir / "km_at_risk_table.csv"
    mutation_csv = results_dir / "mutation_enrichment.csv"
    summary_csv = results_dir / "clinical_summary.csv"
    magnitude_csv = results_dir / "magnitude_kruskal.csv"

    km_fig = figures_dir / "km_curves.pdf"
    mutation_fig = figures_dir / "mutation_enrichment_heatmap.pdf"
    overview_fig = figures_dir / "clinical_overview_heatmap.pdf"
    magnitude_fig = figures_dir / "magnitude_by_cluster.pdf"
    figure_paths = [km_fig, mutation_fig, overview_fig, magnitude_fig]

    omnibus_stat, omnibus_p, n_pairwise = run_survival_analysis(
        data,
        time_col,
        event_col,
        cluster_palette,
        survival_csv,
        at_risk_csv,
        km_fig,
        dpi,
        logger,
    )

    n_sig_mutations = run_mutation_enrichment(
        data,
        mutation_cols,
        mutation_csv,
        mutation_fig,
        dpi,
        logger,
    )

    summary = build_clinical_summary(data, stage_col, mutation_cols)
    save_csv(summary, str(summary_csv), index=False)
    plot_clinical_overview_heatmap(summary, overview_fig, dpi)

    kw_stat, kw_p = run_magnitude_analysis(
        data,
        cluster_palette,
        magnitude_csv,
        magnitude_fig,
        dpi,
        logger,
    )

    print_validation_summary(
        data,
        omnibus_stat,
        omnibus_p,
        n_pairwise,
        n_sig_mutations,
        kw_stat,
        kw_p,
        figure_paths,
    )

    logger.info("Stage 08 complete.")


if __name__ == "__main__":
    main()
