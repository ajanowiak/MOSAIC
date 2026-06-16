# 07_interpret.py
"""
Biological interpretation of MOFA factors and transformation clusters.

Pipeline position: Stage 07 — depends on Stage 03 (MOFA) and Stage 06 (Clustering).

Section A — Factor interpretation: runs preranked GSEA on each MOFA factor's RNA
loading vector to identify biological programmes concentrated at either pole of
the factor. Independent of ΔZ and clustering.

Section B — Cluster interpretation: computes per-patient ΔExpression (tumour minus
normal, full gene set), runs one-vs-rest Mann-Whitney U per cluster, then preranked
GSEA on median_diff to test whether transformation clusters are transcriptionally
distinct.

Inputs:
  results/mofa/factor_loadings_rna.csv
  results/ingestion/rna.csv
  results/ingestion/metadata.csv
  results/clustering/cluster_assignments.csv
  results/clustering/cluster_sizes.csv
  config/config.yml

Outputs:
  results/interpretation/factor_interpretation/gsea_results.csv
  results/interpretation/cluster_interpretation/de_results.csv
  results/interpretation/cluster_interpretation/gsea_clusters.csv
  figures/interpretation/factor_interpretation/gsea_heatmap.pdf
  figures/interpretation/cluster_interpretation/gsea_cluster_heatmap.pdf
  figures/interpretation/cluster_interpretation/volcano_panel.pdf
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))

from utils.io import load_config, load_csv, save_csv
from utils.plotting import set_style, savefig

# Volcano colours are fixed across all cluster panels so up/down mean the same thing everywhere.
VOLCANO_COLOR_UP = "#C44E52"
VOLCANO_COLOR_DOWN = "#4C72B0"
VOLCANO_COLOR_NS = "#D3D3D3"
HEATMAP_GRID_COLOR = "#E0E0E0"


def setup_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
    )


logger = logging.getLogger("07_interpret")


def get_or_cache_gene_sets(library_name: str, cache_dir: str, organism: str = "Human") -> dict:
    """Fetch a gseapy gene set library, writing a JSON cache to avoid repeat downloads.

    Args:
        library_name: Name recognised by gseapy.get_library().
        cache_dir: Directory for cached JSON files.
        organism: Organism string passed to gseapy.

    Returns:
        Dict mapping pathway name to list of gene symbols.
    """
    import gseapy as gp

    cache_path = Path(cache_dir) / f"{library_name}.json"
    if cache_path.exists():
        logger.info("Loading cached gene sets: %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)
    logger.info("Downloading gene set library: %s", library_name)
    gene_sets = gp.get_library(library_name, organism=organism)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(gene_sets, f)
    logger.info("Cached to %s", cache_path)
    return gene_sets


def run_prerank_gsea(
    ranked_series: pd.Series,
    gene_sets: dict,
    config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Run gseapy preranked GSEA and return a normalised result table.

    Args:
        ranked_series: pd.Series with gene symbols as index and numeric ranking
            scores as values. Higher = more positively associated with the
            comparison. Sorted internally; duplicates dropped.
        gene_sets: Dict mapping pathway name to list of gene symbols.
        config: Pipeline config dict (reads interpretation sub-section).
        seed: Random seed for permutation testing.

    Returns:
        DataFrame with columns Term, NES, NOM_pval, FDR_qval, Lead_genes.
        Returns empty DataFrame on any failure — never raises.
    """
    import gseapy as gp

    ranked = ranked_series.sort_values(ascending=False).dropna()
    ranked = ranked[~ranked.index.duplicated(keep="first")]

    min_size = config["interpretation"]["gsea_min_size"]
    if len(ranked) < min_size * 2:
        logger.warning("Ranked list too short for GSEA: %d genes", len(ranked))
        return pd.DataFrame()

    try:
        res = gp.prerank(
            rnk=ranked,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=config["interpretation"]["gsea_max_size"],
            permutation_num=config["interpretation"]["gsea_permutations"],
            seed=seed,
            verbose=False,
            outdir=None,
            no_plot=True,
        )
        # gseapy res2d structure (integer index):
        #   Name (method="prerank"), Term (pathway), ES, NES, NOM p-val,
        #   FDR q-val, FWER p-val, Tag %, Gene %, Lead_genes
        # ES and NES are stored as object dtype in some gseapy versions.
        df = res.res2d.copy()

        # gseapy column names vary by version; map them to a fixed schema.
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(" ", "_").replace("-", "_").strip()
            if cl == "term":
                col_map[col] = "Term"
            elif cl == "nes":
                col_map[col] = "NES"
            elif cl in ("nom_p_val", "nom_pval") or (cl.startswith("nom") and "p" in cl):
                col_map[col] = "NOM_pval"
            elif "fdr" in cl and "q" in cl:
                col_map[col] = "FDR_qval"
            elif cl == "lead_genes":
                col_map[col] = "Lead_genes"
        df = df.rename(columns=col_map)

        # Coerce numeric columns stored as strings.
        for num_col in ("NES", "NOM_pval", "FDR_qval"):
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        for required in ["Term", "NES", "FDR_qval"]:
            if required not in df.columns:
                logger.warning(
                    "gseapy output missing '%s'. Columns present: %s. "
                    "Returning empty DataFrame.",
                    required,
                    df.columns.tolist(),
                )
                return pd.DataFrame()

        # Ensure NOM_pval and Lead_genes exist even if column was absent.
        if "NOM_pval" not in df.columns:
            df["NOM_pval"] = np.nan
        if "Lead_genes" not in df.columns:
            df["Lead_genes"] = ""

        return df[["Term", "NES", "NOM_pval", "FDR_qval", "Lead_genes"]].reset_index(drop=True)

    except Exception as exc:
        logger.warning("GSEA failed: %s", exc)
        return pd.DataFrame()


def _add_heatmap_grid(ax, n_rows: int, n_cols: int) -> None:
    """Draw a light grid over the heatmap so sparse tiles are easier to read."""
    for x in range(n_cols + 1):
        ax.axvline(x, color=HEATMAP_GRID_COLOR, linewidth=0.6, zorder=5)
    for y in range(n_rows + 1):
        ax.axhline(y, color=HEATMAP_GRID_COLOR, linewidth=0.6, zorder=5)


def run_factor_gsea(rna_loadings: pd.DataFrame, all_gene_sets: dict, config: dict) -> pd.DataFrame:
    """Run preranked GSEA on each MOFA factor's RNA loading vector.

    Returns long-format DataFrame with columns: factor, Term, NES, NOM_pval,
    FDR_qval, Lead_genes.
    """
    fdr_thr = config["interpretation"]["gsea_fdr_threshold"]
    factors = rna_loadings.columns.tolist()
    logger.info("Factor interpretation: running GSEA on %d factors", len(factors))

    all_results = []
    for factor_col in factors:
        ranked_series = rna_loadings[factor_col].rename(factor_col)
        gsea_df = run_prerank_gsea(ranked_series, all_gene_sets, config, seed=42)
        if gsea_df.empty:
            logger.warning("No GSEA results for %s", factor_col)
            continue
        gsea_df.insert(0, "factor", factor_col)
        all_results.append(gsea_df)

    if all_results:
        gsea_results = pd.concat(all_results, ignore_index=True)
    else:
        gsea_results = pd.DataFrame(
            columns=["factor", "Term", "NES", "NOM_pval", "FDR_qval", "Lead_genes"]
        )

    n_factors_sig = gsea_results[gsea_results["FDR_qval"] < fdr_thr]["factor"].nunique()
    logger.info(
        "Factor GSEA complete: %d total rows, %d factors with ≥1 pathway at FDR<%.2f",
        len(gsea_results),
        n_factors_sig,
        fdr_thr,
    )
    return gsea_results


def compute_delta_expr(rna: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient ΔExpression = tumour RNA - normal RNA.

    Uses patient_id from metadata to align samples. Tissue type is read from
    the tissue_type column; no string manipulation of sample IDs is used.

    Args:
        rna: DataFrame with sample_id as index, gene symbols as columns,
            log2(TPM+1) values.
        metadata: DataFrame with sample_id as index, columns including
            patient_id and tissue_type.

    Returns:
        DataFrame of shape (n_patients, n_genes) with patient_id as index.
    """
    # tissue_type values in metadata may be spelled "tumor" or "tumour".
    tumour_vals = {"tumor", "tumour"}
    normal_vals = {"normal"}

    tissue = metadata["tissue_type"].str.lower()
    tumour_meta = metadata[tissue.isin(tumour_vals)]
    normal_meta = metadata[tissue.isin(normal_vals)]

    tumour_rna = rna.loc[rna.index.intersection(tumour_meta.index)].copy()
    tumour_rna.index = tumour_meta.loc[tumour_rna.index, "patient_id"]

    normal_rna = rna.loc[rna.index.intersection(normal_meta.index)].copy()
    normal_rna.index = normal_meta.loc[normal_rna.index, "patient_id"]

    shared = tumour_rna.index.intersection(normal_rna.index)
    delta = tumour_rna.loc[shared] - normal_rna.loc[shared]

    nan_count = delta.isna().sum().sum()
    logger.info(
        "ΔExpression matrix: shape=%s, n_patients=%d, n_genes=%d, NaN=%d",
        delta.shape,
        delta.shape[0],
        delta.shape[1],
        nan_count,
    )
    return delta


def run_de_one_vs_rest(
    delta_expr: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    de_fdr_threshold: float,
) -> pd.DataFrame:
    """One-vs-rest Mann-Whitney U test on ΔExpression per cluster.

    Returns long-format DataFrame with columns:
        cluster, gene, median_diff, raw_pval, fdr_bh
    """
    import scipy
    from statsmodels.stats.multitest import multipletests

    logger.info("scipy version: %s", scipy.__version__)

    from scipy.stats import mannwhitneyu

    cluster_labels = sorted(cluster_assignments["cluster"].unique())
    all_de = []

    for c in cluster_labels:
        mask_in = (cluster_assignments["cluster"] == c).values
        mask_out = ~mask_in
        n_in = mask_in.sum()
        n_out = mask_out.sum()

        arr_in = delta_expr.values[mask_in]
        arr_out = delta_expr.values[mask_out]

        # scipy 1.8+ supports vectorised Mann-Whitney U along axis=0.
        try:
            result = mannwhitneyu(arr_in, arr_out, axis=0, alternative="two-sided")
            raw_pvals = result.pvalue
        except TypeError:
            logger.warning(
                "Vectorised mannwhitneyu failed (scipy < 1.8?). "
                "Falling back to per-gene loop — this will be slow."
            )
            raw_pvals = np.array(
                [
                    mannwhitneyu(arr_in[:, g], arr_out[:, g], alternative="two-sided").pvalue
                    for g in range(arr_in.shape[1])
                ]
            )

        median_diff = np.median(arr_in, axis=0) - np.median(arr_out, axis=0)

        _, fdr, _, _ = multipletests(raw_pvals, method="fdr_bh")

        n_up = ((fdr < de_fdr_threshold) & (median_diff > 0)).sum()
        n_dn = ((fdr < de_fdr_threshold) & (median_diff < 0)).sum()
        logger.info(
            "Cluster %d (n=%d vs %d): %d up, %d down at FDR<%.2f",
            c,
            n_in,
            n_out,
            n_up,
            n_dn,
            de_fdr_threshold,
        )

        cluster_de = pd.DataFrame(
            {
                "cluster": c,
                "gene": delta_expr.columns,
                "median_diff": median_diff,
                "raw_pval": raw_pvals,
                "fdr_bh": fdr,
            }
        )
        all_de.append(cluster_de)

    de_results = pd.concat(all_de, ignore_index=True)
    total_sig = (de_results["fdr_bh"] < de_fdr_threshold).sum()
    logger.info(
        "DE complete: %d total gene-cluster rows, %d significant at FDR<%.2f",
        len(de_results),
        total_sig,
        de_fdr_threshold,
    )
    return de_results


def run_cluster_gsea(
    de_results: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    all_gene_sets: dict,
    config: dict,
) -> pd.DataFrame:
    """Run preranked GSEA per cluster using median_diff as the ranking score.

    Genes ranked by median_diff (cluster vs rest): positive = higher tumour-vs-normal
    shift in the cluster relative to all other patients.

    Returns long-format DataFrame with columns: cluster, Term, NES, NOM_pval,
    FDR_qval, Lead_genes.
    """
    fdr_thr = config["interpretation"]["gsea_fdr_threshold"]
    cluster_labels = sorted(cluster_assignments["cluster"].unique())

    all_gsea = []
    for c in cluster_labels:
        sub_de = de_results[de_results["cluster"] == c]
        ranked_series = pd.Series(
            sub_de["median_diff"].values,
            index=sub_de["gene"].values,
        )
        gsea_df = run_prerank_gsea(ranked_series, all_gene_sets, config, seed=42 + c)
        if gsea_df.empty:
            logger.warning("No GSEA results for cluster %d", c)
            continue
        gsea_df.insert(0, "cluster", c)
        all_gsea.append(gsea_df)

    if all_gsea:
        gsea_clusters = pd.concat(all_gsea, ignore_index=True)
    else:
        gsea_clusters = pd.DataFrame(
            columns=["cluster", "Term", "NES", "NOM_pval", "FDR_qval", "Lead_genes"]
        )

    for c in cluster_labels:
        n_sig = ((gsea_clusters["cluster"] == c) & (gsea_clusters["FDR_qval"] < fdr_thr)).sum()
        logger.info("Cluster %d: %d significant pathways at FDR<%.2f", c, n_sig, fdr_thr)

    return gsea_clusters


def plot_gsea_heatmap(
    gsea_results: pd.DataFrame,
    rna_loadings: pd.DataFrame,
    config: dict,
    out_path: str,
) -> None:
    """Heatmap of all significant pathways × MOFA factors (NES from RNA loadings)."""
    fdr_thr = config["interpretation"]["gsea_fdr_threshold"]
    factors = rna_loadings.columns.tolist()

    sig = gsea_results[gsea_results["FDR_qval"] < fdr_thr].copy()
    display_pathways = sig["Term"].unique().tolist()

    n_factors = len(factors)
    n_pathways = len(display_pathways)

    fig, ax = plt.subplots(
        figsize=(max(7, n_factors * 0.9), max(5, n_pathways * 0.35 + 2))
    )

    if n_pathways == 0:
        ax.set_title(
            f"No significant factor enrichments at FDR < {fdr_thr}",
            fontsize=9,
        )
        logger.error("Factor interpretation: no significant pathways — saving blank figure.")
        savefig(fig, out_path)
        return

    # rows = pathways, columns = factors
    pivot = sig.pivot_table(
        index="Term", columns="factor", values="NES", aggfunc="first"
    ).reindex(columns=factors)
    pivot = pivot.reindex(display_pathways)

    yticklabels = [t[:45] for t in display_pathways]

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0,
        mask=pivot.isnull(),
        cbar_kws={"label": "NES (RNA loading weights)"},
        xticklabels=factors,
        yticklabels=yticklabels,
    )
    _add_heatmap_grid(ax, len(display_pathways), len(factors))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"MOFA factor pathway enrichment (GSEA on RNA loadings, FDR<{fdr_thr})",
        fontsize=9,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_gsea_cluster_heatmap(
    gsea_clusters: pd.DataFrame,
    cluster_sizes: pd.DataFrame,
    config: dict,
    out_path: str,
) -> None:
    """Heatmap of all significant pathways × clusters (NES from ΔExpression)."""
    fdr_thr = config["interpretation"]["gsea_fdr_threshold"]
    cluster_labels = sorted(gsea_clusters["cluster"].unique())
    k_final = len(cluster_labels)

    sig = gsea_clusters[gsea_clusters["FDR_qval"] < fdr_thr].copy()

    # keep a placeholder row when a cluster has no significant pathways
    selected_rows = []
    for c in cluster_labels:
        sig_c = sig[sig["cluster"] == c]
        if sig_c.empty:
            placeholder = pd.DataFrame(
                [{"cluster": c, "Term": "No significant enrichment", "NES": 0.0}]
            )
            selected_rows.append(placeholder)
        else:
            selected_rows.append(sig_c)

    selected = pd.concat(selected_rows, ignore_index=True)
    display_pathways = selected["Term"].unique().tolist()

    pivot = selected.pivot_table(
        index="Term", columns="cluster", values="NES", aggfunc="first"
    ).reindex(columns=cluster_labels)
    pivot = pivot.reindex(display_pathways)

    n_pathways = len(display_pathways)
    fig, ax = plt.subplots(
        figsize=(max(5, k_final * 2.2), max(6, n_pathways * 0.4 + 2))
    )

    size_map = cluster_sizes.set_index("cluster")["n_patients"]
    xtick_labels = [f"Cluster {c}\n(n={size_map.get(c, '?')})" for c in cluster_labels]
    yticklabels = [t[:45] for t in display_pathways]

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0,
        mask=pivot.isnull(),
        cbar_kws={"label": "NES (ΔExpression, one-vs-rest)"},
        xticklabels=xtick_labels,
        yticklabels=yticklabels,
    )
    _add_heatmap_grid(ax, len(display_pathways), len(cluster_labels))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"Cluster pathway enrichment — ΔExpression GSEA (FDR<{fdr_thr})",
        fontsize=9,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def plot_volcano_panel(
    de_results: pd.DataFrame,
    cluster_sizes: pd.DataFrame,
    config: dict,
    out_path: str,
) -> None:
    """Volcano plot grid: one panel per cluster showing differential ΔExpression."""
    de_fdr_thr = config["interpretation"]["de_fdr_threshold"]
    cluster_labels = sorted(de_results["cluster"].unique())
    k_final = len(cluster_labels)
    size_map = cluster_sizes.set_index("cluster")["n_patients"]

    fig, axes = plt.subplots(1, k_final, figsize=(5 * k_final, 5), squeeze=False)
    axes = axes[0]

    for idx, c in enumerate(cluster_labels):
        ax = axes[idx]
        sub = de_results[de_results["cluster"] == c].copy()
        sub["fdr_bh_clipped"] = sub["fdr_bh"].clip(lower=1e-300)

        x = sub["median_diff"].values
        y = -np.log10(sub["fdr_bh_clipped"].values)

        sig_up = (sub["fdr_bh"] < de_fdr_thr) & (sub["median_diff"] > 0)
        sig_dn = (sub["fdr_bh"] < de_fdr_thr) & (sub["median_diff"] < 0)

        colours = np.where(
            sig_up.values,
            VOLCANO_COLOR_UP,
            np.where(sig_dn.values, VOLCANO_COLOR_DOWN, VOLCANO_COLOR_NS),
        )

        ax.scatter(x, y, c=colours, s=8, alpha=0.6, rasterized=True)
        ax.axhline(-np.log10(de_fdr_thr), color="grey", lw=0.8, ls="--")
        ax.axvline(-0.5, color="grey", lw=0.8, ls="--")
        ax.axvline(0.5, color="grey", lw=0.8, ls="--")

        # label the strongest up and down genes among those passing FDR
        sig_genes = sub[sub["fdr_bh"] < de_fdr_thr].copy()
        sig_genes_sorted = sig_genes.reindex(
            sig_genes["median_diff"].abs().sort_values(ascending=False).index
        )
        top_up = sig_genes_sorted[sig_genes_sorted["median_diff"] > 0].head(3)
        top_dn = sig_genes_sorted[sig_genes_sorted["median_diff"] < 0].head(2)
        to_label = pd.concat([top_up, top_dn])

        for _, row in to_label.iterrows():
            gx = row["median_diff"]
            gy = -np.log10(max(row["fdr_bh_clipped"], 1e-300))
            ax.annotate(
                row["gene"],
                xy=(gx, gy),
                xytext=(gx + 0.1 * np.sign(gx), gy + 0.5),
                fontsize=6,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
                clip_on=True,
            )

        n_up = sig_up.sum()
        n_dn = sig_dn.sum()
        ax.text(
            0.97,
            0.97,
            f"↑{n_up}  ↓{n_dn}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

        ax.set_xlabel("Median ΔExpr: cluster - rest (log2 TPM)", fontsize=8)
        if idx == 0:
            ax.set_ylabel("-log10(FDR)", fontsize=8)
        n_patients = size_map.get(c, "?")
        ax.set_title(f"Cluster {c} (n={n_patients})", fontsize=8)

    legend_handles = [
        mpatches.Patch(color=VOLCANO_COLOR_UP, label="Higher ΔExpr in cluster"),
        mpatches.Patch(color=VOLCANO_COLOR_DOWN, label="Lower ΔExpr in cluster"),
        mpatches.Patch(color=VOLCANO_COLOR_NS, label="Not significant"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        fontsize=7,
        frameon=False,
    )

    fig.suptitle(
        "Differential ΔExpression per cluster (one-vs-rest, Mann-Whitney U)",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, out_path)


def validate_inputs(paths: list[str]) -> None:
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required input not found: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSAIC Stage 07 — Biological Interpretation")
    parser.add_argument("--config", required=True, help="Path to config.yml")
    args = parser.parse_args()

    setup_logging("log/07_interpret.log")
    logger.info("Stage 07 — Biological Interpretation")

    config = load_config(args.config)
    set_style()

    cache_dir = config["interpretation"]["gene_set_cache_dir"]
    de_fdr_thr = config["interpretation"]["de_fdr_threshold"]

    input_paths = [
        "results/mofa/factor_loadings_rna.csv",
        "results/ingestion/rna.csv",
        "results/ingestion/metadata.csv",
        "results/clustering/cluster_assignments.csv",
        "results/clustering/cluster_sizes.csv",
    ]
    validate_inputs(input_paths)

    all_gene_sets: dict = {}
    for lib in config["interpretation"]["gsea_libraries"]:
        all_gene_sets.update(get_or_cache_gene_sets(lib, cache_dir))
    logger.info("Gene sets loaded: %d total", len(all_gene_sets))

    logger.info("=== Section A: Factor interpretation ===")

    rna_loadings = load_csv("results/mofa/factor_loadings_rna.csv")
    logger.info(
        "RNA loadings: %d genes × %d factors. Factors: %s",
        rna_loadings.shape[0],
        rna_loadings.shape[1],
        rna_loadings.columns.tolist(),
    )

    gsea_results = run_factor_gsea(rna_loadings, all_gene_sets, config)

    save_csv(gsea_results, "results/interpretation/factor_interpretation/gsea_results.csv", index=False)

    Path("figures/interpretation/factor_interpretation").mkdir(parents=True, exist_ok=True)
    plot_gsea_heatmap(
        gsea_results,
        rna_loadings,
        config,
        "figures/interpretation/factor_interpretation/gsea_heatmap.pdf",
    )

    logger.info("=== Section B: Cluster interpretation ===")

    rna = load_csv("results/ingestion/rna.csv")
    metadata = load_csv("results/ingestion/metadata.csv")
    cluster_assignments = load_csv("results/clustering/cluster_assignments.csv")
    cluster_sizes = pd.read_csv("results/clustering/cluster_sizes.csv")

    delta_expr = compute_delta_expr(rna, metadata)

    shared_patients = delta_expr.index.intersection(cluster_assignments.index)
    n_missing = len(cluster_assignments) - len(shared_patients)
    if n_missing > 0:
        logger.warning(
            "%d cluster patients not found in RNA data and will be excluded from cluster interpretation.",
            n_missing,
        )
    delta_expr = delta_expr.loc[shared_patients]
    cluster_assignments = cluster_assignments.loc[shared_patients]

    cluster_sizes_b = cluster_assignments["cluster"].value_counts().sort_index()
    for c, n in cluster_sizes_b.items():
        if n < 10:
            logger.warning(
                "Cluster %d contains only %d patients. DE and GSEA results for "
                "this cluster should be interpreted cautiously. Mann-Whitney U "
                "is underpowered at this sample size.",
                c,
                n,
            )
    logger.info("Cluster sizes entering DE: %s", cluster_sizes_b.to_dict())

    de_results = run_de_one_vs_rest(delta_expr, cluster_assignments, de_fdr_thr)
    save_csv(de_results, "results/interpretation/cluster_interpretation/de_results.csv", index=False)

    gsea_clusters = run_cluster_gsea(de_results, cluster_assignments, all_gene_sets, config)
    save_csv(gsea_clusters, "results/interpretation/cluster_interpretation/gsea_clusters.csv", index=False)

    Path("figures/interpretation/cluster_interpretation").mkdir(parents=True, exist_ok=True)
    plot_gsea_cluster_heatmap(
        gsea_clusters,
        cluster_sizes,
        config,
        "figures/interpretation/cluster_interpretation/gsea_cluster_heatmap.pdf",
    )
    plot_volcano_panel(
        de_results,
        cluster_sizes,
        config,
        "figures/interpretation/cluster_interpretation/volcano_panel.pdf",
    )

    logger.info("Stage 07 complete.")


if __name__ == "__main__":
    main()
