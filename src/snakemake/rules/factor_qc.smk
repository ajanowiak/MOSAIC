rule factor_qc:
    input:
        factor_scores="results/mofa/factor_scores.csv",
        r2="results/mofa/r2_per_factor_per_view.csv",
        metadata="results/ingestion/metadata.csv",
    output:
        delta_z="results/delta/delta_z.csv",
        factor_tissue_stats="results/factor_qc/factor_tissue_stats.csv",
        metadata_corr_delta_rho="results/factor_qc/metadata_corr_delta_rho.csv",
        metadata_corr_tumour_rho="results/factor_qc/metadata_corr_tumour_rho.csv",
        pca_all_samples="results/factor_qc/pca_all_samples.csv",
        umap_all_samples="results/factor_qc/umap_all_samples.csv",
        metadata_corr_delta="figures/factor_qc/metadata_corr_delta.pdf",
        metadata_corr_tumour="figures/factor_qc/metadata_corr_tumour.pdf",
        trajectory_pca="figures/factor_qc/trajectory_arrows_pca.pdf",
        trajectory_umap="figures/factor_qc/trajectory_arrows_umap.pdf",
        trajectory_pca_no_factor1="figures/factor_qc/trajectory_arrows_pca_no_factor1.pdf",
        trajectory_umap_no_factor1="figures/factor_qc/trajectory_arrows_umap_no_factor1.pdf",
        factor_violin="figures/factor_qc/factor_violin_by_tissue.pdf",
    conda:
        "../../env/analysis.yml"
    log:
        "log/factor_qc.log"
    params:
        config="config/config.yml"
    shell:
        "mkdir -p log && python src/py/04_factor_qc.py --config {params.config} > {log} 2>&1"
