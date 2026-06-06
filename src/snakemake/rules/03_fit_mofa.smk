rule fit_mofa:
    input:
        rna="results/preprocessing/rna_preprocessed.csv",
        proteomics="results/preprocessing/proteomics_preprocessed.csv",
        phospho="results/preprocessing/phospho_preprocessed.csv",
    output:
        factor_scores="results/mofa/factor_scores.csv",
        model_elbo="results/mofa/model_elbo.csv",
        r2_per_factor_per_view="results/mofa/r2_per_factor_per_view.csv",
        factor_loadings_rna="results/mofa/factor_loadings_rna.csv",
        factor_loadings_proteomics="results/mofa/factor_loadings_proteomics.csv",
        factor_loadings_phospho="results/mofa/factor_loadings_phospho.csv",
        variance_explained_heatmap="figures/mofa/variance_explained_heatmap.pdf",
        elbo_convergence="figures/mofa/elbo_convergence.pdf",
    conda:
        "../../env/mofa.yml"
    log:
        "log/03_fit_mofa.log"
    params:
        config="config/config.yml"
    shell:
        "mkdir -p log && python src/py/03_fit_mofa.py --config {params.config} > {log} 2>&1"
