rule interpret:
    input:
        loadings_rna = "results/mofa/factor_loadings_rna.csv",
        rna_full     = "results/ingestion/rna.csv",
        metadata     = "results/ingestion/metadata.csv",
        assignments  = "results/clustering/cluster_assignments.csv",
        sizes        = "results/clustering/cluster_sizes.csv",
    output:
        gsea_factors = "results/interpretation/factor_interpretation/gsea_results.csv",
        de_results   = "results/interpretation/cluster_interpretation/de_results.csv",
        gsea_clusters = "results/interpretation/cluster_interpretation/gsea_clusters.csv",
        fig_factors  = "figures/interpretation/factor_interpretation/gsea_heatmap.pdf",
        fig_cluster_heat = "figures/interpretation/cluster_interpretation/gsea_cluster_heatmap.pdf",
        fig_volcano  = "figures/interpretation/cluster_interpretation/volcano_panel.pdf",
    conda:
        "../../env/analysis.yml"
    log:
        "log/07_interpret.log"
    params:
        config = "config/config.yml"
    shell:
        "python src/py/07_interpret.py --config {params.config} > {log} 2>&1"
