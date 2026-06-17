rule clinical:
    input:
        assignments = "results/clustering/cluster_assignments.csv",
        magnitude   = "results/delta/delta_magnitude.csv",
        metadata    = "results/ingestion/metadata.csv",
    output:
        survival    = "results/clinical/survival_logrank.csv",
        at_risk     = "results/clinical/km_at_risk_table.csv",
        mutations   = "results/clinical/mutation_enrichment.csv",
        summary     = "results/clinical/clinical_summary.csv",
        magnitude   = "results/clinical/magnitude_kruskal.csv",
        km          = "figures/clinical/km_curves.pdf",
        mut_fig     = "figures/clinical/mutation_enrichment_heatmap.pdf",
        overview    = "figures/clinical/clinical_overview_heatmap.pdf",
        mag_fig     = "figures/clinical/magnitude_by_cluster.pdf",
    conda:
        "../../../env/analysis.yml"
    log:
        "log/08_clinical.log"
    params:
        config = "config/config.yml"
    shell:
        "mkdir -p log && python src/py/08_clinical.py --config {params.config} > {log} 2>&1"
