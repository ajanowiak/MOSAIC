rule clustering:
    input:
        raw_all   = "results/delta/delta_z.csv",
        raw_nof1  = "results/delta/delta_z_no_f1.csv",
        norm_all  = "results/delta/delta_z_normalized.csv",
        norm_nof1 = "results/delta/delta_z_normalized_no_f1.csv",
        magnitude = "results/delta/delta_magnitude.csv",
        metadata  = "results/ingestion/metadata.csv",
        emb_rp    = expand(
            "results/delta/repr_comparison/embeddings/{name}.csv",
            name=[
                "raw_all_pca", "raw_all_umap",
                "raw_nof1_pca", "raw_nof1_umap",
                "norm_all_pca", "norm_all_umap",
                "norm_nof1_pca", "norm_nof1_umap",
            ],
        ),
    output:
        assignments    = "results/clustering/cluster_assignments.csv",
        all_k          = "results/clustering/all_k_assignments.csv",
        k_final        = "results/clustering/k_final.txt",
        decision_table = "results/clustering/decision_table.csv",
        cross_ari      = "results/clustering/cross_repr_ari.csv",
        centroids_raw  = "results/clustering/cluster_centroids_raw.csv",
        centroids_norm = "results/clustering/cluster_centroids_norm.csv",
        sizes          = "results/clustering/cluster_sizes.csv",
        summary        = "results/clustering/cluster_summary.csv",
        dispersion     = "results/clustering/within_cluster_dispersion.csv",
        centroid_dists = "results/clustering/centroid_distances.csv",
        fig_selection  = "figures/clustering/model_selection.pdf",
        fig_ari        = "figures/clustering/cross_repr_ari.pdf",
        fig_heatmap    = "figures/clustering/delta_heatmap.pdf",
        fig_centroids  = "figures/clustering/cluster_centroids.pdf",
        fig_dispersion = "figures/clustering/dispersion_summary.pdf",
    conda:
        "../../../env/analysis.yml"
    log:
        "log/06_clustering.log"
    params:
        config = "config/config.yml"
    shell:
        "mkdir -p log && python src/py/06_clustering.py --config {params.config} > {log} 2>&1"
