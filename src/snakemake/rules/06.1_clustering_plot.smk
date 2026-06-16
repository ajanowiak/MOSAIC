rule clustering_plot:
    input:
        assignments    = "results/clustering/cluster_assignments.csv",
        norm_all_pca   = "results/delta/repr_comparison/embeddings/norm_all_pca.csv",
        norm_all_umap  = "results/delta/repr_comparison/embeddings/norm_all_umap.csv",
        raw_all_pca    = "results/delta/repr_comparison/embeddings/raw_all_pca.csv",
        raw_all_umap   = "results/delta/repr_comparison/embeddings/raw_all_umap.csv",
        norm_nof1_pca  = "results/delta/repr_comparison/embeddings/norm_nof1_pca.csv",
        norm_nof1_umap = "results/delta/repr_comparison/embeddings/norm_nof1_umap.csv",
        raw_nof1_pca   = "results/delta/repr_comparison/embeddings/raw_nof1_pca.csv",
        raw_nof1_umap  = "results/delta/repr_comparison/embeddings/raw_nof1_umap.csv",
    output:
        fig_umap = "figures/clustering/cluster_umap.pdf",
        fig_pca  = "figures/clustering/cluster_pca.pdf",
    conda:
        "../../env/analysis.yml"
    log:
        "log/06.1_clustering_plot.log"
    params:
        config = "config/config.yml"
    shell:
        "mkdir -p log && python src/py/06.1_clustering_plot.py --config {params.config} > {log} 2>&1"
