rule preprocess:
    input:
        rna="results/ingestion/rna.csv",
        proteomics="results/ingestion/proteomics.csv",
        phospho="results/ingestion/phospho.csv",
        metadata="results/ingestion/metadata.csv",
    output:
        rna_preprocessed="results/preprocessing/rna_preprocessed.csv",
        proteomics_preprocessed="results/preprocessing/proteomics_preprocessed.csv",
        phospho_preprocessed="results/preprocessing/phospho_preprocessed.csv",
        feature_counts="results/preprocessing/feature_counts.json",
        mad_distributions="figures/preprocessing/mad_distributions.pdf",
        sample_correlation="figures/preprocessing/sample_correlation_matrix.pdf",
    conda:
        "../../env/analysis.yml"
    log:
        "log/preprocess.log"
    params:
        config="config/config.yml"
    shell:
        "mkdir -p log && python src/py/02_preprocess.py --config {params.config} > {log} 2>&1"
