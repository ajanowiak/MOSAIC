rule ingest:
    output:
        rna="results/ingestion/rna.csv",
        proteomics="results/ingestion/proteomics.csv",
        phospho="results/ingestion/phospho.csv",
        metadata="results/ingestion/metadata.csv",
    params:
        missing_threshold=config["preprocessing"]["missing_value_threshold"],
    conda:
        "../../../env/mosaic.yaml"
    log:
        "log/01_ingest.log"
    shell:
        """
        mkdir -p log && python src/py/01_ingest.py \
          --output-dir results/ingestion/ \
          --missing-threshold {params.missing_threshold} \
          --log {log}
        """
