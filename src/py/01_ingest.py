# 01_ingest.py
"""
Download CPTAC-LUAD multi-omics data, filter to matched tumour/normal pairs,
preprocess each view, and write aligned CSVs for downstream MOFA analysis.

Pipeline stage 1 (ingestion). Inputs are fetched from the cptac package at runtime.
Outputs: rna.csv, proteomics.csv, phospho.csv, and metadata.csv under --output-dir.
"""

import argparse
import logging
from pathlib import Path

import cptac
import numpy as np
import pandas as pd


CLINICAL_COLUMN_MAP = {
    "age_at_diagnosis": "age",
    "sex": "sex",
    "smoking_status": "tobacco_smoking_history",
    "tumor_stage": "tumor_stage_pathological",
    "overall_survival_days": "Overall survival, days",
    "overall_survival_event": "Survival status (1, dead; 0, alive)",
}

MUTATION_GENES = ("KRAS", "EGFR", "TP53")
METADATA_COLUMNS = [
    "sample_id",
    "patient_id",
    "tissue_type",
    "age_at_diagnosis",
    "sex",
    "smoking_status",
    "tumor_stage",
    "overall_survival_days",
    "overall_survival_event",
    "KRAS_mutation",
    "EGFR_mutation",
    "TP53_mutation",
    "ALK_fusion",
]


def parse_arguments(args=None):
    """Parse CLI arguments for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess CPTAC-LUAD data for the MOSAIC pipeline."
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        required=True,
        help="Drop features missing in more than this fraction of samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ingestion output CSVs.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the log file.",
    )
    return parser.parse_args(args)


def setup_logging(log_path):
    """Configure file and console logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mosaic.ingest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def load_cptac_data():
    """Download CPTAC-LUAD transcriptomics, proteomics, phosphoproteomics, and clinical data."""
    luad = cptac.Luad()
    rna_raw = luad.get_transcriptomics(source="bcm")
    prot_raw = luad.get_proteomics(source="umich")
    phospho_raw = luad.get_phosphoproteomics(source="umich")
    clin_raw = luad.get_clinical(source="mssm")
    mut_raw = luad.get_somatic_mutation(source="harmonized")
    return luad, rna_raw, prot_raw, phospho_raw, clin_raw, mut_raw


def patient_id_from_sample(sample_id):
    """Return the patient ID by stripping the normal-tissue '.N' suffix."""
    if str(sample_id).endswith(".N"):
        return str(sample_id)[:-2]
    return str(sample_id)


def tissue_type_from_sample(sample_id):
    """Classify a sample as tumour or normal based on its ID suffix."""
    return "normal" if str(sample_id).endswith(".N") else "tumor"


def patients_with_tissue(df, tissue_type):
    """Return patient IDs that have the requested tissue type in a view."""
    if tissue_type == "normal":
        return {patient_id_from_sample(s) for s in df.index if str(s).endswith(".N")}
    return {patient_id_from_sample(s) for s in df.index if not str(s).endswith(".N")}


def get_matched_patients(rna_raw, prot_raw, phospho_raw, logger):
    """Identify patients with tumour and normal samples present in all three omics views."""
    views = {"rna": rna_raw, "proteomics": prot_raw, "phospho": phospho_raw}
    tumor_sets = [patients_with_tissue(df, "tumor") for df in views.values()]
    normal_sets = [patients_with_tissue(df, "normal") for df in views.values()]

    all_patients = set.intersection(*tumor_sets, *normal_sets)

    tumor_matched = set.intersection(*tumor_sets)
    normal_matched = set.intersection(*normal_sets)
    dropped = sorted((tumor_matched | normal_matched) - all_patients)

    logger.info(
        "Patients before matched-pair filtering: %d",
        len(tumor_matched),
    )
    logger.info("Patients after matched-pair filtering: %d", len(all_patients))

    for patient in dropped:
        logger.warning(
            "Dropping patient %s: not present as both tumour and normal in all three views",
            patient,
        )

    return sorted(all_patients)


def ordered_sample_ids(matched_patients):
    """Build a consistent sample order: tumour then normal for each matched patient."""
    sample_ids = []
    for patient in matched_patients:
        sample_ids.append(patient)
        sample_ids.append(f"{patient}.N")
    return sample_ids


def flatten_multiindex_columns(df):
    """Flatten MultiIndex column headers to gene names, suffixing duplicate names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()

    seen = {}
    new_columns = []
    for col in df.columns:
        name = str(col[0])
        count = seen.get(name, 0) + 1
        seen[name] = count
        new_columns.append(name if count == 1 else f"{name}_{count}")

    flattened = df.copy()
    flattened.columns = new_columns
    return flattened


def drop_high_missing_features(df, missing_threshold, view_name, logger):
    """Drop features with missing values above the configured fraction."""
    missing_fraction = df.isna().mean(axis=0)
    keep_mask = missing_fraction <= missing_threshold
    n_dropped = int((~keep_mask).sum())
    logger.info("%s: dropped %d features above missing threshold", view_name, n_dropped)
    return df.loc[:, keep_mask]


def median_impute(df, view_name, logger):
    """Median-impute remaining missing values column-wise."""
    n_missing = int(df.isna().sum().sum())
    if n_missing == 0:
        logger.info("%s: imputed 0 values", view_name)
        return df

    imputed = df.copy()
    medians = imputed.median(axis=0, skipna=True)
    imputed = imputed.fillna(medians)
    logger.info("%s: imputed %d values", view_name, n_missing)
    return imputed


def preprocess_view(df, missing_threshold, view_name, logger, log_transform=False):
    """Apply optional log transform, feature filtering, and median imputation."""
    processed = df.copy()
    if log_transform:
        processed = np.log2(processed.astype(float) + 1.0)

    processed = drop_high_missing_features(processed, missing_threshold, view_name, logger)
    processed = median_impute(processed, view_name, logger)
    return processed


def encode_mutation_status(status):
    """Encode cptac mutation status as 1 (mutated), 0 (wildtype), or NaN."""
    if pd.isna(status):
        return np.nan
    if status in {"Wildtype_Tumor", "Wildtype_Normal"}:
        return 0
    if status in {"Single_mutation", "Multiple_mutation"}:
        return 1
    return np.nan


def get_patient_mutation_table(luad):
    """Join driver-gene mutation calls at patient level using the cptac API."""
    return luad.join_metadata_to_mutations(
        metadata_name="clinical",
        metadata_source="mssm",
        metadata_cols=[],
        mutations_source="harmonized",
        mutations_genes=list(MUTATION_GENES),
    )


def get_alk_fusion_by_patient(mut_raw):
    """Return patient-level ALK fusion flags derived from COSMIC fusion annotations."""
    if "COSMIC_fusion_genes" not in mut_raw.columns:
        return {}

    fusion_mask = mut_raw["COSMIC_fusion_genes"].astype(str).str.contains(
        "ALK", regex=False, na=False
    )
    fusion_samples = mut_raw.loc[fusion_mask].index.unique()
    fusion_patients = {patient_id_from_sample(s) for s in fusion_samples}
    return {patient: 1 for patient in fusion_patients}


def map_clinical_column(clin_raw, target_name, source_name, logger):
    """Map a clinical source column to a standard metadata field."""
    if source_name not in clin_raw.columns:
        logger.warning(
            "Clinical column %r not found; filling %s with NaN",
            source_name,
            target_name,
        )
        return pd.Series(np.nan, index=clin_raw.index, dtype=float)

    series = clin_raw[source_name]
    if target_name in {"overall_survival_days", "overall_survival_event"}:
        return pd.to_numeric(series, errors="coerce")
    return series


def build_metadata(clin_raw, luad, mut_raw, sample_ids, logger):
    """Build per-sample metadata aligned to the omics matrices."""
    clinical_fields = {
        target_name: map_clinical_column(clin_raw, target_name, source_name, logger)
        for target_name, source_name in CLINICAL_COLUMN_MAP.items()
    }
    mutation_table = get_patient_mutation_table(luad)
    alk_fusion_patients = get_alk_fusion_by_patient(mut_raw)

    rows = []
    for sample_id in sample_ids:
        patient_id = patient_id_from_sample(sample_id)
        row = {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "tissue_type": tissue_type_from_sample(sample_id),
        }

        for target_name, series in clinical_fields.items():
            row[target_name] = series.get(patient_id, np.nan)

        for gene in MUTATION_GENES:
            col_name = f"{gene}_mutation"
            status_col = f"{gene}_Mutation_Status"
            if patient_id in mutation_table.index and status_col in mutation_table.columns:
                row[col_name] = encode_mutation_status(
                    mutation_table.loc[patient_id, status_col]
                )
            else:
                row[col_name] = np.nan

        if patient_id in mutation_table.index:
            row["ALK_fusion"] = 1 if patient_id in alk_fusion_patients else 0
        else:
            row["ALK_fusion"] = np.nan

        rows.append(row)

    metadata = pd.DataFrame(rows)
    return metadata.set_index("sample_id")[METADATA_COLUMNS[1:]]


def align_views_to_samples(view_dfs, sample_ids):
    """Subset and align all views to the same sample order."""
    aligned = {}
    for view_name, df in view_dfs.items():
        missing_samples = [s for s in sample_ids if s not in df.index]
        if missing_samples:
            raise ValueError(
                f"{view_name} is missing matched samples: {missing_samples[:5]}"
            )
        aligned[view_name] = df.loc[sample_ids]
    return aligned


def set_sample_index(df):
    """Rename the sample index to sample_id for downstream joins."""
    indexed = df.copy()
    indexed.index.name = "sample_id"
    return indexed


def write_outputs(output_dir, rna_df, prot_df, phospho_df, metadata_df):
    """Write aligned ingestion outputs to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("rna.csv", rna_df),
        ("proteomics.csv", prot_df),
        ("phospho.csv", phospho_df),
        ("metadata.csv", metadata_df),
    ]:
        set_sample_index(frame).to_csv(output_dir / name, index_label="sample_id")


def main(args=None):
    """Run CPTAC-LUAD download, preprocessing, and export."""
    parsed = parse_arguments(args)
    logger = setup_logging(parsed.log)

    logger.info("Starting CPTAC-LUAD ingestion")
    logger.info("Missing-value threshold: %.3f", parsed.missing_threshold)

    luad, rna_raw, prot_raw, phospho_raw, clin_raw, mut_raw = load_cptac_data()

    matched_patients = get_matched_patients(rna_raw, prot_raw, phospho_raw, logger)
    sample_ids = ordered_sample_ids(matched_patients)

    rna_flat = flatten_multiindex_columns(rna_raw)
    prot_flat = flatten_multiindex_columns(prot_raw)
    phospho_flat = flatten_multiindex_columns(phospho_raw)

    view_dfs = {
        "rna": preprocess_view(
            rna_flat,
            parsed.missing_threshold,
            "rna",
            logger,
            log_transform=True,
        ),
        "proteomics": preprocess_view(
            prot_flat,
            parsed.missing_threshold,
            "proteomics",
            logger,
            log_transform=False,
        ),
        "phospho": preprocess_view(
            phospho_flat,
            parsed.missing_threshold,
            "phospho",
            logger,
            log_transform=False,
        ),
    }

    aligned_views = align_views_to_samples(view_dfs, sample_ids)
    metadata = build_metadata(clin_raw, luad, mut_raw, sample_ids, logger)
    metadata.index.name = "sample_id"

    write_outputs(
        parsed.output_dir,
        aligned_views["rna"],
        aligned_views["proteomics"],
        aligned_views["phospho"],
        metadata,
    )

    logger.info("Wrote %d samples to %s", len(sample_ids), parsed.output_dir)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
