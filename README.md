# MOSAIC

**Multi-Omics Subtype Analysis In Cancer** — a Snakemake pipeline for studying how lung adenocarcinoma patients transform at the molecular level.

Most multi-omics studies characterize the molecular state of a tumour at diagnosis. MOSAIC instead studies the molecular transformation from matched normal tissue to tumour by representing each patient as a latent shift (ΔZ) in a shared MOFA+ space:

```
ΔZ_i = Z_tumour_i − Z_normal_i
```

The pipeline uses CPTAC-LUAD matched tumour–normal pairs (transcriptomics, proteomics, phosphoproteomics; ~102 patients). Clustering ΔZ vectors groups patients by transformation pattern; downstream stages link the resulting modes to pathways and clinical variables.

A successful run generates:

- MOFA latent factors
- ΔZ representations
- patient clusters
- biological interpretation (GSEA, differential expression)
- clinical association analyses

---

## Quick start

**Requirements:** Linux or macOS, [Miniforge](https://github.com/conda-forge/miniforge), network access (CPTAC download in Stage 01, gene sets in Stage 07).

```bash
git clone <repository-url>
cd MOSAIC

conda env create -f env/mosaic.yaml
conda activate mosaic

snakemake --snakefile src/snakemake/Snakefile --use-conda -j <cores>
```

Or `bash run_snakemake.sh` (32 cores, from repo root). All commands must be run from the repository root.

The default [`config/config.yml`](config/config.yml) reproduces the analysis in the report. MOFA fitting (Stage 03) takes 1–3 hours; everything else is much faster.

> `results/` and `figures/` are gitignored — a fresh clone has no outputs until you run the pipeline.

---

## What the pipeline does

```
01 ingest → 02 preprocess → 03 MOFA+ → 04 factor QC & ΔZ
    → 05 ΔZ embeddings → 06 cluster → 06.1 visualise → 07 interpret → 08 clinical
```

| Stage | What it does |
|-------|--------------|
| 01–02 | Download CPTAC-LUAD; select high-MAD features; z-score |
| 03 | Fit MOFA+ on all 204 samples (shared latent space) |
| 04–05 | Compute ΔZ; explore representations (PCA/UMAP) |
| 06 | Cluster patients by ΔZ; produce diagnostics |
| 06.1 | Colour embeddings by cluster assignment (no clustering logic) |
| 07–08 | GSEA / differential expression; survival and mutation associations |

Scripts live in `src/py/`; Snakemake rules in `src/snakemake/rules/`. Parameters are in `config/config.yml` only.

**Three conda environments** (Snakemake creates the stage-specific ones via `--use-conda`):

- `env/mosaic.yaml` → `mosaic` — Snakemake + Stage 01
- `env/mofa.yml` → `mosaic-mofa` — Stage 03
- `env/analysis.yml` → `mosaic-analysis` — all other stages

---

## Clustering workflow

Clustering is the main decision point. Stage **06** fits the model and writes diagnostics; Stage **06.1** only plots the result on pre-computed embeddings from Stage 05. Change clustering parameters → re-run from Stage 06. Change only the plot embedding → edit `visualization_representation` and re-run Stage 06.1.

ΔZ representations (set under `clustering.manual.representation` or `clustering.automatic.representation`):

| Value | Meaning |
|-------|---------|
| `norm_all` | L2-normalised ΔZ, all factors — **report default** |
| `norm_nof1` | L2-normalised, Factor1 excluded |
| `raw_all` | Raw ΔZ, all factors |
| `raw_nof1` | Raw ΔZ, Factor1 excluded |

### Option 1 — Reproduce the report (default)

Leave `config/config.yml` unchanged (`clustering_parameter_selection: manual`, k = 3, K-means, `norm_all`) and run Snakemake once.

### Option 2 — Automatic k selection

After changing the feature set or other upstream settings, switch to automatic selection:

```yaml
clustering:
  clustering_parameter_selection: automatic
  automatic:
    strategy: silhouette
    method: kmeans
    representation: norm_all
```

Stage 06 picks k with the highest silhouette score. Then:

```bash
snakemake --snakefile src/snakemake/Snakefile --use-conda -j <cores> --forcerun clustering
```

### Option 3 — Explore, then choose manually

```bash
# 1. Run through clustering diagnostics
snakemake --snakefile src/snakemake/Snakefile --use-conda -j <cores> --until clustering

# 2. Review figures/clustering/model_selection.pdf, dispersion_summary.pdf,
#    results/clustering/decision_table.csv, cross_repr_ari.csv
#    (Stage 06 also prints guidance in log/06_clustering.log)

# 3. Set your choice in config.yml under clustering.manual

# 4. Rebuild clustering and everything downstream
snakemake --snakefile src/snakemake/Snakefile --use-conda -j <cores> --forcerun clustering
```

---

## Outputs

All tables are stored in the `results/` directory (by stage: `ingestion/`, `preprocessing/`, `mofa/`, `delta/`, `clustering/`, etc.). Figures are stored in `figures/` with the same stage-based layout.

---

## Tips

- **Dry run:** `snakemake --snakefile src/snakemake/Snakefile --use-conda -n`
- **Single stage:** `python src/py/05_delta.py --config config/config.yml` (activate the right conda env first)
- **Skip MOFA refit** if models exist: `python src/py/03_fit_mofa.py --config config/config.yml --skip-fit`
- **Wrong directory?** Snakemake expects `config/config.yml` relative to the repo root.
- Parameter details and inline documentation: [`config/config.yml`](config/config.yml)

---

## References

- Gillette et al., *Cell* 2020 — [CPTAC-LUAD](https://cptac-data-portal.georgetown.edu/)
- Argelaguet et al., *Genome Biology* 2020 — MOFA+

Licensed under GPL-3.0 ([LICENSE](LICENSE)).
