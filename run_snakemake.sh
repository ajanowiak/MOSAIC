#!/bin/bash
# Run the full MOSAIC pipeline from the repository root.
# Requires Snakemake (e.g. conda activate mosaic) and network access for Stage 01.
set -euo pipefail
cd "$(dirname "$0")"
snakemake --snakefile src/snakemake/Snakefile --use-conda -j 32 -k
