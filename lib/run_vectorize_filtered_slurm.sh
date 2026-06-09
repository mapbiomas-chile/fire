#!/bin/bash
# Deprecated: use vectorize/run_vectorize_pipeline_slurm.sh
echo "[WARN] lib/run_vectorize_filtered_slurm.sh is deprecated. Use: sbatch vectorize/run_vectorize_pipeline_slurm.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "${SCRIPT_DIR}/../vectorize/run_vectorize_pipeline_slurm.sh" "$@"
