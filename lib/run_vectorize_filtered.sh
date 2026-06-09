#!/usr/bin/env bash
# Deprecated: use vectorize/run_vectorize_pipeline.sh
echo "[WARN] lib/run_vectorize_filtered.sh is deprecated. Use: bash vectorize/run_vectorize_pipeline.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../vectorize/run_vectorize_pipeline.sh" "$@"
