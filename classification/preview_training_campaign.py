#!/usr/bin/env python3
"""List which sample TIFFs match each Chile training campaign job."""

from __future__ import annotations

import sys
from pathlib import Path

from fire_model_common import CHILE_TRAINING_CAMPAIGN, extract_sample_year, select_training_files


def main() -> None:
  samples_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/flepin/samples_col1")
  sample_version = sys.argv[2] if len(sys.argv) > 2 else "v1"

  if not samples_dir.is_dir():
    raise SystemExit(f"Not a directory: {samples_dir}")

  print(f"Samples dir: {samples_dir}")
  print(f"Filename token: {sample_version}\n")

  for spec in CHILE_TRAINING_CAMPAIGN:
    region = str(spec["region"])
    model_version = str(spec["model_version"])
    start_year = int(spec["sample_start_year"])
    end_year = int(spec["sample_end_year"])

    files = select_training_files(
      samples_dir,
      region,
      sample_version=sample_version,
      sample_start_year=start_year,
      sample_end_year=end_year,
    )
    years = sorted({extract_sample_year(path) for path in files})

    print(f"=== col1_chile_{model_version}_{region}_rnn_lstm_ckpt  ({start_year}-{end_year}) ===")
    print(f"  files: {len(files)}   years: {years}")
    for path in files:
      print(f"    {path.name}")
    if not files:
      print("  [WARNING] no files matched")
    print()


if __name__ == "__main__":
  main()
