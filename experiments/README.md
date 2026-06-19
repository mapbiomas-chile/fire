# experiments/

Archived configuration files from development A/B runs.
These are **not production** and are kept for reproducibility only.

## Conservative campaign (A/B vs classification_20260618)

These envs trained and classified with a conservative recipe (no spatial window, no
oversample, fixed threshold 0.55, IoU checkpoint) to diagnose overestimated burned area.
Results were validated as **better** than the 20260618 campaign and promoted to
**production** under `classification_20260619`.

| File | Role |
|------|------|
| `classification/cluster_paths.train_conservative.env.leftraru` | Train → `models_col1_conservative` |
| `classification/cluster_paths.classify_conservative.env.leftraru` | Classify → `classification_conservative` |
| `filtering/cluster_paths.conservative.env.leftraru` | Filter `classification_conservative` |

**Production env (same recipe):** `classification/cluster_paths.20260619.env.leftraru`
