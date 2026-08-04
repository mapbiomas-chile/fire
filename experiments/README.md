# experiments/

[Español](README.es.md) | **English**

Archived **configuration files** from development A/B runs. They are retained for **methodological reproducibility** of design decisions and are **not** required for the current production path once artifacts under `models_col1_20260619` / `classification_20260619` exist.

---

## Conservative campaign (promoted to 20260619)

These environments trained and classified with a conservative recipe (no spatial window, no burned oversampling, fixed decision threshold **0.55**, IoU as selection metric) to diagnose overestimated burned area relative to campaign `classification_20260618`. The recipe was promoted to production as **`classification_20260619`**.

| File | Historical role |
|------|-----------------|
| `classification/cluster_paths.train_conservative.env.leftraru` | Train → `models_col1_conservative` |
| `classification/cluster_paths.classify_conservative.env.leftraru` | Classify → `classification_conservative` |
| `filtering/cluster_paths.conservative.env.leftraru` | Filter that tree |

**Production env (same recipe, canonical paths):**  
`classification/cluster_paths.20260619.env.leftraru` and `filtering/cluster_paths.20260619.env.leftraru`.

See [classification README](../classification/README.md) § experimental configurations.
