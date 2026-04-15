# Release Notes (2026-04)

## Scope
This release introduces benchmark reporting (stats + figures), CV robustness fixes, and CI hardening.

## Data Source
- UCI Student Performance dataset (ID 320): `student-mat.csv`

## Reporting Added
- New `analysis/` package with:
  - `report.py`, `plotting.py`, `stats_utils.py`, `json_util.py`, module entrypoint
- Generated outputs:
  - `reports/summary.json`
  - `reports/REPORT.md`
  - `reports/figures/roc_auc_model_comparison.png`
  - `reports/figures/folds_*.png`

## Latest Report Snapshot (ROC-AUC mean ± std)
- `catboost`: `0.6968 ± 0.0642`
- `xgboost`: `0.6858 ± 0.0333`
- `random_forest`: `0.6746 ± 0.0394`

## Modeling/CV Fixes
- Replaced fragile scorer path with explicit StratifiedKFold evaluation loop.
- Set XGBoost classifier objective explicitly for binary classification.
- Preserved stable per-fold metrics and leaderboard sorting.

## Reliability and CI
- Added ZIP-based UCI fetch fallback via `app/uci_fetch.py`.
- Ensured local/offline stability with vendored `data/student-mat.csv`.
- CI runs tests and `python -m analysis` smoke step.
- Upgraded actions to:
  - `actions/checkout@v6`
  - `actions/setup-python@v6`

## Latest CI Status
- Latest successful run: https://github.com/milos-plavsic/tabular-ensemble-arena/actions/runs/24447653654

## Dependency Notes
- `xgboost` pinned to `>=2.0,<2.1` to reduce CUDA-related install issues in generic CPU CI.
