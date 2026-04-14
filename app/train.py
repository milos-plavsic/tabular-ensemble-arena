from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def build_models(random_state: int = 42) -> dict[str, object]:
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=4,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    xgb = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
    )
    cb = CatBoostClassifier(
        iterations=120,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        verbose=False,
        random_seed=random_state,
        allow_writing_files=False,
    )
    return {"random_forest": rf, "xgboost": xgb, "catboost": cb}


def run_comparison(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 3,
    random_state: int = 42,
) -> dict[str, dict]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    models = build_models(random_state)
    results: dict[str, dict] = {}
    for name, clf in models.items():
        scores = cross_val_score(
            clf,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        results[name] = {
            "roc_auc_mean": float(np.mean(scores)),
            "roc_auc_std": float(np.std(scores)),
            "folds": [float(s) for s in scores],
        }
    return results


def leaderboard(results: dict[str, dict]) -> list[tuple[str, float, float]]:
    rows = [(name, v["roc_auc_mean"], v["roc_auc_std"]) for name, v in results.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
