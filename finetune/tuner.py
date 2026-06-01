from __future__ import annotations

import os

import numpy as np
from ml_core import configure_logging
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from app.data import DATA_SOURCE, load_student_pass_fail_encoded

logger = configure_logging("finetune.tuner")


def run_rf_hyperparam_finetune(random_state: int = 42) -> dict:
    """Tune RandomForest (same family as leaderboard RF) via randomized search."""
    n_iter = int(os.getenv("FINETUNE_N_ITER", "10"))
    X, y = load_student_pass_fail_encoded()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    param = {
        "n_estimators": randint(60, 250),
        "max_depth": [None, 8, 10, 14],
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None],
    }
    base = RandomForestClassifier(
        random_state=random_state, n_jobs=2, class_weight="balanced_subsample"
    )
    search = RandomizedSearchCV(
        base,
        param,
        n_iter=n_iter,
        cv=3,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    proba = search.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    best = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in search.best_params_.items()
    }
    return {
        "model": "random_forest",
        "best_params": best,
        "test_roc_auc": auc,
        "n_iter": n_iter,
        "data_source": DATA_SOURCE,
    }


def main() -> None:
    """Main."""
    out = run_rf_hyperparam_finetune()
    logger.info("RF hyperparameter fine-tune (UCI pass/fail)")
    for k, v in out.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
