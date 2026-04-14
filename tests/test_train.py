import numpy as np

from app.data import load_synthetic
from app.train import leaderboard, run_comparison


def test_run_comparison_three_models() -> None:
    X, y = load_synthetic(n_samples=800, n_features=12)
    out = run_comparison(X, y, cv_splits=3, random_state=0)
    assert set(out.keys()) == {"random_forest", "xgboost", "catboost"}
    for name, metrics in out.items():
        assert 0.5 <= metrics["roc_auc_mean"] <= 1.0, name
        assert metrics["roc_auc_std"] >= 0
        assert len(metrics["folds"]) == 3


def test_leaderboard_sorted() -> None:
    X, y = load_synthetic(n_samples=600, n_features=10)
    out = run_comparison(X, y, cv_splits=3, random_state=1)
    board = leaderboard(out)
    means = [m for _, m, _ in board]
    assert means == sorted(means, reverse=True)
