"""Comprehensive unit tests for tabular-ensemble-arena.

Covers:
  - Data loading and encoding
  - run_comparison cross-validation (ROC-AUC > 0.6)
  - leaderboard ordering
  - DiversityMetrics with known-value verification (Q-stat, disagreement)
  - ModelRegistry store/retrieve/persist
  - API endpoints: health, train, predict, data-profile, leaderboard, models
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from app.data import load_student_pass_fail_encoded
from app.ensemble_builder import DiversityMetrics
from app.model_registry import ModelRegistry
from app.train import leaderboard, run_comparison

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_dataset():
    """Load the UCI student dataset once per module."""
    X, y = load_student_pass_fail_encoded()
    return X, y


@pytest.fixture(scope="module")
def small_dataset(full_dataset):
    """150-row stratified subset for fast tests."""
    X, y = full_dataset
    rng = np.random.default_rng(42)
    idx_pass = np.where(y == 1)[0]
    idx_fail = np.where(y == 0)[0]
    chosen = np.concatenate(
        [
            rng.choice(idx_pass, size=min(90, len(idx_pass)), replace=False),
            rng.choice(idx_fail, size=min(60, len(idx_fail)), replace=False),
        ]
    )
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


@pytest.fixture
def fresh_registry():
    """A fresh ModelRegistry with no persisted state."""
    return ModelRegistry()


# ---------------------------------------------------------------------------
# 1. Data loading tests
# ---------------------------------------------------------------------------


def test_data_returns_numpy_arrays(full_dataset):
    """load_student_pass_fail_encoded must return two numpy arrays."""
    X, y = full_dataset
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"


def test_data_shape_minimum(full_dataset):
    """Dataset must have at least 300 rows and at least 20 encoded features."""
    X, y = full_dataset
    assert X.shape[0] >= 300, f"Expected >= 300 rows, got {X.shape[0]}"
    assert X.shape[1] >= 20, f"Expected >= 20 features after encoding, got {X.shape[1]}"


def test_data_target_is_binary(full_dataset):
    """Target must be binary (0/1 only)."""
    _, y = full_dataset
    unique = set(np.unique(y))
    assert unique == {0, 1}, f"y should be binary, got {unique}"


def test_data_class_balance(full_dataset):
    """Neither class should represent fewer than 10% of samples."""
    _, y = full_dataset
    pass_rate = y.mean()
    assert 0.10 <= pass_rate <= 0.90, f"Unusual class balance: {pass_rate:.2f}"


def test_data_no_nan(full_dataset):
    """Encoded feature matrix must not contain NaN values."""
    X, _ = full_dataset
    assert np.isfinite(X).all(), "X contains NaN or Inf after encoding"


def test_data_with_prior_grades_has_more_features(full_dataset):
    """include_prior_grades=True should produce more features than the default."""
    X_base, _ = full_dataset
    X_grades, _ = load_student_pass_fail_encoded(include_prior_grades=True)
    assert X_grades.shape[1] > X_base.shape[1], (
        f"Prior-grades dataset should have more features: "
        f"{X_grades.shape[1]} vs {X_base.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 2. run_comparison tests
# ---------------------------------------------------------------------------


def test_run_comparison_returns_three_models(small_dataset):
    """run_comparison must return results for exactly three model families."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=0)
    assert set(results.keys()) == {"random_forest", "xgboost", "catboost"}


def test_run_comparison_roc_auc_above_0_6(small_dataset):
    """Every model's mean ROC-AUC must exceed 0.6 on the UCI dataset."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=7)
    for name, metrics in results.items():
        assert (
            metrics["roc_auc_mean"] > 0.6
        ), f"{name}: ROC-AUC mean={metrics['roc_auc_mean']:.4f} must exceed 0.6"


def test_run_comparison_folds_count_matches_cv_splits(small_dataset):
    """Number of fold scores must equal cv_splits."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=3, random_state=0)
    for name, metrics in results.items():
        assert len(metrics["folds"]) == 3, f"{name}: expected 3 folds"


def test_run_comparison_std_non_negative(small_dataset):
    """Standard deviation of fold scores must be non-negative."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=5)
    for name, metrics in results.items():
        assert metrics["roc_auc_std"] >= 0.0, f"{name}: negative std"


def test_run_comparison_mean_matches_fold_average(small_dataset):
    """Reported mean must equal np.mean of reported folds."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=3, random_state=2)
    for name, metrics in results.items():
        expected = float(np.mean(metrics["folds"]))
        assert (
            abs(metrics["roc_auc_mean"] - expected) < 1e-9
        ), f"{name}: mean mismatch: reported={metrics['roc_auc_mean']} expected={expected}"


def test_run_comparison_reproducible(small_dataset):
    """Same random_state must produce identical results."""
    X, y = small_dataset
    r1 = run_comparison(X, y, cv_splits=2, random_state=99)
    r2 = run_comparison(X, y, cv_splits=2, random_state=99)
    for name in r1:
        assert abs(r1[name]["roc_auc_mean"] - r2[name]["roc_auc_mean"]) < 1e-9


# ---------------------------------------------------------------------------
# 3. leaderboard tests
# ---------------------------------------------------------------------------


def test_leaderboard_sorted_descending(small_dataset):
    """leaderboard() must be sorted best-first by mean ROC-AUC."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=3)
    board = leaderboard(results)
    means = [m for _, m, _ in board]
    assert means == sorted(means, reverse=True), f"Not sorted: {means}"


def test_leaderboard_correct_length(small_dataset):
    """leaderboard() must return one row per model."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=0)
    board = leaderboard(results)
    assert len(board) == 3


# ---------------------------------------------------------------------------
# 4. DiversityMetrics — Q-statistic with known inputs
# ---------------------------------------------------------------------------


def test_q_statistic_identical_predictors_return_zero():
    """Identical predictors: n11=|correct|, n10=n01=0, n00=|wrong|.

    When all predictions are identical and completely correct:
    n11=5, n10=0, n01=0, n00=0 → denominator=0 → Q returns 0.0 by convention.

    When predictions are identical but some are wrong:
    n11>0, n00>0, n01=n10=0 → Q = (n11*n00 - 0)/(n11*n00 + 0) = 1.0
    """
    # Case 1: all correct — denom=0, returns 0.0
    y_all_correct = np.array([0, 1, 0, 1, 1])
    pred_all_correct = np.array([0, 1, 0, 1, 1])
    q1 = DiversityMetrics.q_statistic(pred_all_correct, pred_all_correct, y_all_correct)
    assert q1 == 0.0  # denom = n11*n00 + n01*n10 = 5*0 + 0*0 = 0

    # Case 2: some wrong — Q = 1.0 (identical predictors always agree)
    y_mixed = np.array([0, 1, 0, 1, 1, 0])
    pred_mixed = np.array([0, 1, 0, 1, 0, 1])  # wrong on indices 4 and 5
    # n11=4 (both right on 0,1,2,3), n00=2 (both wrong on 4,5), n01=0, n10=0
    # Q = (4*2 - 0*0)/(4*2 + 0*0) = 8/8 = 1.0
    q2 = DiversityMetrics.q_statistic(pred_mixed, pred_mixed, y_mixed)
    assert abs(q2 - 1.0) < 1e-9, f"Expected Q=1.0, got Q={q2}"


def test_q_statistic_known_values():
    """Verify Q-statistic formula: Q=(n11*n00 - n01*n10)/(n11*n00 + n01*n10).

    Constructed example:
      y     = [1, 1, 1, 1, 0, 0, 0, 0]
      pred_i = [1, 1, 0, 0, 0, 0, 1, 1]   ← correct on samples 0,1,4,5
      pred_j = [1, 0, 1, 0, 0, 1, 0, 1]   ← correct on samples 0,2,4,6

      Correct_i: samples where pred_i==y → {0,1,4,5}
      Correct_j: samples where pred_j==y → {0,2,4,6}

      n11 (both correct):          samples {0,4}        → 2
      n10 (i correct, j wrong):    samples {1,5}        → 2
      n01 (i wrong, j correct):    samples {2,6}        → 2
      n00 (both wrong):            samples {3,7}        → 2

      Q = (2*2 - 2*2) / (2*2 + 2*2) = (4-4)/(4+4) = 0/8 = 0.0
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pred_i = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    pred_j = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    q = DiversityMetrics.q_statistic(pred_i, pred_j, y)
    assert abs(q - 0.0) < 1e-9, f"Expected Q=0.0, got Q={q}"


def test_q_statistic_complement():
    """When models are perfect complements (one right when other wrong), Q=-1.

    n11=0, n00=0, n01=N/2, n10=N/2
    Q = (0*0 - (N/2)*(N/2)) / (0*0 + (N/2)*(N/2)) = -(N/2)^2 / (N/2)^2 = -1.0
    """
    y = np.array([1, 1, 0, 0])
    pred_i = np.array([1, 0, 0, 1])  # correct on {0, 2}
    pred_j = np.array([0, 1, 1, 0])  # correct on {1, 3}
    # n11=0, n10=2, n01=2, n00=0  → Q=(0-4)/(0+4)=-1.0
    q = DiversityMetrics.q_statistic(pred_i, pred_j, y)
    assert abs(q - (-1.0)) < 1e-9, f"Expected Q=-1.0, got Q={q}"


def test_q_statistic_denominator_zero_returns_zero():
    """When denominator is 0, Q-statistic must return 0.0 without error."""
    # All predictions correct for both models, and never both wrong:
    y = np.array([1, 1])
    pred_i = np.array([1, 1])
    pred_j = np.array([1, 1])
    # n11=2, n10=0, n01=0, n00=0 → denom = 2*0 + 0*0 = 0 → return 0.0
    q = DiversityMetrics.q_statistic(pred_i, pred_j, y)
    assert q == 0.0


def test_disagreement_rate_known_values():
    """Verify disagreement formula: D = (n01 + n10) / N.

    Using the same example as test_q_statistic_known_values:
      n10=2, n01=2, N=8  →  D = 4/8 = 0.5
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pred_i = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    pred_j = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    d = DiversityMetrics.disagreement(pred_i, pred_j, y)
    assert abs(d - 0.5) < 1e-9, f"Expected disagreement=0.5, got {d}"


def test_disagreement_identical_predictors():
    """Two identical predictors should have disagreement rate of 0."""
    y = np.array([1, 0, 1, 0])
    pred = np.array([1, 0, 0, 1])
    d = DiversityMetrics.disagreement(pred, pred, y)
    assert d == 0.0, f"Identical predictors should disagree 0 times, got {d}"


def test_disagreement_fully_complementary():
    """Perfectly complementary predictors disagree on every sample."""
    y = np.array([1, 1, 0, 0])
    pred_i = np.array([1, 0, 0, 1])  # correct on {0,2}
    pred_j = np.array([0, 1, 1, 0])  # correct on {1,3}
    # i and j predict opposite on every sample → disagreement = 4/4 = 1.0
    d = DiversityMetrics.disagreement(pred_i, pred_j, y)
    assert abs(d - 1.0) < 1e-9, f"Expected 1.0, got {d}"


def test_pairwise_q_matrix_shape():
    """pairwise_q_statistics must return (n_models, n_models) matrix."""
    y = np.array([1, 0, 1, 0, 1])
    preds = np.column_stack(
        [
            np.array([1, 0, 1, 0, 1]),
            np.array([1, 1, 0, 0, 1]),
            np.array([0, 0, 1, 0, 1]),
        ]
    )
    q_mat = DiversityMetrics.pairwise_q_statistics(preds, y)
    assert q_mat.shape == (3, 3), f"Expected (3,3), got {q_mat.shape}"
    # Diagonal must be 1.0
    for i in range(3):
        assert q_mat[i, i] == 1.0, f"Diagonal element [{i},{i}] should be 1.0"


def test_pairwise_disagreement_matrix_symmetric():
    """pairwise_disagreements matrix must be symmetric."""
    y = np.array([1, 0, 1, 0, 1, 0])
    preds = np.column_stack(
        [
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([1, 1, 0, 0, 1, 0]),
        ]
    )
    d_mat = DiversityMetrics.pairwise_disagreements(preds, y)
    assert abs(d_mat[0, 1] - d_mat[1, 0]) < 1e-9, "Disagreement matrix not symmetric"


def test_ambiguity_with_identical_probabilities():
    """Ambiguity must be 0.0 when all models produce the same probabilities."""
    probs = np.array([[0.7, 0.7, 0.7], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5]])
    amb = DiversityMetrics.ambiguity(probs)
    assert abs(amb) < 1e-9, f"Expected 0.0, got {amb}"


def test_ambiguity_known_value():
    """Verify ambiguity formula with a manually computed case.

    For probs = [[0.0, 1.0], [1.0, 0.0]]:
      Row 0: mean=0.5, deviations=(-0.5)^2 + (0.5)^2 = 0.5, mean=0.25
      Row 1: mean=0.5, deviations=(0.5)^2 + (-0.5)^2 = 0.5, mean=0.25
      Overall mean = 0.25
    """
    probs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    amb = DiversityMetrics.ambiguity(probs)
    assert abs(amb - 0.25) < 1e-9, f"Expected 0.25, got {amb}"


# ---------------------------------------------------------------------------
# 5. ModelRegistry tests
# ---------------------------------------------------------------------------


def _make_fitted_model(random_state: int = 0):
    """Return a fitted RandomForestClassifier on synthetic data."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=5, random_state=random_state)
    clf.fit(X, y)
    return clf, X, y


def test_registry_register_and_list(fresh_registry):
    """Registered model appears in list_models()."""
    clf, _, _ = _make_fitted_model()
    fresh_registry.register("rf_test", clf, {"roc_auc_mean": 0.75})
    entries = fresh_registry.list_models()
    assert len(entries) == 1
    assert entries[0]["name"] == "rf_test"
    assert entries[0]["roc_auc_mean"] == 0.75


def test_registry_get_best_returns_highest_auc(fresh_registry):
    """get_best() must return the model with the highest roc_auc_mean."""
    clf_a, _, _ = _make_fitted_model(0)
    clf_b, _, _ = _make_fitted_model(1)
    fresh_registry.register("model_a", clf_a, {"roc_auc_mean": 0.70})
    fresh_registry.register("model_b", clf_b, {"roc_auc_mean": 0.85})
    best_name, _ = fresh_registry.get_best()
    assert best_name == "model_b", f"Expected model_b, got {best_name}"


def test_registry_get_raises_key_error(fresh_registry):
    """get() on a non-existent name must raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        fresh_registry.get("nonexistent")


def test_registry_get_best_raises_on_empty(fresh_registry):
    """get_best() on empty registry must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="No models registered"):
        fresh_registry.get_best()


def test_registry_predict_returns_valid_probability(fresh_registry):
    """predict() must return probabilities in [0, 1]."""
    clf, X, _ = _make_fitted_model()
    fresh_registry.register("rf", clf, {"roc_auc_mean": 0.80})
    proba = fresh_registry.predict(X[:5])
    assert proba.shape == (5,), f"Unexpected shape: {proba.shape}"
    assert np.all((proba >= 0) & (proba <= 1)), f"Proba out of [0,1]: {proba}"


def test_registry_predict_achieves_roc_auc(fresh_registry):
    """Model retrieved from registry must achieve ROC-AUC > 0.6 on held-out data."""
    X, y = load_student_pass_fail_encoded()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=32, random_state=42)
    clf.fit(X_tr, y_tr)
    fresh_registry.register("rf_auc", clf, {"roc_auc_mean": 0.78})
    proba = fresh_registry.predict(X_te)
    auc = roc_auc_score(y_te, proba)
    assert auc > 0.6, f"Registry model ROC-AUC={auc:.4f} must exceed 0.6"


def test_registry_persist_and_reload(tmp_path):
    """Model persisted to disk must be loadable by a new registry instance."""
    clf, X, _ = _make_fitted_model()
    reg1 = ModelRegistry(persist_dir=tmp_path / "models")
    reg1.register("persisted_rf", clf, {"roc_auc_mean": 0.77})

    reg2 = ModelRegistry(persist_dir=tmp_path / "models")
    entries = reg2.list_models()
    assert len(entries) == 1, "Expected 1 model after reload"
    assert entries[0]["name"] == "persisted_rf"


def test_registry_clear(fresh_registry):
    """clear() must remove all models from memory."""
    clf, _, _ = _make_fitted_model()
    fresh_registry.register("m1", clf, {"roc_auc_mean": 0.7})
    fresh_registry.clear()
    assert fresh_registry.list_models() == []


# ---------------------------------------------------------------------------
# 6. API endpoint tests
# ---------------------------------------------------------------------------


def test_api_health():
    """GET /health must return 200 with status ok."""
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_api_data_profile_returns_200():
    """GET /v1/data-profile must return 200."""
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    r = client.get("/v1/data-profile")
    assert r.status_code == 200


def test_api_data_profile_shape():
    """GET /v1/data-profile must include shape with rows >= 300."""
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    body = client.get("/v1/data-profile").json()
    assert "shape" in body
    assert body["shape"]["rows"] >= 300


def test_api_data_profile_class_balance():
    """GET /v1/data-profile must include valid class balance."""
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    body = client.get("/v1/data-profile").json()
    cb = body["class_balance"]
    assert 0.0 < cb["pass_rate"] < 1.0


def test_api_models_lists_three():
    """GET /v1/models must list all three model families."""
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    body = client.get("/v1/models").json()
    assert set(body["models"].keys()) == {"random_forest", "xgboost", "catboost"}


def test_api_models_has_hyperparameters():
    """GET /v1/models each entry must include non-empty hyperparameters."""
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    body = client.get("/v1/models").json()
    for name, spec in body["models"].items():
        assert "hyperparameters" in spec, f"{name} missing 'hyperparameters'"
        assert len(spec["hyperparameters"]) > 0


def test_api_leaderboard_empty_before_training():
    """GET /v1/leaderboard before training should return empty list."""
    from fastapi.testclient import TestClient

    from app.api import app
    from app.model_registry import registry

    client = TestClient(app)
    registry.clear()
    body = client.get("/v1/leaderboard").json()
    assert body["leaderboard"] == []


def test_api_train_returns_three_models():
    """POST /v1/train must return leaderboard with three entries."""
    from fastapi.testclient import TestClient

    from app.api import app
    from app.model_registry import registry

    client = TestClient(app)
    registry.clear()
    r = client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    assert r.status_code == 200
    body = r.json()
    assert len(body["leaderboard"]) == 3
    registry.clear()


def test_api_train_roc_auc_above_threshold():
    """POST /v1/train all models must report ROC-AUC > 0.5."""
    from fastapi.testclient import TestClient

    from app.api import app
    from app.model_registry import registry

    client = TestClient(app)
    registry.clear()
    r = client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    body = r.json()
    for entry in body["leaderboard"]:
        assert entry["roc_auc_mean"] > 0.5, f"{entry['name']} ROC-AUC too low"
    registry.clear()


def test_api_predict_requires_prior_training():
    """POST /v1/predict without training must return 400."""
    from fastapi.testclient import TestClient

    from app.api import app
    from app.model_registry import registry

    client = TestClient(app)
    registry.clear()
    r = client.post("/v1/predict", json={"features": [0.0] * 10})
    assert r.status_code == 400


def test_api_predict_after_training():
    """POST /v1/predict after training must return a valid probability."""
    from fastapi.testclient import TestClient

    from app.api import app
    from app.model_registry import registry

    client = TestClient(app)
    registry.clear()
    client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})

    # Determine n_features from data profile
    profile = client.get("/v1/data-profile").json()
    n_features = profile["shape"]["columns"]
    features = [0.0] * n_features
    r = client.post("/v1/predict", json={"features": features})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_class_1"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    registry.clear()


def test_api_leaderboard_populated_after_training():
    """GET /v1/leaderboard after POST /v1/train must list three models."""
    from fastapi.testclient import TestClient

    from app.api import app
    from app.model_registry import registry

    client = TestClient(app)
    registry.clear()
    client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    body = client.get("/v1/leaderboard").json()
    assert len(body["leaderboard"]) == 3
    assert body["best_model"] is not None
    registry.clear()
