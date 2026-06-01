"""Production-grade ensemble builders with diversity metrics.

Classes:
    StackingEnsemble   — sklearn StackingClassifier wrapper with a
                         LogisticRegression meta-learner.
    VotingEnsemble     — sklearn VotingClassifier (hard and soft voting).
    DiversityMetrics   — computes pairwise disagreement, Q-statistic, and
                         ambiguity decomposition between base models.

All classes conform to the sklearn estimator API (fit / predict /
predict_proba) so they can be used inside pipelines or evaluated with
cross_validate.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from ml_core import configure_logging
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

logger = configure_logging("app.ensemble_builder")


# ---------------------------------------------------------------------------
# StackingEnsemble
# ---------------------------------------------------------------------------


class StackingEnsemble:
    """Stacking ensemble with four heterogeneous base learners.

    Base estimators:
        - LogisticRegression  (linear)
        - RandomForestClassifier  (bagging)
        - GradientBoostingClassifier  (boosting)
        - KNeighborsClassifier  (non-parametric)

    Meta-learner:
        - LogisticRegression (trained on out-of-fold predictions)

    Usage::

        se = StackingEnsemble(random_state=42)
        se.fit(X_train, y_train)
        proba = se.predict_proba(X_test)
        preds = se.predict(X_test)
    """

    def __init__(
        self,
        *,
        random_state: int = 42,
        cv: int = 5,
        passthrough: bool = False,
    ) -> None:
        self.random_state = random_state
        self.cv = cv
        self.passthrough = passthrough
        self._clf: StackingClassifier | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self) -> StackingClassifier:
        estimators: list[tuple[str, Any]] = [
            (
                "lr",
                LogisticRegression(
                    max_iter=500,
                    random_state=self.random_state,
                    solver="lbfgs",
                    C=1.0,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=32,
                    max_depth=8,
                    random_state=self.random_state,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=32,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                ),
            ),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=7, n_jobs=1),
            ),
        ]
        meta_learner = LogisticRegression(
            max_iter=500,
            random_state=self.random_state,
            solver="lbfgs",
            C=1.0,
        )
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=self.cv,
            passthrough=self.passthrough,
            n_jobs=1,
        )

    # ------------------------------------------------------------------
    # sklearn-compatible API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> StackingEnsemble:
        """Fit all base estimators and the meta-learner."""
        logger.info("Fitting StackingEnsemble on %d samples", len(X))
        self._clf = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._clf.fit(X, y)
        logger.info("StackingEnsemble fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions."""
        self._check_fitted()
        return self._clf.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the meta-learner."""
        self._check_fitted()
        return self._clf.predict_proba(X)  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("StackingEnsemble must be fitted before predicting")

    @property
    def classes_(self) -> np.ndarray:
        self._check_fitted()
        return self._clf.classes_  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# VotingEnsemble
# ---------------------------------------------------------------------------


class VotingEnsemble:
    """Voting ensemble supporting both hard and soft voting.

    Uses the same four base learners as StackingEnsemble.

    Usage::

        ve_soft = VotingEnsemble(voting="soft", random_state=42)
        ve_soft.fit(X_train, y_train)
        preds = ve_soft.predict(X_test)

        ve_hard = VotingEnsemble(voting="hard", random_state=42)
        ve_hard.fit(X_train, y_train)
        preds = ve_hard.predict(X_test)
    """

    def __init__(
        self,
        *,
        voting: str = "soft",
        random_state: int = 42,
    ) -> None:
        if voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft', got {voting!r}")
        self.voting = voting
        self.random_state = random_state
        self._clf: VotingClassifier | None = None

    def _build(self) -> VotingClassifier:
        estimators: list[tuple[str, Any]] = [
            (
                "lr",
                LogisticRegression(
                    max_iter=500,
                    random_state=self.random_state,
                    solver="lbfgs",
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=32,
                    max_depth=8,
                    random_state=self.random_state,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=32,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                ),
            ),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=7, n_jobs=1),
            ),
        ]
        return VotingClassifier(estimators=estimators, voting=self.voting, n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> VotingEnsemble:
        logger.info("Fitting VotingEnsemble (voting=%s) on %d samples", self.voting, len(X))
        self._clf = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._clf.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.  Only available for soft voting."""
        self._check_fitted()
        if self.voting == "hard":
            raise AttributeError("predict_proba is not available with hard voting")
        return self._clf.predict_proba(X)  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("VotingEnsemble must be fitted before predicting")

    @property
    def classes_(self) -> np.ndarray:
        self._check_fitted()
        return self._clf.classes_  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# DiversityMetrics
# ---------------------------------------------------------------------------


class DiversityMetrics:
    """Diversity metrics for binary classification ensembles.

    Implements three complementary measures:

    **Pairwise disagreement**
        The proportion of samples on which two classifiers disagree:
        ``D(i,j) = (n01 + n10) / N``
        where n01 = model-i correct, model-j wrong; n10 = vice versa.

    **Q-statistic** (Yule 1900)
        ``Q(i,j) = (n11*n00 - n01*n10) / (n11*n00 + n01*n10)``
        Q ∈ [-1, 1]; Q=0 means statistically independent, Q→1 means
        models agree, Q→-1 means models complement each other.

    **Ambiguity decomposition**
        Decomposes ensemble error into bias and variance terms:
        ``ambiguity = mean_i(E[(f_i(x) - f_bar(x))^2])``
        where f_bar is the ensemble mean prediction.

    All methods accept binary prediction arrays of shape (n_samples,) per
    model or a stacked matrix of shape (n_samples, n_models).
    """

    @staticmethod
    def _contingency(
        pred_i: np.ndarray, pred_j: np.ndarray, y_true: np.ndarray
    ) -> tuple[int, int, int, int]:
        """Compute the 2x2 oracle-oracle contingency table for two classifiers.

        Returns:
            (n11, n10, n01, n00)
            n11 — both correct
            n10 — i correct, j wrong
            n01 — i wrong, j correct
            n00 — both wrong
        """
        correct_i = (pred_i == y_true).astype(int)
        correct_j = (pred_j == y_true).astype(int)
        n11 = int(np.sum((correct_i == 1) & (correct_j == 1)))
        n10 = int(np.sum((correct_i == 1) & (correct_j == 0)))
        n01 = int(np.sum((correct_i == 0) & (correct_j == 1)))
        n00 = int(np.sum((correct_i == 0) & (correct_j == 0)))
        return n11, n10, n01, n00

    @classmethod
    def q_statistic(
        cls,
        pred_i: np.ndarray,
        pred_j: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Pairwise Q-statistic between classifiers i and j.

        Formula:
            Q(i,j) = (n11*n00 - n01*n10) / (n11*n00 + n01*n10)

        Args:
            pred_i: Hard predictions of classifier i, shape (n_samples,).
            pred_j: Hard predictions of classifier j, shape (n_samples,).
            y_true: Ground-truth labels, shape (n_samples,).

        Returns:
            Q in [-1, 1].  Returns 0.0 if denominator is zero.
        """
        n11, n10, n01, n00 = cls._contingency(pred_i, pred_j, y_true)
        numerator = n11 * n00 - n01 * n10
        denominator = n11 * n00 + n01 * n10
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    @classmethod
    def disagreement(
        cls,
        pred_i: np.ndarray,
        pred_j: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Pairwise disagreement rate between classifiers i and j.

        Formula: D(i,j) = (n01 + n10) / N

        Returns a value in [0, 1]; higher means more diverse.
        """
        n11, n10, n01, n00 = cls._contingency(pred_i, pred_j, y_true)
        n_total = n11 + n10 + n01 + n00
        if n_total == 0:
            return 0.0
        return float((n01 + n10) / n_total)

    @classmethod
    def pairwise_q_statistics(
        cls,
        predictions: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """Compute the full (n_models x n_models) Q-statistic matrix.

        Args:
            predictions: Integer array of shape (n_samples, n_models).
            y_true: Ground-truth labels, shape (n_samples,).

        Returns:
            Square symmetric matrix of Q-statistics.
        """
        n_models = predictions.shape[1]
        matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    matrix[i, j] = 1.0  # Q with itself = 1
                else:
                    matrix[i, j] = cls.q_statistic(predictions[:, i], predictions[:, j], y_true)
        return matrix

    @classmethod
    def pairwise_disagreements(
        cls,
        predictions: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """Compute the full (n_models x n_models) disagreement matrix.

        Args:
            predictions: Integer array of shape (n_samples, n_models).
            y_true: Ground-truth labels, shape (n_samples,).

        Returns:
            Square symmetric matrix of disagreement rates.
        """
        n_models = predictions.shape[1]
        matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    matrix[i, j] = 0.0  # disagrees with itself = 0
                else:
                    matrix[i, j] = cls.disagreement(predictions[:, i], predictions[:, j], y_true)
        return matrix

    @staticmethod
    def ambiguity(
        probabilities: np.ndarray,
    ) -> float:
        """Ambiguity (bias-variance) decomposition for an ensemble.

        Measures how much individual models deviate from the ensemble mean
        prediction. Higher values indicate more diversity.

        Args:
            probabilities: Soft probabilities of shape (n_samples, n_models).

        Returns:
            Mean squared deviation of individual model predictions from the
            ensemble average, i.e.
            ``mean_i( mean_x( (f_i(x) - f_bar(x))^2 ) )``
        """
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("probabilities must have shape (n_samples, n_models) with n_models>=2")
        ensemble_mean = probabilities.mean(axis=1, keepdims=True)  # (n_samples, 1)
        deviations = (probabilities - ensemble_mean) ** 2  # (n_samples, n_models)
        return float(deviations.mean())

    @classmethod
    def compute_all(
        cls,
        predictions: np.ndarray,
        y_true: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Convenience method: compute all diversity metrics at once.

        Args:
            predictions: Hard predictions, shape (n_samples, n_models).
            y_true: Ground-truth labels, shape (n_samples,).
            probabilities: Optional soft probabilities, shape (n_samples, n_models).
                           Required for ambiguity computation.

        Returns:
            Dictionary with keys:
              - ``mean_q_statistic``: mean of upper-triangle Q values
              - ``mean_disagreement``: mean of upper-triangle disagreements
              - ``q_matrix``: full Q-statistic matrix (as nested list)
              - ``disagreement_matrix``: full disagreement matrix (as nested list)
              - ``ambiguity``: (only if probabilities provided)
        """
        n_models = predictions.shape[1]
        q_mat = cls.pairwise_q_statistics(predictions, y_true)
        d_mat = cls.pairwise_disagreements(predictions, y_true)

        # Upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n_models, n_models), dtype=bool), k=1)
        mean_q = float(q_mat[mask].mean()) if mask.any() else 0.0
        mean_d = float(d_mat[mask].mean()) if mask.any() else 0.0

        result: dict[str, Any] = {
            "mean_q_statistic": mean_q,
            "mean_disagreement": mean_d,
            "q_matrix": q_mat.tolist(),
            "disagreement_matrix": d_mat.tolist(),
        }
        if probabilities is not None:
            result["ambiguity"] = cls.ambiguity(probabilities)
        return result
