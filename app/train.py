"""Ensemble training with diversity metrics."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from ml_core import configure_logging
from ml_core.exceptions import ApplicationError as ModelError
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = configure_logging("app.train")


class _ValidatorShim:
    """Shim that adapts ml_core.validate_array to the legacy validator.validate_arrays API."""

    @staticmethod
    def validate_arrays(X, y=None, *, allow_inf: bool = False) -> None:
        import numpy as np

        arr = np.asarray(X)
        if arr.size == 0:
            raise ValueError("X array is empty")
        if not allow_inf and not np.isfinite(arr).all():
            raise ValueError("X contains NaN or inf values")
        if y is not None:
            yarr = np.asarray(y)
            if len(yarr) != len(arr):
                raise ValueError(f"X and y have different lengths: {len(arr)} vs {len(yarr)}")
            if not allow_inf and not np.isfinite(yarr).all():
                raise ValueError("y contains NaN or inf values")


validator = _ValidatorShim()


class EnsembleTrainer:
    """Train and validate ensemble models."""

    @staticmethod
    def calculate_diversity(predictions: np.ndarray) -> dict[str, float]:
        """Calculate ensemble diversity metrics.

        Args:
            predictions: Array of shape (n_samples, n_models)

        Returns:
            Dictionary with diversity metrics
        """
        if predictions.shape[1] < 2:
            raise ModelError("Need at least 2 models for diversity calculation")

        # Calculate correlations between model predictions
        correlations = np.corrcoef(predictions.T)

        # Remove diagonal (correlation with self)
        mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
        off_diag_corr = correlations[mask]

        return {
            "mean_correlation": float(np.mean(off_diag_corr)),
            "max_correlation": float(np.max(off_diag_corr)),
            "min_correlation": float(np.min(off_diag_corr)),
            "std_correlation": float(np.std(off_diag_corr)),
        }

    @staticmethod
    def validate_diversity(
        diversity_metrics: dict[str, float],
        max_correlation: float = 0.95,
    ) -> None:
        """Validate ensemble diversity is sufficient.

        Args:
            diversity_metrics: Metrics from calculate_diversity
            max_correlation: Maximum allowed mean correlation

        Raises:
            ModelError: If diversity is too low
        """
        mean_corr = diversity_metrics["mean_correlation"]

        if mean_corr > max_correlation:
            logger.warning(
                f"Ensemble diversity low: mean correlation {mean_corr:.4f} > "
                f"threshold {max_correlation}"
            )
            raise ModelError(
                f"Ensemble models too correlated (r={mean_corr:.4f}). "
                f"Models may be too similar to benefit ensemble."
            )

        logger.info(f"Ensemble diversity acceptable: mean correlation {mean_corr:.4f}")

    @classmethod
    def train_ensemble(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        random_state: int = 42,
    ) -> tuple[list[object], np.ndarray, dict[str, float]]:
        """Train ensemble with diversity validation.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Validation feature matrix
            y_val: Validation target vector
            random_state: Random seed

        Returns:
            Tuple of (models, weights, metrics)

        Raises:
            ModelError: If training fails or diversity is low
        """
        # Validate inputs
        validator.validate_arrays(X_train, y_train, allow_inf=False)
        validator.validate_arrays(X_val, y_val, allow_inf=False)

        logger.info(f"Training ensemble on {len(X_train)} samples")

        # Train individual models
        models = [
            RandomForestRegressor(
                n_estimators=32,
                max_depth=10,
                random_state=random_state,
            ),
            GradientBoostingRegressor(
                n_estimators=32,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
            ),
            Ridge(alpha=1.0),
        ]

        predictions = []
        scores = []

        for i, model in enumerate(models):
            try:
                logger.info(f"Training model {i+1}/{len(models)}: {model.__class__.__name__}")
                model.fit(X_train, y_train)

                # Get validation predictions
                y_pred = model.predict(X_val)

                # Validate predictions
                if not np.isfinite(y_pred).all():
                    raise ModelError(f"Model {i} produced NaN/Inf predictions")

                r2 = r2_score(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)

                logger.info(f"  Model {i+1}: R² = {r2:.4f}, MSE = {mse:.4f}")

                predictions.append(y_pred)
                scores.append(max(0.0, r2))  # Clamp to [0, 1]

            except Exception as e:
                raise ModelError(f"Failed to train model {i}: {e}") from e

        # Stack predictions
        predictions_array = np.column_stack(predictions)

        # Calculate diversity
        diversity = cls.calculate_diversity(predictions_array)
        logger.info(
            f"Diversity metrics: "
            f"mean_corr={diversity['mean_correlation']:.4f}, "
            f"max_corr={diversity['max_correlation']:.4f}"
        )

        # Validate diversity
        cls.validate_diversity(diversity)

        # Calculate weights based on validation performance
        scores_array = np.array(scores)
        weights = scores_array / np.sum(scores_array)

        logger.info(f"Ensemble weights: {weights}")

        return models, weights, diversity


# ---------------------------------------------------------------------------
# Backward-compatible module-level helpers used by app.langgraph_compare
# ---------------------------------------------------------------------------


def run_comparison(
    X,
    y,
    *,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict]:
    """Cross-validated ROC-AUC comparison for the three classifier families."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    validator.validate_arrays(X_arr, y_arr, allow_inf=False)

    cv = StratifiedKFold(n_splits=max(2, int(cv_splits)), shuffle=True, random_state=random_state)
    factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=32, max_depth=10, random_state=random_state, n_jobs=1
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=32,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        ),
        "catboost": lambda: CatBoostClassifier(
            iterations=32,
            depth=4,
            learning_rate=0.1,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
    }

    results: dict[str, dict] = {}
    for name, make_model in factories.items():
        fold_scores: list[float] = []
        for train_idx, val_idx in cv.split(X_arr, y_arr):
            X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
            model = make_model()
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            fold_scores.append(float(roc_auc_score(y_va, proba)))
        arr = np.asarray(fold_scores, dtype=float)
        results[name] = {
            "roc_auc_mean": float(arr.mean()),
            "roc_auc_std": float(arr.std()),
            "folds": [float(s) for s in fold_scores],
        }
    return results


def leaderboard(results: dict[str, dict]) -> list[tuple[str, float, float]]:
    """Format ``run_comparison`` output as a list of (name, mean, std), best-first."""
    rows = [
        (
            name,
            float(metrics.get("roc_auc_mean", 0.0)),
            float(metrics.get("roc_auc_std", 0.0)),
        )
        for name, metrics in results.items()
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
