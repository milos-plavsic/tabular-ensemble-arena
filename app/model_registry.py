"""In-memory (and optional disk-backed) model registry for trained ensemble models.

Supports registering models after training, retrieving the best model by
ROC-AUC, and running predictions through the best model.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np
from ml_core import configure_logging

logger = configure_logging("app.model_registry")


class ModelRegistry:
    """Thread-safe store for trained sklearn-compatible estimators.

    Models are always kept in memory.  When ``persist_dir`` is provided
    each registered model is also written to disk as a pickle file so it
    survives process restarts.

    Usage::

        registry = ModelRegistry()
        registry.register("random_forest", model, {"roc_auc_mean": 0.80})
        name, model = registry.get_best()
        predictions = registry.predict(X_arr)
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._persist_dir: Path | None = None
        if persist_dir is not None:
            self._persist_dir = Path(persist_dir)
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str, model: Any, metadata: dict[str, Any]) -> None:
        """Store a trained model under *name*.

        Args:
            name: Unique identifier for this model (e.g. "random_forest").
            model: Any sklearn-compatible estimator.
            metadata: Arbitrary key-value pairs; must contain ``roc_auc_mean``
                      (float) to participate in best-model selection.
        """
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        if model is None:
            raise ValueError("model must not be None")

        entry = {"model": model, "metadata": dict(metadata)}

        with self._lock:
            self._store[name] = entry
            logger.info("Registered model '%s'  metadata=%s", name, metadata)

        if self._persist_dir is not None:
            self._save_to_disk(name, entry)

    def get_best(self) -> tuple[str, Any]:
        """Return ``(name, model)`` for the model with the highest ``roc_auc_mean``.

        Raises:
            RuntimeError: If no models have been registered yet.
        """
        with self._lock:
            if not self._store:
                raise RuntimeError("No models registered in the registry")

            best_name = max(
                self._store,
                key=lambda n: float(self._store[n]["metadata"].get("roc_auc_mean", 0.0)),
            )
            return best_name, self._store[best_name]["model"]

    def get(self, name: str) -> Any:
        """Return the model registered under *name*.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        with self._lock:
            if name not in self._store:
                raise KeyError(f"Model '{name}' not found in registry")
            return self._store[name]["model"]

    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for all registered models, best-first by ROC-AUC."""
        with self._lock:
            rows = [{"name": n, **entry["metadata"]} for n, entry in self._store.items()]
        rows.sort(key=lambda r: float(r.get("roc_auc_mean", 0.0)), reverse=True)
        return rows

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the best model's ``predict_proba`` on *X*.

        Returns:
            1-D array of positive-class probabilities, shape ``(n_samples,)``.

        Raises:
            RuntimeError: If no models are registered.
        """
        _, model = self.get_best()
        proba = model.predict_proba(X)
        # Binary classification: return P(class=1)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def clear(self) -> None:
        """Remove all registered models from memory (disk files are kept)."""
        with self._lock:
            self._store.clear()
        logger.info("ModelRegistry cleared")

    # ------------------------------------------------------------------
    # Disk persistence helpers
    # ------------------------------------------------------------------

    def _save_to_disk(self, name: str, entry: dict[str, Any]) -> None:
        assert self._persist_dir is not None
        path = self._persist_dir / f"{name}.pkl"
        try:
            with open(path, "wb") as fh:
                pickle.dump(entry, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Persisted model '%s' to %s", name, path)
        except Exception as exc:
            logger.warning("Could not persist model '%s': %s", name, exc)

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for pkl_path in self._persist_dir.glob("*.pkl"):
            name = pkl_path.stem
            try:
                with open(pkl_path, "rb") as fh:
                    entry = pickle.load(fh)
                self._store[name] = entry
                logger.info("Loaded model '%s' from disk", name)
            except Exception as exc:
                logger.warning("Could not load model '%s' from %s: %s", name, pkl_path, exc)


# Module-level singleton — importable directly.
registry = ModelRegistry()
