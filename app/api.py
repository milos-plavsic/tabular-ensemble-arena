"""FastAPI application for Tabular Ensemble Arena.

Endpoints:
  GET  /health                    — liveness probe
  GET  /metrics                   — Prometheus metrics
  POST /v1/compare                — agentic compare (langgraph)
  POST /v1/finetune/rf_search     — RF hyper-parameter search (legacy)
  POST /v1/train                  — train all models, return leaderboard
  POST /v1/predict                — predict with best registered model
  GET  /v1/models                 — list model types and default hyper-parameters
  GET  /v1/data-profile           — dataset statistics
  GET  /v1/leaderboard            — latest leaderboard from registry
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from ml_core import configure_logging, install_middleware
from ml_core.observability import metrics_router, observe_request
from pydantic import BaseModel, Field

from app.data import DATA_SOURCE, load_student_pass_fail_encoded
from app.langgraph_compare import run_agentic_compare
from app.model_registry import registry
from app.train import leaderboard, run_comparison
from finetune.tuner import run_rf_hyperparam_finetune

logger = configure_logging("app.api")

app = FastAPI(title="Tabular Ensemble Arena", version="0.3.0")

# -- Middleware ----------------------------------------------------------
install_middleware(app, cors_allow_origins=("*",))
app.include_router(metrics_router)


@app.middleware("http")
async def _observe(request: Request, call_next):
    return await observe_request(request, call_next)


# -----------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------


class CompareRequest(BaseModel):
    """Pydantic schema for the compare request."""

    cv_splits: int = Field(3, ge=2, le=10)
    confidence_threshold: float = Field(0.69, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=8)


class TrainRequest(BaseModel):
    """Pydantic schema for POST /v1/train."""

    cv_splits: int = Field(3, ge=2, le=10, description="Number of CV folds")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    include_prior_grades: bool = Field(False, description="If true, include G1/G2 as features")


class PredictRequest(BaseModel):
    """Pydantic schema for POST /v1/predict."""

    features: list[float] = Field(
        ..., description="Feature vector matching the training columns (encoded)"
    )


# -----------------------------------------------------------------------
# Routes — liveness
# -----------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# -----------------------------------------------------------------------
# Routes — legacy
# -----------------------------------------------------------------------


@app.post("/v1/compare")
async def compare(body: CompareRequest) -> dict:
    """Agentic comparison (wraps langgraph graph)."""
    out = run_agentic_compare(
        cv_splits=body.cv_splits,
        confidence_threshold=body.confidence_threshold,
        max_iterations=body.max_iterations,
    )
    return {**out, "data_source": DATA_SOURCE}


@app.post("/v1/finetune/rf_search")
async def finetune_rf_search() -> dict:
    """Run RF hyper-parameter search (legacy)."""
    return run_rf_hyperparam_finetune()


# -----------------------------------------------------------------------
# Routes — v1 ML API
# -----------------------------------------------------------------------


@app.post("/v1/train")
async def train(body: TrainRequest) -> dict:
    """Train all model families and return a leaderboard.

    Side effect: registers all trained models in the in-process model registry
    so that subsequent calls to ``POST /v1/predict`` work without retraining.
    """
    try:
        X, y = load_student_pass_fail_encoded(include_prior_grades=body.include_prior_grades)
        results = run_comparison(X, y, cv_splits=body.cv_splits, random_state=body.random_state)
        board = leaderboard(results)

        # Factories for each model family (train once on full data after CV)
        from catboost import CatBoostClassifier
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        factories = {
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=32,
                max_depth=10,
                random_state=body.random_state,
                n_jobs=1,
            ),
            "xgboost": lambda: XGBClassifier(
                n_estimators=32,
                max_depth=3,
                learning_rate=0.1,
                random_state=body.random_state,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=1,
            ),
            "catboost": lambda: CatBoostClassifier(
                iterations=32,
                depth=4,
                learning_rate=0.1,
                random_seed=body.random_state,
                verbose=False,
                allow_writing_files=False,
            ),
        }

        for name, make_model in factories.items():
            model = make_model()
            model.fit(X, y)
            registry.register(
                name,
                model,
                {
                    "roc_auc_mean": results[name]["roc_auc_mean"],
                    "roc_auc_std": results[name]["roc_auc_std"],
                    "cv_splits": body.cv_splits,
                    "random_state": body.random_state,
                },
            )

        leaderboard_response = [
            {"name": name, "roc_auc_mean": mean, "roc_auc_std": std} for name, mean, std in board
        ]
        return {
            "leaderboard": leaderboard_response,
            "data_source": DATA_SOURCE,
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
        }
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/predict")
async def predict(body: PredictRequest) -> dict:
    """Return positive-class probability from the best registered model.

    The registry must be populated first by calling ``POST /v1/train``.
    """
    if not registry.list_models():
        raise HTTPException(
            status_code=400,
            detail="No models trained yet. Call POST /v1/train first.",
        )
    try:
        X = np.array([body.features], dtype=np.float32)
        probabilities = registry.predict(X)
        best_name, _ = registry.get_best()
        return {
            "best_model": best_name,
            "probability_class_1": float(probabilities[0]),
            "predicted_class": int(probabilities[0] >= 0.5),
        }
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/models")
async def list_models() -> dict:
    """Return the available model families and their default hyper-parameters."""
    model_specs: dict[str, dict[str, Any]] = {
        "random_forest": {
            "type": "RandomForestClassifier",
            "preprocessing": "pd.get_dummies one-hot encoding",
            "hyperparameters": {
                "n_estimators": 32,
                "max_depth": 10,
            },
        },
        "xgboost": {
            "type": "XGBClassifier",
            "preprocessing": "pd.get_dummies one-hot encoding",
            "hyperparameters": {
                "n_estimators": 32,
                "max_depth": 3,
                "learning_rate": 0.1,
            },
        },
        "catboost": {
            "type": "CatBoostClassifier",
            "preprocessing": "pd.get_dummies one-hot encoding",
            "hyperparameters": {
                "iterations": 32,
                "depth": 4,
                "learning_rate": 0.1,
            },
        },
    }

    registered = {m["name"] for m in registry.list_models()}
    for name in model_specs:
        model_specs[name]["registered"] = name in registered

    return {"models": model_specs, "total": len(model_specs)}


@app.get("/v1/data-profile")
async def data_profile() -> dict:
    """Return dataset statistics: shape, feature count, class balance."""
    try:
        from pathlib import Path

        import pandas as pd

        path = Path(__file__).resolve().parent.parent / "data" / "student-mat.csv"
        df_raw = pd.read_csv(path, sep=";")
        y_raw = (df_raw["G3"] >= 10).astype(int).to_numpy()

        X, y = load_student_pass_fail_encoded()

        n_total = int(len(y_raw))
        n_pass = int(np.sum(y_raw == 1))
        n_fail = int(np.sum(y_raw == 0))

        # Basic stats per encoded column
        col_means = X.mean(axis=0).tolist()
        col_stds = X.std(axis=0).tolist()

        return {
            "shape": {"rows": int(X.shape[0]), "columns": int(X.shape[1])},
            "class_balance": {
                "total": n_total,
                "pass_n": n_pass,
                "fail_n": n_fail,
                "pass_rate": round(n_pass / n_total, 4),
            },
            "feature_means": col_means[:10],  # first 10 for brevity
            "feature_stds": col_stds[:10],
            "data_source": DATA_SOURCE,
        }
    except Exception as exc:
        logger.exception("Data profile failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/leaderboard")
async def leaderboard_endpoint() -> dict:
    """Return the current registry leaderboard (populated after POST /v1/train)."""
    models = registry.list_models()
    if not models:
        return {"leaderboard": [], "message": "No models trained yet. Call POST /v1/train first."}
    return {
        "leaderboard": models,
        "best_model": models[0]["name"] if models else None,
    }
