from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.data import load_synthetic
from app.train import leaderboard, run_comparison

app = FastAPI(title="Tabular Ensemble Arena", version="0.1.0")


class CompareRequest(BaseModel):
    n_samples: int = Field(2000, ge=500, le=50_000)
    n_features: int = Field(20, ge=4, le=200)
    cv_splits: int = Field(3, ge=2, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compare")
def compare(body: CompareRequest) -> dict:
    X, y = load_synthetic(n_samples=body.n_samples, n_features=body.n_features)
    results = run_comparison(X, y, cv_splits=body.cv_splits)
    board = [{"model": n, "roc_auc_mean": m, "roc_auc_std": s} for n, m, s in leaderboard(results)]
    return {"leaderboard": board, "raw": results}
