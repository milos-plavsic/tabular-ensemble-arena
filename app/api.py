from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.data import DATA_SOURCE, load_student_pass_fail_encoded
from app.train import leaderboard, run_comparison
from finetune.tuner import run_rf_hyperparam_finetune

app = FastAPI(title="Tabular Ensemble Arena", version="0.1.0")


class CompareRequest(BaseModel):
    cv_splits: int = Field(3, ge=2, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compare")
def compare(body: CompareRequest) -> dict:
    X, y = load_student_pass_fail_encoded()
    results = run_comparison(X, y, cv_splits=body.cv_splits)
    board = [{"model": n, "roc_auc_mean": m, "roc_auc_std": s} for n, m, s in leaderboard(results)]
    return {"leaderboard": board, "raw": results, "data_source": DATA_SOURCE}


@app.post("/v1/finetune/rf_search")
def finetune_rf_search() -> dict:
    return run_rf_hyperparam_finetune()
