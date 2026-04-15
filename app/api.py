from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.data import DATA_SOURCE
from app.langgraph_compare import run_agentic_compare
from finetune.tuner import run_rf_hyperparam_finetune

app = FastAPI(title="Tabular Ensemble Arena", version="0.2.0")


class CompareRequest(BaseModel):
    cv_splits: int = Field(3, ge=2, le=10)
    confidence_threshold: float = Field(0.69, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=8)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compare")
def compare(body: CompareRequest) -> dict:
    out = run_agentic_compare(
        cv_splits=body.cv_splits,
        confidence_threshold=body.confidence_threshold,
        max_iterations=body.max_iterations,
    )
    return {**out, "data_source": DATA_SOURCE}


@app.post("/v1/finetune/rf_search")
def finetune_rf_search() -> dict:
    return run_rf_hyperparam_finetune()
