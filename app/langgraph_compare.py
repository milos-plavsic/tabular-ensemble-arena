from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from app.data import DATA_SOURCE, load_student_pass_fail_encoded
from app.train import leaderboard, run_comparison


class IterSummary(TypedDict):
    iteration: int
    include_prior_grades: bool
    cv_splits: int
    best_model: str
    best_auc: float
    best_std: float
    confidence_score: float


class CompareState(TypedDict, total=False):
    cv_splits: int
    confidence_threshold: float
    max_iterations: int
    random_state: int

    iteration: int
    include_prior_grades: bool
    results: dict[str, dict]
    leaderboard_rows: list[tuple[str, float, float]]

    confidence_score: float
    confidence_label: str
    continue_loop: bool
    stop_reason: str

    history: list[IterSummary]


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _validate(state: CompareState) -> CompareState:
    return {
        "cv_splits": max(2, min(10, int(state.get("cv_splits", 3)))),
        "confidence_threshold": _clip01(float(state.get("confidence_threshold", 0.69))),
        "max_iterations": max(1, int(state.get("max_iterations", 3))),
        "random_state": int(state.get("random_state", 42)),
        "iteration": 0,
        "history": [],
    }


def _plan(state: CompareState) -> CompareState:
    it = int(state["iteration"]) + 1
    return {
        "iteration": it,
        "include_prior_grades": it >= 2,
        "cv_splits": min(10, state["cv_splits"] + (1 if it >= 2 else 0)),
    }


def _compare(state: CompareState) -> CompareState:
    X, y = load_student_pass_fail_encoded(include_prior_grades=state["include_prior_grades"])
    results = run_comparison(
        X,
        y,
        cv_splits=state["cv_splits"],
        random_state=state["random_state"],
    )
    board = leaderboard(results)
    return {"results": results, "leaderboard_rows": board}


def _assess(state: CompareState) -> CompareState:
    best_model, best_auc, best_std = state["leaderboard_rows"][0]
    margin = best_auc - state["leaderboard_rows"][1][1]
    confidence = _clip01(0.65 * best_auc + 0.20 * _clip01(margin * 5.0) + 0.15 * (1.0 - min(best_std, 1.0)))
    conf_label = _label(confidence)

    reached_conf = confidence >= state["confidence_threshold"]
    reached_limit = state["iteration"] >= state["max_iterations"]
    continue_loop = not reached_conf and not reached_limit

    if reached_conf:
        reason = "confidence_threshold_reached"
    elif reached_limit:
        reason = "max_iterations_reached"
    else:
        reason = "retry_with_additional_information"

    h: IterSummary = {
        "iteration": state["iteration"],
        "include_prior_grades": state["include_prior_grades"],
        "cv_splits": state["cv_splits"],
        "best_model": best_model,
        "best_auc": float(best_auc),
        "best_std": float(best_std),
        "confidence_score": confidence,
    }

    return {
        "confidence_score": confidence,
        "confidence_label": conf_label,
        "continue_loop": continue_loop,
        "stop_reason": reason,
        "history": [*state["history"], h],
    }


def _route(state: CompareState) -> Literal["plan", "finalize"]:
    return "plan" if state["continue_loop"] else "finalize"


def _finalize(state: CompareState) -> CompareState:
    return {}


def build_compare_graph():
    g = StateGraph(CompareState)
    g.add_node("validate", _validate)
    g.add_node("plan", _plan)
    g.add_node("compare", _compare)
    g.add_node("assess", _assess)
    g.add_node("finalize", _finalize)

    g.set_entry_point("validate")
    g.add_edge("validate", "plan")
    g.add_edge("plan", "compare")
    g.add_edge("compare", "assess")
    g.add_conditional_edges("assess", _route, {"plan": "plan", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_COMPARE_GRAPH = build_compare_graph()


def run_agentic_compare(
    *,
    cv_splits: int = 3,
    confidence_threshold: float = 0.69,
    max_iterations: int = 3,
    random_state: int = 42,
) -> dict:
    out = _COMPARE_GRAPH.invoke(
        {
            "cv_splits": cv_splits,
            "confidence_threshold": confidence_threshold,
            "max_iterations": max_iterations,
            "random_state": random_state,
        }
    )
    board = [
        {"model": n, "roc_auc_mean": m, "roc_auc_std": s}
        for n, m, s in out["leaderboard_rows"]
    ]
    return {
        "leaderboard": board,
        "raw": out["results"],
        "confidence_score": out["confidence_score"],
        "confidence_label": out["confidence_label"],
        "confidence_threshold": out["confidence_threshold"],
        "iterations": out["iteration"],
        "loop_terminated_reason": out["stop_reason"],
        "iteration_history": out["history"],
        "data_source": DATA_SOURCE,
    }
