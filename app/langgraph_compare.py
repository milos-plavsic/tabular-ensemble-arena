from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.data import DATA_SOURCE, load_student_pass_fail_encoded
from app.orchestration_policy import (
    confidence_label,
    decide_loop,
    normalize_threshold,
    normalized_stability,
    weighted_confidence,
)
from app.train import leaderboard, run_comparison


class IterSummary(TypedDict):
    """Implementation of the iter summary."""

    iteration: int
    include_prior_grades: bool
    cv_splits: int
    best_model: str
    best_auc: float
    best_std: float
    confidence_score: float


class CompareState(TypedDict, total=False):
    """Implementation of the compare state."""

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


def _validate(state: CompareState) -> CompareState:
    """Internal helper that handles validate."""
    return {
        "cv_splits": max(2, min(10, int(state.get("cv_splits", 3)))),
        "confidence_threshold": normalize_threshold(state.get("confidence_threshold", 0.69)),
        "max_iterations": max(1, int(state.get("max_iterations", 3))),
        "random_state": int(state.get("random_state", 42)),
        "iteration": 0,
        "history": [],
    }


def _plan(state: CompareState) -> CompareState:
    """Internal helper that handles plan."""
    it = int(state["iteration"]) + 1
    return {
        "iteration": it,
        "include_prior_grades": it >= 2,
        "cv_splits": min(10, state["cv_splits"] + (1 if it >= 2 else 0)),
    }


def _compare(state: CompareState) -> CompareState:
    """Internal helper that handles compare."""
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
    """Internal helper that handles assess."""
    best_model, best_auc, best_std = state["leaderboard_rows"][0]
    margin = best_auc - state["leaderboard_rows"][1][1]
    components = {
        "primary_quality": best_auc,
        "secondary_quality": min(max(margin * 5.0, 0.0), 1.0),
        "stability": normalized_stability(best_std),
    }
    confidence = weighted_confidence(components)
    conf_label = confidence_label(confidence)

    loop = decide_loop(
        confidence_score=confidence,
        confidence_threshold=state["confidence_threshold"],
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )

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
        "continue_loop": loop["continue_loop"],
        "stop_reason": loop["stop_reason"],
        "history": [*state["history"], h],
    }


def _route(state: CompareState) -> Literal["plan", "finalize"]:
    """Internal helper that handles route."""
    return "plan" if state["continue_loop"] else "finalize"


def _finalize(state: CompareState) -> CompareState:
    """Internal helper that handles finalize."""
    return {}


def build_compare_graph():
    """Execute the build compare graph routine."""
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
    """Execute the run agentic compare routine."""
    out = _COMPARE_GRAPH.invoke(
        {
            "cv_splits": cv_splits,
            "confidence_threshold": confidence_threshold,
            "max_iterations": max_iterations,
            "random_state": random_state,
        }
    )
    board = [
        {"model": n, "roc_auc_mean": m, "roc_auc_std": s} for n, m, s in out["leaderboard_rows"]
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
