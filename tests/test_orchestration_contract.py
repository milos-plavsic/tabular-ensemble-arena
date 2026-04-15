from app.langgraph_compare import run_agentic_compare


REQUIRED = {
    "confidence_score",
    "confidence_label",
    "confidence_threshold",
    "iterations",
    "loop_terminated_reason",
    "iteration_history",
}


def test_orchestration_contract_fields_present() -> None:
    out = run_agentic_compare(cv_splits=3, confidence_threshold=0.7, max_iterations=2, random_state=0)
    assert REQUIRED.issubset(set(out.keys()))
    assert 0.0 <= out["confidence_score"] <= 1.0
    assert out["confidence_label"] in {"low", "medium", "high"}
    assert out["loop_terminated_reason"] in {
        "confidence_threshold_reached",
        "max_iterations_reached",
        "retry_with_additional_information",
    }
    assert isinstance(out["iteration_history"], list)
    assert len(out["iteration_history"]) == out["iterations"]
