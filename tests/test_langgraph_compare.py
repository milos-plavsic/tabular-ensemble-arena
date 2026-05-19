from app.langgraph_compare import run_agentic_compare


def test_compare_loop_hits_max_iterations_for_strict_threshold() -> None:
    """Execute the test compare loop hits max iterations for strict threshold routine."""
    out = run_agentic_compare(
        cv_splits=3, confidence_threshold=0.99, max_iterations=2, random_state=0
    )
    assert out["iterations"] == 2
    assert out["loop_terminated_reason"] == "max_iterations_reached"


def test_compare_output_contains_confidence() -> None:
    """Execute the test compare output contains confidence routine."""
    out = run_agentic_compare(
        cv_splits=3, confidence_threshold=0.1, max_iterations=3, random_state=0
    )
    assert 0.0 <= out["confidence_score"] <= 1.0
    assert out["confidence_label"] in {"low", "medium", "high"}
