from pathlib import Path

from analysis.report import generate_report


def test_generate_report_smoke(tmp_path: Path) -> None:
    out = generate_report(tmp_path, cv_splits=3)
    assert out["n_models"] == 3
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "figures" / "roc_auc_model_comparison.png").is_file()
