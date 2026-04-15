from __future__ import annotations

from pathlib import Path

from app.data import DATA_SOURCE, load_student_pass_fail_encoded
from app.train import run_comparison
from analysis.json_util import dumps_pretty
from analysis.plotting import bar_with_errors, fold_line_plot
from analysis.stats_utils import comparison_table


def generate_report(out_dir: Path | None = None, cv_splits: int = 3) -> dict:
    out = Path(out_dir or "reports")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_student_pass_fail_encoded()
    results = run_comparison(X, y, cv_splits=cv_splits)
    table = comparison_table(results)

    summary = {
        "data_source": DATA_SOURCE,
        "task": "binary_classification_pass_fail",
        "n_samples": int(X.shape[0]),
        "n_features_encoded": int(X.shape[1]),
        "cv_splits": cv_splits,
        "models": table,
    }
    (out / "summary.json").write_text(dumps_pretty(summary), encoding="utf-8")

    labels = [r["model"] for r in table]
    means = [r["roc_auc_mean"] for r in table]
    stds = [r["roc_auc_std"] for r in table]
    bar_with_errors(
        labels,
        means,
        stds,
        fig_dir / "roc_auc_model_comparison.png",
        title="UCI student math — pass/fail (ROC-AUC by model)",
        ylabel="ROC-AUC",
    )

    for name, m in results.items():
        safe = name.replace("/", "_").replace(" ", "_")
        fold_line_plot(
            name,
            m["folds"],
            fig_dir / f"folds_{safe}.png",
            metric_name="ROC-AUC",
        )

    md = _markdown_summary(summary, fig_dir)
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    return {"output_dir": str(out.resolve()), "n_models": len(table)}


def _markdown_summary(summary: dict, fig_dir: Path) -> str:
    lines = [
        "# Ensemble benchmark — statistical summary",
        "",
        f"**Data:** {summary['data_source']}",
        f"**Samples:** {summary['n_samples']} | **Features (encoded):** {summary['n_features_encoded']}",
        "",
        "## Model comparison (mean ± std over CV folds)",
        "",
        "| Model | ROC-AUC mean | ROC-AUC std |",
        "|---|---:|---:|",
    ]
    for m in summary["models"]:
        lines.append(
            f"| {m['model']} | {m['roc_auc_mean']:.4f} | {m['roc_auc_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- ![ROC-AUC bar](figures/roc_auc_model_comparison.png)",
            "",
            "Per-fold line plots: see `figures/folds_*.png`.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    out = generate_report()
    print(dumps_pretty(out))


if __name__ == "__main__":
    main()
