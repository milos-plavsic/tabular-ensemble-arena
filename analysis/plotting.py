from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_with_errors(
    labels: list[str],
    means: list[float],
    stds: list[float],
    out_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    """Execute the bar with errors routine."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, yerr=stds, capsize=6, color="#3b6ea5", ecolor="#333333", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fold_line_plot(
    model_name: str,
    fold_scores: list[float],
    out_path: Path,
    *,
    metric_name: str,
) -> None:
    """Execute the fold line plot routine."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(fold_scores) + 1), fold_scores, marker="o", color="#c45c3e")
    ax.set_xlabel("CV fold")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{model_name} — per-fold scores")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
