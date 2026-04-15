from __future__ import annotations

import math
from typing import Any


def fold_statistics(folds: list[float]) -> dict[str, float]:
    xs = [float(x) for x in folds]
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return {
        "mean": m,
        "std_sample": math.sqrt(v),
        "min": min(xs),
        "max": max(xs),
        "n_folds": float(len(xs)),
    }


def comparison_table(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, m in results.items():
        fs = m.get("folds", [])
        st = fold_statistics(fs) if fs else {}
        rows.append(
            {
                "model": name,
                "roc_auc_mean": m.get("roc_auc_mean"),
                "roc_auc_std": m.get("roc_auc_std"),
                **{f"fold_stat_{k}": v for k, v in st.items()},
            }
        )
    rows.sort(key=lambda r: r.get("roc_auc_mean") or 0, reverse=True)
    return rows
