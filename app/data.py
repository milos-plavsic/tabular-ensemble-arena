from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ml_core import validate_dataframe as _ml_validate_dataframe

from app.uci_fetch import fetch_uci_student_csv

DATA_SOURCE = (
    "UCI — Student Performance (Math), secondary schools Portugal. "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    """Project root."""
    return Path(__file__).resolve().parent.parent


def _ensure_student_mat_csv(path: Path) -> None:
    """Ensure student mat csv.."""
    if path.exists():
        return
    try:
        fetch_uci_student_csv("student-mat.csv", path)
    except Exception as e:
        raise RuntimeError("Could not obtain student-mat.csv from UCI") from e


def load_student_pass_fail_encoded(
    *, include_prior_grades: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Binary pass if final grade G3 >= 10; optionally include G1/G2 as additional information."""
    path = project_root() / "data" / "student-mat.csv"
    _ensure_student_mat_csv(path)
    df = pd.read_csv(path, sep=";")
    _ml_validate_dataframe(df)
    y = (df["G3"] >= 10).astype(int).to_numpy()
    if not include_prior_grades:
        df["prior_grade_mean"] = df[["G1", "G2"]].mean(axis=1)
    drop_cols = ["G3"] if include_prior_grades else ["G3", "G1", "G2"]
    X = df.drop(columns=drop_cols)
    X = pd.get_dummies(X, drop_first=True)
    return X.to_numpy(dtype=np.float32), y
