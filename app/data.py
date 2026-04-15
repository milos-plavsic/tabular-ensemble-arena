from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.uci_fetch import fetch_uci_student_csv

DATA_SOURCE = (
    "UCI — Student Performance (Math), secondary schools Portugal. "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_student_mat_csv(path: Path) -> None:
    if path.exists():
        return
    try:
        fetch_uci_student_csv("student-mat.csv", path)
    except Exception as e:
        raise RuntimeError("Could not obtain student-mat.csv from UCI") from e


def load_student_pass_fail_encoded() -> tuple[np.ndarray, np.ndarray]:
    """Binary pass if final math grade G3 >= 10; exclude prior grades for a harder task."""
    path = project_root() / "data" / "student-mat.csv"
    _ensure_student_mat_csv(path)
    df = pd.read_csv(path, sep=";")
    y = (df["G3"] >= 10).astype(int).to_numpy()
    X = df.drop(columns=["G3", "G1", "G2"])
    X = pd.get_dummies(X, drop_first=True)
    return X.to_numpy(dtype=np.float32), y
