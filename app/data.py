from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_SOURCE = (
    "UCI — Student Performance (Math), secondary schools Portugal. "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_student_pass_fail_encoded() -> tuple[np.ndarray, np.ndarray]:
    """Binary pass if final math grade G3 >= 10; exclude prior grades for a harder task."""
    path = project_root() / "data" / "student-mat.csv"
    df = pd.read_csv(path, sep=";")
    y = (df["G3"] >= 10).astype(int).to_numpy()
    X = df.drop(columns=["G3", "G1", "G2"])
    X = pd.get_dummies(X, drop_first=True)
    return X.to_numpy(dtype=np.float32), y
