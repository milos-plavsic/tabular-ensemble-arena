from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_SOURCE = (
    "UCI — Student Performance (Math), secondary schools Portugal. "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


_UCI_STUDENT_MAT = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
)
_UCI_STUDENT_MAT_HTTP = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
)


def _download_first_working(urls: tuple[str, ...], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; portfolio-report/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as e:
            last_err = e
    raise RuntimeError("Could not download student-mat.csv from UCI") from last_err


def _ensure_student_mat_csv(path: Path) -> None:
    if path.exists():
        return
    _download_first_working((_UCI_STUDENT_MAT, _UCI_STUDENT_MAT_HTTP), path)


def load_student_pass_fail_encoded() -> tuple[np.ndarray, np.ndarray]:
    """Binary pass if final math grade G3 >= 10; exclude prior grades for a harder task."""
    path = project_root() / "data" / "student-mat.csv"
    _ensure_student_mat_csv(path)
    df = pd.read_csv(path, sep=";")
    y = (df["G3"] >= 10).astype(int).to_numpy()
    X = df.drop(columns=["G3", "G1", "G2"])
    X = pd.get_dummies(X, drop_first=True)
    return X.to_numpy(dtype=np.float32), y
