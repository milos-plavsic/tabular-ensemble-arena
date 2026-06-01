import os

from ml_core import configure_logging

from app.data import load_student_pass_fail_encoded
from app.train import leaderboard, run_comparison

logger = configure_logging("app.main")


def main() -> None:
    """Main."""
    cv_splits = int(os.getenv("CV_SPLITS", "3"))
    X, y = load_student_pass_fail_encoded()
    results = run_comparison(X, y, cv_splits=cv_splits)
    logger.info("Tabular Ensemble Arena — UCI student math (pass/fail) — ROC-AUC")
    for name, mean, std in leaderboard(results):
        logger.info(f"  {name:16s}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
