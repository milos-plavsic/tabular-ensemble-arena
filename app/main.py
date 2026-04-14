import os

from app.data import load_synthetic
from app.train import leaderboard, run_comparison


def main() -> None:
    n_samples = int(os.getenv("N_SAMPLES", "2000"))
    n_features = int(os.getenv("N_FEATURES", "20"))
    cv_splits = int(os.getenv("CV_SPLITS", "3"))
    X, y = load_synthetic(n_samples=n_samples, n_features=n_features)
    results = run_comparison(X, y, cv_splits=cv_splits)
    print("Tabular Ensemble Arena — ROC-AUC (higher is better)")
    for name, mean, std in leaderboard(results):
        print(f"  {name:16s}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
