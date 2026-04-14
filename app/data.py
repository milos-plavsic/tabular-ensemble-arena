import numpy as np
from sklearn.datasets import make_classification


def load_synthetic(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 12,
    n_redundant: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.int64)
