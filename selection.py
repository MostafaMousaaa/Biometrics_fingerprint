import numpy as np

class PCAProjector:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        total_var = (S**2).sum() + 1e-9
        self.explained_variance_ratio_ = (S[:k]**2) / total_var
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
