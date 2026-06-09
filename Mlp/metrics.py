"""Métricas e utilitários sem scikit-learn."""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Projeta X (m, d) em 2D via PCA (SVD)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T
