"""Loss functions."""

import numpy as np


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Cross-entropy with one-hot or integer labels.
    probs: (m, C), y_true: (m,) int or (m, C) one-hot
    """
    m = probs.shape[0]
    eps = 1e-12
    if y_true.ndim == 1:
        log_likelihood = -np.log(probs[np.arange(m), y_true.astype(int)] + eps)
    else:
        log_likelihood = -np.sum(y_true * np.log(probs + eps), axis=1)
    return float(np.mean(log_likelihood))


def cross_entropy_gradient(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of CE w.r.t. logits when combined with softmax in last layer.
    Returns dL/dz for shape (m, C).
    """
    m = probs.shape[0]
    if y_true.ndim == 1:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(m), y_true.astype(int)] = 1.0
        y = one_hot
    else:
        y = y_true
    return (probs - y) / m
