"""Activation functions and their derivatives."""

import numpy as np

ACTIVATIONS = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
}


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax for batch shape (m, n_classes)."""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return relu, relu_derivative
    if name == "sigmoid":
        return sigmoid, sigmoid_derivative
    if name == "tanh":
        return tanh, tanh_derivative
    raise ValueError(f"Unknown activation: {name}")
