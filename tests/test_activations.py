"""Unit tests for activation functions and gradients."""

import numpy as np
import pytest

from mlp.activations import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    tanh,
    tanh_derivative,
)


def numerical_derivative(fn, z, eps=1e-5):
    grad = np.zeros_like(z)
    it = np.nditer(z, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = z[idx]
        z[idx] = orig + eps
        f_plus = fn(z.copy())
        z[idx] = orig - eps
        f_minus = fn(z.copy())
        z[idx] = orig
        grad[idx] = (f_plus[idx] - f_minus[idx]) / (2 * eps)
        it.iternext()
    return grad


class TestReLU:
    def test_forward(self):
        z = np.array([-1.0, 0.0, 2.0])
        assert np.allclose(relu(z), [0.0, 0.0, 2.0])

    def test_derivative(self):
        z = np.random.randn(10, 20)
        analytic = relu_derivative(z)
        numeric = numerical_derivative(relu, z)
        assert np.allclose(analytic, numeric, atol=1e-5)


class TestSigmoid:
    def test_derivative(self):
        z = np.random.randn(5, 10)
        analytic = sigmoid_derivative(z)
        numeric = numerical_derivative(sigmoid, z)
        assert np.allclose(analytic, numeric, atol=1e-5)


class TestTanh:
    def test_derivative(self):
        z = np.random.randn(5, 10)
        analytic = tanh_derivative(z)
        numeric = numerical_derivative(tanh, z)
        assert np.allclose(analytic, numeric, atol=1e-5)


class TestSoftmax:
    def test_sums_to_one(self):
        z = np.random.randn(32, 10)
        p = softmax(z)
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_positive(self):
        z = np.random.randn(16, 10)
        p = softmax(z)
        assert np.all(p >= 0)
