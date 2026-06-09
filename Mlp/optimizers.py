"""Optimizers for weight updates."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class SGDMomentum:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity: Dict[int, np.ndarray] = {}

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        for i in range(len(params)):
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(params[i])
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]


class Adam:
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        self.t += 1
        for i in range(len(params)):
            if i not in self.m:
                self.m[i] = np.zeros_like(params[i])
                self.v[i] = np.zeros_like(params[i])
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def get_optimizer(name: str, learning_rate: float):
    name = name.lower()
    if name == "sgd":
        return SGD(learning_rate)
    if name in ("momentum", "sgd_momentum"):
        return SGDMomentum(learning_rate)
    if name == "adam":
        return Adam(learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")
