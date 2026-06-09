"""Multi-Layer Perceptron implemented with NumPy only."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .activations import get_activation, softmax
from .losses import cross_entropy_gradient, cross_entropy_loss
from .optimizers import get_optimizer


class MLP:
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        learning_rate: float = 0.1,
        optimizer: str = "sgd",
        seed: Optional[int] = 42,
    ):
        if len(layer_sizes) < 3:
            raise ValueError("Need input, at least one hidden, and output layer sizes.")
        if layer_sizes[-1] != 10:
            raise ValueError("Output layer must have 10 neurons for MNIST.")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.act_fn, self.act_deriv = get_activation(activation)

        if seed is not None:
            np.random.seed(seed)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            if i < self.n_layers - 1:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros((1, fan_out)))

        self.optimizer = get_optimizer(optimizer, learning_rate)
        self._cache: Dict[str, list] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. Returns softmax probabilities."""
        self._cache = {"z": [], "a": [X]}
        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self._cache["z"].append(z)
            if i < self.n_layers - 1:
                a = self.act_fn(z)
            else:
                a = softmax(z)
            self._cache["a"].append(a)
        return a

    def backward(self, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backpropagation. Returns (dW list, db list)."""
        probs = self._cache["a"][-1]
        dz = cross_entropy_gradient(probs, y)

        dW: List[np.ndarray] = []
        db: List[np.ndarray] = []

        for i in reversed(range(self.n_layers)):
            a_prev = self._cache["a"][i]
            dW.insert(0, a_prev.T @ dz)
            db.insert(0, np.sum(dz, axis=0, keepdims=True))

            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self.act_deriv(self._cache["z"][i - 1])

        return dW, db

    def update(self, dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        params = self.weights + self.biases
        grads = dW + db
        self.optimizer.step(params, grads)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.forward(X)
        loss = cross_entropy_loss(probs, y)
        dW, db = self.backward(y)
        self.update(dW, db)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        labels = y if y.ndim == 1 else np.argmax(y, axis=1)
        return float(np.mean(preds == labels))

    def evaluate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return cross_entropy_loss(self.forward(X), y)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        n = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                Xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                epoch_loss += self.train_step(Xb, yb)
                n_batches += 1

            train_loss = epoch_loss / n_batches
            val_loss = self.evaluate_loss(X_val, y_val)
            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                    f"acc: {train_acc:.4f} | val_acc: {val_acc:.4f}"
                )

        return history

    def gradient_check(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-5,
        tolerance: float = 1e-5,
        num_checks: int = 20,
    ) -> Tuple[bool, float]:
        """
        Numerical gradient check on a random subset of parameters.
        Returns (passed, max_relative_error).
        """
        self.forward(X)
        dW_analytic, db_analytic = self.backward(y)
        max_error = 0.0

        def loss_at_params():
            return cross_entropy_loss(self.forward(X), y)

        all_params = self.weights + self.biases
        all_grads = dW_analytic + db_analytic

        rng = np.random.default_rng(0)
        for _ in range(num_checks):
            layer_idx = int(rng.integers(0, len(all_params)))
            param = all_params[layer_idx]
            grad = all_grads[layer_idx]
            flat_idx = int(rng.integers(0, param.size))
            idx = np.unravel_index(flat_idx, param.shape)

            orig = param[idx]
            param[idx] = orig + epsilon
            loss_plus = loss_at_params()
            param[idx] = orig - epsilon
            loss_minus = loss_at_params()
            param[idx] = orig

            num_grad = (loss_plus - loss_minus) / (2 * epsilon)
            ana_grad = grad[idx]
            denom = max(abs(num_grad), abs(ana_grad), 1e-8)
            rel_err = abs(num_grad - ana_grad) / denom
            max_error = max(max_error, rel_err)

        return max_error < tolerance, float(max_error)

    def get_hidden_representation(self, X: np.ndarray, layer: int = -1) -> np.ndarray:
        """Return activations after hidden layer `layer` (0 = first hidden, -1 = last hidden)."""
        a = X
        hidden_count = 0
        n_hidden = self.n_layers - 1
        for i in range(n_hidden):
            z = a @ self.weights[i] + self.biases[i]
            a = self.act_fn(z)
            target = layer if layer >= 0 else n_hidden + layer
            if hidden_count == target:
                return a
            hidden_count += 1
        return a
