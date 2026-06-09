"""Train MLP on MNIST and save plots / metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlp.data import load_mnist
from mlp.metrics import confusion_matrix
from mlp.network import MLP

RESULTS_DIR = Path(__file__).parent / "results"


def plot_history(history: dict, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — {name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy — {name}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / f"history_{name}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


def plot_confusion_matrix(y_true, y_pred, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {name}")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    path = RESULTS_DIR / f"confusion_{name}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


def run_experiment(
    name: str,
    layer_sizes: list,
    lr: float,
    optimizer: str,
    epochs: int,
    batch_size: int,
    activation: str = "relu",
    seed: int = 42,
) -> dict:
    x_train, y_train, x_test, y_test = load_mnist()
    # Use test set as validation during training (common for MNIST homework)
    model = MLP(
        layer_sizes=layer_sizes,
        activation=activation,
        learning_rate=lr,
        optimizer=optimizer,
        seed=seed,
    )

    # Gradient check on small subset
    X_check = x_train[:5]
    y_check = y_train[:5]
    passed, err = model.gradient_check(X_check, y_check)
    print(f"Gradient check ({name}): passed={passed}, max_rel_error={err:.2e}")

    history = model.train(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
    )

    test_acc = model.accuracy(x_test, y_test)
    preds = model.predict(x_test)
    plot_history(history, name)
    plot_confusion_matrix(y_test, preds, name)

    result = {
        "name": name,
        "layer_sizes": layer_sizes,
        "lr": lr,
        "optimizer": optimizer,
        "activation": activation,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_accuracy": test_acc,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "gradient_check_passed": bool(passed),
        "gradient_check_error": float(err),
    }
    print(f"Test accuracy ({name}): {test_acc:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--experiment", default="all", choices=["baseline", "adam", "deep", "all"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    experiments = []
    if args.experiment in ("baseline", "all"):
        experiments.append(
            ("baseline_sgd", [784, 256, 128, 10], 0.1, "sgd", args.epochs)
        )
    if args.experiment in ("adam", "all"):
        experiments.append(
            ("adam_lr001", [784, 256, 128, 10], 0.001, "adam", args.epochs)
        )
    if args.experiment in ("deep", "all"):
        experiments.append(
            ("deep_3hidden", [784, 512, 256, 128, 10], 0.05, "sgd", max(args.epochs, 20))
        )

    results = []
    for name, sizes, lr, opt, epochs in experiments:
        results.append(
            run_experiment(
                name,
                sizes,
                lr,
                opt,
                epochs,
                args.batch_size,
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "experiments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
