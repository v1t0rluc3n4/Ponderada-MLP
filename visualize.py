"""Visualizações: PCA (NumPy) e t-SNE opcional (scikit-learn, se disponível)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlp.data import load_mnist
from mlp.metrics import pca_2d
from mlp.network import MLP

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP([784, 256, 128, 10], learning_rate=0.1, optimizer="sgd", seed=42)
    model.train(
        x_train[:10000], y_train[:10000], x_test, y_test,
        epochs=5, batch_size=128, verbose=False,
    )

    n_samples = 2000
    X = x_test[:n_samples]
    y = y_test[:n_samples]
    hidden = model.get_hidden_representation(X, layer=-1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    emb_pca = pca_2d(hidden)
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], c=y, cmap="tab10", s=5, alpha=0.7)
    ax.set_title("PCA — última camada oculta (128D → 2D)")
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_embeddings.png", dpi=120)
    plt.close()
    print(f"Saved {RESULTS_DIR / 'pca_embeddings.png'}")

    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
        emb_tsne = tsne.fit_transform(hidden[:1000])
        fig, ax = plt.subplots(figsize=(8, 7))
        scatter = ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=y[:1000], cmap="tab10", s=8, alpha=0.7)
        ax.set_title("t-SNE — última camada oculta (1000 amostras)")
        plt.colorbar(scatter, ax=ax)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tsne_embeddings.png", dpi=120)
        plt.close()
        print(f"Saved {RESULTS_DIR / 'tsne_embeddings.png'}")
    except ImportError as e:
        print("t-SNE ignorado (scikit-learn indisponível neste ambiente):", e)


if __name__ == "__main__":
    main()
