"""Load and preprocess MNIST (no deep-learning framework required)."""

from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = CACHE_DIR / "mnist.npz"


def load_mnist(normalize: bool = True) -> Tuple[np.ndarray, ...]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists():
        print(f"Downloading MNIST to {CACHE_FILE} ...")
        urlretrieve(MNIST_URL, CACHE_FILE)

    with np.load(CACHE_FILE) as data:
        x_train = data["x_train"].reshape(-1, 784).astype(np.float64)
        y_train = data["y_train"].astype(np.int64)
        x_test = data["x_test"].reshape(-1, 784).astype(np.float64)
        y_test = data["y_test"].astype(np.int64)

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    return x_train, y_train, x_test, y_test
