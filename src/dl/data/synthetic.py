from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_classification_dataset(
    num_samples: int = 2048,
    input_dim: int = 2,
    noise: float = 0.6,
    seed: int = 42,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    half = num_samples // 2

    class0 = rng.normal(loc=-1.5, scale=noise, size=(half, input_dim))
    class1 = rng.normal(loc=1.5, scale=noise, size=(num_samples - half, input_dim))

    # Add a light nonlinear twist so the task is not entirely trivial.
    if input_dim >= 2:
        class0[:, 1] += np.sin(class0[:, 0]) * 0.8
        class1[:, 1] += math.cos(1.2) + np.sin(class1[:, 0]) * 0.8

    x = np.concatenate([class0, class1], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros((half,), dtype=np.int64),
            np.ones((num_samples - half,), dtype=np.int64),
        ],
        axis=0,
    )

    indices = rng.permutation(num_samples)
    x = x[indices]
    y = y[indices]

    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
