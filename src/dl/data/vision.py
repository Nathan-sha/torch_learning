from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def make_mnist_datasets(data_root: str | Path = "data") -> tuple[Dataset, Dataset]:
    root = Path(data_root)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=str(root),
        train=True,
        download=True,
        transform=transform,
    )
    val_dataset = datasets.MNIST(
        root=str(root),
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, val_dataset
