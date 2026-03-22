from __future__ import annotations

import torch


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a user-facing device string into a torch device."""
    normalized = device_arg.strip().lower()

    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动回退到 CPU。")
        return torch.device("cpu")

    return torch.device(normalized)
