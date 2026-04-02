from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_run_metrics_plots(run_dir: Path) -> list[Path]:
    """Read ``run_dir / metrics.jsonl`` and save curve figures under ``run_dir / plots``."""
    metrics_path = run_dir / "metrics.jsonl"
    records = load_metrics_jsonl(metrics_path)
    if not records:
        return []

    steps = [int(r["step"]) for r in records]
    keys: list[str] = []
    for r in records:
        for k, v in r.items():
            if k == "step" or not isinstance(v, (int, float)):
                continue
            if k not in keys:
                keys.append(k)

    if not keys:
        return []

    series = {k: [float(r[k]) for r in records] for k in keys}
    loss_keys = [k for k in keys if "loss" in k.lower()]
    acc_keys = [k for k in keys if "acc" in k.lower()]
    other_keys = [k for k in keys if k not in loss_keys and k not in acc_keys]

    panels: list[tuple[str, list[str]]] = []
    if loss_keys:
        panels.append(("loss", loss_keys))
    if acc_keys:
        panels.append(("accuracy", acc_keys))
    if other_keys:
        panels.append(("value", other_keys))

    if not panels:
        return []

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.5 * len(panels)), squeeze=False)
    for ax, (ylabel, ks) in zip(axes.ravel(), panels):
        for k in ks:
            ax.plot(steps, series[k], label=k, marker="o", markersize=2)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "metrics_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]
