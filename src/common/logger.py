from __future__ import annotations

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        record = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()
