from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def make_run_dir(task: str, exp_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUTS_ROOT / task / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def save_args(run_dir: Path, args_dict: dict) -> None:
    output_path = run_dir / "args.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
