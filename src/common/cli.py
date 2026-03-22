from __future__ import annotations

import argparse


def add_common_args(parser: argparse.ArgumentParser, default_exp_name: str) -> argparse.ArgumentParser:
    parser.add_argument("--exp-name", type=str, default=default_exp_name)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    return parser
