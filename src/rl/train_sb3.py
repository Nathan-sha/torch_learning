from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.common.cli import add_common_args
from src.common.device import resolve_device
from src.common.paths import make_run_dir, save_args
from src.common.seed import set_seed
from src.rl.envs.make_env import make_env


class MetricsCallback(BaseCallback):
    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.metrics_path = run_dir / "metrics.jsonl"

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            episode_info = self.model.ep_info_buffer[-1]
            with self.metrics_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": self.num_timesteps,
                            "episode_return": float(episode_info["r"]),
                            "episode_length": float(episode_info["l"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stable-baselines3 RL baseline.")
    parser = add_common_args(parser, default_exp_name="sb3")
    parser.set_defaults(device="cpu")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--total-steps", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_dir = make_run_dir(task="rl", exp_name=f"{args.exp_name}_{args.algo}")
    save_args(run_dir, vars(args))

    def env_factory():
        env = make_env(args.env, seed=args.seed)
        return Monitor(env)

    vec_env = DummyVecEnv([env_factory])

    common_kwargs = {
        "env": vec_env,
        "seed": args.seed,
        "verbose": 1,
        "tensorboard_log": str(run_dir / "tb"),
        "device": device.type,
    }
    if args.algo == "ppo":
        model = PPO("MlpPolicy", **common_kwargs)
    else:
        model = DQN("MlpPolicy", **common_kwargs)

    callback = MetricsCallback(run_dir)
    model.learn(total_timesteps=args.total_steps, callback=callback)
    model.save(str(run_dir / "checkpoints" / f"{args.algo}_final"))
    vec_env.close()
    print(f"[RL-SB3] done. output_dir={run_dir}")


if __name__ == "__main__":
    main()
