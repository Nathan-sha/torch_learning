from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.checkpoint import save_checkpoint
from src.common.cli import add_common_args
from src.common.device import resolve_device
from src.common.logger import ExperimentLogger
from src.common.paths import make_run_dir, save_args
from src.common.seed import set_seed
from src.rl.envs.make_env import make_env
from src.rl.on_policy import ActorCritic, RolloutBuffer, check_supported_spaces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VPG agent with a value baseline.")
    parser = add_common_args(parser, default_exp_name="vpg")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-iters", type=int, default=80)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


def to_obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    env = make_env(args.env, seed=args.seed)
    obs_dim, act_dim = check_supported_spaces(env)

    run_dir = make_run_dir(task="rl", exp_name=args.exp_name)
    save_args(run_dir, vars(args))
    logger = ExperimentLogger(run_dir)

    model = ActorCritic(obs_dim, act_dim, args.hidden_dim).to(device)
    policy_optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.policy_lr)
    value_optimizer = torch.optim.Adam(model.critic.parameters(), lr=args.value_lr)
    value_loss_fn = nn.MSELoss()

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    total_episodes = 0

    for epoch in range(args.epochs):
        buffer = RolloutBuffer(obs_dim, args.steps_per_epoch, args.gamma, args.lam, device)
        epoch_returns: list[float] = []
        epoch_lengths: list[int] = []

        for step in range(args.steps_per_epoch):
            obs_tensor = to_obs_tensor(obs, device)
            action_tensor, log_prob_tensor, value_tensor = model.step(obs_tensor)

            action = int(action_tensor.item())
            log_prob = float(log_prob_tensor.item())
            value = float(value_tensor.item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.store(obs, action, reward, value, log_prob)
            episode_return += reward
            episode_length += 1
            obs = next_obs

            timeout = step == args.steps_per_epoch - 1
            if done or timeout:
                if done:
                    last_value = 0.0
                else:
                    with torch.no_grad():
                        last_value = float(model.value(to_obs_tensor(obs, device)).item())

                buffer.finish_path(last_value=last_value)

                if done:
                    total_episodes += 1
                    epoch_returns.append(float(episode_return))
                    epoch_lengths.append(episode_length)
                    logger.log_metrics(
                        total_episodes,
                        {
                            "episode/return": float(episode_return),
                            "episode/length": float(episode_length),
                        },
                    )
                    obs, _ = env.reset()
                    episode_return = 0.0
                    episode_length = 0

        batch = buffer.get()
        dist = model.distribution(batch["obs"])
        log_prob = dist.log_prob(batch["act"])
        policy_loss = -(log_prob * batch["adv"]).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_loss = torch.tensor(0.0, device=device)
        for _ in range(args.value_iters):
            values = model.value(batch["obs"])
            value_loss = value_loss_fn(values, batch["ret"])
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        entropy = dist.entropy().mean()
        approx_kl = (batch["logp"] - log_prob.detach()).mean().abs()
        avg_return = float(np.mean(epoch_returns)) if epoch_returns else 0.0
        avg_length = float(np.mean(epoch_lengths)) if epoch_lengths else 0.0

        logger.log_metrics(
            epoch,
            {
                "train/policy_loss": float(policy_loss.item()),
                "train/value_loss": float(value_loss.item()),
                "train/entropy": float(entropy.item()),
                "train/approx_kl": float(approx_kl.item()),
                "epoch/avg_return": avg_return,
                "epoch/avg_length": avg_length,
            },
        )

        if (epoch + 1) % args.log_interval == 0:
            print(
                f"[RL-VPG] epoch={epoch + 1}/{args.epochs} "
                f"avg_return={avg_return:.2f} avg_length={avg_length:.2f} "
                f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}"
            )

    save_checkpoint(
        run_dir / "checkpoints" / "policy_final.pt",
        {
            "model_state": model.state_dict(),
            "args": vars(args),
        },
    )
    logger.close()
    env.close()
    print(f"[RL-VPG] done. output_dir={run_dir}")


if __name__ == "__main__":
    main()
