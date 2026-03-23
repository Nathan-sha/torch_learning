from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    parser = argparse.ArgumentParser(description="Train a PPO-Clip agent.")
    parser = add_common_args(parser, default_exp_name="ppo_clip")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.policy_lr)
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
        dataset = TensorDataset(
            batch["obs"],
            batch["act"],
            batch["ret"],
            batch["adv"],
            batch["logp"],
        )
        loader = DataLoader(dataset, batch_size=min(args.batch_size, args.steps_per_epoch), shuffle=True)

        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        approx_kl_value = 0.0
        clip_frac_value = 0.0
        update_count = 0

        for _ in range(args.train_iters):
            should_stop = False
            for obs_batch, act_batch, ret_batch, adv_batch, logp_old_batch in loader:
                dist = model.distribution(obs_batch)
                logp = dist.log_prob(act_batch)
                ratio = torch.exp(logp - logp_old_batch)

                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_loss = -(torch.min(ratio * adv_batch, clipped_ratio * adv_batch)).mean()

                value_pred = model.value(obs_batch)
                value_loss = value_loss_fn(value_pred, ret_batch)
                entropy = dist.entropy().mean()

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (logp_old_batch - logp).mean().abs()
                    clip_frac = ((ratio > 1.0 + args.clip_ratio) | (ratio < 1.0 - args.clip_ratio)).float().mean()

                policy_loss_value += float(policy_loss.item())
                value_loss_value += float(value_loss.item())
                entropy_value += float(entropy.item())
                approx_kl_value += float(approx_kl.item())
                clip_frac_value += float(clip_frac.item())
                update_count += 1

                if approx_kl.item() > 1.5 * args.target_kl:
                    should_stop = True
                    break

            if should_stop:
                break

        denom = max(update_count, 1)
        avg_return = float(np.mean(epoch_returns)) if epoch_returns else 0.0
        avg_length = float(np.mean(epoch_lengths)) if epoch_lengths else 0.0

        logger.log_metrics(
            epoch,
            {
                "train/policy_loss": policy_loss_value / denom,
                "train/value_loss": value_loss_value / denom,
                "train/entropy": entropy_value / denom,
                "train/approx_kl": approx_kl_value / denom,
                "train/clip_frac": clip_frac_value / denom,
                "epoch/avg_return": avg_return,
                "epoch/avg_length": avg_length,
            },
        )

        if (epoch + 1) % args.log_interval == 0:
            print(
                f"[RL-PPO] epoch={epoch + 1}/{args.epochs} "
                f"avg_return={avg_return:.2f} avg_length={avg_length:.2f} "
                f"policy_loss={policy_loss_value / denom:.4f} "
                f"value_loss={value_loss_value / denom:.4f} "
                f"kl={approx_kl_value / denom:.4f}"
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
    print(f"[RL-PPO] done. output_dir={run_dir}")


if __name__ == "__main__":
    main()
