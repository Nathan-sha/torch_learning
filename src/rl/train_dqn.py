from __future__ import annotations

import argparse
import random
import sys
from collections import deque, namedtuple
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

Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple DQN agent.")
    parser = add_common_args(parser, default_exp_name="dqn")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--train-interval", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    mix = min(step / max(decay_steps, 1), 1.0)
    return start + mix * (end - start)


def to_tensor(array, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    env = make_env(args.env, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    run_dir = make_run_dir(task="rl", exp_name=args.exp_name)
    save_args(run_dir, vars(args))
    logger = ExperimentLogger(run_dir)

    policy_net = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    target_net = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(args.buffer_size)

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    episode_idx = 0

    for global_step in range(1, args.total_steps + 1):
        epsilon = linear_epsilon(
            global_step,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(to_tensor(obs, device).unsqueeze(0))
                action = int(q_values.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_return += reward
        episode_length += 1

        if done:
            episode_idx += 1
            logger.log_metrics(
                episode_idx,
                {
                    "episode/return": float(episode_return),
                    "episode/length": float(episode_length),
                    "exploration/epsilon": float(epsilon),
                },
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        if (
            global_step >= args.warmup_steps
            and len(replay_buffer) >= args.batch_size
            and global_step % args.train_interval == 0
        ):
            batch = replay_buffer.sample(args.batch_size)
            obs_batch = to_tensor(batch.obs, device)
            action_batch = torch.as_tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
            reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
            next_obs_batch = to_tensor(batch.next_obs, device)
            done_batch = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = policy_net(obs_batch).gather(1, action_batch)
            with torch.no_grad():
                next_q_values = target_net(next_obs_batch).max(dim=1, keepdim=True)[0]
                target_values = reward_batch + args.gamma * (1.0 - done_batch) * next_q_values

            loss = nn.functional.mse_loss(q_values, target_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log_metrics(
                global_step,
                {
                    "train/loss": float(loss.item()),
                    "exploration/epsilon_step": float(epsilon),
                },
            )

        if global_step % args.target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if global_step % args.log_interval == 0:
            print(
                f"[RL-DQN] step={global_step}/{args.total_steps} "
                f"episodes={episode_idx} epsilon={epsilon:.3f} buffer={len(replay_buffer)}"
            )

    save_checkpoint(
        run_dir / "checkpoints" / "policy_final.pt",
        {
            "model_state": policy_net.state_dict(),
            "args": vars(args),
        },
    )
    logger.close()
    env.close()
    print(f"[RL-DQN] done. output_dir={run_dir}")


if __name__ == "__main__":
    main()
