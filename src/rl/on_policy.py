from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(x))):
        running = x[idx] + discount * running
        result[idx] = running
    return result


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.actor(obs)
        return Categorical(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def step(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            dist = self.distribution(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value(obs)
        return action, log_prob, value


class RolloutBuffer:
    def __init__(
        self,
        obs_dim: int,
        size: int,
        gamma: float,
        lam: float,
        device: torch.device,
    ) -> None:
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int64)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.max_size = size
        self.ptr = 0
        self.path_start_idx = 0

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
    ) -> None:
        if self.ptr >= self.max_size:
            raise RuntimeError("RolloutBuffer 已满，不能继续写入。")

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.logp_buf[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_value: float = 0.0) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rew_buf[path_slice], last_value)
        values = np.append(self.val_buf[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
        if self.ptr != self.max_size:
            raise RuntimeError("RolloutBuffer 尚未装满，不能读取批数据。")

        advantages = self.adv_buf.copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data = {
            "obs": torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device),
            "act": torch.as_tensor(self.act_buf, dtype=torch.long, device=self.device),
            "ret": torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device),
            "adv": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "logp": torch.as_tensor(self.logp_buf, dtype=torch.float32, device=self.device),
        }
        self.ptr = 0
        self.path_start_idx = 0
        return data


def check_supported_spaces(env) -> tuple[int, int]:
    obs_space = env.observation_space
    act_space = env.action_space

    if not isinstance(obs_space, Box) or len(obs_space.shape) != 1:
        raise ValueError("当前 on-policy 模板仅支持一维 Box 观测空间。")
    if not isinstance(act_space, Discrete):
        raise ValueError("当前 on-policy 模板仅支持 Discrete 动作空间。")

    return obs_space.shape[0], act_space.n
