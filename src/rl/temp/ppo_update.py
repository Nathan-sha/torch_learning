import torch.nn as nn
import torch
import torch.nn.functional as F

# 写一个PPO的核心update函数，输入是一批数据(states, actions, old_log_probs, advantages, returns)，
# 输出是actor loss和critic loss。用PyTorch写，不需要完整训练循环，只需要loss计算部分。

def ppo_update(
    policy: nn, # neural network
    critic: nn,
    states: torch.Tensor, #(B, state_dim)
    actions: torch.Tensor, #(B, action_dim)
    old_log_probs: torch.Tensor, #(B, )
    advantages: torch.Tensor, #(B, )
    returns: torch.Tensor, #(B, )
    epsilon: float) -> tuple[torch.Tensor, torch.Tensor]: #(B, )
    # (B, state_dim/action_dim/1/1/1)


    dist = policy.distribution(states)
    # 连续动作，输入状态，得到一个多维动作的概率高斯分布

    new_log_probs = dist.log_prob(actions).sum(dim = -1) # (B,)


    ratio = torch.exp(new_log_probs - old_log_probs)    # (B,)
    actor_loss = -torch.mean((torch.min(ratio * advantages, torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages)))

    values = critic(states).squeeze(-1)
    critic_loss = F.mse_loss(values, returns)

    return actor_loss, critic_loss
