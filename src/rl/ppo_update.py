import torch.nn as nn
import torch

# 写一个PPO的核心update函数，输入是一批数据(states, actions, old_log_probs, advantages, returns)，
# 输出是actor loss和critic loss。用PyTorch写，不需要完整训练循环，只需要loss计算部分。

def ppo_update(
    states: torch.Tensor, #(B, state_dim)
    actions: torch.Tensor, #(B, action_dim)
    old_log_probs: torch.Tensor, #(B, )
    advantages: torch.Tensor, #(B, )
    returns: torch.Tensor): #(B, )
    # (B, state_dim/action_dim/1/1/1)
    epsilon = 0.2

    new_log_probs = dist.log_prob(actions).sum(dim=-1)  # (B,)

    ratio = torch.exp(new_log_probs - old_log_probs)    # (B,)
    ratio = 

    actor_loss = -(min(ratio * advantages, torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages))


    return actor_loss, critic_loss


# import torch
# import torch.nn.functional as F

# from src.rl.on_policy import ActorCritic


# def ppo_update(
#     model: ActorCritic,
#     states: torch.Tensor,  # (B, state_dim)
#     actions: torch.Tensor,  # (B,) int64
#     old_log_probs: torch.Tensor,  # (B,)
#     advantages: torch.Tensor,  # (B,)
#     returns: torch.Tensor,  # (B,)
#     epsilon: float = 0.2,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     dist = model.distribution(states)
#     new_log_probs = dist.log_prob(actions)

#     ratio = torch.exp(new_log_probs - old_log_probs)
#     clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
#     actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()

#     values = model.value(states)
#     critic_loss = F.mse_loss(values, returns)

#     return actor_loss, critic_loss