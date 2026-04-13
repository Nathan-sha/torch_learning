import torch.nn as nn
import torch
import torch.nn.functional as F

# 手写GAE（Generalized Advantage Estimation），
# 输入是一条trajectory的rewards和values，输出advantages。λ和γ是参数。

def GAE(rewards: torch.Tensor, # (B, N)
        states: torch.Tensor, # 
        critic: nn,
        lam: float,
        gamma: float) -> tuple[torch.Tensor]: # (B, N+1)
    
    values = critic(states) # (B, N+1)
    T = rewards.shape[1]
    for t in reversed(range(T)):
        delta_t = rewards[:,t] + gamma * values[:, t+1] - values[:, t]
    
    gae = torch.zeros_like(rewards)

    for t in reversed(range(T)):
        i = T-t
        gae[:,t] = delta_t[t]
        for j in range(i-1):
            gae[:,t] += lam * gamma * delta_t[t+j+1]
        
    
    return gae