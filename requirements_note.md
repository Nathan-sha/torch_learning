# 环境说明

当前框架默认复用已有的 `env_isaaclab` 环境，不额外强制创建新环境。

本项目第一版默认依赖这些已存在的包：

- `torch`
- `torchvision`
- `gymnasium`
- `stable-baselines3`
- `tensorboard`

如果后续某个实验需要额外依赖，建议单独记录到该实验的说明文件中，而不是一开始把整个环境继续做重。
