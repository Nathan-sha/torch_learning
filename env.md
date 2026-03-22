在 `torch` 文件夹中开展我的深度学习个人研究。

优先复用目前已存在的虚拟环境 `env_isaaclab`，因为 `pytorch` 体积较大，重新下载可能很慢。

## 环境名称

- `env_isaaclab`

## 环境位置

- `D:\anaconda3\envs\env_isaaclab`

## 激活方式

```powershell
conda activate env_isaaclab
```

## 基础信息

- Python: `3.11.15`
- Python 可执行文件: `D:\anaconda3\envs\env_isaaclab\python.exe`
- 平台: `Windows-10-10.0.26200-SP0`

## PyTorch / CUDA

- PyTorch: `2.7.0+cu128`
- TorchVision: `0.22.0+cu128`
- PyTorch 编译 CUDA 版本: `12.8`
- `torch.cuda.is_available()`: `True`
- 可见 GPU 数量: `1`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- NVIDIA 驱动版本: `580.88`
- `nvidia-smi` 报告 CUDA 版本: `13.0`

## 关键深度学习与强化学习依赖

- `isaaclab==0.47.2`
- `isaaclab-assets==0.2.3`
- `isaaclab-mimic==1.0.15`
- `isaaclab-rl==0.4.4`
- `isaaclab-tasks==0.11.6`
- `ray==2.52.1`
- `stable-baselines3==2.7.1`
- `rsl-rl-lib==3.0.1`
- `skrl==1.4.3`
- `rl-games==1.6.1`
- `transformers==5.3.0`
- `tensorboard==2.20.0`
- `wandb==0.25.1`
- `onnx==1.20.1`
- `pandas==3.0.1`
- `einops==0.8.2`

## 说明

- 该环境已经可以正常识别 CUDA，可直接用于 PyTorch GPU 训练。
- 该环境偏向 Isaac Lab / 强化学习 / 深度学习研究用途，适合先直接复用，再按项目逐步补充依赖。
