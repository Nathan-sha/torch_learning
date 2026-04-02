# Torch 小实验框架

我是神经网络新手，基于该项目框架，逐步帮助我学会pytorch编程。
注意你的输出需要一步一步引导我了解目前的框架，同时又熟练掌握核心知识
从手写数字识别开始
注意不要一次性输出太多东西，不要直接输出答案，引导我学习python 神经网络编程

这个项目用于在当前 `env_isaaclab` 环境中快速开展深度学习（DL）和强化学习（RL）小实验。

## 目标

- 用尽量少的样板代码启动新实验
- 统一管理随机种子、设备、日志、checkpoint 和输出目录
- 同时支持原生 PyTorch 的 DL/RL 模板，以及 `stable-baselines3` 的 RL baseline

## 目录结构

```text
torch/
  env.md
  outputs/
  requirements_note.md
  scripts/
    run_dl.ps1
    run_rl.ps1
  src/
    common/
    dl/
    rl/
```

## 快速开始

```powershell
conda activate env_isaaclab
python -m src.dl.train_classifier --epochs 20 --batch-size 64 --device cuda
python -m src.dl.train_classifier --dataset mnist --epochs 5 --batch-size 128 --device cuda
python -m src.rl.train_dqn --env CartPole-v1 --total-steps 20000 --device cuda
python -m src.rl.train_vpg --env CartPole-v1 --epochs 50 --steps-per-epoch 2000 --device cpu
python -m src.rl.train_ppo_clip --env CartPole-v1 --epochs 50 --steps-per-epoch 2000 --device cpu
python -m src.rl.train_sb3 --env CartPole-v1 --algo ppo --total-steps 50000 --device cpu
```

也可以用脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_dl.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_rl.ps1 -Algo dqn
```

## 输出约定

每次运行会自动生成目录：

```text
outputs/<task>/<exp_name>/<timestamp>/
```

其中通常包含：

- `args.json`：本次实验参数
- `metrics.jsonl`：逐步记录的指标
- `checkpoints/`：保存的模型
- TensorBoard 日志文件

## 当前内置实验

- `src.dl.train_classifier`
  - 一个支持合成二维分类数据和 `MNIST` 的 MLP 训练模板
- `src.rl.train_dqn`
  - 一个尽量简洁、便于修改的原生 PyTorch DQN 模板
- `src.rl.train_vpg`
  - 一个带 value baseline 的原生 VPG 模板，适合 on-policy 小实验
- `src.rl.train_ppo_clip`
  - 一个原生 PPO-Clip 模板，包含 GAE、clip objective 和多轮更新
- `src.rl.train_sb3`
  - 一个快速 RL baseline 入口，封装 `stable-baselines3`

## 当前 RL 范围

- `DQN`
  - 适合离散动作空间的小实验
- `VPG`
  - 当前实现为离散动作空间、一维 Box 观测空间的 on-policy 版本
- `PPO-Clip`
  - 当前实现为离散动作空间、一维 Box 观测空间的 on-policy 版本

示例：

```powershell
python -m src.rl.train_vpg --env CartPole-v1 --epochs 20 --steps-per-epoch 2000 --device cpu
python -m src.rl.train_ppo_clip --env CartPole-v1 --epochs 20 --steps-per-epoch 2000 --device cpu
```

## 扩展建议

- 新增 DL 实验时，复用 `src/common/` 中的公共能力，替换数据集与模型即可
- 新增 RL 实验时，可继续复用 `src/rl/envs/make_env.py`
- 如果后续要接入 `Isaac Lab`，建议在 `src/rl/envs/` 下新增独立环境工厂而不是修改现有轻量流程

## 当前可用 DL 数据集

- `synthetic`
  - 默认数据集，二维合成分类数据，适合快速验证训练流程
- `mnist`
  - 手写数字分类数据集，首次运行会自动下载到 `data/`

示例：

```powershell
python -m src.dl.train_classifier --dataset mnist --epochs 3 --batch-size 128 --device cuda
```
