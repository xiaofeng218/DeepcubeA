# DeepCubeA: 基于深度强化学习的魔方求解器

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/DeepcubeA)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

## 项目概述

本项目是 [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf) 方法的复现，训练使用 PyTorch Lightning 框架。该方法结合深度强化学习和搜索算法来解决魔方问题。原始论文展示了如何通过结合神经网络和搜索技术来解决复杂的组合优化问题，如魔方。

## 安装指南

### 训练环境

- Python 3.10.16
- PyTorch 2.5.1
- CUDA (可选，用于加速训练)

### 安装步骤

1. 克隆仓库：

   ```bash
   git clone https://github.com/yourusername/DeepcubeA.git
   cd DeepcubeA
   ```

2. 创建环境并安装依赖项：

   ```bash
   conda create -n deepcubea python=3.10.16
   conda activate deepcubea
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

## 使用方法

### 训练模型

```bash
python train.py
```

### 求解魔方

下载训练好的 [converged_model_K_30.pth](https://drive.google.com/file/d/1eb1SCPS4v-6LkPokrgWh0xfh6GcV_F_a/view?usp=drive_link) 模型，将其放在checkpoint文件夹下。

注意修改 config 中的 model_path 为 `checkpoint/converged_model_K_30.pth`

随机打乱魔方并求解：

```bash
python inference.py --model_path path/to/model.pth
```

指定初始状态求解，可选 action 为 `U, R, F, D, L, B, U_inv, R_inv, F_inv, D_inv, L_inv, B_inv`，多个动作之间用空格分隔。

```bash
python inference.py --model_path path/to/model.pth --actions "U R F D L_inv B_inv"
```

运行`inference.py`脚本后，会生成一个HTML文件 `rubiks_solution.html`，用于可视化求解过程。

### 配置参数说明

主要配置参数 (在config.py中定义)：

- `--batch_size`: 训练批次大小 (默认: 10000)
- `--num_workers`: 数据加载线程数 (默认: 16)
- `--K`: 最大打乱次数 (默认: 30)
- `--max_epochs`: 最大训练轮数 (默认: 100)
- `--learning_rate`: 学习率 (默认: 1e-3)
- `--convergence_threshold`: 收敛阈值 (默认: 0.05)
- `--compile`: 是否编译加速模型 (默认: True)

## 实现细节

详细的实现方法和算法说明请参阅 [Implement.md](Implement.md) 文件，包括：

- 魔方状态表示
- 动作表示
- 深度近似值迭代算法
- 训练伪代码
- BWAS搜索算法
- 神经网络架构

## 结果展示

### 训练结果

不同K值模型收敛（损失小于0.20）所需的epoch数（`1000 step/epoch`）：
  ![k_convergence_epochs](assets/k_convergence_epochs.png)

### 测试结果（K=30训练获得的最终收敛模型）

#### 模型在不同打乱次数下状态输入的cost-to-go预测值统计（平均值，最大值）

  ![model_output](assets/model_output_vs_shuffles.png)

#### 打乱25步的魔方，求解结果及所需时间

按照原文的训练和推理设置，我们的模型目前无法在可接受时间内求解打乱30步的魔方。

有可能是因为在K值较高的情况下，我们设置的收敛阈值过大，模型实际上并未完全收敛就终止了训练，导致模型不能比较好地处理距离较远的状态。

如果在进行推理的过程中发现过长时间没有搜索出结果，请尽量降低打乱的步数。

下面给出的是对一个打乱25步的魔方进行还原的测试结果：

| 指标 | 值 |
| --- | --- |
| 打乱步数 | 25 |
| 解决方案路径长度 | 16 |
| 求解时间 | 28.0831 秒 |
| 打乱路径 | `['B_inv' 'D' 'R' 'D' 'B' 'D_inv' 'U' 'U_inv' 'D' 'U_inv' 'D' 'R' 'R_inv' 'U_inv' 'U' 'F_inv' 'U_inv' 'F_inv' 'R_inv' 'U' 'U' 'F_inv' 'R_inv' 'D_inv' 'D']` |
| 解决方案路径 | `['R', 'F', 'U', 'U', 'R', 'F', 'U', 'F', 'D_inv', 'U', 'B_inv', 'D_inv', 'R_inv', 'D_inv', 'B']` |

[查看魔方还原过程](https://xiaofeng218.github.io/DeepcubeA/assets/rubiks_solution.html)

## 引用

如果您在研究中使用了本项目的代码，请引用原始论文：

```bibtex
@article{agostinelli2019solving,
  title={Solving the Rubik’s cube with deep reinforcement learning and search},
  author={Agostinelli, Forest and McAleer, Stephen and Shmakov, Alexander and Baldi, Pierre},
  journal={Nature Machine Intelligence},
  volume={1},
  number={8},
  pages={356--363},
  year={2019},
  publisher={Nature Publishing Group UK London}
}
```
