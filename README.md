# Ada-MVTGC: 自适应多视图时序图聚类

基于 MVTGC（IEEE TNNLS 2025）的改进版本，将手工设定的静态视图权重替换为**可学习的自适应视图融合机制**，并新增训练监控模块（WatchLogger），实时追踪视图权重变化与聚类指标。

## 改进点

| 模块 | 原版 MVTGC | Ada-MVTGC |
|------|-----------|-----------|
| 视图融合 | 手工设定 `r_RW`、`r_PE`（每个数据集调参） | MLP 自适应学习每节点视图权重（`compute_alpha`） |
| 熵正则化 | 无 | 基于 beta 衰减的熵正则化，防止权重坍缩 |
| 训练监控 | 仅终端打印 loss + 聚类指标 | WatchLogger 完整记录每轮 alpha 统计、分量损失、聚类指标 |
| 损失分量 | 未单独追踪 | 每条日志含 temporal NCE / L_d / L_x / L_ent 四项分量 |

### 自适应融合架构

```
三视图特征 (RW / PE / MP)
      ↓
 scoring_fc1 → ReLU → scoring_fc2   ← 两层 MLP
      ↓
 softmax(s / tau) → α_rw, α_pe, α_a
      ↓
 加权融合嵌入 + 熵正则化
```

alpha 权重不在 main.py 里手动指定，而是由模型根据数据自动学习。训练过程中 WatchLogger 记录每轮各视图 alpha 的均值/标准差，可观察权重从初始均匀分布到收敛的全过程。

## 项目结构

```
Ada-MVTGC/
├── code/
│   ├── main.py                    # 训练入口，超参数配置
│   └── model/
│       ├── MVTGC.py               # 核心模型（自适应融合 + 训练逻辑）
│       ├── DataSet.py             # 数据加载、负采样、多视图特征读取
│       ├── evaluation.py          # 聚类评估（ACC / NMI / ARI / F1）
│       └── WatchLogger.py         # 训练日志模块
├── data/                          # 数据集（patent, school, dblp, brain, arxivAI）
├── Watch/                         # 训练日志输出（自动创建）
└── docs/superpowers/              # 设计文档与实现计划
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+（CUDA 推荐）
- scikit-learn
- numpy
- munkres

```bash
conda create -n MVCTgc python=3.10
conda activate MVCTgc
pip install torch numpy scikit-learn munkres
```

## 数据格式

时序边文件 `data/{dataset}/{dataset}.txt`：

```
source_node  target_node  timestamp
162          212          0.000000
0            275          0.000000
```

多视图特征 `data/{dataset}/MVC Features/` 下三个文件：
- `View_RW.txt` — 随机游走嵌入（局部结构）
- `View_PE.txt` — 位置编码嵌入（全局位置）
- `View_MP.txt` — 消息传递嵌入（邻居聚合）

标签文件：`data/{dataset}/label.txt`、`data/{dataset}/node2label.txt`

## 使用方法

```bash
cd code
python main.py
```

`main.py` 中可配置的参数：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名 | patent |
| `--r_RW` | RW 视图初始融合权重 | 0.5 |
| `--r_PE` | PE 视图初始融合权重 | 0.1 |
| `--epoch` | 训练轮数 | 30 |
| `--batch_size` | 批次大小 | 128 |
| `--emb_size` | 嵌入维度 | 128 |
| `--neg_size` | 负采样数 | 3 |
| `--hist_len` | 历史邻居窗口 | 2 |
| `--learning_rate` | 学习率 | 0.01 |
| `--d_a` | 视图评分 MLP 隐藏维度 | 32 |
| `--tau` | softmax 温度系数 | 1.0 |
| `--beta_0` | 初始熵正则化权重 | 1.0 |
| `--rho` | beta 衰减率 | 0.05 |
| `--beta_min` | beta 下界 | 0.01 |

## WatchLogger 日志格式

每轮运行会在 `Watch/` 目录下生成 `{dataset}_{timestamp}.log` 文件。

### Header

```
# MVTGC Training Log
# Dataset: patent  |  Time: 2026-05-27 19:11:06
# Epochs: 30  |  Batch: 128  |  LR: 0.01  |  Emb: 128  |  Neg: 3  |  Hist: 2
# Init weights: RW=0.50  PE=0.10  MP=0.40
```

### 数据列

| 列名 | 含义 |
|------|------|
| loss | 总损失 |
| temp_NCE | 时序负采样对比损失 |
| L_d | KL 聚类蒸馏损失 |
| L_x | 特征保真度损失（MSE） |
| L_ent | 熵正则化损失 |
| ACC / NMI / ARI / F1 | 聚类指标 |
| RW_mean / RW_std | RW 视图 alpha 均值/标准差 |
| PE_mean / PE_std | PE 视图 alpha 均值/标准差 |
| MP_mean / MP_std | MP 视图 alpha 均值/标准差 |
| beta | 当前熵正则化系数 |

### 训练摘要

日志末尾自动附加 6 条摘要记录：

```
# best_ACC epoch 26
26        0.9907    0.5040    0.3239    0.2794    0.3937    ...
# second_ACC epoch 33
33        0.9749    0.4908    0.3479    0.2795    0.3862    ...
# best_NMI epoch 33
33        0.9749    0.4908    0.3479    0.2795    0.3862    ...
# best_ARI epoch 26
26        0.9907    0.5040    0.3239    0.2794    0.3937    ...
# best_F1 epoch 26
26        0.9907    0.5040    0.3239    0.2794    0.3937    ...
# best_AVG epoch 26  avg:0.3753
26        0.9907    0.5040    0.3239    0.2794    0.3937    ...
```

## 引用

原版 MVTGC 论文：

> Meng Liu, Ke Liang, Hao Yu, Lingyuan Meng, Siwei Wang, Sihang Zhou, Xinwang Liu. "Multiview Temporal Graph Clustering." IEEE TNNLS, 2025.

原版代码：https://github.com/MGitHubL/Data4TGC
