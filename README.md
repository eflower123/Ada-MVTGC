# Alpha-Freeze 策略：自适应融合的稳定训练方案

基于 `feat-adaptive-fusion` 分支的改进版本，引入 **alpha 冻结策略**解决自适应视图融合在训练后期因权重振荡导致的性能退化问题，并新增训练时长日志回填功能。

## 背景问题

在 `feat-adaptive-fusion` 分支中，alpha 权重（`scoring_fc1` / `scoring_fc2`）在整个训练过程中持续更新。训练后期，随着 beta（熵正则化系数）衰减趋近于零，alpha 容易坍缩到单一视图，导致聚类性能急剧下降。而早期 ACC 高点对应的 alpha 分布往往质量更好。

## Alpha 冻结策略

**核心思路**：训练前期让 alpha 自由学习，当 ACC 连续多轮不再提升时，冻结 alpha 评分网络并恢复到最佳状态，使后续训练专注于 GNN 参数（embedding / delta / cluster_layer）的收敛。

### 两阶段触发机制

1. **前 `--min_train_epochs`（默认 25）轮**：强制不冻结，只追踪历史最优 ACC（`first_best_acc`）及其发生轮次
2. **第 25 轮起**：启动 `--patience`（默认 5）机制——连续 5 轮 ACC 未创新高即触发冻结

### 冻结时的操作

```
追踪 best ACC (first_best_acc) + 备份最佳 scoring 网络参数 (deepcopy)
                ↓
      no_improve_count >= patience 且 epoch >= min_train_epochs - 1 ?
        否 ↓                        是 ↓
      继续更新 alpha              1. 恢复 scoring_fc1 / scoring_fc2 至最佳权重
                                  2. 设置 requires_grad = False（冻结 alpha 参数）
                                  3. 恢复 beta 至冻结时刻值（停止衰减）
                                  4. 从损失中移除 L_ent（熵正则化不再需要）
                                  5. 写入冻结标记行到 Watch 日志
```

### 冻结后继续训练

冻结后 alpha 评分网络不再更新，但 embedding / delta / cluster_layer 正常训练。时序 NCE、L_d、L_x 三项损失继续优化。冻结后 ACC 可能继续提升——因为 GNN 参数仍在基于固定的优质 alpha 进行优化。

### 关键行为变化

| 阶段 | L_ent 计入损失 | beta 衰减 | alpha 更新 | 损失公式 |
|------|--------------|----------|-----------|---------|
| 冻结前 | 是 | 是（指数衰减） | 是 | `L_d + L_x + beta * L_ent` |
| 冻结后 | 否 | 否（保持冻结时刻值） | 否（梯度关闭） | `L_d + L_x` |

## 训练时长日志回填

WatchLogger 在日志头部写入占位行，训练结束时自动回填实际耗时。

### 写入占位行（训练开始时）

```
# Total training time: --                      ← 固定宽度占位
```

### 回填实际耗时（训练结束时 close()）

```
# Total training time: 1h 23m 45s
```

实现通过记录占位行的文件偏移量（`time_placeholder_pos`），`close()` 时 `seek` 回该位置覆盖写入，再 `seek` 回文件末尾。

## 项目结构

```
Ada-MVTGC/
├── code/
│   ├── main.py                    # 新增 --min_train_epochs / --patience 参数
│   └── model/
│       ├── MVTGC.py               # 核心模型（自适应融合 + alpha 冻结逻辑 + 损失分支）
│       ├── DataSet.py             # 数据加载、负采样、多视图特征读取
│       ├── evaluation.py          # 聚类评估（ACC / NMI / ARI / F1）
│       └── WatchLogger.py         # 训练日志（冻结标记 + 时长占位/回填 + 训练摘要）
├── data/                          # 数据集（patent, school, dblp, brain, arxivAI）
├── Watch/                         # 训练日志输出（自动创建）
└── docs/superpowers/              # 设计文档与实现计划
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+（推荐 CUDA）
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
| `--epoch` | 训练轮数 | 100 |
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
| `--min_train_epochs` | alpha 冻结前最少训练轮数 | 25 |
| `--patience` | ACC 无提升连续容忍轮数 | 5 |

## WatchLogger 日志格式

每轮运行在 `Watch/` 目录下生成 `{dataset}_{timestamp}.log` 文件。

### Header

```
# MVTGC Training Log
# Dataset: patent  |  Time: 2026-05-29 10:30:00
# Epochs: 100  |  Batch: 128  |  LR: 0.01  |  Emb: 128  |  Neg: 3  |  Hist: 2
# Init weights: RW=0.50  PE=0.10  MP=0.40
# Total training time: 1h 23m 45s                 ← 训练结束时回填
```

### 数据列

| 列名 | 含义 |
|------|------|
| loss | 总损失 |
| temp_NCE | 时序负采样对比损失 |
| L_d | KL 聚类蒸馏损失 |
| L_x | 特征保真度损失（MSE） |
| L_ent | 熵正则化损失（冻结后为 0） |
| ACC / NMI / ARI / F1 | 聚类指标 |
| RW_mean / RW_std | RW 视图 alpha 均值/标准差 |
| PE_mean / PE_std | PE 视图 alpha 均值/标准差 |
| MP_mean / MP_std | MP 视图 alpha 均值/标准差 |
| beta | 当前熵正则化系数（冻结后不变） |

### 冻结标记

alpha 触发冻结时，在日志中插入标记行：

```
# first_best_ACC epoch 13 (ACC=0.4137), alpha frozen at epoch 25
```

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

## 与 feat-adaptive-fusion 的区别

| 特性 | feat-adaptive-fusion | feat-alpha-freeze |
|------|---------------------|-------------------|
| alpha 更新策略 | 全程更新 | 两阶段（自由学习 → 冻结） |
| L_ent 损失 | 始终计入 | 冻结后移除 |
| beta 衰减 | 全程指数衰减 | 冻结后保持不变 |
| 训练时长记录 | 仅打印到终端 | 回填到日志文件头部 |
| 冻结标记 | 无 | 写入 Watch 日志 |
| 默认 epochs | 30 | 100 |
| 新增 CLI 参数 | 无 | `--min_train_epochs`, `--patience` |

## 设计文档

- `docs/superpowers/specs/2026-05-27-alpha-freeze-design.md` — alpha 冻结策略设计说明
- `docs/superpowers/plans/2026-05-27-alpha-freeze.md` — alpha 冻结实现计划
- `docs/superpowers/specs/2026-05-29-add-training-duration-to-log-design.md` — 训练时长日志设计
- `docs/superpowers/plans/2026-05-29-add-training-duration-to-log.md` — 训练时长日志实现计划

## 引用

原版 MVTGC 论文：

> Meng Liu, Ke Liang, Hao Yu, Lingyuan Meng, Siwei Wang, Sihang Zhou, Xinwang Liu. "Multiview Temporal Graph Clustering." IEEE TNNLS, 2025.

原版代码：https://github.com/MGitHubL/Data4TGC
