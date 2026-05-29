# MVTGC: 多视图时序图聚类

基于论文 **"Multiview Temporal Graph Clustering"** (IEEE TNNLS 2025) 的官方实现，使用静态手工设定的视图权重进行多视图融合。

> 数据集采用 Data4TGC，感谢开源！链接：https://github.com/MGitHubL/Deep-Temporal-Graph-Clustering

## 核心思想

将时序图拆分为三个互补视图（随机游走 RW、位置编码 PE、消息传递 MP），分别学习节点嵌入后通过手工设定的权重参数进行加权融合，最后联合优化聚类损失和图结构损失。

## 项目结构

```
MVTGC/
├── code/
│   ├── main.py                    # 训练入口，超参数配置
│   └── model/
│       ├── MVTGC.py               # 核心模型（GCN 编码器、时序对比、聚类蒸馏）
│       ├── DataSet.py             # 数据加载、负采样、多视图特征读取
│       └── evaluation.py          # 聚类评估（ACC / NMI / ARI / F1）
├── data/                          # 数据集（patent, school, dblp, brain, arxivAI）
└── docs/                          # 设计文档
```

## 三视图融合

```
节点特征 → GCN 编码 → 三视图嵌入 (H_RW, H_PE, H_MP)
                              ↓
               融合嵌入 = r_RW * H_RW + r_PE * H_PE + (1-r_RW-r_PE) * H_MP
                              ↓
                        聚类分配 + 图结构重建
```

视图权重 `r_RW`、`r_PE` 在 `main.py` 中按数据集手工预设，不参与训练。

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

在 `main.py` 中修改数据集和对应参数：

| 参数 | 含义 | 示例值 (dblp) |
|------|------|---------------|
| `data` | 数据集名 | `'dblp'` |
| `r_RW` | RW 视图融合权重 | `0.5` |
| `r_PE` | PE 视图融合权重 | `0.3` |
| `--epoch` | 训练轮数 | `10` |
| `--batch_size` | 批次大小 | `64` |
| `--emb_size` | 嵌入维度 | `128` |
| `--neg_size` | 负采样数 | `3` |
| `--hist_len` | 历史邻居窗口 | `2` |
| `--learning_rate` | 学习率 | `0.01` |

各数据集预设的权重与聚类数：

| 数据集 | r_RW | r_PE | 聚类数 |
|--------|------|------|--------|
| arxivAI | 1.0 | 0.0 | 5 |
| school | 0.8 | 0.1 | 9 |
| dblp | 0.5 | 0.3 | 10 |
| brain | 0.5 | 0.5 | 10 |
| patent | 0.6 | 0.1 | 6 |

## 局限性

- 视图融合权重需手工设定，每个数据集需要单独调参
- 训练过程中仅终端输出 loss 和聚类指标，无可追溯日志
- 无损失分量追踪，难以分析各损失项的贡献

> 针对上述问题，请参见 `feat-adaptive-fusion` 和 `feat-alpha-freeze` 分支的改进版本。

## 引用

> Meng Liu, Ke Liang, Hao Yu, Lingyuan Meng, Siwei Wang, Sihang Zhou, Xinwang Liu. "Multiview Temporal Graph Clustering." IEEE TNNLS, 2025.

原版代码：https://github.com/MGitHubL/Data4TGC
