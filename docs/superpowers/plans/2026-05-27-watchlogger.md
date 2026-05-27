# WatchLogger 自适应融合训练监控模块 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 WatchLogger 模块，在 MVTGC 训练时记录每 epoch 的视图 alpha 权重统计、损失和聚类指标到日志文件。

**Architecture:** 新建 `code/model/WatchLogger.py` 封装日志文件的创建与写入。在 `MVTGC.forward()` 中缓存每 batch 的 alpha 值，`train()` 循环中每 epoch 汇总统计并调用 logger 写入。`main.py` 负责创建 logger 并传入模型。

**Tech Stack:** Python, PyTorch, NumPy

---

### Task 1: 创建 WatchLogger 类

**Files:**
- Create: `code/model/WatchLogger.py`

- [ ] **Step 1: 实现 WatchLogger 类**

```python
import os
from datetime import datetime


class WatchLogger:
    def __init__(self, log_dir, dataset):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        filepath = os.path.join(log_dir, f'{dataset}_{timestamp}.log')
        self.file = open(filepath, 'w', encoding='utf-8')

    def write_header(self, rw, pe, mp, args):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mp_weight = round(1 - rw - pe, 4)
        self.file.write(f'# MVTGC Training Log\n')
        self.file.write(f'# Dataset: {args.dataset}  |  Time: {timestamp}\n')
        self.file.write(f'# Epochs: {args.epoch}  |  Batch: {args.batch_size}  |  '
                        f'LR: {args.learning_rate}  |  Emb: {args.emb_size}  |  '
                        f'Neg: {args.neg_size}  |  Hist: {args.hist_len}\n')
        self.file.write(f'# Init weights: RW={rw}  PE={pe}  MP={mp_weight}\n')
        self.file.write('# epoch\tloss\tACC\tNMI\tARI\tF1\t'
                        'RW_mean\tRW_std\tPE_mean\tPE_std\tMP_mean\tMP_std\tbeta\n')
        self.file.flush()

    def log_epoch(self, epoch, loss, acc, nmi, ari, f1, alpha_stats, beta):
        s = alpha_stats
        self.file.write(f'{epoch + 1}\t{loss:.4f}\t{acc:.4f}\t{nmi:.4f}\t{ari:.4f}\t{f1:.4f}\t'
                        f'{s["rw_mean"]:.4f}\t{s["rw_std"]:.4f}\t'
                        f'{s["pe_mean"]:.4f}\t{s["pe_std"]:.4f}\t'
                        f'{s["mp_mean"]:.4f}\t{s["mp_std"]:.4f}\t'
                        f'{beta:.4f}\n')
        self.file.flush()

    def close(self):
        self.file.close()
```

- [ ] **Step 2: 提交**

```bash
git add code/model/WatchLogger.py
git commit -m "feat: add WatchLogger module for training monitoring"
```

---

### Task 2: 在 MVTGC 中集成日志记录

**Files:**
- Modify: `code/model/MVTGC.py`

- [ ] **Step 1: 修改 `__init__` 接受 logger 参数**

在 `__init__` 的参数签名中增加 `logger=None`，位置放在 `self.epochs = args.epoch` 之前：

```python
self.logger = logger
```

- [ ] **Step 2: 在 `forward()` 中缓存 batch alpha**

在 `forward()` 方法的 `compute_alpha` 调用之后（当前代码约第 173 行），`return total_loss` 之前，添加一行将 alpha detach 并移到 CPU：

```python
self._batch_alpha = alpha.detach().cpu().numpy()
```

- [ ] **Step 3: 在 `train()` 中收集 alpha、计算统计、调用 logger**

在 `train()` 方法中，在 `for epoch in range(self.epochs):` 内部、`self.loss = 0.0` 之后，初始化 epoch 级 alpha 列表：

```python
epoch_alphas = []
```

在 `self.update(...)` 调用之后（即 batch 循环内部末尾，当前代码约第 252 行之后），收集该 batch 的 alpha：

```python
epoch_alphas.append(self._batch_alpha)
```

在 epoch 评估块（`eva(...)` 调用之后，约第 254 行）和 `if acc > self.best_acc:` 之间，加入以下日志逻辑：

```python
if self.logger is not None:
    all_alphas = np.concatenate(epoch_alphas, axis=0)  # (total_nodes_in_epoch, 3)
    rw_mean, rw_std = float(all_alphas[:, 0].mean()), float(all_alphas[:, 0].std())
    pe_mean, pe_std = float(all_alphas[:, 1].mean()), float(all_alphas[:, 1].std())
    mp_mean, mp_std = float(all_alphas[:, 2].mean()), float(all_alphas[:, 2].std())
    alpha_stats = {
        'rw_mean': rw_mean, 'rw_std': rw_std,
        'pe_mean': pe_mean, 'pe_std': pe_std,
        'mp_mean': mp_mean, 'mp_std': mp_std,
    }
    epoch_loss = self.loss.cpu().numpy() / len(self.data)
    self.logger.log_epoch(epoch, epoch_loss, acc, nmi, ari, f1, alpha_stats, self.beta)
```

- [ ] **Step 4: 提交**

```bash
git add code/model/MVTGC.py
git commit -m "feat: integrate WatchLogger into MVTGC training loop"
```

---

### Task 3: 在 main.py 中挂接 WatchLogger

**Files:**
- Modify: `code/main.py`

- [ ] **Step 1: 导入 WatchLogger 并创建实例**

在 `main.py` 的 import 区域添加导入：

```python
from model.WatchLogger import WatchLogger
```

在 `main_train(args)` 函数中，创建 MVTGC 实例之前，初始化 logger：

```python
def main_train(args):
    start = datetime.datetime.now()
    log_dir = os.path.join('..', 'Watch')
    logger = WatchLogger(log_dir, args.dataset)
    rw = float(args.r_RW)
    pe = float(args.r_PE)
    logger.write_header(rw, pe, args)
    the_train = MVTGC.MVTGC(args, logger=logger)
    the_train.train()
    logger.close()
    end = datetime.datetime.now()
    print('Training Complete with Time: %s' % str(end - start))
```

在调用 `main_train(args)` 之前也需要导入 `os`（当前 `main.py` 已有 `import os`）。

- [ ] **Step 2: 提交**

```bash
git add code/main.py
git commit -m "feat: wire WatchLogger into main training entry point"
```

---

### Task 4: 验证

- [ ] **Step 1: 运行训练并检查输出**

```bash
cd code && python main.py
```

预期：训练正常完成，`Watch/` 目录下生成 `patent_YYYY-MM-DD_HHMMSS.log` 文件，内容包含 header（5 行 `#` 注释）和 30 行数据（每个 epoch 一行）。

- [ ] **Step 2: 检查日志文件格式**

```bash
head -6 Watch/patent_*.log
```

预期输出示例（数值可不同）：

```
# MVTGC Training Log
# Dataset: patent  |  Time: 2026-05-27 15:30:00
# Epochs: 30  |  Batch: 128  |  LR: 0.01  |  Emb: 128  |  Neg: 3  |  Hist: 2
# Init weights: RW=0.5  PE=0.1  MP=0.4
# epoch  loss      ACC     NMI     ARI     F1      RW_mean  RW_std  PE_mean  PE_std  MP_mean  MP_std  beta
1       1.2345     0.4500  0.3200  0.1500  0.4100  0.5200   0.1500  0.2800   0.1200  0.2000   0.1000  1.0000
```

- [ ] **Step 3: 提交（如有修正）**

```bash
git add -A && git commit -m "chore: verify WatchLogger output format"
```
