# Alpha 冻结训练策略 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现两阶段 alpha 冻结策略——前 25 轮强制不冻结，之后连续 5 轮 ACC 未创新高时冻结视图权重，后续只训练图神经网络参数。

**Architecture:** 在 MVTGC 中新增冻结状态变量，每次 ACC 创新高时 deepcopy 保存 scoring 网络状态，满足冻结条件后恢复最优参数并设 requires_grad=False。forward() 中根据冻结状态跳过 l_ent。WatchLogger 新增冻结标记行。

**Tech Stack:** Python, PyTorch, copy

---

### Task 1: MVTGC 中新增冻结状态变量

**Files:**
- Modify: `code/model/MVTGC.py`

- [ ] **Step 1: 添加 `import copy`**

在文件顶部 import 区域（约第 6 行附近，与其他标准库 import 放在一起）：

```python
import copy
```

- [ ] **Step 2: 在 `__init__` 中添加冻结相关属性**

在 `self._batch_alpha = None`（第 90 行）之后、`self.opt = SGD(...)`（第 92 行）之前，添加：

```python
        self.min_train_epochs = getattr(args, 'min_train_epochs', 25)
        self.patience = getattr(args, 'patience', 5)
        self.alpha_frozen = False
        self.no_improve_count = 0
        self.first_best_acc = 0.0
        self.first_best_acc_epoch = 0
        self._best_scoring_fc1_state = copy.deepcopy(self.scoring_fc1.state_dict())
        self._best_scoring_fc2_state = copy.deepcopy(self.scoring_fc2.state_dict())
        self._best_beta = self.beta
```

- [ ] **Step 3: 提交**

```bash
git add code/model/MVTGC.py
git commit -m "feat: add alpha freeze state variables to MVTGC"
```

---

### Task 2: 在 train() 中添加 ACC 追踪和冻结判定

**Files:**
- Modify: `code/model/MVTGC.py`

- [ ] **Step 1: 在 eval 后添加追踪逻辑**

找到 `train()` 中 `if acc > self.best_acc:` 块（约第 268 行）。在该块之后、`sys.stdout.write(...)` 行之后，添加冻结追踪和判定。插入在 `if self.logger is not None:` 块之前即可——追踪逻辑应独立于 logger 存在。

具体修改：在 `if acc > self.best_acc:` 完整块之后、logger 日志块之前，插入以下代码：

```python
            # --- alpha freeze: track best ACC and trigger ---
            if acc > self.first_best_acc:
                self.first_best_acc = acc
                self.first_best_acc_epoch = epoch
                self.no_improve_count = 0
                if not self.alpha_frozen:
                    self._best_scoring_fc1_state = copy.deepcopy(self.scoring_fc1.state_dict())
                    self._best_scoring_fc2_state = copy.deepcopy(self.scoring_fc2.state_dict())
                    self._best_beta = self.beta
            elif not self.alpha_frozen:
                self.no_improve_count += 1

            if not self.alpha_frozen and epoch >= self.min_train_epochs - 1 and self.no_improve_count >= self.patience:
                self.alpha_frozen = True
                self.scoring_fc1.load_state_dict(self._best_scoring_fc1_state)
                self.scoring_fc2.load_state_dict(self._best_scoring_fc2_state)
                self.beta = self._best_beta
                for p in self.scoring_fc1.parameters():
                    p.requires_grad = False
                for p in self.scoring_fc2.parameters():
                    p.requires_grad = False
                if self.logger is not None:
                    self.logger.write_freeze_marker(
                        self.first_best_acc_epoch, self.first_best_acc, epoch)
```

- [ ] **Step 2: 修改 beta 衰减逻辑，冻结后跳过**

找到 `self.beta = max(self.beta * math.exp(-self.rho), self.beta_min)` 行（约第 297 行），包裹条件：

```python
            if not self.alpha_frozen:
                self.beta = max(self.beta * math.exp(-self.rho), self.beta_min)
```

- [ ] **Step 3: 提交**

```bash
git add code/model/MVTGC.py
git commit -m "feat: add alpha freeze tracking and trigger logic in train()"
```

---

### Task 3: 修改 forward() 处理冻结后的损失

**Files:**
- Modify: `code/model/MVTGC.py`

- [ ] **Step 1: 冻结时 l_ent 不参与损失**

找到 `l_framework = l_d + l_x + self.beta * l_ent` 行（约第 201 行），改为：

```python
        if self.alpha_frozen:
            l_framework = l_d + l_x
        else:
            l_framework = l_d + l_x + self.beta * l_ent
```

- [ ] **Step 2: 提交**

```bash
git add code/model/MVTGC.py
git commit -m "feat: exclude l_ent from loss when alpha is frozen"
```

---

### Task 4: WatchLogger 新增冻结标记方法

**Files:**
- Modify: `code/model/WatchLogger.py`

- [ ] **Step 1: 添加 `write_freeze_marker` 方法**

在 `write_summary` 方法之前添加：

```python
    def write_freeze_marker(self, best_epoch, best_acc, freeze_epoch):
        self.file.write(
            f'# first_best_ACC epoch {best_epoch + 1} (ACC={best_acc:.4f}), '
            f'alpha frozen at epoch {freeze_epoch + 1}\n'
        )
        self.file.flush()
```

- [ ] **Step 2: 提交**

```bash
git add code/model/WatchLogger.py
git commit -m "feat: add write_freeze_marker to WatchLogger"
```

---

### Task 5: main.py 新增命令行参数

**Files:**
- Modify: `code/main.py`

- [ ] **Step 1: 添加两个新参数**

在 `parser.add_argument('--directed', ...)` 行之前添加：

```python
    parser.add_argument('--min_train_epochs', type=int, default=25,
                        help='minimum epochs before alpha freeze can trigger')
    parser.add_argument('--patience', type=int, default=5,
                        help='consecutive epochs without ACC improvement to trigger freeze')
```

- [ ] **Step 2: 提交**

```bash
git add code/main.py
git commit -m "feat: add --min_train_epochs and --patience CLI args"
```

---

### Task 6: 验证

- [ ] **Step 1: 运行训练，确认冻结逻辑触发**

```bash
cd code && python main.py
```

预期：训练正常完成。冻结应在前 25 轮之后的某个位置触发（取决于 ACC 变化）。

- [ ] **Step 2: 检查日志文件中冻结标记和 l_ent 列变化**

```bash
grep "first_best_ACC\|alpha frozen" Watch/patent_*.log
```

预期：输出类似 `# first_best_ACC epoch 7 (ACC=0.4794), alpha frozen at epoch 12`。

冻结后 `l_ent` 列值应保持不变（alpha 固定，熵不变），`beta` 列也应保持不变。

- [ ] **Step 3: 提交（如有修正）**

```bash
git add -A && git commit -m "chore: verify alpha freeze behavior"
```
