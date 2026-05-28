# Alpha 冻结训练策略

> 当聚类指标 ACC 达到早期高点后，固定自适应视图权重（alpha），后续只训练图神经网络参数，防止 alpha 坍缩导致聚类性能下降。

## 触发条件

两阶段判定：

1. **前 `min_train_epochs`（默认 25）轮**：强制不冻结，只追踪 best_ACC 及其发生的 epoch
2. **第 25 轮起**：启动 patience=5 机制。连续 5 轮 ACC 未创新高即触发冻结

## 冻结机制

**保存与回退**：每次刷新 best_ACC 时，用 `copy.deepcopy` 保存 `scoring_fc1`/`scoring_fc2` 的状态字典。冻结时恢复，alpha 精确回到 best_ACC 时刻的值。

**冻结后的变化**：
- `compute_alpha` 不再被调用，alpha 固定为冻结时刻的值
- 损失中移除 `l_ent`（alpha 常量，熵正则化无意义）
- `beta` 停止衰减
- 冻结后的 alpha 值仍然每轮记录到日志（列值不变）

**冻结后不变化**：
- embedding / delta / cluster_layer 继续正常训练
- 时序 NCE、L_d、L_x 三项损失继续优化
- 聚类指标仍每轮评估

## WatchLogger 日志变化

冻结发生时，在对应数据行后插入标记行：

```
# first_best_ACC epoch 7 (ACC=0.4794), alpha frozen at epoch 12
```

冻结后数据行继续正常写入（alpha 列保持不变），最终摘要中 `best_ACC` 仍标出全局最高。

## 新增参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--min_train_epochs` | 最小训练轮数（在此之前不触发冻结） | 25 |
| `--patience` | 连续未创新高容忍轮数 | 5 |

## 实现涉及文件

| 文件 | 修改内容 |
|------|----------|
| `code/model/MVTGC.py` | 新增冻结状态变量、保存/恢复 scoring 网络、判定逻辑、修改 forward() 和 train() |
| `code/model/WatchLogger.py` | 新增 `write_freeze_marker()` 方法 |
| `code/main.py` | 新增两个命令行参数 |
