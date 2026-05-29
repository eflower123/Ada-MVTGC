# Add Training Duration to Log

## 概述

在 WatchLogger 训练日志中增加总训练时长记录。采用占位符 + seek 回填方案：
训练开始时写入占位行，训练结束时计算实际耗时（格式 `Xh Ym Zs`），
seek 回去覆盖写入。时长行位于 header 之后、epoch 数据之前。

## Goal

Record total training time in the WatchLogger log file, placed right after
the header and before per-epoch data rows.

## Design

### File changed: `code/model/WatchLogger.py`

**`__init__`** — record training start time:

- Add `self.start_time = datetime.now()` (reuses the existing `timestamp`
  variable as the start time)
- Add `self.time_placeholder_pos = None`

**`write_header`** — write a placeholder line after the column header:

- After the column header line, record current file position via
  `self.time_placeholder_pos = self.file.tell()`
- Write a placeholder: `# Total training time: --              \n`
- The trailing spaces ensure the placeholder is wide enough to be
  overwritten by any reasonable real duration string

**`close`** — fill in the actual duration:

- Compute `elapsed = datetime.now() - self.start_time`
- Format as `Xh Ym Zs` (e.g. `0h 5m 23s`, `1h 12m 8s`)
- If `self.time_placeholder_pos` is set, seek to that position, write a
  new placeholder line, write `# Total training time: <formatted>`, then
  pad with spaces to cover any leftover placeholder characters
- Seek back to file end before closing

### Not changed: `code/main.py`

The existing `print('Training Complete with Time: %s' % str(end - start))`
remains as console output — redundant but harmless.

## Format example

```
# MVTGC Training Log
# Dataset: school  |  Time: 2026-05-28 18:14:46
# Epochs: 30  |  Batch: 128  |  LR: 0.01  |  Emb: 128  |  Neg: 3  |  Hist: 2
# Init weights: RW=0.5  PE=0.1  MP=0.4
# epoch   loss      ...
# Total training time: 0h 5m 23s
1         -0.0074   196.1856  ...
...
```
