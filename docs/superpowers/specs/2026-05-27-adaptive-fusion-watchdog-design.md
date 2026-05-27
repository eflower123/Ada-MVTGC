# WatchLogger: Adaptive Fusion Training Monitor

## Summary

Add a `WatchLogger` module that records per-epoch view weight statistics, loss, and
clustering metrics during MVTGC training, enabling analysis of how the adaptive
view-scoring mechanism evolves over time.

## Files

| File | Action |
|------|--------|
| `code/model/WatchLogger.py` | New — WatchLogger class |
| `code/model/MVTGC.py` | Modify — call logger in train() loop |
| `code/main.py` | Modify — instantiate logger, pass init weights |
| `Watch/` | New directory (auto-created) |

## WatchLogger class

Located in `code/model/WatchLogger.py`. Exposes three methods:

- `__init__(log_dir, dataset)` — creates `Watch/` directory if missing, opens log file
  named `Watch/{dataset}_{YYYY-MM-DD_HHMMSS}.log`
- `write_header(rw, pe, mp, args)` — writes metadata block (dataset, timestamp,
  hyperparams, initial static view weights)
- `log_epoch(epoch, loss, acc, nmi, ari, f1, alpha_stats, beta)` — appends one
  tab-separated data row. `alpha_stats` is `{rw_mean, rw_std, pe_mean, pe_std, 
  mp_mean, mp_std}`

## Log file format

```
# MVTGC Training Log
# Dataset: patent  |  Time: 2026-05-27 14:30:52
# Epochs: 30  |  Batch: 128  |  LR: 0.01  |  Emb: 128  |  Neg: 3  |  Hist: 2
# Init weights: RW=0.50  PE=0.10  MP=0.40
# epoch  loss      ACC     NMI     ARI     F1      RW_mean  RW_std  PE_mean  PE_std  MP_mean  MP_std  beta
1       1.2345     0.4500  0.3200  0.1500  0.4100  0.5200   0.1500  0.2800   0.1200  0.2000   0.1000  1.0000
```

Header uses `#` comment lines for human readability. Data rows are tab-separated
for easy parsing.

## Alpha accumulation

During each `forward()` call, per-node alpha values (shape `[batch, 3]`) are
appended to an epoch-level buffer. After each epoch, the buffer is aggregated
into per-view mean/std and passed to `log_epoch()`. The buffer is cleared at
the start of each epoch.

## Changes to MVTGC.train()

After the existing evaluation block (line 254–262 in current code), add a call
to `self.logger.log_epoch(...)`. The `log_epoch` call happens after `eva()` so
loss and clustering metrics are available.

## Changes to main.py

- Add `import os` (if not already present)
- Instantiate `WatchLogger` before `main_train(args)`, pass it into MVTGC
- Pass the static RW/PE weights to `write_header()`
