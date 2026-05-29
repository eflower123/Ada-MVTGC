# Add Training Duration to Log — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record total training time (`Xh Ym Zs`) in the WatchLogger log file between the header and epoch data rows.

**Architecture:** Single-file change to `WatchLogger.py`. `__init__` captures start time; `write_header` writes a placeholder line and remembers its file position; `close()` seeks back to overwrite the placeholder with the actual formatted duration.

**Tech Stack:** Python stdlib (datetime, os)

---

### Task 1: Update `__init__` to record start time

**Files:**
- Modify: `code/model/WatchLogger.py:6-11`

- [ ] **Step 1: Add `self.start_time` and `self.time_placeholder_pos`**

In `__init__`, replace the `timestamp = datetime.now()...` line to reuse the same `datetime.now()` call for the start time, and initialize the placeholder position tracker.

```python
def __init__(self, log_dir, dataset):
    os.makedirs(log_dir, exist_ok=True)
    self.start_time = datetime.now()
    timestamp = self.start_time.strftime('%Y-%m-%d_%H%M%S')
    filepath = os.path.join(log_dir, f'{dataset}_{timestamp}.log')
    self.file = open(filepath, 'w', encoding='utf-8')
    self.epochs_data = []
    self.time_placeholder_pos = None
```

- [ ] **Step 2: Commit**

```bash
git add code/model/WatchLogger.py
git commit -m "feat: add start_time and placeholder_pos tracking to WatchLogger"
```

---

### Task 2: Write placeholder line in `write_header`

**Files:**
- Modify: `code/model/WatchLogger.py:13-27`

- [ ] **Step 1: Append placeholder line after column header**

After the `header_line` write and `self.file.flush()` at the end of `write_header`, add the placeholder logic:

```python
def write_header(self, rw, pe, args):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mp_weight = round(1 - rw - pe, 4)
    self.file.write(f'# MVTGC Training Log\n')
    self.file.write(f'# Dataset: {args.dataset}  |  Time: {timestamp}\n')
    self.file.write(f'# Epochs: {args.epoch}  |  Batch: {args.batch_size}  |  '
                    f'LR: {args.learning_rate}  |  Emb: {args.emb_size}  |  '
                    f'Neg: {args.neg_size}  |  Hist: {args.hist_len}\n')
    self.file.write(f'# Init weights: RW={rw}  PE={pe}  MP={mp_weight}\n')
    headers = ["# epoch", "loss", "temp_NCE", "L_d", "L_x", "L_ent",
               "ACC", "NMI", "ARI", "F1",
               "RW_mean", "RW_std", "PE_mean", "PE_std", "MP_mean", "MP_std", "beta"]
    header_line = "".join(f"{h:<10}" for h in headers) + "\n"
    self.file.write(header_line)

    self.time_placeholder_pos = self.file.tell()
    self.file.write('# Total training time: --              \n')
    self.file.flush()
```

- [ ] **Step 2: Commit**

```bash
git add code/model/WatchLogger.py
git commit -m "feat: write duration placeholder line in log header"
```

---

### Task 3: Fill in actual duration in `close`

**Files:**
- Modify: `code/model/WatchLogger.py:124-127`

- [ ] **Step 1: Add duration backfill logic to `close`**

Replace `close` to compute elapsed time, seek back to the placeholder, and overwrite it:

```python
def close(self):
    if self.file and not self.file.closed:
        if self.time_placeholder_pos is not None:
            elapsed = datetime.now() - self.start_time
            total_seconds = int(elapsed.total_seconds())
            h = total_seconds // 3600
            m = (total_seconds % 3600) // 60
            s = total_seconds % 60
            duration_str = f'{h}h {m}m {s}s'
            line = f'# Total training time: {duration_str}'
            line = line.ljust(50) + '\n'
            self.file.seek(self.time_placeholder_pos)
            self.file.write(line)
            self.file.seek(0, os.SEEK_END)
        self.file.close()
```

- [ ] **Step 2: Commit**

```bash
git add code/model/WatchLogger.py
git commit -m "feat: backfill actual training duration on log close"
```

---

### Task 4: Verify with a training run

**Files:**
- None (verification only)

- [ ] **Step 1: Run a short training to produce a log**

```bash
cd code && python main.py --epoch 5 --min_train_epochs 3 --patience 2
```

- [ ] **Step 2: Inspect the generated log file**

Check that the `Watch/` directory contains a new log file and that the line after the column header shows the actual duration, e.g.:
```
# Total training time: 0h 0m 12s
```

- [ ] **Step 3: Verify the placeholder is fully overwritten**

Confirm there's no leftover `--` or extra whitespace artifacts on the duration line.

```bash
grep "Total training time" Watch/school_*.log
```
