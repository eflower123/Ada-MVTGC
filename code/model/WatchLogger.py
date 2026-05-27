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
