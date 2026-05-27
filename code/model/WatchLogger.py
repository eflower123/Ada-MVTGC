import os
from datetime import datetime


class WatchLogger:
    def __init__(self, log_dir, dataset):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        filepath = os.path.join(log_dir, f'{dataset}_{timestamp}.log')
        self.file = open(filepath, 'w', encoding='utf-8')

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
        self.file.flush()

    def log_epoch(self, epoch, loss, temp_nce, l_d, l_x, l_ent, acc, nmi, ari, f1, alpha_stats, beta):
        s = alpha_stats
        epoch_str = f"{epoch + 1}"

        self.file.write(
            f'{epoch_str:<10}'
            f'{loss:<10.4f}'
            f'{temp_nce:<10.4f}'
            f'{l_d:<10.4f}'
            f'{l_x:<10.4f}'
            f'{l_ent:<10.4f}'
            f'{acc:<10.4f}'
            f'{nmi:<10.4f}'
            f'{ari:<10.4f}'
            f'{f1:<10.4f}'
            f'{s["rw_mean"]:<10.4f}'
            f'{s["rw_std"]:<10.4f}'
            f'{s["pe_mean"]:<10.4f}'
            f'{s["pe_std"]:<10.4f}'
            f'{s["mp_mean"]:<10.4f}'
            f'{s["mp_std"]:<10.4f}'
            f'{beta:<10.4f}\n'
        )
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()
