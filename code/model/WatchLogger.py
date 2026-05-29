import os
from datetime import datetime


class WatchLogger:
    def __init__(self, log_dir, dataset):
        os.makedirs(log_dir, exist_ok=True)
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime('%Y-%m-%d_%H%M%S')
        filepath = os.path.join(log_dir, f'{dataset}_{timestamp}.log')
        self.file = open(filepath, 'w', encoding='utf-8')
        self.epochs_data = []
        self.time_placeholder_pos = None

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

        self.time_placeholder_pos = self.file.tell()
        self.file.write(f'{"# Total training time: --":<50}\n')
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

        self.epochs_data.append({
            'epoch': epoch,
            'loss': loss, 'temp_nce': temp_nce, 'l_d': l_d, 'l_x': l_x, 'l_ent': l_ent,
            'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1,
            'alpha_stats': s, 'beta': beta,
        })

    def _fmt_row(self, d):
        s = d['alpha_stats']
        return (
            f'{d["epoch"] + 1:<10}'
            f'{d["loss"]:<10.4f}'
            f'{d["temp_nce"]:<10.4f}'
            f'{d["l_d"]:<10.4f}'
            f'{d["l_x"]:<10.4f}'
            f'{d["l_ent"]:<10.4f}'
            f'{d["acc"]:<10.4f}'
            f'{d["nmi"]:<10.4f}'
            f'{d["ari"]:<10.4f}'
            f'{d["f1"]:<10.4f}'
            f'{s["rw_mean"]:<10.4f}'
            f'{s["rw_std"]:<10.4f}'
            f'{s["pe_mean"]:<10.4f}'
            f'{s["pe_std"]:<10.4f}'
            f'{s["mp_mean"]:<10.4f}'
            f'{s["mp_std"]:<10.4f}'
            f'{d["beta"]:<10.4f}\n'
        )

    def write_freeze_marker(self, best_epoch, best_acc, freeze_epoch):
        self.file.write(
            f'# first_best_ACC epoch {best_epoch + 1} (ACC={best_acc:.4f}), '
            f'alpha frozen at epoch {freeze_epoch + 1}\n'
        )
        self.file.flush()

    def write_summary(self):
        if not self.epochs_data:
            return

        sorted_by = lambda key: sorted(self.epochs_data, key=lambda d: d[key], reverse=True)
        best = {}

        acc_sorted = sorted_by('acc')
        best['best_ACC'] = acc_sorted[0]
        best['second_ACC'] = acc_sorted[1] if len(acc_sorted) > 1 else acc_sorted[0]

        best['best_NMI'] = sorted_by('nmi')[0]
        best['best_ARI'] = sorted_by('ari')[0]
        best['best_F1'] = sorted_by('f1')[0]

        avg_sorted = sorted(self.epochs_data,
                            key=lambda d: (d['acc'] + d['nmi'] + d['ari'] + d['f1']) / 4,
                            reverse=True)
        best['best_AVG'] = avg_sorted[0]
        best_avg_val = (best['best_AVG']['acc'] + best['best_AVG']['nmi'] +
                        best['best_AVG']['ari'] + best['best_AVG']['f1']) / 4

        self.file.write('\n')
        order = ['best_ACC', 'second_ACC', 'best_NMI', 'best_ARI', 'best_F1']
        for label in order:
            d = best[label]
            self.file.write(f'# {label} epoch {d["epoch"] + 1}\n')
            self.file.write(self._fmt_row(d))

        d = best['best_AVG']
        self.file.write(f'# best_AVG epoch {d["epoch"] + 1}  avg:{best_avg_val:.4f}\n')
        self.file.write(self._fmt_row(d))
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()
