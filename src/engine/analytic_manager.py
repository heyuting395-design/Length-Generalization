import os
import torch
import pandas as pd

class AnalyticManager:
    def __init__(self, config, working_dir):
        self.config = config
        self.working_dir = working_dir
        self.metrics_path = os.path.join(working_dir, 'metrics.csv')
        self.history = []
        self.snapshot_dir = os.path.join(working_dir, 'analysis/snapshots')
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def log_metrics(self, epoch, train_loss, iid_loss, iid_acc, ood_loss, ood_acc):
        """
        保存所有关键指标。
        由于 evaluate 暂未计算测试 loss，此处保留 iid_acc 和 ood_acc。
        """
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'iid_loss':iid_loss,
            'ood_loss':ood_loss,
            'iid_acc': iid_acc,
            'ood_acc': ood_acc
        }
        self.history.append(row)
        # 高频落盘，防止断电
        if epoch % 5 == 0 or epoch == self.config['train']['epochs']:
            pd.DataFrame(self.history).to_csv(self.metrics_path, index=False)

    def save_snapshot(self, model, epoch):
        """保存权重快照供后续分析"""
        is_zero = (epoch == 0)
        is_early = (epoch < 100 and epoch % 20 == 0)
        is_regular = (epoch > 0 and epoch % 200 == 0)
        is_final = (epoch == self.config['train']['epochs'])

        if is_zero or is_early or is_regular or is_final:
            save_path = os.path.join(self.snapshot_dir, f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), save_path)
            print(f">>> [Snapshot] Saved weights for epoch {epoch}")

    def run_final_logic(self):
        pd.DataFrame(self.history).to_csv(self.metrics_path, index=False)
        print(f"--- 训练数据已全部保存至 {self.working_dir} ---")