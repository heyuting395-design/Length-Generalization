import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.plot_settings import apply_plot_settings, get_functional_colors

def plot_metrics(exp_dir):
    # 1. 加载数据
    metrics_path = os.path.join(exp_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"Error: 找不到文件 {metrics_path}")
        return
    
    df = pd.read_csv(metrics_path)
    apply_plot_settings()
    colors = get_functional_colors()
    
    save_dir = os.path.join(exp_dir, "analysis/plots")
    os.makedirs(save_dir, exist_ok=True)

    # ----- 图1：所有 Loss 曲线（train / iid / ood / test）-----
    loss_cols = [col for col in ['train_loss', 'iid_loss', 'ood_loss', 'test_loss'] 
                 if col in df.columns]
    
    if loss_cols:
        plt.figure(figsize=(7, 5))
        for col in loss_cols:
            # 设置颜色和标签
            if 'train' in col:
                label = 'Train Loss'
                color = colors.get('train', '#4C72B0')
            elif 'iid' in col:
                label = 'IID Loss'
                color = colors.get('iid', '#55A868')
            elif 'ood' in col:
                label = 'OOD Loss'
                color = colors.get('ood', '#C44E52')
            else:
                label = 'Test Loss'
                color = '#9467BD'
            plt.plot(df['epoch'], df[col], label=label, color=color, linewidth=1.5)
        
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.yscale('log')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(save_dir, "loss_curves.pdf"))
        plt.savefig(os.path.join(save_dir, "loss_curves.png"))
        plt.close()
    else:
        print("Warning: No loss columns found, skip loss plot.")

    # ----- 图2：所有 Accuracy 曲线（iid / ood / test）-----
    acc_cols = [col for col in ['iid_acc', 'ood_acc', 'test_acc'] if col in df.columns]
    
    if acc_cols:
        plt.figure(figsize=(7, 5))
        for col in acc_cols:
            if 'iid' in col:
                label = 'IID Accuracy'
                color = colors.get('iid', '#55A868')
                linestyle = '-'
            elif 'ood' in col:
                label = 'OOD Accuracy'
                color = colors.get('ood', '#C44E52')
                linestyle = '--'
            else:
                label = 'Test Accuracy'
                color = '#9467BD'
                linestyle = '-.'
            plt.plot(df['epoch'], df[col], label=label, color=color, 
                     linestyle=linestyle, linewidth=2)
        
        # 动态阶段划分竖线（epoch 最大值的 1/3 和 2/3 处）
        max_epoch = df['epoch'].max()
        if max_epoch >= 20:
            vlines = [max_epoch // 3, max_epoch * 2 // 3]
        else:
            vlines = []  # 数据太少时不画竖线
        
        for x in vlines:
            plt.axvline(x=x, color='black', linestyle=':', linewidth=0.8, alpha=0.2)
        
        # 添加阶段文本（如果竖线存在）
        if vlines:
            text_y = 1.02
            plt.text(vlines[0]/2, text_y, 'Stage 1', ha='center', fontsize=10, 
                     fontweight='bold', alpha=0.6)
            mid = (vlines[0] + vlines[1]) / 2
            plt.text(mid, text_y, 'Stage 2', ha='center', fontsize=10, 
                     fontweight='bold', alpha=0.6)
            end_mid = (vlines[1] + max_epoch) / 2
            plt.text(end_mid, text_y, 'Stage 3', ha='center', fontsize=10, 
                     fontweight='bold', alpha=0.6)
        
        plt.xlabel('Epoch')
        plt.ylabel('Sequence Accuracy')
        plt.ylim(-0.05, 1.12)
        plt.title('Accuracy Curves')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend(loc='lower right', frameon=True)
        plt.savefig(os.path.join(save_dir, "accuracy_curves.pdf"))
        plt.savefig(os.path.join(save_dir, "accuracy_curves.png"))
        plt.close()
    else:
        print("Warning: No accuracy columns found, skip accuracy plot.")

    print(f"成功！图表（PNG & PDF）已保存至: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="实验输出目录路径")
    args = parser.parse_args()
    
    plot_metrics(args.exp_dir)