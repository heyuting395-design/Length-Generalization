import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 路径处理
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.carrier_encoder import InterleavedEncoderAdder
from src.utils.plot_settings import apply_plot_settings, get_color_palette, get_functional_colors

def get_true_carries(a_seq, b_seq):
    """
    根据输入的数字序列，手动计算每一位的真实进位状态。
    返回 shape: (Batch, L+1)，其中每个元素为 0 或 1。
    """
    batch, L = a_seq.shape
    carries = np.zeros((batch, L + 1))
    c = 0
    # 从低位到高位计算 (假设 index 0 是最低位)
    for i in range(L):
        sum_val = a_seq[:, i] + b_seq[:, i] + c
        c = (sum_val >= 10).astype(int)
        carries[:, i+1] = c
    return carries

def prepare_test_batch(num_samples=200, length=7):
    """构造专门用于测试进位分布的数据集：混合简单加法和连续进位"""
    a = torch.randint(0, 10, (num_samples, length))
    b = torch.randint(0, 10, (num_samples, length))
    
    # 强行插入一些“进位链”案例，比如 999... + 1
    a[0, :] = 9
    b[0, :] = 0; b[0, 0] = 1
    
    return a, b

def analyze_checkpoints(exp_dir, epochs):
    apply_plot_settings()
    colors = get_color_palette()
    f_colors = get_functional_colors()
    
    # 准备保存路径
    save_dir = os.path.join(exp_dir, "analysis/carry_space")
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载配置
    import json
    with open(os.path.join(exp_dir, "config_backup.json"), 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterleavedEncoderAdder(config).to(device)
    
    # 准备测试数据
    a_seq, b_seq = prepare_test_batch(num_samples=256)
    true_carries = get_true_carries(a_seq.numpy(), b_seq.numpy())
    
    # 遍历所有指定的 Epoch
    for ep in epochs:
        # 处理路径：snapshot 文件夹或 checkpoints 文件夹
        if isinstance(ep, str): # 处理 model_best
            ckpt_path = os.path.join(exp_dir, "checkpoints", ep)
            label = "Best"
        else:
            ckpt_path = os.path.join(exp_dir, f"analysis/snapshots/model_epoch_{ep}.pt")
            label = f"Epoch {ep}"
            
        if not os.path.exists(ckpt_path):
            print(f"跳过: 找不到 {ckpt_path}")
            continue
            
        # 加载权重
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        # 提取进位向量轨迹
        with torch.no_grad():
            # states shape: (Batch, L+1, d_model)
            states = model.extract_carry_states(a_seq.to(device), b_seq.to(device))
            states = states.cpu().numpy()
            
        # 压平数据进行降维：将 (Batch, Time) 视为样本
        B, T, D = states.shape
        flat_states = states.reshape(-1, D)
        flat_labels = true_carries.reshape(-1)
        
        # 使用 PCA 降维到 2D
        pca = PCA(n_components=2)
        states_2d = pca.fit_transform(flat_states)
        
        # 绘图
        plt.figure(figsize=(8, 7))
        
        # 分别绘制“无进位”和“有进位”的点
        idx_0 = (flat_labels == 0)
        idx_1 = (flat_labels == 1)
        
        plt.scatter(states_2d[idx_0, 0], states_2d[idx_0, 1], 
                    c=f_colors['primary'], label='Carry = 0', alpha=0.4, s=15, edgecolors='none')
        plt.scatter(states_2d[idx_1, 0], states_2d[idx_1, 1], 
                    c=f_colors['ood'], label='Carry = 1', alpha=0.6, s=20, marker='^')
        
        plt.title(f"Latent Carry Space - {label}")
        plt.xlabel(f"PC1 (Var: {pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 (Var: {pca.explained_variance_ratio_[1]:.2%})")
        plt.legend(loc='upper right')
        
        # 保存
        file_name = f"carry_pca_{label.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()
        print(f"已生成分析图: {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    
    # 待分析的节点列表
    target_epochs = [0, 20, 40, 60, 80, 200, 400, 500, "model_best.pth"]
    
    analyze_checkpoints(args.exp_dir, target_epochs)