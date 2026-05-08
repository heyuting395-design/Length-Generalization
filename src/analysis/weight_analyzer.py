import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.utils import plot_settings
def plot_weight_distribution(model, working_dir, epoch):
    """查看模型权重的直方图分布"""
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().view(-1))
    
    all_weights = torch.cat(weights).numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(all_weights, bins=100, color='skyblue', edgecolor='black')
    plt.title(f"Weight Distribution at Epoch {epoch}")
    plt.savefig(f"{working_dir}/analysis/weight_dist_ep{epoch}.png")
    plt.close()