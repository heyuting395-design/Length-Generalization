import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from sklearn.manifold import TSNE

# 路径处理
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.decoder_only import DecoderOnlyAdder
#from src.models.encoder_decoder import InterleavedTransformerAdder
# (if you want to change the model)from src.models.carrier_encoder import InterleavedEncoderAdder
from src.utils.plot_settings import apply_plot_settings, get_functional_colors

def analyze_embeddings_tsne(exp_dir, epochs):
    """
    只生成数字 0-9 的 t-SNE 聚类 PNG 图
    """
    apply_plot_settings()
    colors = get_functional_colors()
    
    save_dir = os.path.join(exp_dir, "analysis/embeddings")
    os.makedirs(save_dir, exist_ok=True)

    # 加载配置
    import json
    config_path = os.path.join(exp_dir, "config_backup.json")
    if not os.path.exists(config_path):
        print(f"Error: 找不到配置文件 {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyAdder(config).to(device)

    ##model = InterleavedEncoderAdder(config).to(device)

    for ep in epochs:
        # 路径适配：处理 checkpoints 里的 model_best 或 snapshots 里的 epoch 文件
        if isinstance(ep, str) and ep.endswith(".pth"):
            ckpt_path = os.path.join(exp_dir, "checkpoints", ep)
            label = ep.split('.')[0] # e.g., model_best
        else:
            ckpt_path = os.path.join(exp_dir, f"analysis/snapshots/model_epoch_{ep}.pt")
            label = f"epoch_{ep}"

        if not os.path.exists(ckpt_path):
            print(f"Skip: 找不到权重文件 {ckpt_path}")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # 1. 提取数字 Embedding (0-9)
        with torch.no_grad():
            # 这里的路径需对应你的模型结构：model.embedding -> DigitEmbedding -> digit_embed
            raw_weights = model.embedding.digit_embed.weight.detach()[:10].cpu().numpy()

        # 2. t-SNE 降维
        # 对于 10 个样本，perplexity 设置为 3 比较稳健
        tsne = TSNE(
            n_components=2, 
            perplexity=3, 
            random_state=42, 
            init='pca', 
            learning_rate='auto'
        )
        embed_2d = tsne.fit_transform(raw_weights)

        # 3. 绘图
        plt.figure(figsize=(6, 6))
        
        # 绘制背景散点（宝石蓝）
        plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=colors['primary'], s=150, alpha=0.15)
        
        # 绘制数字标签（洋红）
        for i in range(10):
            plt.text(
                embed_2d[i, 0], embed_2d[i, 1], str(i), 
                fontsize=16, fontweight='bold', color=colors['ood'],
                ha='center', va='center'
            )

        plt.title(f"t-SNE: Digit Embeddings ({label.replace('_', ' ').capitalize()})")
        plt.grid(True, linestyle=':', alpha=0.3)
        
        # 清理坐标轴，使几何结构更突出
        plt.xticks([])
        plt.yticks([])
        
        # 只保存 PNG
        save_path = os.path.join(save_dir, f"tsne_{label.lower()}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Success: 已生成 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="实验输出路径")
    args = parser.parse_args()
    
    # 指定需要分析的关键节点
    checkpoints_to_analyze = [0, 20, 100, 200, 500, "model_best.pth"]
    
    analyze_embeddings_tsne(args.exp_dir, checkpoints_to_analyze)