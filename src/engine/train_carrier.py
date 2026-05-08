import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# -----------------------------
# 1. 路径处理
# -----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.carrier_encoder import InterleavedEncoderAdder 
from data.dataset import get_dataloaders
from src.engine.lr_scheduler import get_lr_scheduler
from src.engine.analytic_manager import AnalyticManager
from src.utils.io_operate import IOManager

# -----------------------------
# 2. 工具函数
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_actual_length(digits_list):
    """计算补零前的实际有效位数"""
    for i in range(len(digits_list)-1, -1, -1):
        if digits_list[i] != 0: return i + 1
    return 1

def evaluate(model, dataloader, device):
    """评估函数：tf_ratio=0.0，模拟真实推理过程"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for a, b, target in dataloader:
            a, b, target = a.to(device), b.to(device), target.to(device)
            # 预测时不给 target_seq
            logits = model(a, b, target_seq=None, tf_ratio=0.0)
            preds = torch.argmax(logits, dim=-1)
            # 必须全序列相等才算对
            is_correct = (preds == target).all(dim=1)
            correct += is_correct.sum().item()
            total += a.size(0)
    return correct / total

# -----------------------------
# 3. 训练主逻辑
# -----------------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # 1. 初始化
    set_seed(42)
    io = IOManager(args.config)
    config = io.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = os.path.join(io.exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. 数据准备
    data_dir = os.path.join(project_root, "data/data_output")
    raw_train_loader, test_iid_loader, test_ood_loader = get_dataloaders(
        data_dir, batch_size=config["train"]["batch_size"]
    )
    
    # 建立长度索引
    length_to_indices = {i: [] for i in range(1, 11)}
    full_train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    for idx, (a, b, _) in enumerate(full_train_data):
        actual_l = max(get_actual_length(a), get_actual_length(b))
        if actual_l in length_to_indices:
            length_to_indices[actual_l].append(idx)

    # 3. 模型与优化器
    model = InterleavedEncoderAdder(config).to(device)
    
    # 针对 carry_net 进位层单独处理参数（通常不使用 L2 惩罚）
    carry_params = list(model.carry_net.parameters())
    other_params = [p for n, p in model.named_parameters() if 'carry_net' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-2},
        {'params': carry_params, 'weight_decay': 0.0} 
    ], lr=float(config["train"]["lr"]))

    # 4. 分析管理器与调度器
    manager = AnalyticManager(config, io.exp_dir)
    manager.save_snapshot(model, epoch=0) # 记录初始随机状态

    total_steps = config["train"]["epochs"] * len(raw_train_loader) 
    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    criterion = nn.CrossEntropyLoss()

    best_ood_acc = 0.0

    # 5. 训练循环 (500 Epoch)
    for epoch in range(1, config["train"]["epochs"] + 1):
        # --- 三阶段课程学习逻辑 ---
        if epoch <= 20: 
            stage, lens = "Stage 1: 1-2 Digits", [1, 2]
        elif epoch <= 100: 
            stage, lens = "Stage 2: 1-4 Digits", [1, 2, 3, 4]
        else: 
            stage, lens = "Stage 3: Full 1-7 Digits", list(range(1, 8))

        # 筛选子集
        indices = []
        for l in lens:
            indices.extend(length_to_indices[l])
        
        # 兜底：如果选不到数据，回退到全量训练集
        if not indices:
            indices = list(range(len(full_train_data)))
            stage += " (Fallback)"

        train_loader = DataLoader(
            Subset(raw_train_loader.dataset, indices), 
            batch_size=config["train"]["batch_size"], 
            shuffle=True
        )

        model.train()
        epoch_loss = 0.0
        
        # --- Teacher Forcing 衰减 ---
        # 配合 500 Epoch：前 400 轮线性降至 min_tf，最后 100 轮保持低 TF 进行攻坚
        min_tf = config["train"].get("min_tf", 0.3)
        tf_ratio = max(min_tf, 1.0 - (epoch / (config["train"]["epochs"] * 0.8)))

        pbar = tqdm(train_loader, desc=f"Ep {epoch} [{stage}]")
        for a, b, target in pbar:
            a, b, target = a.to(device), b.to(device), target.to(device)
            optimizer.zero_grad()
            
            # 前向传播：传入 target_seq 用于 TF 训练
            logits = model(a, b, target_seq=target, tf_ratio=tf_ratio)
            
            loss = criterion(logits.view(-1, 10), target.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}", TF=f"{tf_ratio:.2f}")

        # 6. 评估与持久化
        avg_train_loss = epoch_loss / len(train_loader)
        iid_acc = evaluate(model, test_iid_loader, device)
        ood_acc = evaluate(model, test_ood_loader, device)
        
        io.logger.info(f"Ep {epoch} | {stage} | Loss: {avg_train_loss:.4f} | IID: {iid_acc:.2%} | OOD: {ood_acc:.2%}")
        
        # 通过统一接口保存数据
        manager.log_metrics(epoch, avg_train_loss, iid_acc, ood_acc)
        manager.save_snapshot(model, epoch)

        # 保存最优
        if ood_acc >= best_ood_acc:
            best_ood_acc = ood_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_best.pth"))

    # 7. 收尾
    manager.run_final_logic()
    io.logger.info(f"训练结束。最终 OOD 准确率: {best_ood_acc:.2%}")

if __name__ == "__main__":
    train()