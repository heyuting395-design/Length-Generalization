import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# 路径处理
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 注意：这里导入你新写的 DecoderOnly 模型
from src.models import DecoderOnlyAdder 
from data.dataset import get_dataloaders
from src.engine.lr_scheduler import get_lr_scheduler
from src.engine.analytic_manager import AnalyticManager 
from src.utils.io_operate import IOManager

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_actual_length(digits_list):
    for i in range(len(digits_list)-1, -1, -1):
        if digits_list[i] != 0: return i + 1
    return 1

def prepare_decoder_only_data(a_seq, b_seq, target_seq, device):
    batch, L = a_seq.shape
    SOS_TOKEN = 12
    
    # 1. 拼接输入操作数 (Batch, 2L)
    src_seq = torch.stack([a_seq, b_seq], dim=2).view(batch, 2*L).to(device)
    src_types = torch.tensor([0, 1] * L, device=device).repeat(batch, 1)
    
    # 2. 准备答案部分
    target_seq = target_seq.to(device)
    sos = torch.full((batch, 1), SOS_TOKEN, device=device)
    
    # 3. 构造模型输入全序列: [Src] + [SOS] + [Target的前N-1位]
    full_in = torch.cat([src_seq, sos, target_seq[:, :-1]], dim=1)
    
    # 类型映射: 0/1 是加数, 2 是答案部分
    types_in = torch.cat([src_types, torch.full((batch, target_seq.size(1)), 2, device=device)], dim=1)
    
    # 4. 构造 Loss 标签: 输入部分全部用 -100 忽略，只对目标输出计算 Loss
    ignore_labels = torch.full((batch, 2*L + 1), -100, device=device)
    # 注意：使用 target_seq[:, 1:] 而不是 target_seq，长度对齐
    full_labels = torch.cat([ignore_labels, target_seq[:, 1:]], dim=1)
    
    return full_in, types_in, full_labels, src_seq, src_types

def evaluate(model, dataloader, device, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    
    with torch.no_grad():
        for a, b, target in dataloader:
            batch, L = a.shape
            # 准备数据用于计算 Loss
            full_in, types_in, full_labels, src_seq, src_types = prepare_decoder_only_data(a, b, target, device)
            
            # 1. 计算验证集 Loss
            logits, _ = model(full_in, types_in)
            loss = criterion(logits.view(-1, 10), full_labels.view(-1))
            total_loss += loss.item()
            
            # 2. 自动回归预测准确率 (使用 KV Cache 提升速度)
            preds = model.predict(src_seq, src_types, max_len=target.size(1)) 
            is_correct = (preds == target.to(device)).all(dim=1)
            
            correct += is_correct.sum().item()
            total += batch
            
    return correct / total, total_loss / len(dataloader)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    io = IOManager(args.config) 
    config = io.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = os.path.join(io.exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dir = os.path.join(project_root, "data/data_output")
    raw_train_loader, test_iid_loader, test_ood_loader = get_dataloaders(data_dir, batch_size=config["train"]["batch_size"])
    
    # 建立课程学习索引
    length_to_indices = {i: [] for i in range(1, 11)}
    full_train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    for idx, (a, b, _) in enumerate(full_train_data):
        actual_l = max(get_actual_length(a), get_actual_length(b))
        if actual_l in length_to_indices: length_to_indices[actual_l].append(idx)
    
    # 初始化 Decoder-Only 模型
    model = DecoderOnlyAdder(config).to(device)
    manager = AnalyticManager(config, io.exp_dir)
    manager.save_snapshot(model, epoch=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["train"]["lr"]), 
        weight_decay=config["train"].get("weight_decay", 1e-2)
    )
    
    total_steps = config["train"]["epochs"] * len(raw_train_loader)
    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    
    # 关键：ignore_index=-100 忽略输入部分的标签
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_ood_acc = 0.0

    for epoch in range(1, config["train"]["epochs"] + 1):
        # 课程学习阶段设置
        if epoch <= 20: 
            stage, lens = "Stage 1: 1-2 Digits", [1, 2]
        elif epoch <= 100: 
            stage, lens = "Stage 2: 1-4 Digits", [1, 2, 3, 4]
        else: 
            stage, lens = "Stage 3: Full 1-7 Digits", list(range(1, 8))

        indices = []
        for l in lens: indices.extend(length_to_indices[l])
        train_subset_loader = DataLoader(Subset(raw_train_loader.dataset, indices), 
                                         batch_size=config["train"]["batch_size"], 
                                         shuffle=True)

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_subset_loader, desc=f"Ep {epoch} [{stage}]")
        
        for a, b, target in pbar:
            optimizer.zero_grad()
            full_in, types_in, full_labels, _, _ = prepare_decoder_only_data(a, b, target, device)
            
            logits, _ = model(full_in, types_in)
            loss = criterion(logits.view(-1, 10), full_labels.view(-1))
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_subset_loader)
        
        # 评估 IID 和 OOD (同时获取 Acc 和 Loss)
        iid_acc, iid_loss = evaluate(model, test_iid_loader, device, criterion)
        ood_acc, ood_loss = evaluate(model, test_ood_loader, device, criterion)
        
        # 记录日志
        io.logger.info(f"Ep {epoch} | T-Loss: {avg_train_loss:.4f} | IID Loss/Acc: {iid_loss:.4f}/{iid_acc:.2%} | OOD Loss/Acc: {ood_loss:.4f}/{ood_acc:.2%}")
        
        # 保存到 AnalyticManager (确保你的 manager 支持记录这些字段)
        manager.log_metrics(epoch, avg_train_loss, iid_loss, iid_acc, ood_loss, ood_acc)
        # 如果你想额外保存 test_loss，可以在 manager 里加一个方法或者直接存入 snapshot
        manager.save_snapshot(model, epoch)

        if ood_acc > best_ood_acc:
            best_ood_acc = ood_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_best.pth"))

    manager.run_final_logic()

if __name__ == "__main__":
    train()