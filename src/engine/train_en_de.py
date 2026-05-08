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

from src.models.encoder_decoder import InterleavedTransformerAdder
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

def prepare_ntp_data(a_seq, b_seq, target_seq, device):
    batch, L = a_seq.shape
    SOS_TOKEN = 12 
    src_seq = torch.stack([a_seq, b_seq], dim=2).view(batch, 2*L).to(device)
    src_types = torch.tensor([0, 1] * L, device=device).repeat(batch, 1)
    
    sos = torch.full((batch, 1), SOS_TOKEN, device=device, dtype=torch.long)
    target_seq = target_seq.to(device)
    tgt_in = torch.cat([sos, target_seq[:, :-1]], dim=1)
    tgt_types = torch.full(tgt_in.shape, 2, device=device, dtype=torch.long)
    
    return src_seq, src_types, tgt_in, tgt_types, target_seq

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for a, b, target in dataloader:
            batch, L = a.shape
            src_seq, src_types, _, _, tgt_out = prepare_ntp_data(a, b, target, device)
            preds = model.predict(src_seq, src_types, max_len=L+1) 
            is_correct = (preds == tgt_out).all(dim=1)
            correct += is_correct.sum().item()
            total += batch
    return correct / total

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
    
    # 建立长度索引用于课程学习
    length_to_indices = {i: [] for i in range(1, 11)}
    full_train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    for idx, (a, b, _) in enumerate(full_train_data):
        actual_l = max(get_actual_length(a), get_actual_length(b))
        if actual_l in length_to_indices: length_to_indices[actual_l].append(idx)
    
    # 模型与分析器初始化
    model = InterleavedTransformerAdder(config).to(device)
    manager = AnalyticManager(config, io.exp_dir)
    
    # 【核心】保存 Epoch 0 初始随机状态
    manager.save_snapshot(model, epoch=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["train"]["lr"]), 
        weight_decay=config["train"].get("weight_decay", 1e-2)
    )
    
    total_steps = config["train"]["epochs"] * len(raw_train_loader)
    scheduler = get_lr_scheduler(optimizer, config, total_steps)
    criterion = nn.CrossEntropyLoss()

    best_ood = 0.0

    for epoch in range(1, config["train"]["epochs"] + 1):
        # 三阶段课程学习逻辑
        if epoch <= 20: 
            stage, lens = "Stage 1: 1-2 Digits", [1, 2]
        elif epoch <= 100: 
            stage, lens = "Stage 2: 1-4 Digits", [1, 2, 3, 4]
        else: 
            stage, lens = "Stage 3: Full 1-7 Digits", list(range(1, 8))

        indices = []
        for l in lens: indices.extend(length_to_indices[l])
        train_loader = DataLoader(Subset(raw_train_loader.dataset, indices), batch_size=config["train"]["batch_size"], shuffle=True)

        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch} [{stage}]")
        
        for a, b, target in pbar:
            optimizer.zero_grad()
            src, st, tin, tt, tout = prepare_ntp_data(a, b, target, device)
            logits = model(src, st, tin)
            loss = criterion(logits.view(-1, 10), tout.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # 评估
        avg_train_loss = epoch_loss / len(train_loader)
        iid_acc = evaluate(model, test_iid_loader, device)
        ood_acc = evaluate(model, test_ood_loader, device)
        
        io.logger.info(f"Ep {epoch} | Loss: {avg_train_loss:.4f} | IID: {iid_acc:.2%} | OOD: {ood_acc:.2%}")
        
        # 保存指标与快照
        manager.log_metrics(epoch, avg_train_loss, iid_acc, ood_acc)
        manager.save_snapshot(model, epoch)

        if ood_acc > best_ood:
            best_ood = ood_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_best.pth"))

    manager.run_final_logic()

if __name__ == "__main__":
    train()