import random
import torch
import os
import json
from typing import List, Tuple, Set, Optional

# -----------------------------
# 加载配置文件
# -----------------------------
def load_config(config_path=None):
    if config_path is None:
        # 获取当前脚本所在的绝对路径
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "config_data.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()

# -----------------------------
# 基础工具
# -----------------------------
def int_to_digits(num: int, length: int) -> List[int]:
    digits = [int(d) for d in str(num)[::-1]]
    return digits + [0] * (length - len(digits))

def count_carries(a: int, b: int) -> int:
    sa, sb = str(a)[::-1], str(b)[::-1]
    max_len = max(len(sa), len(sb))
    carry, carries = 0, 0
    for i in range(max_len):
        da = int(sa[i]) if i < len(sa) else 0
        db = int(sb[i]) if i < len(sb) else 0
        if da + db + carry >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
    return carries

# -----------------------------
# 核心生成逻辑
# -----------------------------
def generate_data(
    num_samples: int,
    min_digits: int,
    max_digits: int,
    total_L: int,
    min_carries: Optional[int] = None,
    exclude_set: Set[Tuple[int, int]] = None,
) -> List[Tuple[List[int], List[int], List[int]]]:
    data = []
    seen = set()
    exclude_set = exclude_set or set()
    max_attempts = num_samples * 50 
    attempts = 0

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        cur_digits = random.randint(min_digits, max_digits)
        a = random.randint(10**(cur_digits-1) if cur_digits > 1 else 0, 10**cur_digits - 1)
        b = random.randint(0, 10**cur_digits - 1)

        if (a, b) in exclude_set or (a, b) in seen:
            continue
            
        carries = count_carries(a, b)
        if min_carries is not None and carries < min_carries:
            continue

        data.append((
            int_to_digits(a, total_L),
            int_to_digits(b, total_L),
            int_to_digits(a + b, total_L + 1),
        ))
        seen.add((a, b))
    return data

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    # 设定随机种子 (从 json 读取)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Loaded config from JSON. Generating {config['num_train']} training samples...")

    # 1. 训练集
    train_data = generate_data(
        num_samples=config['num_train'],
        min_digits=1,
        max_digits=config['train_max_digits'],
        total_L=config['L'],
        min_carries=config['train_min_carries']
    )

    # 提取数值用于去重
    train_val_set = set()
    for a_l, b_l, _ in train_data:
        a_v = int("".join(map(str, a_l[::-1])))
        b_v = int("".join(map(str, b_l[::-1])))
        train_val_set.add((a_v, b_v))

    # 2. IID 测试集
    print(f"Generating {config['num_test_iid']} IID samples...")
    test_iid = generate_data(
        num_samples=config['num_test_iid'],
        min_digits=1,
        max_digits=config['iid_max_digits'],
        total_L=config['L'],
        min_carries=config['iid_min_carries'],
        exclude_set=train_val_set
    )

    # 3. OOD 测试集
    print(f"Generating {config['num_test_ood']} OOD samples...")
    test_ood = generate_data(
        num_samples=config['num_test_ood'],
        min_digits=config['ood_min_digits'],
        max_digits=config['ood_max_digits'],
        total_L=config['L'],
        min_carries=config['ood_min_carries']
    )

    # 保存文件
    torch.save(train_data, os.path.join(config['output_dir'], "train_data.pt"))
    torch.save(test_iid, os.path.join(config['output_dir'], "test_iid.pt"))
    torch.save(test_ood, os.path.join(config['output_dir'], "test_ood.pt"))

    # 科研习惯：顺便把本次使用的 config 备份到数据目录下，方便以后查阅
    with open(os.path.join(config['output_dir'], "config_log.json"), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nDone! Files saved in {config['output_dir']}")
    print(f"Train: {len(train_data)} | IID: {len(test_iid)} | OOD: {len(test_ood)}")