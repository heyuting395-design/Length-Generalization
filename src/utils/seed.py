import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    设置全局随机种子以保证实验的可复现性。
    涵盖了 Python 基础、NumPy、PyTorch CPU/GPU 以及 CuDNN 算法。
    """
    # 1. 基础 Python 随机种子
    random.seed(seed)
    
    # 2. NumPy 随机种子
    np.random.seed(seed)
    
    # 3. PyTorch 随机种子 (CPU)
    torch.manual_seed(seed)
    
    # 4. PyTorch 随机种子 (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 针对多 GPU 情况
    
    # 5. 设置 CUDNN 确定性
    # 注意：这可能会稍微降低运行效率，但能确保每次计算结果完全一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 6. 设置环境变量（可选，进一步限制随机性）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"--- 随机种子已设置为: {seed} (CuDNN 已开启确定性模式) ---")