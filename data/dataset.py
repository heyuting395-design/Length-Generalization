import torch
import os
from torch.utils.data import Dataset, DataLoader

class AdditionDataset(Dataset):
    def __init__(self, file_path, vocab_config=None):
        """
        vocab_config: 用于定义特殊 token 的 ID，例如 {'pad': 10, 'sep': 11}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
            
        data = torch.load(file_path)
        
        # 将列表转换为 Tensor
        # item[0]: list of digits for 'a'
        # item[1]: list of digits for 'b'
        # item[2]: list of digits for 'sum'
        self.a = torch.tensor([item[0] for item in data], dtype=torch.long)
        self.b = torch.tensor([item[1] for item in data], dtype=torch.long)
        self.sum = torch.tensor([item[2] for item in data], dtype=torch.long)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        # 返回原始数字序列，后续在模型输入层进行拼接（Concatenation）
        # 这种做法最严谨，因为你可以灵活实验不同的拼接顺序 (A+B or B+A)
        return self.a[idx], self.b[idx], self.sum[idx]

def get_dataloaders(data_dir, batch_size=128, num_workers=4):
    """
    增加了 num_workers 和更合理的默认 batch_size
    """
    # 定义路径
    train_path = os.path.join(data_dir, "train_data.pt")
    iid_path = os.path.join(data_dir, "test_iid.pt")
    ood_path = os.path.join(data_dir, "test_ood.pt")

    train_ds = AdditionDataset(train_path)
    iid_ds = AdditionDataset(iid_path)
    ood_ds = AdditionDataset(ood_path)

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True if torch.cuda.is_available() else False
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    test_iid_loader = DataLoader(iid_ds, shuffle=False, **loader_args)
    test_ood_loader = DataLoader(ood_ds, shuffle=False, **loader_args)

    return train_loader, test_iid_loader, test_ood_loader