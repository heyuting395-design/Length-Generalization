# src/models/utils.py
import torch.nn as nn

def init_weights(model, init_type='xavier', gain=1.0):
    """
    灵活的模型初始化函数
    init_type: 'xavier', 'kaiming', 'orthogonal', 'normal'
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
        # Transformer Layer 的特殊初始化 (参考 GPT-2/DeepNet)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)