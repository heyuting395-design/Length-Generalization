import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    带预热的余弦退火调度器。
    在前 warmup_steps 中学习率线性增加，随后按余弦曲线下降。
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # 线性预热阶段
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

def get_lr_scheduler(optimizer, config, total_steps):
    """
    根据配置返回调度器
    """
    sched_config = config['train'].get('scheduler', {})
    sched_type = sched_config.get('type', 'cosine')
    
    if sched_type == 'cosine':
        warmup_steps = sched_config.get('warmup_steps', 0)
        min_lr = float(sched_config.get('min_lr', 1e-6))
        return WarmupCosineScheduler(
            optimizer, 
            warmup_steps=warmup_steps, 
            total_steps=total_steps, 
            min_lr=min_lr
        )
    elif sched_type == 'step':
        step_size = sched_config.get('step_size', 100)
        gamma = sched_config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        # 默认返回恒定学习率 (LambdaLR)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)