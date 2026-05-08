import torch
import torch.nn as nn
import math

class DigitEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size=13, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 1. 数字编码 (0-9: 数字, 10: 填充, 11: 符号, 12: SOS)
        self.digit_embed = nn.Embedding(vocab_size, d_model)
        
        # 2. 类型编码 (0: 加数A, 1: 加数B, 2: 结果位)
        self.type_embed = nn.Embedding(3, d_model)
        
        # 3. RoPE 频率定义
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def interleave_and_embed(self, a_seq, b_seq):
        """
          此处处理数字并进行交错合并
          a_seq, b_seq: (Batch, L)
        返回: 
            src_emb: (Batch, 2*L, d_model)
            cos, sin: (2*L, head_dim)
        """
        batch, L = a_seq.shape
        device = a_seq.device
        
        # 数字编码 + 类型标记 (A=0, B=1)
        a_emb = self.digit_embed(a_seq) + self.type_embed(torch.full_like(a_seq, 0))
        b_emb = self.digit_embed(b_seq) + self.type_embed(torch.full_like(b_seq, 1))
        
        # 物理交错: [a1, b1, a2, b2, ...]
        # 形状变换: (Batch, L, 2, d_model) -> (Batch, 2*L, d_model)
        src_emb = torch.stack([a_emb, b_emb], dim=2).view(batch, 2 * L, self.d_model)
        
        #RoPE 缓存
        cos, sin = self.get_rope_cache(2 * L, device)
        
        return src_emb, cos, sin

    def get_rope_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def apply_rope(x, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        # x shape: (B, H, L, D_head)
        # cos/sin shape: (L, D_head)
        def rotate_half(tensor):
            x1, x2 = tensor.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (rotate_half(x) * sin)

    def forward(self, x, type_ids=None):
        """
        x: (batch, seq_len) token ids
        type_ids: (batch, seq_len) 类型标签 (0,1,2)，如果不提供则默认为2
        """
        if type_ids is None:
            type_ids = torch.full_like(x, 2)
        return self.digit_embed(x) + self.type_embed(type_ids)