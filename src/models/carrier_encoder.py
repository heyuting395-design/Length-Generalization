import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_weights
from .embedding import DigitEmbedding

class RoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, q, k, v, cos, sin, mask=None):
        B, L_q, _ = q.shape
        L_k = k.shape[1]

        # 投影并分头
        q = self.q_proj(q).view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)

        # 应用 RoPE (调用 Embedding 类中的静态旋转逻辑)
        q = DigitEmbedding.apply_rope(q, cos, sin)
        k = DigitEmbedding.apply_rope(k, cos, sin)

        # 高效注意力计算
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.out_proj(attn_out)

class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, cos, sin, mask=None):
        # 残差连接 + LayerNorm
        x = self.norm1(x + self.self_attn(x, x, x, cos, sin, mask=mask))
        x = self.norm2(x + self.mlp(x))
        return x
    
class InterleavedEncoderAdder(nn.Module):
    def __init__(self, config):
        super().__init__()
        m_cfg = config['model']
        self.d_model = m_cfg['d_model']
        self.nhead = m_cfg['nhead']
        self.embedding = DigitEmbedding(
            self.d_model, 
            m_cfg['num_digits'], 
            self.nhead
        )
        
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=m_cfg['dim_feedforward'],
                dropout=m_cfg['dropout']
            ) for _ in range(m_cfg['num_layers'])
        ])
        
        self.fc_out = nn.Linear(self.d_model, 10)
        
        # 显式进位演化层 (保持不变)
        self.carry_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh()
        )
        
        init_method = m_cfg.get('init_method', 'xavier')
        init_gain = m_cfg.get('init_gain', 1.0)
        self.apply(lambda m: init_weights(m, init_type=init_method, gain=init_gain))
        # ===================
        print(f"Model initialized with {init_method} (gain={init_gain}), RoPE enabled.")

    def forward(self, a_seq, b_seq, target_seq=None, tf_ratio=0.0):
        batch, L = a_seq.shape
        device = a_seq.device
        
        src, cos, sin = self.embedding.interleave_and_embed(a_seq, b_seq)
        
        memory = src
        for layer in self.layers:
            memory = layer(memory, cos, sin)
        
        outputs = []
        carry_vec = torch.zeros(batch, self.d_model, device=device)
        
        for t in range(L + 1):
            # 获取当前位的 bit_feature
            bit_feat = memory[:, t*2 : t*2+2, :].mean(dim=1) if t < L else torch.zeros(batch, self.d_model, device=device)
            
            # 融合进位
            combined = bit_feat + carry_vec
            logits = self.fc_out(combined)
            outputs.append(logits.unsqueeze(1))
            
            # 状态演化 (隐式模拟进位逻辑)
            carry_vec = self.carry_net(combined)
            
        return torch.cat(outputs, dim=1)

    def extract_carry_states(self, a_seq, b_seq):
        """
        适配新 Embedding 的提取逻辑
        """
        self.eval()
        device = a_seq.device
        batch, L = a_seq.shape
        
        # 1. 重现 forward 的编码逻辑
        src, cos, sin = self.embedding.interleave_and_embed(a_seq, b_seq)
        
        memory = src
        for layer in self.layers:
            memory = layer(memory, cos, sin)
        
        carry_history = []
        carry_vec = torch.zeros(batch, self.d_model, device=device)
        
        # 2. 模拟循环并记录每一位的 carry_vec
        for t in range(L + 1):
            bit_feat = memory[:, t*2 : t*2+2, :].mean(dim=1) if t < L else torch.zeros(batch, self.d_model, device=device)
            combined = bit_feat + carry_vec
            
            # 记录当前的 carry_vec (这是传给下一位的“进位信号”)
            carry_history.append(carry_vec.unsqueeze(1))
            
            # 更新 carry_vec
            carry_vec = self.carry_net(combined)
            
        return torch.cat(carry_history, dim=1)