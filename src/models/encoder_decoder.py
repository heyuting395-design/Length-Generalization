import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import DigitEmbedding
from .utils import init_weights
# -----------------------------
# 1. 由于transfomer没有rope的库
# -----------------------------
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

    def forward(self, q, k, v, cos_all, sin_all, mask=None):
        B, L_q, _ = q.shape
        L_k = k.shape[1]

        q = self.q_proj(q).view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, L_k, self.nhead, self.head_dim).transpose(1, 2)

        q = DigitEmbedding.apply_rope(q, cos_all[:L_q], sin_all[:L_q])
        k = DigitEmbedding.apply_rope(k, cos_all[:L_k], sin_all[:L_k])

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.out_proj(attn_out)

# -----------------------------
# 2.交错式 Encoder-Decoder Adder
# -----------------------------
class InterleavedTransformerAdder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.nhead = config['model']['nhead']
        self.num_layers = config['model']['num_layers']
        d_ff = config['model']['dim_feedforward']
        init_type = config['model'].get('init_type', 'xavier')
        init_gain = config['model'].get('init_gain', 1.0)
        self.embedding = DigitEmbedding(self.d_model, config['model']['num_digits'], self.nhead)
        
        # 定义层
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': RoPEAttention(self.d_model, self.nhead),
                'cross_attn': RoPEAttention(self.d_model, self.nhead),
                'norm1': nn.LayerNorm(self.d_model),
                'norm2': nn.LayerNorm(self.d_model),
                'norm3': nn.LayerNorm(self.d_model),
                'mlp': nn.Sequential(
                    nn.Linear(self.d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, self.d_model),
                    nn.Dropout(0.1)
                )
            }) for _ in range(self.num_layers)
        ])
        
        self.fc_out = nn.Linear(self.d_model, 10)
        self.apply(lambda m: init_weights(m, init_type=init_type, gain=init_gain))
    def forward(self, a_seq, b_seq, tgt_seq):
        """
        a_seq, b_seq: 输入的两个加数 (Batch, L)
        tgt_seq: 训练时的目标序列 (Batch, T)
        """
        device = a_seq.device
        # src_emb 形状: (Batch, 2*L, d_model)
        src_emb, src_cos, src_sin = self.embedding.interleave_and_embed(a_seq, b_seq)
        
        # tgt_emb 形状: (Batch, T, d_model)
        tgt_emb = self.embedding(tgt_seq, type_id=2)
        
        max_len = max(src_emb.size(1), tgt_emb.size(1))
        cos, sin = self.embedding.get_rope_cache(max_len, device)
        
        # --- Encoder 过程 ---
        memory = src_emb
        for layer in self.layers:
            memory = layer['norm1'](memory + layer['self_attn'](memory, memory, memory, cos, sin))
            memory = layer['norm3'](memory + layer['mlp'](memory))
            
        # --- Decoder 过程 ---
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(device)
        x = tgt_emb
        for layer in self.layers:
            # Self Attention (带因果掩码)
            x = layer['norm1'](x + layer['self_attn'](x, x, x, cos, sin, mask=tgt_mask))
            # Cross Attention (Query 是 x, Key/Value 是 memory)
            x = layer['norm2'](x + layer['cross_attn'](x, memory, memory, cos, sin))
            x = layer['norm3'](x + layer['mlp'](x))
            
        return self.fc_out(x)

    @torch.no_grad()
    def predict(self, a_seq, b_seq, max_len=12):
        self.eval()
        device = a_seq.device
        batch = a_seq.size(0)
        SOS_TOKEN = 12 # 12 是 <SOS>
        
        # 1. 编码一次 Source
        src_emb, src_cos, src_sin = self.embedding.interleave_and_embed(a_seq, b_seq)
        
        # 预计算足够长的 RoPE
        rope_len = max(src_emb.size(1), max_len + 1)
        cos, sin = self.embedding.get_rope_cache(rope_len, device)
        
        memory = src_emb
        for layer in self.layers:
            memory = layer['norm1'](memory + layer['self_attn'](memory, memory, memory, cos, sin))
            memory = layer['norm3'](memory + layer['mlp'](memory))
            
        # 2. 自回归解码
        curr_tgt = torch.full((batch, 1), SOS_TOKEN, dtype=torch.long, device=device)
        all_outputs = []
        
        for i in range(max_len):
            tgt_emb = self.embedding(curr_tgt, type_id=2)
            t_mask = nn.Transformer.generate_square_subsequent_mask(curr_tgt.size(1)).to(device)
            
            x = tgt_emb
            for layer in self.layers:
                x = layer['norm1'](x + layer['self_attn'](x, x, x, cos, sin, mask=t_mask))
                x = layer['norm2'](x + layer['cross_attn'](x, memory, memory, cos, sin))
                x = layer['norm3'](x + layer['mlp'](x))
            
            # 取最后一个时刻的预测结果
            logits = self.fc_out(x[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)
            all_outputs.append(next_token)
            
            # 拼接预测值
            curr_tgt = torch.cat([curr_tgt, next_token], dim=1)
            
        return torch.cat(all_outputs, dim=1)