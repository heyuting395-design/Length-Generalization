import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import DigitEmbedding
from .utils import init_weights

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

    def forward(self, x, cos_all, sin_all, mask=None, is_causal=False, past_key_value=None):
        B, L, _ = x.shape
        
        # 1. 投影
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # 2. 应用 RoPE
        # 注意：推理时 L 可能为 1，但它的位置索引取决于已缓存的长度
        start_pos = 0 if past_key_value is None else past_key_value[0].shape[2]
        q = DigitEmbedding.apply_rope(q, cos_all[start_pos : start_pos + L], sin_all[start_pos : start_pos + L])
        k = DigitEmbedding.apply_rope(k, cos_all[start_pos : start_pos + L], sin_all[start_pos : start_pos + L])

        # 3. KV Cache 处理
        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        new_kv = (k, v) # 返回给下一轮使用

        # 4. Attention (使用高效算子)
        # 如果 is_causal=True 且没有手动传入 mask，PyTorch 会自动处理下三角掩码
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and mask is None and L > 1 # 训练时 L > 1 开启，推理时 L=1 不需要掩码
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(attn_out), new_kv

class DecoderOnlyAdder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.nhead = config['model']['nhead']
        self.num_layers = config['model']['num_layers']
        d_ff = config['model']['dim_feedforward']
        init_type = config['model'].get('init_type', 'xavier')
        init_gain = config['model'].get('init_gain', 1.0)
        self.embedding = DigitEmbedding(self.d_model, config['model']['num_digits'], self.nhead)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': RoPEAttention(self.d_model, self.nhead),
                'norm1': nn.LayerNorm(self.d_model),
                'norm2': nn.LayerNorm(self.d_model),
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

    def forward(self, x_tokens, type_ids, past_key_values=None):
        B, L = x_tokens.shape
        device = x_tokens.device
        
        # 1. Embedding
        x = self.embedding(x_tokens, type_ids)
        cos, sin = self.embedding.get_rope_cache(2048, device) # 预存足够长的 cache
        
        new_kvs = []
        for i, layer in enumerate(self.layers):
            # 获取当前层的缓存
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            # --- Post-LN 结构: x = Norm(x + Sublayer(x)) ---
            attn_res, kv = layer['attn'](x, cos, sin, is_causal=True, past_key_value=past_kv)
            x = layer['norm1'](x + attn_res)
            
            mlp_res = layer['mlp'](x)
            x = layer['norm2'](x + mlp_res)
            
            new_kvs.append(kv)
            
        return self.fc_out(x), new_kvs

    @torch.no_grad()
    def predict(self, src_seq, src_types, max_len=12):
        self.eval()
        device = src_seq.device
        batch = src_seq.size(0)
        SOS_TOKEN = 12
        
        # 1. Prefill 阶段：处理输入序列
        curr_tokens = torch.cat([src_seq, torch.full((batch, 1), SOS_TOKEN, device=device)], dim=1)
        curr_types = torch.cat([src_types, torch.full((batch, 1), 2, device=device)], dim=1)
        
        logits, kvs = self.forward(curr_tokens, curr_types)
        
        # 2. Decoding 阶段：逐个生成，使用 KV Cache
        next_token = torch.argmax(logits[:, -1:, :], dim=-1)
        preds = [next_token]
        
        for _ in range(max_len - 1):
            # 此时只需传入最新的 token，L = 1
            logits, kvs = self.forward(next_token, torch.full((batch, 1), 2, device=device), past_key_values=kvs)
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            preds.append(next_token)
            
        return torch.cat(preds, dim=1)