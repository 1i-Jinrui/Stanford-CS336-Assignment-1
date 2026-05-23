import torch
import torch.nn as nn
from .linear_module import Linear_module
from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.q_proj = Linear_module(d_model, d_model, bias=False)
        self.k_proj = Linear_module(d_model, d_model, bias=False)
        self.v_proj = Linear_module(d_model, d_model, bias=False)
        self.o_proj = Linear_module(d_model, d_model, bias=False)
        self.attention = scaled_dot_product_attention()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(out)
