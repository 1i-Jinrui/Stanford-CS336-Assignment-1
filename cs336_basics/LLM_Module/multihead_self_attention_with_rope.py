import torch
import torch.nn as nn
import math
from .linear_module import Linear_module
from .softmax import softmax


class MultiheadSelfAttentionWithRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = 2048
        self.rope = rope
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        if self.rope is not None:
            if self.head_dim % 2 != 0:
                raise ValueError("head_dim must be even when using RoPE")
        self.Wq = Linear_module(d_model, d_model, bias=False)
        self.Wk = Linear_module(d_model, d_model, bias=False)
        self.Wv = Linear_module(d_model, d_model, bias=False)
        self.Wo = Linear_module(d_model, d_model, bias=False)
        self.register_buffer(name='causal_mask', tensor=torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        positions = torch.arange(0, seq_len, device=x.device)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:seq_len, :seq_len] == 0, -float('inf'))
        attn_score = softmax(attn, dim=-1)
        out = torch.matmul(attn_score, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        return self.Wo(out)
