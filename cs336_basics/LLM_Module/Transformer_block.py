import torch
import torch.nn as nn
from .RMS_norm import RMS_norm
from .multihead_self_attention_with_rope import MultiheadSelfAttentionWithRope
from .SwiGLU import SwiGlU
from .ROPE import RoPE


class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if self.d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.dff = dff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.RMS_norm1 = RMS_norm(d_model=d_model, eps=1e-5)
        self.RMS_norm2 = RMS_norm(d_model=d_model, eps=1e-5)
        self.RoPE = RoPE(theta=self.theta, max_seq_len=self.max_seq_len, dim=d_model // num_heads)
        self.MHA = MultiheadSelfAttentionWithRope(d_model=d_model, num_heads=num_heads, rope=self.RoPE)
        self.SwiGlU = SwiGlU(d_model=d_model, d_ff=dff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA(self.RMS_norm1(x))
        x = x + self.SwiGlU(self.RMS_norm2(x))
        return x
