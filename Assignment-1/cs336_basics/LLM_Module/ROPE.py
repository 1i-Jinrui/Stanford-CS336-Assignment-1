import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta: float, max_seq_len: int, dim: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dim = dim
        if self.dim % 2 != 0:
            raise ValueError('dim must be even number')

        freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=self.device) / self.dim))
        position = torch.arange(0, max_seq_len, device=self.device)
        sinusoids = torch.outer(position, freq)  # (max_seq_len, d_model // 2)
        self.register_buffer('sin_cache', sinusoids.sin())
        self.register_buffer('cos_cache', sinusoids.cos())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin  # (*, seq_len, d_model // 2)
        out_odd = x_even * sin + x_odd * cos  # (*, seq_len, d_model // 2)
        out = torch.stack((out_even, out_odd), dim=-1)  # (*, seq_len, d_model // 2, 2)
        return out.flatten(-2)