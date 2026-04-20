import torch
import torch.nn as nn


class RMS_norm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))  # 对应可学习参数γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_type = x.dtype
        x = x.to(torch.float32)
        inv_RMS = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * inv_RMS
        return x.to(origin_type) * self.weight


