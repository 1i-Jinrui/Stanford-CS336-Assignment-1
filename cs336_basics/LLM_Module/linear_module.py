import torch
import torch.nn as nn
import math


class Linear_module(nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, device=self.device, dtype=self.dtype))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.weight.T)
        return out + self.bias if self.bias is not None else out


