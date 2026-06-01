import torch
import torch.nn as nn
from .linear_module import Linear_module


class SwiGlU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = Linear_module(in_features=d_model, out_features=d_ff, bias=False)
        self.W2 = Linear_module(in_features=d_ff, out_features=d_model, bias=False)
        self.W3 = Linear_module(in_features=d_model, out_features=d_ff, bias=False)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = self.silu(self.W1(x))

        return self.W2(x_gate * self.W3(x))


