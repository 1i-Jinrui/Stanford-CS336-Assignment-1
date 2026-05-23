import math
import torch
import torch.nn as nn
from .softmax import softmax


class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        d_k = query.shape[-1]
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -float('inf'))
        attention = softmax(attention, dim=-1)
        return torch.matmul(attention, value)
