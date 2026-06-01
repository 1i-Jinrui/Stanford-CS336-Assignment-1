import torch
import torch.nn as nn


class Embedding_module(nn.Module):
    def __init__(self, vocab_size: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding_matrix = nn.Parameter(
            torch.empty(self.vocab_size, self.embedding_dim, device=self.device, dtype=dtype))
        std = 1
        nn.init.trunc_normal_(self.embedding_matrix, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[x]
