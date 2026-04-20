import torch
import torch.nn as nn
from .embedding_module import Embedding_module
from .Transformer_block import Transformer_block
from .RMS_norm import RMS_norm
from .linear_module import Linear_module


class Transformer_LM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding_module(vocab_size=vocab_size, embedding_dim=d_model)

        self.layers = nn.ModuleList([
            Transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                dff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            )
            for _ in range(num_layers)
        ])

        self.RMS_final = RMS_norm(d_model=d_model, eps=1e-5)
        self.lm_head = Linear_module(d_model, vocab_size, bias=False)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        in_indices: [batch_size, sequence_length]
        return: [batch_size, sequence_length, vocab_size]
        """

        batch_size, seq_len = in_indices.shape
        if seq_len > self.context_length:
            raise ValueError("sequence_length exceeds context_length")

        x = self.token_embeddings(in_indices)  # [B, T, d_model]

        for layer in self.layers:
            x = layer(x)

        x = self.RMS_final(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits
