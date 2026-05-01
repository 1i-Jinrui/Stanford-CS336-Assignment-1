import torch
import torch.nn as nn
import math
from dataclasses import dataclass


@dataclass
class LLM_Config:
    vocab_size: int = 10000
    d_model: int = 512
    d_ff: int = int(d_model * 8 / 3)
    num_heads: int = 8
    num_layers: int = 6
    context_len: int = 1000
    batch_size: int = 64
    rope_theta: float = 10000


class Linear_Module(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, device=self.device, dtype=self.dtype))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.matmul(x, self.weight.T) + self.bias
        else:
            return torch.matmul(x, self.weight.T)


class Embeddings(nn.Module):
    def __init__(self, config: LLM_Config, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.device = device
        self.dtype = dtype
        self.embedding_matrix = nn.Parameter(
            torch.empty(self.vocab_size, self.d_model, device=self.device, dtype=self.dtype)
        )
        std = 1.0
        nn.init.trunc_normal_(self.embedding_matrix, std=std, a=-3 * std, b=3 * std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[input_ids]


class RMS_norm(nn.Module):
    def __init__(self, config: LLM_Config,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = config.d_model
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_type = x.dtype
        x = x.to(torch.float32)
        inv_RMS = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x * inv_RMS).to(origin_type) * self.weight.to(origin_type)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


class RoPE(nn.Module):
    def __init__(self, config: LLM_Config, device: torch.device | None = None):
        super().__init__()
        self.theta = config.rope_theta
        self.d_model = config.d_model
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.dim = config.d_model // config.num_heads  # 这里不应该用d_model，因为实际传入的张量为每个Head，应为每个Head的dim
        self.device = device
        self.max_seq_len = config.context_len
        if self.dim % 2 != 0:
            raise ValueError('dim must be even')
        freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=self.device) / self.dim))
        positions = torch.arange(0, self.max_seq_len, device=self.device)
        sinusoids = torch.outer(positions, freq)
        self.register_buffer('sin_cache', sinusoids.sin())
        self.register_buffer('cos_cache', sinusoids.cos())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(0, x.shape[-2], device=x.device)
        x_odd = x[..., 1::2]
        x_even = x[..., 0::2]
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.stack((out_even, out_odd), dim=-1)
        return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: LLM_Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.max_seq_len = config.context_len
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads
        self.Wqkv = Linear_Module(in_features=self.d_model, out_features=self.d_model * 3, bias=False)
        self.Wo = Linear_Module(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.register_buffer('causal_mask',
                             torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)))
        self.RoPE = RoPE(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.Wqkv(x)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.RoPE(q)
        k = self.RoPE(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(~self.causal_mask[: seq_len, : seq_len], float('-inf'))
        attn = softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.Wo(out)


class SwiGLU(nn.Module):
    def __init__(self, config: LLM_Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.W1 = Linear_Module(in_features=self.d_model, out_features=self.d_ff, bias=False)
        self.W2 = Linear_Module(in_features=self.d_ff, out_features=self.d_model, bias=False)
        self.W3 = Linear_Module(in_features=self.d_model, out_features=self.d_ff, bias=False)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))


class Block(nn.Module):
    def __init__(self, config: LLM_Config):
        super().__init__()
        self.RMS_norm1 = RMS_norm(config=config, eps=1e-5)
        self.RMS_norm2 = RMS_norm(config=config, eps=1e-5)
        self.MHA = MultiHeadAttention(config=config)
        self.SwiGLU = SwiGLU(config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA(self.RMS_norm1(x))
        x = x + self.SwiGLU(self.RMS_norm2(x))
        return x


class LLM(nn.Module):
    def __init__(self, config: LLM_Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.embedding = Embeddings(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.num_layers)])
        self.RMS_final = RMS_norm(config=config, eps=1e-5)
        self.Linear_final = Linear_Module(in_features=self.d_model, out_features=self.vocab_size, bias=False)

    def forward(self, input_idx: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_idx)
        for block in self.blocks:
            x = block(x)
        x = self.RMS_final(x)
        x = self.Linear_final(x)
        return x
