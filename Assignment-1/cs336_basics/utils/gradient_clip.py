import torch
from typing import Iterable

class GradientClipper:

    def __init__(self, params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
        self.params = list(params)
        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def __call__(self):
        grads = [p.grad for p in self.params if p.grad is not None]
        all_grads = torch.cat([grad.flatten() for grad in grads])
        all_grads_norm = torch.norm(all_grads, p=2)
        if all_grads_norm > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (all_grads_norm + self.eps)
            for grad in grads:
                grad.mul_(clip_coeff)

