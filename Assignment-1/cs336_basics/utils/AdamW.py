import torch
from torch import optim
from typing import Optional, Callable


class AdamW(optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
        defaults: dict = dict[str, float | tuple[float, float]](lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):

        with torch.enable_grad():
             loss = None if closure is None else closure()
        with torch.no_grad():
            '''
            param_groups是一个list,例如:
            [
                {
                    "params": [...],
                    "lr": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01
                },
                {
                    "params": [...],
                    "lr": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0
                }
            ]
            '''
            for group in self.param_groups:

                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p)
                        state['v'] = torch.zeros_like(p)
                    
                    state['step'] += 1
                    m = state['m']
                    v = state['v']
                    step = state['step']

                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    # 计算一阶矩和二阶矩
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # 计算偏差修正
                    bias_correction1 = 1 - beta1 ** step 
                    bias_correction2 = 1 - beta2 ** step 

                    # 等价于使用 m_hat = m / bias_correction1
                    step_size = lr / bias_correction1

                    # denom = sqrt(v_hat) + eps
                    denom = (v / bias_correction2).sqrt().add_(eps)

                    # 参数更新
                    # p = p - step_size * m / denom
                    p.addcdiv_(m, denom, value=-step_size)

        return loss
