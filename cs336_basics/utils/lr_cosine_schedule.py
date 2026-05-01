import math


class Cosine_schedule:
    def __init__(self, max_lr, min_lr, warmup_iters, cosine_cycle_iters) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
    
    def __call__(self, it: int) -> int:
        if it < self.warmup_iters:
            return self.max_lr * it / self.warmup_iters
        elif it > self.cosine_cycle_iters:
            return self.min_lr
        else:
            return self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters))) / 2
        


        