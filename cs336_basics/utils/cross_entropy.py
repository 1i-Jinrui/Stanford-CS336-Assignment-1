import torch


class CrossEntropyLoss:
    def __init__(self, inputs:torch.Tensor, targets:torch.Tensor):
        self.inputs = inputs # (batch_size, vocab_size)
        self.targets = targets.long() # (batch_size,) 索引必须是整数类型，通常是 torch.long。
        self.vocab_size = self.inputs.shape[1]
        self.batch_size = self.inputs.shape[0]
    def forward(self) -> torch.Tensor:
        logits = self.inputs
        max_logits = torch.max(logits, dim=-1, keepdim=True).values # torch.max返回的是一个元组，包含最大值和最大值的索引
        # 防止exp(logits)溢出
        shifted_logits = logits - max_logits
        # 稳定计算 log(sum(exp(logits)))    
        log_sum_exp = torch.log(
            torch.sum(torch.exp(shifted_logits), dim=-1)
        ) + max_logits.squeeze(dim=-1)
        batch_indices = torch.arange(self.batch_size, device=self.inputs.device)
        # 取出每个样本真实类别的 logit
        correct_logits = logits[batch_indices, self.targets]
        loss = log_sum_exp - correct_logits
        return loss.mean()




