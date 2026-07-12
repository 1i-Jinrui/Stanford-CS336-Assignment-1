'''
input_ids
   ↓
输入模型 model
   ↓
得到 logits
(batch_size, sequence_length, vocab_size)
   ↓
log_softmax
   ↓
得到整个词表上的 log probability
   ↓
根据 labels
gather 出真实 token 对应的 log probability
   ↓
得到每个 token 的 log_probs
(batch_size, sequence_length)
   ↓
如果 return_token_entropy=True
   ↓
计算每个 token 位置的 entropy
   ↓
返回：
log_probs
token_entropy
'''

import torch
import torch.nn.functional as F
from compute_entropy import run_compute_entropy
def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    outputs = model(input_ids, return_dict=True)
    logits = outputs.logits # (batch_size, sequence_length, vocab_size)

    probs = F.log_softmax(logits, dim=-1)
    labels = labels.unsqueeze(-1) # (batch_size, sequence_length, 1)
    log_probs = torch.gather(probs, dim=-1, index=labels) # (batch_size, sequence_length)
    log_probs = log_probs.squeeze(-1) # (batch_size, sequence_length)
    if return_token_entropy:
        entropy = run_compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}
    else:
        return {"log_probs": log_probs}