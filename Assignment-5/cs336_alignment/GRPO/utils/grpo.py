from typing import Callable, List, Tuple

import torch
from einops import rearrange
from torch.nn import functional as F
from transformers import PreTrainedModel
from typing_extensions import Literal


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:

    """
    计算一批 rollouts 的组内归一化优势值。

    参数：
        reward_fn: 奖励函数，接收 response 和 ground truth，返回一个包含奖励值的字典。
        rollout_responses: 模型生成的 responses 列表，大小为 n_prompts_per_rollout_batch * group_size。
        repeated_ground_truths: 重复了 `group_size` 次的 ground truths 列表，大小与 rollout_responses 相同。
        group_size: 每个组的大小。
        advantage_eps: 加到分母上的 epsilon，用于数值稳定性。
        normalize_by_std: 是否使用组内标准差对优势值进行归一化。

    返回：
        一个元组，包含 (advantages, raw_rewards, stats)。
    """

    # 每个 rollout 对应一个包含 format_reward、answer_reward 和 reward 的字典
    rewards = [reward_fn(response, ground_truth) for response, ground_truth in zip(rollout_responses, repeated_ground_truths)]

    # tensor -> 从 n_grp * group_size 重塑为 (n_grp, group_size)
    raw_rewards = torch.tensor([r['reward'] for r in rewards])
    grp_rewards = rearrange(raw_rewards, '(n_grp group_size) -> n_grp group_size', group_size=group_size)

    # 计算均值和标准差，形状为 (n_grp,)
    mean = grp_rewards.mean(dim=-1)
    std = grp_rewards.std(dim=-1)

    # 计算优势值
    grp_advantages = grp_rewards - mean.unsqueeze(-1)
    if normalize_by_std:
        grp_advantages = grp_advantages / (std.unsqueeze(-1) + advantage_eps)

    # 从 (n_grp, group_size) 重塑回 n_grp * group_size
    advantages = rearrange(grp_advantages, 'n_grp group_size -> (n_grp group_size)', group_size=group_size)

    return advantages, raw_rewards, {'mean': mean, 'std': std, 'rewards': rewards}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor) -> torch.Tensor:
    """
    计算朴素策略梯度损失。

    参数：
        raw_rewards_or_advantages: 每个 rollout 的标量奖励或优势值，形状为 (bs, 1)。
        policy_log_probs: rollout 中各 token 的 log 概率，形状为 (bs, seq_len)。

    返回：
        每个 token 的策略梯度损失，形状为 (bs, seq_len)。
    """
    # 形状：(bs, 1) * (bs, seq_len) -> (bs, seq_len)
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    clip: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GRPO clip 损失。

    参数：
        advantages: 优势值张量，形状为 (bs, 1)。
        policy_log_probs: 当前策略的 log 概率张量，形状为 (bs, seq_len)。
        old_log_probs: 旧策略的 log 概率张量，形状为 (bs, seq_len)。
        cliprange: clip 比例范围。
        clip: 是否对 ratio 进行裁剪。

    返回：
        每个 token 的 GRPO clipped loss，形状为 (bs, seq_len)。
        以及相关元数据。
    """
    ratios = torch.exp(policy_log_probs - old_log_probs)
    scores = ratios * advantages
    mean_ratio = ratios.mean()

    if not clip:
        return -scores, {"clip_fraction": 0.0, "mean_ratio": mean_ratio}

    # 对 ratios 进行裁剪
    clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    clipped_scores = clipped_ratios * advantages

    # 计算被裁剪 token 的比例
    clip_fraction = (~torch.isclose(scores, clipped_scores)).float().mean()

    return -torch.min(scores, clipped_scores), {
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
    }


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    clip: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算策略梯度损失。

    参数：
        policy_log_probs: 当前策略的 log 概率张量，形状为 (bs, seq_len)。
        loss_type: 要计算的损失类型。
        raw_rewards: 原始奖励张量，形状为 (bs, 1)。
        advantages: 优势值张量，形状为 (bs, 1)。
        old_log_probs: 旧策略的 log 概率张量，形状为 (bs, seq_len)。
        cliprange: clip 比例范围。
        clip: 是否对 ratio 进行裁剪。

    返回：
        每个 token 的策略梯度损失，形状为 (bs, seq_len)。
        以及相关元数据。
    """
    
    # 检查 loss_type 是否合法
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"], f"Invalid loss type: {loss_type}"

    if loss_type == "no_baseline":
        assert raw_rewards is not None, f"raw_rewards is required for {loss_type} loss type"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    
    assert advantages is not None, f"advantages is required for {loss_type} loss type"
    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    
    assert old_log_probs is not None, f"old_log_probs is required for {loss_type} loss type"
    if loss_type == "grpo_no_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange, clip=False)
    
    assert cliprange is not None, f"cliprange is required for {loss_type} loss type"
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange, clip)


def masked_mean(
    tensor: torch.Tensor, 
    mask: torch.Tensor,
    dim: int | None = None
    ) -> torch.Tensor:
    """
    沿指定维度计算 tensor 的平均值，只考虑 mask == 1 的元素。

    参数：
        tensor: 要计算平均值的张量。
        mask: 掩码，只考虑其中值为 1 的元素。
        dim: 要计算平均值的维度。

    返回：
        沿指定维度、仅基于 mask == 1 的元素计算得到的平均值。
    """
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,) -> torch.Tensor:
    """
    沿指定维度求和，并用常数进行归一化，只考虑 mask 值为 1 的元素。
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    clip: bool = True,
    norm_mode: Literal["mean", "constant", "microbatch"] = "mean",
    norm_constant: float | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GRPO microbatch 训练步骤。

    参数：
        policy_log_probs: 当前策略的 log 概率张量，形状为 (bs, seq_len)。
        response_mask: response 掩码张量，形状为 (bs, seq_len)。
        gradient_accumulation_steps: 梯度累积步数。
        loss_type: 要计算的损失类型。
        raw_rewards: 原始奖励张量，形状为 (bs, 1)。
        advantages: 优势值张量，形状为 (bs, 1)。
        old_log_probs: 旧策略的 log 概率张量，形状为 (bs, seq_len)。
        cliprange: clip 比例范围。
        clip: 是否对 ratio 进行裁剪。
        norm_mode: 归一化模式。
        norm_constant: 用于归一化的常数。
    """

    # 计算策略梯度损失
    policy_gradient_loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange, clip)

    # 归一化
    if norm_mode == "mean":
        policy_gradient_loss = masked_mean(policy_gradient_loss, response_mask, dim=-1)
    elif norm_mode == "constant":
        assert norm_constant is not None, f"norm_constant is required for {norm_mode} norm mode"
        policy_gradient_loss = masked_normalize(policy_gradient_loss, response_mask, dim=-1, normalize_constant=norm_constant)
    elif norm_mode == "microbatch":
        # 使用当前 batch 中最长 response 的长度作为归一化常数
        norm_constant = response_mask.sum(dim=-1).max().item()
        policy_gradient_loss = masked_normalize(policy_gradient_loss, response_mask, dim=-1, normalize_constant=norm_constant)
    else:
        raise ValueError(f"Invalid norm mode: {norm_mode}")
    
    # 梯度累积
    policy_gradient_loss = policy_gradient_loss.mean() / gradient_accumulation_steps

    # 反向传播
    policy_gradient_loss.backward()

    return policy_gradient_loss, metadata


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    """
    获取在给定 prompt 条件下 response 的条件 log 概率，
    并可选地返回 next-token 预测的熵。

    参数：
        model: 用于计算 log 概率的模型。
        input_ids: prompt 的 input ids。
        labels: response 的 labels。
        return_token_entropy: 是否返回 next-token 预测的熵。

    返回：
        一个字典，包含在给定 prompt 条件下 response 的 log 概率，以及 next-token 预测的熵。
    """
    response_dict = {}
    # 从模型中获取 logits
    logits = model(input_ids).logits
    # 获取在给定 prompt 条件下 response 的 log 概率
    log_probs = F.log_softmax(logits, dim=-1) # (batch_size, sequence_length, vocab_size)
    # 获取实际出现 token 对应位置的 log 概率
    # labels: (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
    response_dict["log_probs"] = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, sequence_length)
    
    if return_token_entropy:
        response_dict["token_entropy"] = compute_entropy(logits)
    
    return response_dict


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算 next-token 预测的熵，也就是在词表维度上的熵。
    entropy = -sum(p * log(p))

    参数：
        logits: torch.Tensor，形状为 (batch_size, sequence_length, vocab_size)，包含未归一化的 logits。

    返回：
        torch.Tensor，形状为 (batch_size, sequence_length)，表示 next-token 预测的熵。
    """
    # 使用 logsumexp 可以让计算始终保持在 log 空间中，避免在取 log 前出现溢出或塌缩为 0 的问题。
    # 如果先做 softmax 再取 log，极小的概率可能变成 0，而 log(0) 会变成 -inf，从而破坏熵的计算。
    with torch.no_grad():
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True) # (batch_size, sequence_length, vocab_size)
        return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1) # (batch_size, sequence_length)