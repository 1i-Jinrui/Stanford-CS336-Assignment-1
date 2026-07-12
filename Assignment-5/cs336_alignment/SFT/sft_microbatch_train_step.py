import torch
from mask_normalize import run_masked_normalize


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
    per_token_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # 每条样本有效 response token 的数量，形状为 [batch_size]
    response_token_counts = response_mask.sum(dim=1).clamp(min=1)

    if per_token_loss:
        # 每条样本分别按照自己的 response token 数量归一化
        normalized_policy_log_probs = (
            (policy_log_probs * response_mask).sum(dim=1)
            / response_token_counts
        )
    else:
        if normalize_constant is None:
            raise ValueError(
                "normalize_constant cannot be None when per_token_loss=False"
            )

        normalized_policy_log_probs = run_masked_normalize(
            policy_log_probs,
            response_mask,
            dim=1,
            normalize_constant=normalize_constant,
        )

    scaled_loss = (
        -normalized_policy_log_probs.mean()
        / gradient_accumulation_steps
    )

    scaled_loss.backward()

    return scaled_loss, {
        "response_token_counts": response_token_counts.detach(),
    }