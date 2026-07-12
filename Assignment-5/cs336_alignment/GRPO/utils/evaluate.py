import random
from typing import Callable, List

from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable,
    val_dataset: List[dict],
    prompt_template: str,
    sampling_params: SamplingParams,
    max_val_examples: int,
) -> dict:
    """
    使用 vLLM 在 val_dataset 的随机子集上评估模型。

    参数：
        vllm_model: 用于生成结果的 vLLM 模型。
        reward_fn: 奖励函数，接收 (response, ground_truth)，并返回一个字典，
                   字典中包含 'reward'、'format_reward' 和 'answer_reward' 等键。
        val_dataset: 字典列表，每个字典包含 'problem' 和 'answer' 两个键。
        prompt_template: 提示词模板字符串，其中包含 '{question}' 占位符。
        sampling_params: 用于生成的 vLLM SamplingParams 参数。
        max_val_examples: 最大评估样本数量。

    返回：
        一个字典，包含以下键：
            'mean_reward': float，平均总奖励。
            'mean_format_reward': float，平均格式奖励。
            'mean_answer_reward': float，平均答案奖励。
            'n_examples': int，实际评估样本数量。
    """
    if max_val_examples is None:
        n = len(val_dataset)
    else:
        n = min(max_val_examples, len(val_dataset))
        
    subset = random.sample(val_dataset, n)

    prompts = [prompt_template.replace("{question}", ex['problem']) for ex in subset]
    ground_truths = [ex['answer'] for ex in subset]

    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    rewards = [reward_fn(response, gt) for response, gt in zip(responses, ground_truths)]

    mean_reward = sum(r['reward'] for r in rewards) / n
    mean_format_reward = sum(r['format_reward'] for r in rewards) / n
    mean_answer_reward = sum(r['answer_reward'] for r in rewards) / n

    return {
        'mean_reward': mean_reward,
        'mean_format_reward': mean_format_reward,
        'mean_answer_reward': mean_answer_reward,
        'n_examples': n,
    }