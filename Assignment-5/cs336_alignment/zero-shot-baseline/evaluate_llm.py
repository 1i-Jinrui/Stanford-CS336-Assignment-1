import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vllm import LLM, SamplingParams
from typing import Callable, List
import json
from collections import Counter, defaultdict
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


PROMPTS_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
    output_file: str,
    summary_file: str = "summary.json",
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    # 1. 批量推理
    # outputs 是 RequestOutput 对象列表
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []

    # 统计整体 metrics
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    # 统计三类情况：
    # 1. format_reward = 1, answer_reward = 1
    # 2. format_reward = 1, answer_reward = 0
    # 3. format_reward = 0, answer_reward = 0
    category_counter = Counter()
    category_examples = defaultdict(list)

    # 2. 遍历结果并评分
    for i, output_obj in enumerate(outputs):
        generated_text = output_obj.outputs[0].text
        ground_truth = ground_truths[i]

        # 调用提供的评分函数解析答案并打分
        scores = reward_fn(generated_text, ground_truth)

        format_reward = float(scores.get("format_reward", 0.0))
        answer_reward = float(scores.get("answer_reward", 0.0))
        reward = float(scores.get("reward", 0.0))

        total_format_reward += format_reward
        total_answer_reward += answer_reward
        total_reward += reward

        if format_reward == 1.0 and answer_reward == 1.0:
            category = "format_1_answer_1"
        elif format_reward == 1.0 and answer_reward == 0.0:
            category = "format_1_answer_0"
        elif format_reward == 0.0 and answer_reward == 0.0:
            category = "format_0_answer_0"
        else:
            category = "other"

        category_counter[category] += 1

        result_entry = {
            "index": i,
            "prompt": prompts[i],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "scores": scores,
            "category": category,
        }
        results.append(result_entry)

        # 每类保存前 10 个例子
        if len(category_examples[category]) < 10:
            category_examples[category].append(result_entry)

    # 3. 保存逐条结果到磁盘
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 4. 计算并保存整体 summary
    num_examples = len(results)

    summary = {
        "num_examples": num_examples,
        "metrics": {
            "average_reward": total_reward / num_examples if num_examples > 0 else 0.0,
            "format_accuracy": total_format_reward / num_examples if num_examples > 0 else 0.0,
            "answer_accuracy": total_answer_reward / num_examples if num_examples > 0 else 0.0,
        },
        "category_counts": {
            "format_1_answer_1": category_counter["format_1_answer_1"],
            "format_1_answer_0": category_counter["format_1_answer_0"],
            "format_0_answer_0": category_counter["format_0_answer_0"],
            "other": category_counter["other"],
        },
        "category_examples_first_10": dict(category_examples),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 5. 在终端打印 summary，方便直接查看
    print("Evaluation finished.")
    print(f"Saved detailed results to: {output_file}")
    print(f"Saved summary to: {summary_file}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # HF repo id（会触发在线下载）
    # MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"

    # 使用本地模型目录
    MODEL_PATH = str(
    (Path(__file__).parent / "models" / "Qwen2.5-Math-1.5B").resolve()
)

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],  # 遇到结束标签即停止
        include_stop_str_in_output=True,
    )

    validation_file = project_root / "MATH" / "val.jsonl"

    with open(validation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取出问题和答案

    prompts = [
    PROMPTS_TEMPLATE.format(question=str(item["problem"]))
    for item in data
]

    ground_truths = [
        "" if item.get("expected_answer") is None
        else str(item["expected_answer"])
        for item in data
    ]
    llm = LLM(model=MODEL_PATH)

    evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        sampling_params,
        ground_truths,
        "results.jsonl",
        "summary.json",
    )