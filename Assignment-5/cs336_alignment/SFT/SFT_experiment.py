import os

# 关闭 vLLM V1 多进程模式。
# 在 vLLM 0.11.0 等版本中，部分环境下可能出现：
# 'LLMEngine' object has no attribute 'model_executor'
# 因此这里显式关闭 V1 multiprocessing。
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import sys
import gc
import json
import math
import time
import random
import inspect
import argparse
from pathlib import Path
from typing import Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams



# 当前脚本所在目录
current_dir = Path(__file__).parent

# 项目根目录
project_root = current_dir.parent

# 将当前目录和项目根目录加入 Python 搜索路径，
# 方便导入同目录或项目根目录下的自定义模块。
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))


# 奖励函数：用于判断模型输出是否符合格式，以及答案是否正确
from drgrpo_grader import r1_zero_reward_fn


from tokenize_prompt_and_output import (
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
)

from get_response_log_probs import (
    run_get_response_log_probs as get_response_log_probs,
)


from sft_microbatch_train_step import (
    run_sft_microbatch_train_step as sft_microbatch_train_step,
)


# 训练和评估统一 prompt 模板。
PROMPTS_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def load_json_data(path: str | Path) -> list[dict]:
    """
    同时支持两种格式：

    1. JSON 数组：
       [
         {...},
         {...}
       ]

    2. JSONL：
       {...}
       {...}
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Data file is empty: {path}")

    # JSON 数组
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON array in {path}: {exc}"
            ) from exc

        if not isinstance(data, list):
            raise TypeError(
                f"The root object in {path} must be a list."
            )

    # JSONL
    else:
        data = []

        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {exc}"
                ) from exc

            data.append(item)

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Sample {index} in {path} must be a JSON object, "
                f"but got {type(item).__name__}."
            )

    return data


def build_sft_data(train_file: str | Path) -> list[dict]:
    """
    根据当前训练集格式构造 reasoning SFT 数据。

    原始字段：
        problem
        reasoning_trace
        extracted_answer
        expected_answer

    训练字段：
        prompt
        response
        expected_answer
        extracted_answer
    """
    raw = load_json_data(train_file)
    data = []

    required_fields = {
        "problem",
        "reasoning_trace",
        "expected_answer",
    }

    for index, item in enumerate(raw):
        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise KeyError(
                f"Training sample {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        problem = str(item["problem"])
        response = str(item["reasoning_trace"])
        response = response.replace(r"\end{think>", "</think>")

        if not response.strip():
            raise ValueError(
                f"Training sample {index} has an empty reasoning_trace."
            )

        # prompt 已经以 <think> 结尾。
        # 如果某些数据的 reasoning_trace 又以 <think> 开头，
        # 则删除重复的开头标签。
        response_without_leading_space = response.lstrip()

        if response_without_leading_space.startswith("<think>"):
            response = response_without_leading_space[len("<think>"):]

        # 检查 SFT 目标是否包含完整格式
        required_response_tags = [
            "</think>",
            "<answer>",
            "</answer>",
        ]

        missing_tags = [
            tag for tag in required_response_tags
            if tag not in response
        ]

        if missing_tags:
            raise ValueError(
                f"Training sample {index} reasoning_trace is missing tags: "
                f"{missing_tags}\n"
                f"reasoning_trace={response[:300]!r}"
            )

        prompt = PROMPTS_TEMPLATE.format(question=problem)

        data.append(
            {
                "prompt": prompt,
                "response": response,
                "expected_answer": str(item["expected_answer"]),
                "extracted_answer": item.get("extracted_answer"),
            }
        )

    return data


def build_val_data(
    val_file: str | Path,
    max_examples: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    根据当前验证集字段构造验证数据。

    验证集字段：
        problem
        expected_answer
    """
    raw = load_json_data(val_file)

    if max_examples is not None:
        raw = raw[:max_examples]

    prompts = []
    ground_truths = []

    for index, item in enumerate(raw):
        required_fields = {
            "problem",
            "expected_answer",
        }

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise KeyError(
                f"Validation sample {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        prompts.append(
            PROMPTS_TEMPLATE.format(
                question=str(item["problem"])
            )
        )

        ground_truths.append(
            str(item["expected_answer"])
        )

    return prompts, ground_truths

class SFTDataLoaderLite:
    """
    一个轻量级 SFT 数据加载器。

    特点：
    1. 不依赖 torch DataLoader。
    2. 每次返回 micro_batch_size 条数据。
    3. 一个 epoch 结束后会重新 shuffle。
    """

    def __init__(
        self,
        data: list[dict],
        micro_batch_size: int,
        seed: int,
    ):
        self.data = data
        self.micro_batch_size = micro_batch_size

        # 使用独立 random.Random，避免影响全局随机状态
        self.rng = random.Random(seed)

        # 样本索引列表
        self.indices = list(range(len(data)))

        # 当前读取指针
        self.ptr = 0

        # 初始化时打乱一次
        self.rng.shuffle(self.indices)

    def get_batch(self) -> list[dict]:
        """
        获取一个 micro-batch。

        如果剩余样本不足一个 micro-batch，
        则重新 shuffle 并从头开始取。
        """
        if self.ptr + self.micro_batch_size > len(self.indices):
            self.rng.shuffle(self.indices)
            self.ptr = 0

        batch_indices = self.indices[self.ptr:self.ptr + self.micro_batch_size]
        self.ptr += self.micro_batch_size

        return [self.data[i] for i in batch_indices]


def init_vllm(
    model_path: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.2,
    max_model_len: int = 2048,
    max_num_seqs: int = 128,
) -> LLM:
    return LLM(
        model=model_path,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
    )


def load_policy_into_vllm_instance(
    policy: PreTrainedModel,
    llm: LLM,
) -> None:
    """
    将当前训练中的 HF policy 模型权重加载到 vLLM 实例中。

    用途：
    每隔若干 step，需要用当前最新 policy 进行验证集生成。
    由于训练模型是 HF 模型，而评估生成使用 vLLM，
    因此需要把 HF 权重同步给 vLLM。
    """
    # 如果模型经过 torch.compile 包装，原始模型通常在 _orig_mod 中
    if hasattr(policy, "_orig_mod"):
        policy = policy._orig_mod

    state_dict = policy.state_dict()

    # 访问 vLLM 内部模型对象，并加载 HF 模型权重
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_vllm_in_memory(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
) -> dict:
    """
    使用 vLLM 对验证集 prompt 进行生成，并用 reward_fn 评估输出。

    返回：
        results:
            每条样本的 prompt、标准答案、模型输出、奖励分数。
        accuracy:
            平均 reward、格式准确率、答案准确率。
    """
    # 使用 vLLM 批量生成
    outputs = vllm_model.generate(prompts, sampling_params)

    results = []

    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = ground_truths[i]

        # reward_fn 返回通常包含：
        # reward: 总奖励
        # format_reward: 格式奖励
        # answer_reward: 答案正确性奖励
        scores = reward_fn(generated_text, gt)

        reward = float(scores.get("reward", 0.0))
        format_reward = float(scores.get("format_reward", 0.0))
        answer_reward = float(scores.get("answer_reward", 0.0))

        total_reward += reward
        total_format_reward += format_reward
        total_answer_reward += answer_reward

        results.append(
            {
                "index": i,
                "prompt": prompts[i],
                "expected_answer": gt,
                "output": generated_text,
                "reward": scores,
            }
        )

    n = len(results)

    return {
        "results": results,
        "accuracy": {
            "avg_reward": total_reward / n if n else 0.0,
            "avg_format_acc": total_format_reward / n if n else 0.0,
            "avg_acc": total_answer_reward / n if n else 0.0,
        },
    }


@torch.no_grad()
def evaluate_val_loss_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    val_prompts: list[str],
    eval_results: list[dict],
    val_batch_size: int,
    device: str,
    device_type: str,
    dtype: torch.dtype,
) -> dict:
    """
    计算验证集上的额外指标。

    这里不是用标准答案 response 来算验证 loss，
    而是对 vLLM 已经生成出来的 response 计算：
    1. eval_loss：当前 HF 模型对 vLLM 输出的负 log probability
    2. eval_avg_token_entropy：生成 token 的平均熵
    3. eval_avg_correct_response_length：答对样本的平均 response token 长度
    4. eval_avg_incorrect_response_length：答错样本的平均 response token 长度

    注意：
    该函数只评估，不反向传播。
    """
    model.eval()

    loss_accum = 0.0
    entropy_accum = 0.0

    correct_response_tokens = 0.0
    incorrect_response_tokens = 0.0
    num_correct = 0
    num_incorrect = 0
    num_batches = 0

    for i in range(0, len(val_prompts), val_batch_size):
        batch_prompts = val_prompts[i:i + val_batch_size]
        batch_results = eval_results[i:i + val_batch_size]

        # 取出 vLLM 生成的输出
        batch_outputs = [res["output"] for res in batch_results]

        # 根据 answer_reward 判断该条样本是否答对
        batch_correct = [
            float(res["reward"].get("answer_reward", 0.0)) == 1.0
            for res in batch_results
        ]

        # 将 prompt 和模型输出拼接 tokenize，
        # response_mask 用来标记哪些 token 属于 response 部分。
        tokenized = tokenize_prompt_and_output(
            prompt_strs=batch_prompts,
            output_strs=batch_outputs,
            tokenizer=tokenizer,
        )

        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        with torch.autocast(device_type=device_type, dtype=dtype):
            out = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )

            log_probs = out["log_probs"]
            token_entropy = out["token_entropy"]

            # 只在 response token 上计算 loss
            loss = -((log_probs * response_mask).sum() / response_mask.sum().clamp(min=1))

        loss_accum += loss.detach().float().item()

        # 累计 response token 的 entropy
        entropy_accum += (token_entropy * response_mask).sum().detach().float().item()

        # 分别统计答对和答错样本的 response 长度
        for j, flag in enumerate(batch_correct):
            cur_len = response_mask[j].sum().detach().float().item()

            if flag:
                correct_response_tokens += cur_len
                num_correct += 1
            else:
                incorrect_response_tokens += cur_len
                num_incorrect += 1

        num_batches += 1

    total_response_tokens = correct_response_tokens + incorrect_response_tokens

    # 恢复训练模式
    model.train()

    return {
        "eval_loss": loss_accum / max(num_batches, 1),
        "eval_avg_token_entropy": entropy_accum / max(total_response_tokens, 1.0),
        "eval_avg_correct_response_length": correct_response_tokens / max(num_correct, 1),
        "eval_avg_incorrect_response_length": incorrect_response_tokens / max(num_incorrect, 1),
    }


def save_eval_examples(
    output_dir: Path,
    step: int,
    eval_results: dict,
    max_examples: int,
) -> None:
    """
    保存部分验证集生成样例，方便人工查看模型输出。

    保存路径：
        output_dir/eval_examples/eval_examples_step_xxx.jsonl
    """
    examples_dir = output_dir / "eval_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    examples = eval_results["results"][:max_examples]

    with open(examples_dir / f"eval_examples_step_{step}.jsonl", "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    # =========================
    # 路径相关参数
    # =========================

    parser.add_argument(
        "--model_path",
        type=str,
        default=str((project_root / "models" / "Qwen2.5-Math-1.5B").resolve()),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str((project_root / "MATH" / "train.jsonl").resolve()),
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=str((project_root / "MATH" / "val.jsonl").resolve()),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str((project_root / "outputs" / "sft_single_cuda").resolve()),
    )

    # =========================
    # 基础训练配置
    # =========================

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda:0")

    # dtype：
    # bfloat16 更稳定，但部分 GPU 不支持；
    # 如果不支持会自动回退到 float16。
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")

    # attention 实现方式，默认使用 flash_attention_2
    parser.add_argument("--attention_type", type=str, default="flash_attention_2")

    # 是否使用 torch.compile 加速模型
    parser.add_argument("--use_compile", action="store_true")

    # =========================
    # batch 与优化器参数
    # =========================

    # total_batch_size 是逻辑总 batch size。
    # 实际每次 forward/backward 使用 micro_batch_size，
    # 通过梯度累积达到 total_batch_size。
    parser.add_argument("--total_batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=2)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument("--per_token_loss", action="store_true", default=True)
    # =========================
    # 训练步数配置
    # =========================

    parser.add_argument("--num_epochs", type=int, default=1)

    # 如果设置 max_steps，则优先使用 max_steps；
    # 否则根据数据量和 epoch 数自动计算。
    parser.add_argument("--max_steps", type=int, default=None)

    # =========================
    # 评估和保存配置
    # =========================

    parser.add_argument("--eval_interval", type=int, default=8)
    parser.add_argument("--checkpoint_interval", type=int, default=8)

    # 验证集最多使用多少条样本，防止验证太慢
    parser.add_argument("--max_val_examples", type=int, default=1000)

    # 计算验证 loss/entropy 时的 batch size
    parser.add_argument("--val_batch_size", type=int, default=4)

    # 每次评估保存多少条生成样例
    parser.add_argument("--num_eval_examples_to_save", type=int, default=20)

    # =========================
    # vLLM 推理配置
    # =========================

    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.2)
    parser.add_argument("--vllm_max_model_len", type=int, default=2048)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=128)

    args = parser.parse_args()

    # =========================
    # 环境检查与 dtype 设置
    # =========================

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # 如果当前 GPU 不支持 bfloat16，则回退到 float16
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("[warning] bfloat16 not supported, falling back to float16")
        args.dtype = "float16"

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # =========================
    # 随机种子设置
    # =========================

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 允许 TF32 matmul，提高 Ampere 及之后 GPU 上的矩阵乘效率
    torch.set_float32_matmul_precision("high")

    # =========================
    # 输出目录与日志文件
    # =========================

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.jsonl"

    # =========================
    # 数据加载
    # =========================

    train_data = build_sft_data(args.train_file)

    val_prompts, val_ground_truths = build_val_data(
        args.val_file,
        max_examples=args.max_val_examples,
    )

    # total_batch_size 必须能被 micro_batch_size 整除，
    # 否则无法得到整数个梯度累积步数。
    if args.total_batch_size % args.micro_batch_size != 0:
        raise ValueError("total_batch_size must be divisible by micro_batch_size.")

    # 每个 optimizer.step() 前需要累积多少个 micro-batch
    grad_acc_steps = args.total_batch_size // args.micro_batch_size

    # 每个 epoch 的优化器更新步数
    steps_per_epoch = math.ceil(len(train_data) / args.total_batch_size)

    # 确定总训练步数
    if args.max_steps is None:
        max_steps = steps_per_epoch * args.num_epochs
    else:
        max_steps = args.max_steps

    print("=" * 80)
    print(f"train size: {len(train_data)}")
    print(f"val size: {len(val_prompts)}")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"total_batch_size: {args.total_batch_size}")
    print(f"micro_batch_size: {args.micro_batch_size}")
    print(f"grad_acc_steps: {grad_acc_steps}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"max_steps: {max_steps}")
    print(f"eval_interval: {args.eval_interval}")
    print("=" * 80)

    # =========================
    # vLLM 生成参数
    # =========================

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,

        # 遇到 </answer> 停止生成
        stop=["</answer>"],

        # 将停止字符串也保留在输出中，方便格式奖励函数判断
        include_stop_str_in_output=True,
    )

    # =========================
    # 初始化 vLLM 推理模型
    # =========================

    print("[vLLM] initializing on single CUDA")
    vllm_model = init_vllm(
        model_path=args.model_path,
        device=device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
    )

    # =========================
    # 加载 tokenizer 和 HF 训练模型
    # =========================

    print("[HF] loading tokenizer and policy model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 如果 tokenizer 没有 pad token，则使用 eos token 作为 pad token。
    # 这对 batch padding 是必要的。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attention_type,
    ).to(device)

    # 可选：使用 torch.compile 编译模型
    if args.use_compile:
        model = torch.compile(model)

    model.train()

    # =========================
    # 优化器设置
    # =========================

    # 检查当前 PyTorch AdamW 是否支持 fused 参数
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters

    # CUDA 下优先使用 fused AdamW
    use_fused = fused_available and device_type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=use_fused,
    )

    # 训练数据加载器
    train_loader = SFTDataLoaderLite(
        data=train_data,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
    )

    # 记录最佳答案准确率
    best_acc = -1.0
    best_checkpoint = None

    # =========================
    # 主训练循环
    # =========================

    for step in range(1, max_steps + 1):
        t0 = time.time()

        # 每个 step 开始前清空梯度
        optimizer.zero_grad(set_to_none=True)

        # =========================
        # 周期性评估
        # =========================
        # 在 step=1、每隔 eval_interval、最后一步时进行评估。
        if step == 1 or step % args.eval_interval == 0 or step == max_steps:
            print(f"[eval] loading current policy into vLLM at step {step}")

            # 将当前 HF 训练模型权重同步到 vLLM
            load_policy_into_vllm_instance(model, vllm_model)

            eval_t0 = time.time()

            # 使用 vLLM 生成验证集答案，并计算 reward / accuracy
            eval_results = evaluate_vllm_in_memory(
                vllm_model=vllm_model,
                reward_fn=r1_zero_reward_fn,
                prompts=val_prompts,
                ground_truths=val_ground_truths,
                sampling_params=sampling_params,
            )

            torch.cuda.empty_cache()

            # 用 HF 模型进一步计算验证集 loss、entropy、response 长度等指标
            eval_extra = evaluate_val_loss_entropy(
                model=model,
                tokenizer=tokenizer,
                val_prompts=val_prompts,
                eval_results=eval_results["results"],
                val_batch_size=args.val_batch_size,
                device=device,
                device_type=device_type,
                dtype=dtype,
            )

            # 保存部分验证样例，方便后续人工检查
            save_eval_examples(
                output_dir=output_dir,
                step=step,
                eval_results=eval_results,
                max_examples=args.num_eval_examples_to_save,
            )

            eval_log = {
                "type": "eval",
                "step": step,
                "avg_reward": eval_results["accuracy"]["avg_reward"],
                "format_accuracy": eval_results["accuracy"]["avg_format_acc"],
                "answer_accuracy": eval_results["accuracy"]["avg_acc"],
                **eval_extra,
                "dt": time.time() - eval_t0,
            }

            # 将评估日志写入 metrics.jsonl
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_log, ensure_ascii=False) + "\n")

            print(
                f"[eval] step={step} "
                f"loss={eval_log['eval_loss']:.4f} "
                f"entropy={eval_log['eval_avg_token_entropy']:.4f} "
                f"format_acc={eval_log['format_accuracy']:.4f} "
                f"answer_acc={eval_log['answer_accuracy']:.4f} "
                f"dt={eval_log['dt']:.2f}s"
            )

            # 如果当前答案准确率更高，则记录最佳 checkpoint 路径
            # 注意：这里只记录路径，真正保存发生在后面的 checkpoint 逻辑中。
            if eval_log["answer_accuracy"] > best_acc:
                best_acc = eval_log["answer_accuracy"]
                best_checkpoint = output_dir / f"checkpoint_best_step_{step}"

            torch.cuda.empty_cache()

        # =========================
        # 训练一个 optimizer step
        # =========================

        loss_accum = 0.0
        entropy_accum = 0.0
        total_response_tokens = 0.0

        # 梯度累积：
        # 每个 micro_step 处理 micro_batch_size 条样本，
        # 累积 grad_acc_steps 次后再执行 optimizer.step()。
        for micro_step in range(grad_acc_steps):
            batch = train_loader.get_batch()

            # 构造 tokenized batch
            tokenized = tokenize_prompt_and_output(
                prompt_strs=[item["prompt"] for item in batch],
                output_strs=[item["response"] for item in batch],
                tokenizer=tokenizer,
            )

            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)

            # autocast 混合精度训练
            with torch.autocast(device_type=device_type, dtype=dtype):
                out = get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True,
                )

                # 计算 SFT loss，并在函数内部完成 backward
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=out["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=grad_acc_steps,
                    normalize_constant=1.0,
                    per_token_loss=args.per_token_loss,
                )

            loss_accum += loss.detach().float().item()

            # 累计 response token 的 entropy，用于日志记录
            entropy_accum += (out["token_entropy"] * response_mask).sum().detach().float().item()

            # 统计 response token 总数
            total_response_tokens += response_mask.sum().detach().float().item()

        # 梯度裁剪，防止梯度爆炸
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_norm_clip,
        )

        # 参数更新
        optimizer.step()

        avg_entropy = entropy_accum / max(total_response_tokens, 1.0)
        dt = time.time() - t0

        train_log = {
            "type": "train",
            "step": step,
            "loss": loss_accum,
            "avg_token_entropy": avg_entropy,
            "learning_rate": args.learning_rate,
            "grad_norm": float(norm.detach().float().cpu()),
            "dt": dt,
        }

        # 写入训练日志
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(train_log, ensure_ascii=False) + "\n")

        print(
            f"[train] step={step} "
            f"loss={loss_accum:.4f} "
            f"entropy={avg_entropy:.4f} "
            f"norm={train_log['grad_norm']:.4f} "
            f"dt={dt:.2f}s"
        )

        # =========================
        # 周期性保存 checkpoint
        # =========================

        if step % args.checkpoint_interval == 0 or step == max_steps:
            checkpoint_dir = output_dir / f"checkpoint_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 如果模型被 torch.compile 包装，则保存原始模型
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model

            save_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            print(f"[checkpoint] saved to {checkpoint_dir}")

        # =========================
        # 保存最佳 checkpoint
        # =========================

        if best_checkpoint is not None and step % args.eval_interval == 0:
            best_checkpoint.mkdir(parents=True, exist_ok=True)

            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model

            save_model.save_pretrained(best_checkpoint)
            tokenizer.save_pretrained(best_checkpoint)

            print(f"[best] saved to {best_checkpoint}")

            # 保存后清空，避免重复保存同一个 best checkpoint
            best_checkpoint = None

        # 定期进行 Python 垃圾回收和 CUDA 显存清理
        if step % args.eval_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # =========================
    # 保存最终模型
    # =========================

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 保存训练总结信息
    summary = {
        "train_size": len(train_data),
        "val_size": len(val_prompts),
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "total_batch_size": args.total_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "grad_acc_steps": grad_acc_steps,
        "best_answer_accuracy": best_acc,
        "final_model_dir": str(final_dir),
        "metrics_file": str(metrics_file),
    }

    with open(output_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[finished]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()