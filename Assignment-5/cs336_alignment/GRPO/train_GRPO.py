import os
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import inspect
import json
import random
import shutil
import time

import torch

from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEFAULT_CONFIG, DTYPE_MAPPING
from utils.dataset import load_dataset, tokenize_prompt_and_output
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.evaluate import evaluate_vllm
from utils.grpo import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
)
from utils.helper import log_memory, pretty_print, set_seed
from utils.vllm import init_vllm, load_policy_into_vllm_instance
from vllm import SamplingParams


if __name__ == "__main__":

    # -------------------------------------------------------------#
    # 加载配置
    # -------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (overrides defaults)",
    )
    args = parser.parse_args()

    if args.config is not None:
        config = OmegaConf.merge(DEFAULT_CONFIG, OmegaConf.load(args.config))
    else:
        config = DEFAULT_CONFIG

    config_dict = OmegaConf.to_container(config, resolve=True)
    pretty_print(config_dict, title="Config")

    # -------------------------------------------------------------#
    # Local logging
    # -------------------------------------------------------------#
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.jsonl"
    summary_file = output_dir / "final_summary.json"
    best_model_dir = output_dir / "best_model"
    final_model_dir = output_dir / "final_model"

    def append_metrics(record):
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    use_wandb = False
    del config_dict

    # -------------------------------------------------------------#
    # Batch size 检查
    # -------------------------------------------------------------#
    assert (
        config.training.train_batch_size % config.training.gradient_accumulation_steps == 0
    ), "train_batch_size must be divisible by gradient_accumulation_steps"

    micro_train_batch_size = (
        config.training.train_batch_size // config.training.gradient_accumulation_steps
    )
    pretty_print(f"Micro train batch size: {micro_train_batch_size}")

    assert (
        config.training.rollout_batch_size % config.training.group_size == 0
    ), "rollout_batch_size must be divisible by group_size"

    n_prompts_per_rollout_batch = (
        config.training.rollout_batch_size // config.training.group_size
    )
    pretty_print(f"Number of prompts per rollout batch: {n_prompts_per_rollout_batch}")

    assert (
        config.training.train_batch_size >= config.training.group_size
    ), "train_batch_size must be greater than or equal to group_size"

    steps_per_epoch = (
        config.training.rollout_batch_size // config.training.train_batch_size
    )

    # -------------------------------------------------------------#
    # 设置随机种子和精度
    # -------------------------------------------------------------#
    pretty_print(
        f"Setting the seed to {config.training.seed} and using tf32 precision...",
        title="Set Random Seed",
    )
    set_seed(config.training.seed)
    torch.set_float32_matmul_precision("high")

    # -------------------------------------------------------------#
    # 加载训练集和验证集
    # -------------------------------------------------------------#
    pretty_print(None, title="Load datasets")

    pretty_print(f"Loading prompt template from {config.paths.prompt_template_file}...")
    prompt_template = load_dataset(
        data_file=config.paths.prompt_template_file,
        data_type="prompt",
    )
    pretty_print(prompt_template, title="Prompt template", is_sub_title=True)

    pretty_print(f"Loading train dataset from {config.paths.train_data_file}...")
    train_dataset = load_dataset(
        data_file=config.paths.train_data_file,
        data_type="train",
    )
    pretty_print(
        f"Train dataset size: {len(train_dataset)}",
        title="Train dataset",
        is_sub_title=True,
    )
    pretty_print(train_dataset[:5])

    pretty_print(f"Loading val dataset from {config.paths.val_data_file}...")
    val_dataset = load_dataset(
        data_file=config.paths.val_data_file,
        data_type="val",
    )
    pretty_print(
        f"Val dataset size: {len(val_dataset)}",
        title="Val dataset",
        is_sub_title=True,
    )
    pretty_print(val_dataset[:5])

    # -------------------------------------------------------------#
    # 初始化 vLLM 模型
    # -------------------------------------------------------------#
    pretty_print("Initializing the vLLM model...", title="vLLM model initialization")
    vllm_model = init_vllm(config.training.seed, config)

    # -------------------------------------------------------------#
    # 初始化分词器和模型
    # -------------------------------------------------------------#
    pretty_print(None, title="Tokenizer and model initialization")

    pretty_print("Initializing the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.paths.model_path)

    pretty_print("Initializing the model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.paths.model_path,
        dtype=DTYPE_MAPPING[config.training.dtype],
        attn_implementation=config.training.attention_type,
        device_map=config.training.device,
    )

    if config.training.use_gradient_checkpointing:
        pretty_print("Gradient checkpointing enabled...")
        model.gradient_checkpointing_enable()

    if config.training.use_compile:
        print(f"compile flag: {config.training.use_compile}, compiling the model...")
        model = torch.compile(model)
    else:
        print(f"compile flag: {config.training.use_compile}, skipping model compilation...")

    # -------------------------------------------------------------#
    # 设置优化器
    # -------------------------------------------------------------#
    pretty_print("Setup the AdamW optimizer...", title="AdamW optimizer setup")

    if config.training.use_bnb_adamw8bit:
        import bitsandbytes as bnb

        pretty_print("Using bitsandbytes AdamW8bit optimizer.")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps,
        )
    else:
        use_fused = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and config.training.device.startswith("cuda")
        )
        pretty_print(f"Using torch AdamW (fused={use_fused}).")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps,
            fused=use_fused,
        )

    print(optimizer)

    if config.training.track_peak_memory:
        log_memory(
            "after init (model + vLLM + optimizer)",
            config.training.device,
            reset_after=True,
        )

    # -------------------------------------------------------------#
    # 训练循环
    # -------------------------------------------------------------#
    pretty_print("Starting the training loop...", title="GRPO Training loop")

    sampling_params = SamplingParams(
        temperature=config.training.temperature,
        top_p=config.training.top_p,
        max_tokens=config.training.max_tokens,
        stop=[str(s) for s in config.training.stop],
        include_stop_str_in_output=config.training.include_stop_str_in_output,
        min_tokens=config.training.min_tokens,
    )

    reward_fn = r1_zero_reward_fn

    eval_step = 0

    best_answer_accuracy = None
    best_reward = None
    best_format_accuracy = None
    best_eval_step = None
    best_grpo_step = None

    last_answer_accuracy = None
    last_reward = None
    last_format_accuracy = None

    def get_model_for_saving():
        """Return the original Hugging Face model when torch.compile is enabled."""
        return getattr(model, "_orig_mod", model)

    def save_model_and_tokenizer(save_dir, overwrite=False):
        """Save model and tokenizer, optionally replacing an existing directory."""
        save_dir = Path(save_dir)

        if overwrite and save_dir.exists():
            shutil.rmtree(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        get_model_for_saving().save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    def save_best_model(
        phase,
        cur_grpo_step,
        cur_eval_step,
        answer_accuracy,
        reward,
        format_accuracy,
    ):
        """Overwrite best_model/ and record the metrics that selected it."""
        pretty_print(
            f"Saving new best model to {best_model_dir}...",
            title="Best Model",
        )

        save_model_and_tokenizer(best_model_dir, overwrite=True)

        best_model_info = {
            "phase": phase,
            "grpo_step": cur_grpo_step,
            "eval_step": cur_eval_step,
            "answer_accuracy": answer_accuracy,
            "reward": reward,
            "format_accuracy": format_accuracy,
            "selection_metric": "mean_answer_reward",
        }

        with open(
            best_model_dir / "best_model_info.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(best_model_info, f, indent=2, ensure_ascii=False)

        append_metrics(
            {
                "type": "best_model",
                **best_model_info,
                "model_dir": str(best_model_dir.resolve()),
            }
        )

        pretty_print(
            f"New best model saved | "
            f"phase: {phase} | "
            f"grpo_step: {cur_grpo_step} | "
            f"eval_step: {cur_eval_step} | "
            f"ans_rew: {answer_accuracy:.4f} | "
            f"rew: {reward:.4f} | "
            f"fmt_rew: {format_accuracy:.4f}"
        )

    def should_log_rollouts(grpo_step, is_last_step):
        if grpo_step == 0 or is_last_step:
            return True

        if config.training.eval_interval <= 0:
            return False

        return (grpo_step + 1) % config.training.eval_interval == 0

    # -------------------------------------------------------------#
    # 训练前评估
    # -------------------------------------------------------------#
    if config.training.eval_interval > 0:
        pretty_print(
            "Running base model evaluation (pre-training)...",
            title="Base Model Evaluation",
        )

        eval_metrics = evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=reward_fn,
            val_dataset=val_dataset,
            prompt_template=prompt_template,
            sampling_params=sampling_params,
            max_val_examples=config.training.max_val_examples,
        )

        pretty_print(
            f"[EVAL] grpo_step: {0:03d} | "
            f"eval_step: {eval_step:03d} | "
            f"n: {eval_metrics['n_examples']} | "
            f"rew: {eval_metrics['mean_reward']:.4f} | "
            f"ans_rew: {eval_metrics['mean_answer_reward']:.4f} | "
            f"fmt_rew: {eval_metrics['mean_format_reward']:.4f}"
        )

        append_metrics(
            {
                "type": "eval",
                "phase": "pretrain",
                "grpo_step": 0,
                "eval_step": eval_step,
                "n_examples": eval_metrics["n_examples"],
                "reward": eval_metrics["mean_reward"],
                "answer_reward": eval_metrics["mean_answer_reward"],
                "format_reward": eval_metrics["mean_format_reward"],
            }
        )

        last_answer_accuracy = float(eval_metrics["mean_answer_reward"])
        last_reward = float(eval_metrics["mean_reward"])
        last_format_accuracy = float(eval_metrics["mean_format_reward"])

        best_answer_accuracy = last_answer_accuracy
        best_reward = last_reward
        best_format_accuracy = last_format_accuracy
        best_eval_step = eval_step
        best_grpo_step = 0

        # 训练前基线模型是首次评估得到的当前最佳模型。
        save_best_model(
            phase="pretrain",
            cur_grpo_step=best_grpo_step,
            cur_eval_step=best_eval_step,
            answer_accuracy=best_answer_accuracy,
            reward=best_reward,
            format_accuracy=best_format_accuracy,
        )

        eval_step += 1

    # -------------------------------------------------------------#
    # GRPO 主循环
    # -------------------------------------------------------------#
    for grpo_step in range(config.training.n_grpo_steps):
        is_last_step = grpo_step == config.training.n_grpo_steps - 1
        grpo_step_title = (
            f"GRPO step {grpo_step:03d}/{config.training.n_grpo_steps - 1:03d}"
        )
        pretty_print(None, title=grpo_step_title)

        grpo_step_start = time.time()

        # ---------------------------------------------------------#
        # 采样 prompts
        # ---------------------------------------------------------#
        cur_batch = random.sample(train_dataset, n_prompts_per_rollout_batch)

        rep_rollout_prompts = [
            prompt_template.replace("{question}", ex["problem"])
            for ex in cur_batch
            for _ in range(config.training.group_size)
        ]

        rep_rollout_ground_truths = [
            ex["answer"]
            for ex in cur_batch
            for _ in range(config.training.group_size)
        ]

        # ---------------------------------------------------------#
        # 生成 rollouts
        # ---------------------------------------------------------#
        rollout_start = time.time()
        rollout_outputs = vllm_model.generate(rep_rollout_prompts, sampling_params)
        rollout_dt = time.time() - rollout_start

        rollout_responses = [output.outputs[0].text for output in rollout_outputs]

        # ---------------------------------------------------------#
        # 计算奖励
        # ---------------------------------------------------------#
        (
            rollout_advantages,
            rollout_raw_rewards,
            rollout_rewards_meta,
        ) = compute_group_normalized_rewards(
            reward_fn,
            rollout_responses,
            rep_rollout_ground_truths,
            config.training.group_size,
            config.training.advantage_eps,
            config.training.use_std_normalization,
        )

        # ---------------------------------------------------------#
        # 随机保存部分 rollouts
        # ---------------------------------------------------------#
        if (
            config.training.n_rollouts_to_log > 0
            and should_log_rollouts(grpo_step, is_last_step)
        ):
            save_indices = random.sample(
                range(len(rollout_responses)),
                min(config.training.n_rollouts_to_log, len(rollout_responses)),
            )

            rollouts_dir = output_dir / "rollouts"
            rollouts_dir.mkdir(parents=True, exist_ok=True)

            rollout_records = [
                {
                    "prompt": rep_rollout_prompts[i],
                    "response": rollout_responses[i],
                    "ground_truth": rep_rollout_ground_truths[i],
                    "advantage": rollout_advantages[i].item(),
                    "reward": rollout_rewards_meta["rewards"][i]["reward"],
                    "format_reward": rollout_rewards_meta["rewards"][i][
                        "format_reward"
                    ],
                    "answer_reward": rollout_rewards_meta["rewards"][i][
                        "answer_reward"
                    ],
                }
                for i in save_indices
            ]

            rollout_file = rollouts_dir / f"rollouts_step_{grpo_step:03d}.jsonl"

            with open(rollout_file, "w", encoding="utf-8") as f:
                f.write(
                    "\n".join(
                        json.dumps(r, ensure_ascii=False)
                        for r in rollout_records
                    )
                    + "\n"
                )

            pretty_print(f"Saved {len(rollout_records)} rollouts to {rollout_file}")

        if config.training.track_peak_memory:
            log_memory(
                f"[{grpo_step_title}] after rollout generation",
                config.training.device,
                reset_after=True,
            )

        # ---------------------------------------------------------#
        # tokenize rollout 数据
        # ---------------------------------------------------------#
        tokenized_train_data = tokenize_prompt_and_output(
            rep_rollout_prompts,
            rollout_responses,
            tokenizer,
        )

        pretty_print(
            tokenized_train_data,
            title="Tokenized train data",
            is_sub_title=True,
        )

        mean_response_length = (
            tokenized_train_data["response_mask"]
            .sum(dim=1)
            .float()
            .mean()
            .item()
        )

        # ---------------------------------------------------------#
        # 计算 old_log_probs
        # ---------------------------------------------------------#
        old_log_probs = None

        if config.training.loss_type in ["grpo_clip", "grpo_no_clip"]:
            pretty_print(
                "Computing old_log_probs over full rollout_batch_size...",
                title=f"{grpo_step_title} - Old log probs computation",
                is_sub_title=True,
            )

            model.eval()
            old_log_probs = []

            total_train_size = len(tokenized_train_data["input_ids"])
            batch_size = config.training.old_log_probs_train_size

            for idx in range(0, total_train_size, batch_size):
                input_ids, labels = map(
                    lambda x: x[idx : idx + batch_size].to(config.training.device),
                    [
                        tokenized_train_data["input_ids"],
                        tokenized_train_data["labels"],
                    ],
                )

                with torch.no_grad():
                    with torch.autocast(
                        device_type=config.training.device.split(":")[0],
                        dtype=DTYPE_MAPPING[config.training.dtype],
                    ):
                        log_probs = get_response_log_probs(
                            model,
                            input_ids,
                            labels,
                        )["log_probs"]

                    old_log_probs.append(log_probs)

            old_log_probs = torch.cat(old_log_probs, dim=0).cpu()

            old_log_probs_mem_mb = (
                old_log_probs.element_size()
                * old_log_probs.nelement()
                / 1024**2
            )

            pretty_print(
                f"Old log probs shape: {old_log_probs.shape}, "
                f"memory: {old_log_probs_mem_mb:.2f} MB"
            )

            del (
                log_probs,
                input_ids,
                labels,
                total_train_size,
                batch_size,
                old_log_probs_mem_mb,
            )

        # ---------------------------------------------------------#
        # 训练前释放 vLLM 显存
        # ---------------------------------------------------------#
        if config.training.use_vllm_sleep_mode:
            pretty_print(
                "Sleeping vLLM to free its GPU memory "
                "(weights + KV cache) during training..."
            )
            vllm_model.sleep(level=1)

        if config.training.track_peak_memory:
            log_memory(
                f"[{grpo_step_title}] after vLLM sleep",
                config.training.device,
                reset_after=True,
            )

        torch.cuda.empty_cache()
        model.train()

        # ---------------------------------------------------------#
        # GRPO 内层训练
        # ---------------------------------------------------------#
        train_dt = 0.0

        for train_epoch in range(config.training.epochs_per_rollout_batch):
            pretty_print(
                "",
                title=(
                    f"{grpo_step_title} - GRPO epoch "
                    f"{train_epoch:02d}/"
                    f"{config.training.epochs_per_rollout_batch - 1:02d}"
                ),
                is_sub_title=True,
            )

            num_train_steps = (
                config.training.rollout_batch_size
                // config.training.train_batch_size
            )

            for train_step in range(num_train_steps):
                pretty_print(
                    "",
                    title=(
                        f"{grpo_step_title} - GRPO inner step "
                        f"{train_step:02d}/{num_train_steps - 1:02d}"
                    ),
                    is_sub_title=True,
                )

                loss_accum = 0.0
                entropy_accum = 0.0
                clip_fraction_accum = 0.0
                mean_ratio_accum = 0.0
                total_response_tokens = 0

                start_time = time.time()

                for idx in tqdm(
                    range(config.training.gradient_accumulation_steps),
                    desc="Microbatches",
                    leave=False,
                ):
                    base_idx = train_step * config.training.train_batch_size
                    start_idx = base_idx + idx * micro_train_batch_size
                    end_idx = start_idx + micro_train_batch_size

                    microbatch = {
                        k: v[start_idx:end_idx].to(config.training.device)
                        for k, v in tokenized_train_data.items()
                    }

                    with torch.autocast(
                        device_type=config.training.device.split(":")[0],
                        dtype=DTYPE_MAPPING[config.training.dtype],
                    ):
                        cur_log_probs_result = get_response_log_probs(
                            model,
                            microbatch["input_ids"],
                            microbatch["labels"],
                            True,
                        )

                    current_log_probs = cur_log_probs_result["log_probs"]
                    token_entropy = cur_log_probs_result["token_entropy"]

                    old_log_probs_microbatch = (
                        old_log_probs[start_idx:end_idx].to(config.training.device)
                        if old_log_probs is not None
                        else None
                    )

                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=current_log_probs,
                        response_mask=microbatch["response_mask"],
                        gradient_accumulation_steps=(
                            config.training.gradient_accumulation_steps
                        ),
                        loss_type=config.training.loss_type,
                        raw_rewards=rollout_raw_rewards[start_idx:end_idx]
                        .unsqueeze(-1)
                        .to(config.training.device),
                        advantages=rollout_advantages[start_idx:end_idx]
                        .unsqueeze(-1)
                        .to(config.training.device),
                        old_log_probs=old_log_probs_microbatch,
                        cliprange=config.training.cliprange,
                        norm_mode=config.training.normalize_mode,
                        norm_constant=config.training.normalize_constant,
                    )

                    loss_accum += loss.detach()
                    entropy_accum += (
                        token_entropy * microbatch["response_mask"]
                    ).sum().detach()
                    total_response_tokens += (
                        microbatch["response_mask"].sum().detach()
                    )
                    clip_fraction_accum += meta.get("clip_fraction", 0.0)
                    mean_ratio_accum += meta.get("mean_ratio", 0.0)

                    del (
                        microbatch,
                        cur_log_probs_result,
                        current_log_probs,
                        token_entropy,
                        loss,
                        old_log_probs_microbatch,
                    )

                avg_entropy = entropy_accum / total_response_tokens
                avg_clip_fraction = (
                    clip_fraction_accum
                    / config.training.gradient_accumulation_steps
                )
                avg_mean_ratio = (
                    mean_ratio_accum
                    / config.training.gradient_accumulation_steps
                )

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                dt = time.time() - start_time
                train_dt += dt

                step_mean_format_reward = 0.0
                step_mean_answer_reward = 0.0
                step_mean_reward = 0.0

                rewards_len = len(rollout_rewards_meta["rewards"])

                for r in rollout_rewards_meta["rewards"]:
                    step_mean_format_reward += r["format_reward"] / rewards_len
                    step_mean_answer_reward += r["answer_reward"] / rewards_len
                    step_mean_reward += r["reward"] / rewards_len

                pretty_print(
                    f"[TRAIN] grpo_step: {grpo_step:03d} | "
                    f"loss: {loss_accum.item():.4f} | "
                    f"ent: {avg_entropy:.4f} | "
                    f"rew: {step_mean_reward:.4f} | "
                    f"ans_rew: {step_mean_answer_reward:.4f} | "
                    f"fmt_rew: {step_mean_format_reward:.4f} | "
                    f"clip: {avg_clip_fraction:.4f} | "
                    f"ratio: {avg_mean_ratio:.4f} | "
                    f"gnorm: {grad_norm:.4f} | "
                    f"lr: {config.training.learning_rate:.6f} | "
                    f"resp_len: {mean_response_length:.1f}"
                )

                append_metrics(
                    {
                        "type": "train",
                        "grpo_step": grpo_step,
                        "loss": loss_accum.item(),
                        "entropy": avg_entropy.item(),
                        "reward": step_mean_reward,
                        "answer_reward": step_mean_answer_reward,
                        "format_reward": step_mean_format_reward,
                        "clip_fraction": float(avg_clip_fraction),
                        "mean_ratio": float(avg_mean_ratio),
                        "grad_norm": grad_norm.item(),
                        "mean_response_length": mean_response_length,
                    }
                )

                torch.cuda.empty_cache()

        eval_dt = 0.0
        step_dt = time.time() - grpo_step_start

        if config.training.track_peak_memory:
            log_memory(
                f"[{grpo_step_title}] after training inner loop "
                f"(peak = training VRAM)",
                config.training.device,
                reset_after=True,
            )

        # ---------------------------------------------------------#
        # 唤醒 vLLM 并加载更新后的模型权重
        # ---------------------------------------------------------#
        if config.training.use_vllm_sleep_mode:
            vllm_model.wake_up()

        load_policy_into_vllm_instance(model, vllm_model)

        # ---------------------------------------------------------#
        # 中间评估
        # ---------------------------------------------------------#
        if config.training.eval_interval > 0 and (
            (grpo_step + 1) % config.training.eval_interval == 0
            or is_last_step
        ):
            eval_step += 1

            pretty_print(
                f"Running intermediate evaluation on "
                f"{config.training.max_val_examples} val examples...",
                title=f"{grpo_step_title} - Intermediate Evaluation",
                is_sub_title=True,
            )

            eval_start = time.time()

            eval_metrics = evaluate_vllm(
                vllm_model=vllm_model,
                reward_fn=reward_fn,
                val_dataset=val_dataset,
                prompt_template=prompt_template,
                sampling_params=sampling_params,
                max_val_examples=config.training.max_val_examples,
            )

            eval_dt = time.time() - eval_start

            pretty_print(
                f"[EVAL] grpo_step: {grpo_step:03d} | "
                f"eval_step: {eval_step:03d} | "
                f"n: {eval_metrics['n_examples']} | "
                f"rew: {eval_metrics['mean_reward']:.4f} | "
                f"ans_rew: {eval_metrics['mean_answer_reward']:.4f} | "
                f"fmt_rew: {eval_metrics['mean_format_reward']:.4f}"
            )

            append_metrics(
                {
                    "type": "eval",
                    "phase": "intermediate",
                    "grpo_step": grpo_step,
                    "eval_step": eval_step,
                    "n_examples": eval_metrics["n_examples"],
                    "reward": eval_metrics["mean_reward"],
                    "answer_reward": eval_metrics["mean_answer_reward"],
                    "format_reward": eval_metrics["mean_format_reward"],
                }
            )

            last_answer_accuracy = float(eval_metrics["mean_answer_reward"])
            last_reward = float(eval_metrics["mean_reward"])
            last_format_accuracy = float(eval_metrics["mean_format_reward"])

            if (
                best_answer_accuracy is None
                or last_answer_accuracy > best_answer_accuracy
            ):
                best_answer_accuracy = last_answer_accuracy
                best_reward = last_reward
                best_format_accuracy = last_format_accuracy
                best_eval_step = eval_step
                best_grpo_step = grpo_step

                save_best_model(
                    phase="intermediate",
                    cur_grpo_step=best_grpo_step,
                    cur_eval_step=best_eval_step,
                    answer_accuracy=best_answer_accuracy,
                    reward=best_reward,
                    format_accuracy=best_format_accuracy,
                )

        # ---------------------------------------------------------#
        # 打印并保存耗时指标
        # ---------------------------------------------------------#
        pretty_print(
            f"[DT] grpo_step: {grpo_step:03d} | "
            f"rollout_dt: {rollout_dt:.1f}s | "
            f"train_dt: {train_dt:.1f}s | "
            f"eval_dt: {eval_dt:.1f}s | "
            f"step_dt: {step_dt:.1f}s"
        )

        append_metrics(
            {
                "type": "timing",
                "grpo_step": grpo_step,
                "rollout_dt": rollout_dt,
                "train_dt": train_dt,
                "eval_dt": eval_dt,
                "step_dt": step_dt,
            }
        )

        # ---------------------------------------------------------#
        # 保存 checkpoint
        # ---------------------------------------------------------#
        if config.training.checkpoint_interval > 0 and (
            (grpo_step + 1) % config.training.checkpoint_interval == 0
            or is_last_step
        ):
            ckpt_dir = output_dir / f"checkpoint_{grpo_step:03d}"

            pretty_print(
                f"Saving checkpoint to {ckpt_dir}...",
                title=f"{grpo_step_title} - Checkpoint",
                is_sub_title=True,
            )

            save_model_and_tokenizer(ckpt_dir)

    # -------------------------------------------------------------#
    # 保存最终模型
    # -------------------------------------------------------------#
    pretty_print(
        f"Saving final model to {final_model_dir}...",
        title="Final Model",
    )

    save_model_and_tokenizer(final_model_dir, overwrite=True)

    # -------------------------------------------------------------#
    # 写入 final_summary.json
    # -------------------------------------------------------------#
    final_summary = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "max_steps": config.training.n_grpo_steps,
        "steps_per_epoch": steps_per_epoch,
        "epochs_per_rollout_batch": config.training.epochs_per_rollout_batch,
        "rollout_batch_size": config.training.rollout_batch_size,
        "total_batch_size": config.training.train_batch_size,
        "micro_batch_size": micro_train_batch_size,
        "grad_acc_steps": config.training.gradient_accumulation_steps,
        "group_size": config.training.group_size,
        "n_prompts_per_rollout_batch": n_prompts_per_rollout_batch,
        "best_answer_accuracy": best_answer_accuracy,
        "best_reward": best_reward,
        "best_format_accuracy": best_format_accuracy,
        "best_eval_step": best_eval_step,
        "best_grpo_step": best_grpo_step,
        "last_answer_accuracy": last_answer_accuracy,
        "last_reward": last_reward,
        "last_format_accuracy": last_format_accuracy,
        "best_model_dir": (
            str(best_model_dir.resolve()) if best_model_dir.exists() else None
        ),
        "best_model_info_file": (
            str((best_model_dir / "best_model_info.json").resolve())
            if (best_model_dir / "best_model_info.json").exists()
            else None
        ),
        "final_model_dir": str(final_model_dir.resolve()),
        "metrics_file": str(metrics_file.resolve()),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    pretty_print(
        final_summary,
        title=f"Saved final summary to {summary_file}",
    )

    if use_wandb:
        wandb.finish()