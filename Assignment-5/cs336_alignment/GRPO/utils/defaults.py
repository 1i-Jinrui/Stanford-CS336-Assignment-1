from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf


@dataclass
class PathConfig:
    train_data_file: str = "./MATH/train.jsonl"
    val_data_file: str = "./MATH/val.jsonl"
    prompt_template_file: str = "./MATH/r1_zero.prompt"
    model_path: str = "./models/Qwen2.5-Math-1.5B"
    output_dir: str = "results"


@dataclass
class TrainingConfig:
    # common
    seed: int = 2317
    device: str = "cuda:0"
    dtype: str = "bfloat16"

    # model
    attention_type: str = "flash_attention_2"
    use_compile: bool = False


    # AdamW optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


    # sampling parameters
    temperature: float = 1.0
    top_p: float = 1.0
    min_tokens: int = 4
    max_tokens: int = 1024

    stop: List[str] = field(
        default_factory=lambda: ["</answer>"]
    )

    include_stop_str_in_output: bool = True


    # -------------------------------------------------------------#
    # GRPO parameters
    # -------------------------------------------------------------#

    # 总 GRPO 外循环次数
    n_grpo_steps: int = 200

    # reward advantage normalization
    advantage_eps: float = 1e-6

    # rollout
    rollout_batch_size: int = 256
    group_size: int = 8

    # 每个 rollout batch 更新次数
    epochs_per_rollout_batch: int = 2

    # training batch
    train_batch_size: int = 256

    # 梯度累积
    gradient_accumulation_steps: int = 64

    # GRPO loss
    loss_type: str = "grpo_clip"

    use_std_normalization: bool = True

    cliprange: float = 0.2

    max_grad_norm: float = 1.0


    # old policy log probs
    # 注意：这里不是token数量，而是一次forward处理多少条sample
    old_log_probs_train_size: int = 16


    # loss normalization
    normalize_mode: str = "mean"
    normalize_constant: float = 1024.0



    # -------------------------------------------------------------#
    # evaluation
    # -------------------------------------------------------------#

    eval_interval: int = 5

    # None表示整个验证集
    max_val_examples: Optional[int] = 1000



    # -------------------------------------------------------------#
    # checkpoint
    # -------------------------------------------------------------#

    checkpoint_interval: int = 0



    # -------------------------------------------------------------#
    # local logging
    # -------------------------------------------------------------#

    n_rollouts_to_log: int = 16



    # -------------------------------------------------------------#
    # wandb disabled
    # -------------------------------------------------------------#

    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_tags: List[str] = field(
        default_factory=list
    )



    # -------------------------------------------------------------#
    # memory optimization
    # -------------------------------------------------------------#

    track_peak_memory: bool = False

    use_gradient_checkpointing: bool = False

    use_vllm_sleep_mode: bool = True

    use_bnb_adamw8bit: bool = False



@dataclass
class VllmConfig:

    # vLLM占GPU显存比例
    gpu_memory_utilization: float = 0.2

    dtype: str = "bfloat16"

    enable_prefix_caching: bool = True



@dataclass
class Config:

    paths: PathConfig = field(
        default_factory=PathConfig
    )

    vllm: VllmConfig = field(
        default_factory=VllmConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )