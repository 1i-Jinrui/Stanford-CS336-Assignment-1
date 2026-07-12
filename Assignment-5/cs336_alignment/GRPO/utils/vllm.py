from transformers import PreTrainedModel
from vllm import LLM

from utils.defaults import Config
from utils.constants import DTYPE_MAPPING
from utils.helper import pretty_print


# -------------------------------------------------------------#
# 初始化 vLLM 模型，并在训练过程中将 policy 加载到 vLLM 模型中的函数
# -------------------------------------------------------------#
def init_vllm(seed: int, cfg: Config):
    """
    启动推理进程，这里使用 vLLM 在与 policy 相同的 GPU 上持有一个模型。

    """

    vllm_init_params = {
        "model": cfg.paths.model_path,
        "dtype": DTYPE_MAPPING[cfg.vllm.dtype],
        "seed": seed,
        "gpu_memory_utilization": cfg.vllm.gpu_memory_utilization,
        "max_model_len": getattr(cfg.vllm, "max_model_len", 2048),
        "max_num_seqs": getattr(cfg.vllm, "max_num_seqs", 128),
        "enable_prefix_caching": cfg.vllm.enable_prefix_caching,
    }

    pretty_print(vllm_init_params, title="vLLM model initialization parameters")
    return LLM(**vllm_init_params)


# -------------------------------------------------------------#
# 将 policy 加载到 vLLM 模型中
# -------------------------------------------------------------#
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):

    # 如果模型经过了 torch.compile，真实模块位于 ._orig_mod 中
    if hasattr(policy, "_orig_mod"):
        policy_for_state = policy._orig_mod
    else:
        policy_for_state = policy

    state_dict = policy_for_state.state_dict()

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())