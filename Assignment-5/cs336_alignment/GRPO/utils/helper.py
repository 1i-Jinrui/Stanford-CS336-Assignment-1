import json
import random
from typing import Callable, List

import numpy as np
import torch


# -------------------------------------------------------------#
# 记录显存使用情况的函数
# -------------------------------------------------------------#
def log_memory(label: str, device: str, reset_after: bool = False):
    """在指定检查点记录当前显存和峰值显存。可选地在记录后重置峰值统计。"""
    cur_gb  = torch.cuda.memory_allocated(device) / 1024**3
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    pretty_print(f"[MEM] {label}: current={cur_gb:.2f}GB  peak_since_reset={peak_gb:.2f}GB")
    if reset_after:
        torch.cuda.reset_peak_memory_stats(device)


# -------------------------------------------------------------#
# 设置随机种子的函数
# -------------------------------------------------------------#
def set_seed(seed: int):
    """
    设置各个随机数生成器的随机种子。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------#
# 美观打印 input_config
# -------------------------------------------------------------#
def _fmt(val):
    return val.shape if isinstance(val, (torch.Tensor, np.ndarray)) else val


def pretty_print(input_config: dict | list | str | None, title: str | None = None, is_sub_title: bool = False) -> None:
    """
    美观打印 input_config。
    """
    if title is not None:
        if is_sub_title:
            print(f"{'-'*30}\n{title}:\n{'-'*30}")
        else:
            print("="*25 + f" {title} " + "="*25)
    if isinstance(input_config, dict):
        for k, v in input_config.items():
            if isinstance(v, dict):
                print(f"{k:<25}:")
                for kk, vv in v.items():
                    print(f"    {kk:<25}: {_fmt(vv)}")
            else:
                print(f"{k:<25}: {_fmt(v)}")
    elif isinstance(input_config, list):
        for i, v in enumerate(input_config):
            print(f"{i:<25}: {_fmt(v)}")
    elif isinstance(input_config, str):
        print(input_config)
    elif input_config is None:
        pass
    else:
        raise ValueError(f"Unsupported type: {type(input_config)}")