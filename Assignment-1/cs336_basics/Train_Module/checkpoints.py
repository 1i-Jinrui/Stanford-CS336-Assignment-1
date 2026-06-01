import torch
import torch.nn as nn
import os
from torch.optim import Optimizer


def save_checkpoint(model: nn.Module,
                    optimizer: Optimizer, 
                    iteration: int, 
                    save_path: str | os.PathLike):
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)


def load_checkpoint(load_path: str | os.PathLike, 
                    model: nn.Module, 
                    optimizer: Optimizer) -> int:
    checkpoint = torch.load(load_path)
    iteration = checkpoint["iteration"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return iteration