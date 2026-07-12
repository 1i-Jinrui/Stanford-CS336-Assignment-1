import torch
from utils.defaults import Config
from omegaconf import OmegaConf

# MAPPING
DTYPE_MAPPING = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DEFAULT_CONFIG = OmegaConf.structured(Config)
