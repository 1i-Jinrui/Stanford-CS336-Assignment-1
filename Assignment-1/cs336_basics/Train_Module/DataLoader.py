import torch
import numpy as np
from typing import Iterator
import math



class DataLoader:
    def __init__(self, 
        dataset: list[int] | np.ndarray, 
        batch_size: int, 
        context_length: int,
        device: str = "cpu",
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.data_len = len(dataset)
        if self.context_length > self.data_len:
            raise ValueError("context_length must be less than to data_len")

    def get_train_batch_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        随机获取一个训练 batch 数据，每个样本包含 context_length 个时间步
        """
        num_samples = self.data_len - self.context_length # 保证起点不能太靠后，因为每个训练样本都要从序列里连续取出长度为 context_length 的一段
        idx = np.random.randint(low=0, high=num_samples, size=(self.batch_size,))
        x = np.stack([self.dataset[i:i+self.context_length] for i in idx])
        y = np.stack([self.dataset[i+1:i+self.context_length+1] for i in idx])
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        return x, y
    
    def get_valid_batch_data_iter(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        顺序遍历验证 batch 数据，每个样本包含 context_length 个时间步
        """
        num_samples = self.data_len - self.context_length
        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)
            idx = np.arange(batch_start, batch_end)
            x = np.stack([self.dataset[i:i+self.context_length] for i in idx])
            y = np.stack([self.dataset[i+1:i+self.context_length+1] for i in idx])
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            y = torch.tensor(y, dtype=torch.long, device=self.device)
            yield x, y
            
    def __len__(self) -> int:
        """
        返回完整遍历验证集时的 batch 数量。
        """
        num_samples = self.data_len - self.context_length
        return math.ceil(num_samples / self.batch_size)





