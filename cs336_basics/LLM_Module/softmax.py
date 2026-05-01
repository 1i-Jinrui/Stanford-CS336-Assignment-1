import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # .max返回的对象可以看作是一个元组，第一个元素是最大值，第二个元素是最大值所在的索引，只需要最大值
    x_max = x.max(dim=dim, keepdim=True)[0]
    # 防止数值爆炸
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


