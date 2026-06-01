# 交叉熵的数值稳定性

## 公式

CE = log Σⱼ eˣʲ - xₜ

## 数值不稳定的问题

先 softmax 再 log 的实现方式：

```python
y_pred = softmax(inputs)
p = y_pred[batch_indices, targets]
loss = -torch.log(p).mean()
```

逻辑正确，但存在数值问题：
- softmax 虽然做了 `exp(x - x_max)` 防止上溢
- 但当 xⱼ - xₘₐₓ 是很大的负数时（如 -1000），e⁻¹⁰⁰⁰ ≈ 0 会**下溢为 0**
- 随后 log(0) = -∞ → batch 平均后 loss 变成 inf

## 数值稳定的做法：logsumexp trick

将 log Σⱼ eˣʲ 改写为：

log Σⱼ eˣʲ⁻ᵐ + m = log Σⱼ eˣʲ

其中 m = maxⱼ xⱼ，这样 eˣʲ⁻ᵐ 的最大值为 1，既防上溢也避免下溢。

# 优化器内部结构理解

```
optimizer
├── param_groups
│   ├── group 0
│   │   ├── "params"
│   │   │   ├── p0 = layer1.weight
│   │   │   ├── p1 = layer1.bias
│   │   │   ├── p2 = layer2.weight
│   │   │   └── p3 = layer2.bias
│   │   ├── "lr"
│   │   ├── "betas"
│   │   ├── "eps"
│   │   └── "weight_decay"
│   │
│   └── group 1
│       ├── "params"
│       ├── "lr"
│       ├── "betas"
│       ├── "eps"
│       └── "weight_decay"
│
└── state
    ├── p0
    │   ├── "step"
    │   ├── "m"
    │   └── "v"
    ├── p1
    │   ├── "step"
    │   ├── "m"
    │   └── "v"
    └── ...
```