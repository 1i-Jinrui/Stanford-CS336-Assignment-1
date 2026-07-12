# CS336 Assignment 5 — Alignment and Reasoning RL

## 项目简介

本项目以 **Qwen2.5-Math-1.5B Base** 为基础模型，在 MATH 数学推理数据集上实现并比较 Zero-shot、监督微调（Supervised Fine-Tuning，SFT）和组相对策略优化（Group Relative Policy Optimization，GRPO）三类方法。

项目实现内容包括：

- **Zero-shot 基线评测**：使用 `r1_zero` 提示模板构造输入，通过 vLLM 批量生成回答，并使用奖励函数统计 `reward`、`format_reward` 和 `answer_reward`。
- **SFT 基础组件**：实现 prompt/response 独立分词与拼接、response mask、逐 token log-probability、token entropy、masked normalization、梯度累积及 microbatch 训练步骤。
- **完整 SFT 训练流程**：使用带有推理轨迹的数学样本微调基础模型，并定期在验证集上评估答案正确率和格式正确率。
- **训练数据过滤**：使用答案奖励函数检查 SFT 样本，仅保留最终答案正确的推理轨迹，以分析监督数据质量对训练结果的影响。
- **GRPO 基础组件**：实现组内奖励归一化、naive policy-gradient loss、GRPO-Clip loss、策略梯度损失分发、masked mean 以及 GRPO microbatch 更新。
- **完整 GRPO 训练流程**：完成 rollout 生成、奖励计算、组内优势估计、旧策略 log-probability 计算、策略更新、vLLM 权重同步和周期性验证。
- **两种 GRPO 初始化路线**：
  1. 从 Qwen2.5-Math-1.5B Base 直接进行 GRPO；
  2. 从最佳 SFT 模型继续进行 GRPO，以比较监督学习预热对强化学习训练的影响。
- **实验环境**：SFT 实验在单张 NVIDIA GeForce RTX 4090 24GB GPU 上完成；GRPO 实验在单张 NVIDIA RTX PRO 6000 96GB GPU 上完成。
---

## 实验结果分析

### 1. Zero-shot 基线

在未进行任何微调的情况下，使用 Qwen2.5-Math-1.5B Base 对 1,000 个 MATH 验证样本进行评测。

| 指标 | 数值 |
|---|---:|
| 验证样本数 | 1,000 |
| Average Reward | 0.042 |
| Format Accuracy | 21.9% |
| Answer Accuracy | 4.2% |

不同输出类别如下：

| 输出类别 | 数量 | 占比 |
|---|---:|---:|
| 格式正确、答案正确 | 42 | 4.2% |
| 格式正确、答案错误 | 177 | 17.7% |
| 格式错误、答案错误 | 781 | 78.1% |

Zero-shot 模型的答案正确率仅为 **4.2%**，格式正确率为 **21.9%**。

### 2. 完整训练集 SFT

使用未经答案过滤的全部 4,836 条 SFT 推理轨迹进行训练。

| 配置项 | 数值 |
|---|---:|
| 训练集大小 | 4,836 |
| 验证集大小 | 1,000 |
| 训练 GPU | NVIDIA GeForce RTX 4090 24GB |
| Total Batch Size | 128 |
| Micro Batch Size | 1 |
| Gradient Accumulation Steps | 128 |
| 最佳 Answer Accuracy | 57.0% |
| Format Accuracy | 99.0% |

SFT 将答案正确率从 **4.2%** 提升至 **57.0%**，绝对提升 **52.8 个百分点**；格式正确率从 **21.9%** 提升至 **99.0%**，绝对提升 **77.1 个百分点**。结果说明，带有完整推理过程的监督数据能够同时帮助模型学习数学求解模式和固定输出格式。

SFT 后格式正确率已经接近饱和，而答案正确率仍明显低于格式正确率，因此后续性能瓶颈主要由数学推理过程和最终答案判断造成，而不再是输出标签缺失或格式错误。

### 3. 过滤错误推理轨迹后的 SFT

使用奖励函数检查全部 SFT 样本，只保留最终答案正确的推理轨迹。过滤后训练集由 4,836 条减少至 3,496 条。

| 配置项 | 数值 |
|---|---:|
| 原始训练集大小 | 4,836 |
| 验证集大小 | 1,000 |
| 训练 GPU | NVIDIA GeForce RTX 4090 24GB |
| Total Batch Size | 128 |
| Micro Batch Size | 1 |
| Gradient Accumulation Steps | 128 |
| 过滤后训练集大小 | 3,496 |
| 删除样本数 | 1,340 |
| 数据保留率 | 72.29% |
| 最佳 Answer Accuracy | 57.6% |
| Format Accuracy | 99.1% |

过滤后答案正确率由 **57.0%** 小幅提升至 **57.6%**，提升 **0.6 个百分点**。这说明错误答案对应的推理轨迹确实会引入噪声，但简单删除所有最终答案错误的样本并没有带来显著增益。

一种可能的解释是，部分最终答案错误的轨迹仍然包含有效的局部推理步骤；完全删除这些样本虽然提高了监督信号的整体正确性，也同时减少了训练数据规模和推理模式的多样性。因此，数据过滤带来的收益相对有限。

### 4. 逐 token 损失归一化 SFT

在普通序列损失中，较长回答包含更多有效 token，因而可能在总损失和梯度中占据更大权重。未归一化的单样本损失为：

\[
L_i=-\sum_{t=1}^{T_i}\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}),
\]

其中 \(T_i\) 为第 \(i\) 个样本的有效 response token 数量。

启用逐 token 损失归一化后，损失改为：

\[
L_i=-\frac{1}{T_i}\sum_{t=1}^{T_i}
\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}).
\]

实验结果如下：

| 配置项 | 数值 |
|---|---:|
| 训练集大小 | 3,496 |
| 验证集大小 | 1,000 |
| 训练 GPU | NVIDIA GeForce RTX 4090 24GB |
| Total Batch Size | 128 |
| Micro Batch Size | 1 |
| Gradient Accumulation Steps | 128 |
| 最佳 Answer Accuracy | 66.6% |
| Format Accuracy | 99.6% |

在初始 SFT 实验中，观察到训练过程中的梯度范数长期处于较高水平。尽管损失能够持续下降，模型性能也在朝正确方向改善，但不同训练批次之间的梯度波动较大。进一步分析发现，训练样本的 response 长度差异明显：若直接对每个序列的 token 损失求和，长序列会因为包含更多有效 token 而对总损失和梯度产生更大的贡献，从而导致不同长度样本之间的权重失衡，并增大梯度更新的方差。

为解决这一问题，在训练步骤中加入了 `per_token_loss` 选项。启用后，系统会根据每个样本实际包含的 response token 数量对序列损失进行归一化，使长短回答在样本层面具有更加均衡的梯度贡献。采用该方法后，模型的答案正确率由过滤 SFT 的 **57.6%** 提升至 **66.6%**，绝对提升 **9.0 个百分点**；格式正确率也由 **99.1%** 提升至 **99.6%**。除准确率显著提高外，训练过程中的梯度范数也更加平稳，说明逐 token 损失归一化有效缓解了长序列过度主导梯度的问题，降低了不同 microbatch 之间的梯度方差，并提升了训练稳定性。

这一结果表明，在包含可变长度推理轨迹的 SFT 任务中，损失归一化方式会直接影响模型对不同样本的利用效率。按照有效 response token 数量进行归一化，不仅能够改善梯度权重分配，还能使模型更加均衡地学习不同长度的推理过程，从而显著提升数学推理性能。


### 5. 从基础模型直接训练 GRPO

在完整训练集上，从 Qwen2.5-Math-1.5B Base 直接进行 GRPO。主要配置和结果如下：

| 配置项 | 数值 |
|---|---:|
| 训练集大小 | 4,836 |
| 验证集大小 | 1,000 |
| 训练 GPU | NVIDIA RTX PRO 6000 96GB |
| Epochs per Rollout Batch | 2 |
| Rollout Batch Size | 256 |
| Total Batch Size | 256 |
| Micro Batch Size | 4 |
| Gradient Accumulation Steps | 64 |
| Group Size | 8 |
| 每个 Rollout Batch 的问题数 | 32 |
| 最佳 Answer Accuracy | 74.9% |
| 最佳 Reward | 0.749 |
| 最佳 Format Accuracy | 97.4% |



从基础模型直接进行 GRPO 后，最佳答案正确率达到 **74.9%**。相较于 Zero-shot 的 **4.2%**，绝对提升 **70.7 个百分点**；相较于最佳 SFT 模型的 **66.6%**，进一步提升 **8.3 个百分点**。

这一结果说明，基于可验证答案奖励的 GRPO 能够直接强化产生正确解答的策略，能够显著提升模型的数学推理表现。与 SFT 不同，GRPO 不要求每条训练样本都提供固定的目标推理轨迹，而是通过多次 rollout 和组内相对优势，让模型增加高奖励回答的生成概率。


### 6. 从最佳 SFT 模型继续训练 GRPO

以逐 token 损失归一化得到的最佳 SFT 模型为初始化，继续进行 GRPO。在记录到的中间验证节点上得到：

| 指标 | 数值 |
|---|---:|
| Phase | intermediate |
| GRPO Step | 124 |
| Eval Step | 26 |
| 训练 GPU | NVIDIA RTX PRO 6000 96GB |
| Answer Accuracy | 77.1% |
| Reward | 0.771 |
| Format Accuracy | 99.3% |
| Selection Metric | mean_answer_reward |

SFT 初始化的 GRPO 达到 **77.1%** 的答案正确率和 **99.3%** 的格式正确率。与最佳 SFT 模型相比，答案正确率由 **66.6%** 提升至 **77.1%**，绝对提升 **10.5 个百分点**；与从基础模型直接训练 GRPO 的最佳结果相比，答案正确率进一步提高 **2.2个百分点**。

格式正确率由基础模型 GRPO 的最佳 **97.4%** 提升至 **99.3%**，提高 **1.9 个百分点**。这说明 SFT 预热的主要价值不仅体现在最终答案正确率，还体现在为 GRPO 提供一个已经掌握推理结构和输出格式的稳定初始策略。


### 7. 综合对比

| 实验 | 初始化模型 | 训练方式 | 最佳 Answer Accuracy | Format Accuracy |
|---|---|---|---:|---:|
| Zero-shot | Qwen2.5-Math-1.5B Base | 无训练 | 4.2% | 21.9% |
| 全量 SFT | Base | SFT | 57.0% | 99.0% |
| 过滤数据 SFT | Base | SFT | 57.6% | 99.1% |
| 过滤数据 + Per-token Loss | Base | SFT | 66.6% | 99.6% |
| Base + GRPO | Base | GRPO | 74.9% | 97.4% |
| 最佳 SFT + GRPO | 最佳 SFT | GRPO | **77.1%** | **99.3%** |

性能递进关系：

1. **Zero-shot 阶段**同时受限于格式遵循能力和数学推理能力。
2. **SFT 阶段**迅速解决了输出格式问题，并将答案正确率提升至 57.0%。
3. **数据过滤**能够减少错误监督，但单独带来的收益较小。
4. **逐 token 损失归一化**显著改善训练效果，将最佳 SFT 答案正确率提升至 66.6%。
5. **GRPO**进一步利用可验证奖励优化推理策略，将答案正确率提升至 74.9% 以上。
6. **SFT + GRPO**取得当前最佳结果：答案正确率 **77.1%**，格式正确率 **99.3%**。

从结果看，SFT 与 GRPO 具有明显的互补关系。SFT 擅长建立稳定的推理表达方式和输出格式，为策略提供高质量初始化；GRPO 则通过组内相对奖励直接优化答案正确性，突破监督数据所限定的固定推理轨迹。