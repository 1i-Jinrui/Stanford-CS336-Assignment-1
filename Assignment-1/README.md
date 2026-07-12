# CS336 Assignment 1 — 从零构建语言模型

## 项目简介

本项目使用 PyTorch 从底层实现 Transformer 语言模型的全部核心组件，不依赖 `torch.nn.Transformer` 等高层 API。

实现内容包括三大模块：

- **BPE 分词器**：基于 UTF-8 字节的 Byte-Pair Encoding 算法，支持训练、编码、解码，并经过多轮工程优化，吞吐量达 196 万 tokens/秒。
- **LLM 组件**：从 Linear、RMSNorm、Softmax 到 Scaled Dot-Product Attention、Multi-Head Self-Attention、RoPE、SwiGLU、Transformer Block、完整 Transformer Language Model，全部手写实现。
- **训练管线**：含数据加载、AdamW 优化器、余弦学习率调度、梯度裁剪、交叉熵损失（数值稳定的 logsumexp 实现）、checkpoint 管理、文本生成（temperature + nucleus sampling）。

---

## 实验分析

### 1. BPE 分词器训练

分别在 TinyStories（2.23 GB）和 OpenWebText（11.9 GB）两个数据集上训练 BPE 分词器。

| 数据集       | 词表大小 | 合并次数 | 训练耗时  | 压缩率 (bytes/token) |
| ------------ | -------- | -------- | --------- | -------------------- |
| TinyStories  | 10,000   | 9,749    | 407.16 s  | 1.86                 |
| OpenWebText  | 32,000   | 31,744   | 8524.85 s | 2.28                 |

**分析**：

- OpenWebText 由于词表扩大（32k vs 10k）导致每次合并需扫描更大的 token 频次表，再加上多进程预分词带来额外协调开销。

- OpenWebText Tokenizer 的 bytes/token 更高，主要得益于更大的 32K 词表以及更丰富的训练语料，使其能够学习更多常见单词和子词组合。TinyStories 虽然内容模式更集中，但 10K 词表限制了可合并的子词数量，因此其压缩率略低。

---

### 2. 语言模型训练

分别在 TinyStories 和 OpenWebText 上训练 Transformer 语言模型。

| 数据集         |  词表 |   参数量 | 上下文长度 | Batch Size | 训练步数 | 累计训练 Tokens | Val Loss |  PPL |    GPU |
| ------------- | ---: | ------: | --------: | ---------: | -------: | -------------: | -------: | --: | -----: |
| TinyStories   |  10K |  22.7M  |       512 |         64 |     10K  |         327.7M |   0.7555 | 2.13 | 4090  |
| OpenWebText   |  32K |  45.2M  |       512 |         32 |     30K  |         491.5M |   2.1010 | 8.17 | 4090  |

**分析**：

- **TinyStories 模型（PPL 2.13）效果好于 OWT（PPL 8.17）**。 TinyStories 数据分布简单——词汇有限、句式规整、主题集中在童话故事； 2.23 GB 的文本量对 22M 参数的模型足够充分学习。OpenWebText  覆盖新闻、百科、论坛等多样文本，45M 参数的容量相对不足；
---

### 3. 文本生成示例

#### TinyStories 模型

> **Prompt**: *Once upon a time, there was a little girl named Lily who found a tiny golden key in the garden.*
>
> Once upon a time, there was a little girl named Lily who found a tiny golden key in the garden. She was very curious and wanted to see what was inside the golden key.
> Lily walked around the garden and saw a tiny door in the garden. She knew it was a secret door, so she tried to open it. When she opened the door, she found a small room with a big bed and a tiny room. Lily thought it was a magic key, so she took the key and unlocked the door.
> When Lily opened the door, she saw a beautiful garden with lots of flowers and trees. She played in the garden all day and forgot about the secret door. When it was time to go inside, Lily found the secret room and went back to her garden. She had a great day with her new secret garden and couldn't wait to see what else she could find.

> **Prompt**: *Tom was afraid of the dark, but one night he heard a soft voice coming from under his bed.*
>
> Tom was afraid of the dark, but one night he heard a soft voice coming from under his bed. It was the voice of a little bird who had been stuck in a tree.
> Tom was very frightened and he didn't know what to do. He thought and thought about what he could do. Then he remembered his mom had seen him hiding in the tree. So he decided to make it a safe place for the bird to sleep.
> He was happy that he could help the bird. He stayed with the bird until the nest was gone. Then he got back into his bed and went to sleep.
> Tom was never seen again. He was a frightened boy and never woke up the nest again.
> <|endoftext|>

> **Prompt**: *The little rabbit had never left the forest before, but today he decided to follow the river.*
>
> The little rabbit had never left the forest before, but today he decided to follow the river. He hopped and hopped until he found a big tree with a swing. The little rabbit was so excited to swing on the swing that he didn't even notice a sign that said "Win the branch". He was about to take a deep breath and he slowly moved closer to the tree. He was so happy to have so many adventures and he knew he would never forget the day he was so excited to swing on the swing for his adventure.
> <|endoftext|>

> **Prompt**: *Mia opened the old wooden box and found a note that said, "Do not open the red door."*
>
> Mia opened the old wooden box and found a note that said, "Do not open the red door." Mia was excited and asked her mom, "What does the red door open?" Her mom said, "The red door is in the garden, Mia. We can go find out what is inside."
> Mia and her mom went to the garden and found a small box. They opened the box and found a tiny mouse inside. The mouse said, "Hello, I am Max the mouse. I live in this house." Mia and her mom were very surprised. They did not know mice could talk. They asked the mouse, "How can you talk?" The mouse said, "I am sorry, I was wrong. I was lonely in the garden."
> Mia and her mom were happy to meet Max. They all became good friends and played together every day. And from that day on, Mia was never lonely again.
> <|endoftext|>

**分析**：
- TinyStories 模型能够保持简单、自然的语言和短句表达
- 大多数故事具有完整的开头、发展和结尾
- 偶尔会出现情节跳跃或逻辑不一致的问题


#### OpenWebText 模型（30,000 步，PPL 8.17）

> **Prompt**: *In recent years, researchers have increasingly focused on the relationship between artificial intelligence and education.*
>
> In recent years, researchers have increasingly focused on the relationship between artificial intelligence and education. But the study also shows that neuroscientists have created a research laboratory for a number of researchers. Peanut fruits and vegetables and the diversity of different chemicals can be a key part of the brain, with the fact that these nanobots can be a problem for modern metastasis. But such evidence can be traced back to the skeptical side of the brain.
> It is unclear what the effect of research on nanobots has on the use of nicotine in human research. A research team studying nanobots and electronics is currently considering how such nanobots are used to be harmful to the brain. The researchers say that the researchers have found a small research group that they saw a significant increase in the efficacy of nanobots in the brain and is an important part of the process.
> The researchers have found that nanobots have an important role in helping compute molecules with an additive system and differ on the nanobots that can be used to highlight the effects of nanobots. They also found that nanobots are more likely to be found in the future than they are to the future.
> But their research is now focused on the first place in nanobots as well. They look at nanobots

> **Prompt**: *The city council announced on Tuesday that it would reconsider the proposed transportation plan after weeks of public criticism.*
>
> The city council announced on Tuesday that it would reconsider the proposed transportation plan after weeks of public criticism.
> The proposed plan is a three-week deadline for a new funding proposal to pay the same taxes for Toyota ticket plans and to give it a new ticket to Toyota to remain on track at the next round of new riders.
> Toyota plans to launch the new funding legislation in Toyota on Friday after it has reached an agreement with Toyota to comply with the proposed Transportation Act.
> The proposal has been a two-hour long process in Toyota and Toyota — a bike with a bike that includes a dedicated bike — and a proposed additional free agency deal in Toyota to form a free ticket for travel permits for Toyota to complete transportation research in Toyota in Toyota and Toyota in Toyota to drive a bike ticket in Toyota Park.
> Toyota is still planning a year to build a new station in Toyota in Toyota and Toyota to add a new ticket to Toyota to the Toyota National Toyota Tour.
> Toyota will be introduced in Toyota and Toyota to Toyota to make the transportation work the new ticket includes a $950 per barrel that will allow Toyota to use in Toyota for Toyota and Toyota as their

> **Prompt**: *When the company launched its first product, few analysts expected it to become one of the fastest-growing firms in the industry.*
>
> When the company launched its first product, few analysts expected it to become one of the fastest-growing firms in the industry.
> "We are starting to see a year for sales to tens of millions of dollars a year for the rest of the year," said Rep. Ron Johnson, a former employee of the GLA co-founder of the company's founder brand. "We look forward to making a continued supply of in-house customers to the future of our market through a fast growing retail industry."
> Johnson added that the company is currently working with GLA to produce a new product called the GLA V and U.S. company Hold Up Hold Down App to be able to sell its products in the U.S. by 2020.
> "The business has been very different than any other companies have been able to manufacture it," Johnson said. "The company is working with the company to achieve a fresh industry with greater cash than it is at its core."
> Johnson said he and his company worked with Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up Hold Up

> **Prompt**: *There is a common misconception that economic growth automatically leads to improvements in quality of life.*
>
> There is a common misconception that economic growth automatically leads to improvements in quality of life. The new administration has proposed for growth in the last 100 years, but few new firms have made such a big difference. In the last six years, I have seen the growth of the economy over the last 20 years in the last 40 years. I have seen that growth in the workforce as well. It has a very powerful business cycle that has started to recover and has thus far unfolded in the long run. Today is the end of the fiscal year of the year. In our view, the economy can continue to grow and to build growth and we are not the only ones who have lost the balance of the economy. While the most vulnerable economy has been in the industry for a long time, the economy is almost double that of the economy. If we had started to compete in the economy, we would have to get a sense of it. In the past we have to develop the economy at a level that goes far beyond the absolute guarantees of the economy. It is not a question of what it will do to the economy as a result of that problem. And the economy is taking the strength of the economy. It is a thing that is nothing more than a stagnation for a fiscal year of the year

**分析**：
- OpenWebText 模型已经学会了新闻、科技和评论类文章的语言风格，能够生成较为正式的长句
- 但仍存在明显的问题：语义断裂（peanut fruits → nanobots → metastasis）、实体重复（Toyota 循环）、逻辑混乱、主题逐渐漂移
- PPL 8.17 尚不足以支撑连贯的长文本生成，需要更大模型容量或更多训练步数