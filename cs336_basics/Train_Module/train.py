import argparse
import json
import os
import torch
import yaml
from tqdm import tqdm
import logging
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from cs336_basics.LLM_Module.Transformer_LM import Transformer_LM
from cs336_basics.Train_Module.DataLoader import DataLoader
from cs336_basics.Train_Module.checkpoints import save_checkpoint, load_checkpoint
from cs336_basics.utils.AdamW import AdamW
from cs336_basics.utils.cross_entropy import CrossEntropyLoss
from cs336_basics.utils.lr_cosine_schedule import Cosine_schedule
from cs336_basics.utils.gradient_clip import GradientClipper
from cs336_basics.bpe.tokenizer import Tokenizer


class TqdmLoggingHandler(logging.Handler):
    """自定义 logging handler，使用 tqdm.write() 输出日志"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]) -> Tokenizer:
    """
    加载 BPE tokenizer。

    vocab.json 中保存的是 token_id -> token 字符串的映射；
    merges.txt 中保存的是 BPE merge 规则。
    这里需要将 vocab 中的 token 字符串恢复为 bytes 形式
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    vocab = {}

    # 将 vocab.json 中的字符串形式 token 转换为 bytes 形式
    for str_id, token_str in vocab_data.items():
        token_id = int(str_id)

        # 前 256 个 token 通常对应单字节 token，直接构造 bytes
        if token_id < 256:
            vocab[token_id] = bytes([token_id])
        else:
            if isinstance(token_str, str):
                # 处理形如 "\xAB" 的十六进制字节表示
                if token_str.startswith(r'\x') and len(token_str) == 4:
                    vocab[token_id] = bytes([int(token_str[2:], 16)])
                else:
                    # 普通字符串 token 编码为 UTF-8 bytes
                    vocab[token_id] = token_str.encode("utf-8", errors="surrogateescape")
            else:
                # 如果 vocab 中已经是 bytes 或其他兼容格式，则直接保留
                vocab[token_id] = token_str

    merges = []

    # 加载 BPE merge 规则，每一行表示一对可以合并的 token
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading BPE merges", unit="line"):
            # 跳过注释行和空行
            if line.startswith("#") or not line.strip():
                continue

            parts = line.rstrip('\n').split(" ")

            # 每条 merge 至少应包含两个 token
            if len(parts) >= 2:
                merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def load_and_tokenize_data(train_data_path: str, tokenizer: Tokenizer) -> list[int]:
    all_ids = []
    with open(train_data_path, "r", encoding="utf-8", errors="ignore") as f:
        # 按行读取可以保证不会把一个词从中间切断
        for line in tqdm(f, desc="Tokenizing data", unit="line"):
            if not line.strip():
                continue
            ids = tokenizer.encode(line)
            all_ids.extend(ids)
    return all_ids


def load_config(config_path: str) -> dict:
    """
    从 yaml 配置文件加载参数，并进行强制类型转换。

    参数主要分为：
    1. 数据与 tokenizer 路径；
    2. checkpoint 输出与恢复；
    3. 模型结构超参数；
    4. 训练超参数；
    5. 优化器、学习率调度与梯度裁剪参数。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 数据与 tokenizer 路径 - 字符串类型
    config["train_data_path"] = str(config.get("train_data_path", ""))
    config["vocab_path"] = str(config.get("vocab_path", ""))
    config["merges_path"] = str(config.get("merges_path", ""))

    # checkpoint 相关参数 - 字符串类型
    config["output_dir"] = str(config.get("output_dir", "./checkpoints"))
    resume_from = config.get("resume_from")
    config["resume_from"] = str(resume_from) if resume_from is not None else None

    # 训练设备 - 字符串类型
    config["device"] = str(config.get("device", "cuda"))

    # 模型结构超参数 - 整数和浮点数类型
    model_config = config.get("model", {})
    config["model"] = {
        "vocab_size": int(model_config.get("vocab_size", 50257)),
        "context_length": int(model_config.get("context_length", 512)),
        "d_model": int(model_config.get("d_model", 768)),
        "num_layers": int(model_config.get("num_layers", 12)),
        "num_heads": int(model_config.get("num_heads", 12)),
        "d_ff": int(model_config.get("d_ff", 3072)),
        "rope_theta": float(model_config.get("rope_theta", 10000.0))
    }

    # 训练超参数 - 整数类型
    training_config = config.get("training", {})
    config["training"] = {
        "batch_size": int(training_config.get("batch_size", 8)),
        "max_iters": int(training_config.get("max_iters", 100000)),
        "warmup_iters": int(training_config.get("warmup_iters", 1000))
    }

    # 优化器参数 - 浮点数类型
    optimizer_config = config.get("optimizer", {})
    config["optimizer"] = {
        "max_lr": float(optimizer_config.get("max_lr", 3e-4)),
        "min_lr": float(optimizer_config.get("min_lr", 3e-5)),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.01)),
        "betas": tuple(float(b) for b in optimizer_config.get("betas", [0.9, 0.999])),
        "eps": float(optimizer_config.get("eps", 1e-8))
    }

    # 梯度裁剪阈值 - 浮点数类型
    config["max_grad_norm"] = float(config.get("max_grad_norm", 1.0))

    # 日志打印与模型保存间隔 - 整数类型
    config["log_interval"] = int(config.get("log_interval", 10))
    config["save_interval"] = int(config.get("save_interval", 1000))

    return config


def generate_run_name(config: dict) -> str:
    """根据关键训练参数生成唯一运行名称，用于日志和曲线文件命名"""
    m = config["model"]
    t = config["training"]
    o = config["optimizer"]
    name = (f"d{m['d_model']}_l{m['num_layers']}_h{m['num_heads']}_"
            f"ctx{m['context_length']}_bs{t['batch_size']}_"
            f"lr{o['max_lr']:.0e}_seed{config.get('seed', 42)}")
    return name


def get_gpu_memory_str() -> str:
    """获取本次 iteration 的显存峰值"""
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        pct = peak / total * 100
        return f"{peak:.2f}G/{total:.1f}G({pct:.1f}%)"
    else:
        return "N/A"

def main():
    """
    训练入口函数。

    整体流程：
    1. 解析参数并检查文件路径；
    2. 加载 tokenizer;
    3. 读取并 tokenize 训练数据;
    4. 初始化 DataLoader、模型、优化器、学习率调度器和梯度裁剪器;
    5. 如有 checkpoint,则恢复训练状态;
    6. 执行训练循环;
    7. 定期保存 checkpoint;
    8. 训练结束后保存最终模型。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    # 定义命令行参数
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    # 读取终端输入
    args = parser.parse_args()

    config = load_config(args.config)

    # 生成随机种子
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 生成基于关键参数的运行名称
    run_name = generate_run_name(config)
    run_dir = os.path.join(config["output_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 配置详细日志记录
    log_file = os.path.join(run_dir, f"train_log_{run_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            TqdmLoggingHandler()
        ]
    )
    logging.info(f"Starting training run: {run_name}")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Output directory: {run_dir}")

    # 创建 checkpoint 输出目录
    os.makedirs(config["output_dir"], exist_ok=True)

    # 检查必要文件是否存在
    if not os.path.exists(config["train_data_path"]):
        raise FileNotFoundError(f"Data file not found: {config['train_data_path']}")
    if not os.path.exists(config["vocab_path"]):
        raise FileNotFoundError(f"Vocab file not found: {config['vocab_path']}")
    if not os.path.exists(config["merges_path"]):
        raise FileNotFoundError(f"Merges file not found: {config['merges_path']}")

    device = torch.device(config["device"])

    # 加载 BPE tokenizer
    tokenizer = load_tokenizer(config["vocab_path"], config["merges_path"], ["<|endoftext|>"])
    print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")
    logging.info(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")

    # 将原始文本数据转换为 token id 序列
    data_ids = load_and_tokenize_data(config["train_data_path"], tokenizer)
    print(f"Total tokens in dataset: {len(data_ids)}")
    logging.info(f"Total tokens in dataset: {len(data_ids)}")

    # 构造训练数据加载器
    # 每次会返回形状类似 [batch_size, context_length] 的 x 和 y
    dataloader = DataLoader(
        dataset=data_ids,
        batch_size=config["training"]["batch_size"],
        context_length=config["model"]["context_length"],
        device=config["device"]
    )

    # 初始化 Transformer Language Model
    model = Transformer_LM(
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        rope_theta=config["model"]["rope_theta"]
    ).to(device)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 初始化 AdamW 优化器
    optimizer = AdamW(
        params=model.parameters(),
        lr=config["optimizer"]["max_lr"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
        weight_decay=config["optimizer"]["weight_decay"]
    )

    # 初始化 cosine 学习率调度器
    # 训练初期使用 warmup，之后按照 cosine 曲线从 max_lr 衰减到 min_lr
    lr_schedule = Cosine_schedule(
        max_lr=config["optimizer"]["max_lr"],
        min_lr=config["optimizer"]["min_lr"],
        warmup_iters=config["training"]["warmup_iters"],
        cosine_cycle_iters=config["training"]["max_iters"]
    )

    # 初始化梯度裁剪器，用于限制整体梯度 L2 norm
    grad_clipper = GradientClipper(
        params=model.parameters(),
        max_l2_norm=config["max_grad_norm"]
    )

    # 默认从第 0 次迭代开始训练
    start_iter = 0

    # 如果指定了 checkpoint，则恢复模型参数和优化器状态
    if config["resume_from"] is not None:
        if not os.path.exists(config["resume_from"]):
            raise FileNotFoundError(f"Checkpoint not found: {config['resume_from']}")

        start_iter = load_checkpoint(config["resume_from"], model, optimizer)
        print(f"Resumed from checkpoint, starting at iteration {start_iter}")
        logging.info(f"Resumed from checkpoint, starting at iteration {start_iter}")

    # 切换到训练模式，启用 dropout 等训练行为
    model.train()

    # 用于记录损失曲线的数据
    losses = []
    lrs = []
    iters = []

    progress_bar = tqdm(
        range(start_iter, config["training"]["max_iters"]),
        desc="Training",
        unit="iter",
        initial=start_iter,
        total=config["training"]["max_iters"]
    )

    for iteration in progress_bar:
        # 重置显存峰值统计
        torch.cuda.reset_peak_memory_stats()

        # 清空上一轮反向传播残留的梯度
        optimizer.zero_grad()

        # 采样一个训练 batch
        # x 是输入 token 序列，y 是对应的下一个 token 标签
        x, y = dataloader.get_train_batch_data()

        # 前向传播，输出 logits
        # logits 形状通常为 [batch_size, context_length, vocab_size]
        logits = model(x)

        # 将 logits 和 targets 拉平成二维/一维，方便计算交叉熵
        logits_flat = logits.reshape(-1, config["model"]["vocab_size"])
        targets_flat = y.reshape(-1)

        # 计算语言模型的 next-token prediction loss
        loss_fn = CrossEntropyLoss(inputs=logits_flat, targets=targets_flat)
        loss = loss_fn.forward()

        # 反向传播，计算所有可训练参数的梯度
        loss.backward()

        # 梯度裁剪，避免梯度爆炸导致训练不稳定
        grad_clipper()

        # 根据当前 iteration 计算学习率
        current_lr = lr_schedule(iteration)

        # 将调度器计算出的学习率写入优化器
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # 参数更新
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)
        lrs.append(current_lr)
        iters.append(iteration)

        # 定期打印训练日志和更新进度条
        if iteration % config["log_interval"] == 0:
            mem_str = get_gpu_memory_str()
            log_msg = f"Iteration: {iteration:6d} | Loss: {current_loss:.6f} | LR: {current_lr:.6e} | Mem: {mem_str}"
            logging.info(log_msg)
            progress_bar.set_postfix(
                loss=f"{current_loss:.6f}",
                lr=f"{current_lr:.6e}",
                mem=mem_str
            )

        # 定期保存 checkpoint，便于中断后恢复训练
        if iteration % config["save_interval"] == 0 and iteration > 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

            # 额外保存一份 latest checkpoint，方便快速恢复最新训练状态
            latest_path = os.path.join(run_dir, "checkpoint_latest.pt")
            save_checkpoint(model, optimizer, iteration, latest_path)

    # 训练完成后保存最终 checkpoint
    final_path = os.path.join(run_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, config["training"]["max_iters"], final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")
    logging.info(f"Training complete. Final checkpoint saved to {final_path}")

    # 保存详细损失曲线数据 (CSV)
    csv_path = os.path.join(run_dir, f"loss_curve_{run_name}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "loss", "lr"])
        for i, lo, lr_val in zip(iters, losses, lrs):
            writer.writerow([i, lo, lr_val])
    logging.info(f"Loss curve data saved to {csv_path}")

    # 绘制并保存损失曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(iters, losses, label='Training Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve - {run_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(run_dir, f"loss_curve_{run_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Loss curve plot saved to {plot_path}")

    logging.info("=== Training Run Completed Successfully ===")


if __name__ == "__main__":
    main()