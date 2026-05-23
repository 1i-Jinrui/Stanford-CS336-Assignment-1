import argparse
import json
import math
import os
import torch
import yaml
from tqdm import tqdm

from cs336_basics.LLM_Module.Transformer_LM import Transformer_LM
from cs336_basics.utils.AdamW import AdamW
from cs336_basics.utils.cross_entropy import CrossEntropyLoss
from cs336_basics.bpe.tokenizer import Tokenizer


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]) -> Tokenizer:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    vocab = {}

    for str_id, token_str in vocab_data.items():
        token_id = int(str_id)

        if token_id < 256:
            vocab[token_id] = bytes([token_id])
        else:
            if isinstance(token_str, str):
                if token_str.startswith(r"\x") and len(token_str) == 4:
                    vocab[token_id] = bytes([int(token_str[2:], 16)])
                else:
                    vocab[token_id] = token_str.encode("utf-8", errors="surrogateescape")
            else:
                vocab[token_id] = token_str

    merges = []

    with open(merges_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading BPE merges", unit="line"):
            if line.startswith("#") or not line.strip():
                continue

            parts = line.rstrip("\n").split(" ")
            if len(parts) >= 2:
                merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def load_and_tokenize_data(valid_data_path: str, tokenizer: Tokenizer) -> list[int]:
    all_ids = []

    with open(valid_data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Tokenizing validation data", unit="line"):
            if not line.strip():
                continue
            all_ids.extend(tokenizer.encode(line))

    return all_ids


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["valid_data_path"] = str(config.get("valid_data_path", ""))
    config["vocab_path"] = str(config.get("vocab_path", ""))
    config["merges_path"] = str(config.get("merges_path", ""))

    config["output_dir"] = str(config.get("output_dir", "./checkpoints"))
    resume_from = config.get("resume_from")
    config["resume_from"] = str(resume_from) if resume_from is not None else None

    config["device"] = str(config.get("device", "cpu"))

    model_config = config.get("model", {})
    config["model"] = {
        "vocab_size": int(model_config.get("vocab_size", 50257)),
        "context_length": int(model_config.get("context_length", 512)),
        "d_model": int(model_config.get("d_model", 768)),
        "num_layers": int(model_config.get("num_layers", 12)),
        "num_heads": int(model_config.get("num_heads", 12)),
        "d_ff": int(model_config.get("d_ff", 3072)),
        "rope_theta": float(model_config.get("rope_theta", 10000.0)),
    }

    training_config = config.get("training", {})
    config["training"] = {
        "batch_size": int(training_config.get("batch_size", 8)),
        "max_iters": int(training_config.get("max_iters", 100000)),
        "warmup_iters": int(training_config.get("warmup_iters", 1000)),
    }

    optimizer_config = config.get("optimizer", {})
    config["optimizer"] = {
        "max_lr": float(optimizer_config.get("max_lr", 3e-4)),
        "min_lr": float(optimizer_config.get("min_lr", 3e-5)),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.01)),
        "betas": tuple(float(b) for b in optimizer_config.get("betas", [0.9, 0.999])),
        "eps": float(optimizer_config.get("eps", 1e-8)),
    }

    return config


def build_model(config: dict, device: torch.device) -> Transformer_LM:
    model = Transformer_LM(
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        rope_theta=config["model"]["rope_theta"],
    ).to(device)

    return model


def build_optimizer(config: dict, model: Transformer_LM) -> AdamW:
    return AdamW(
        params=model.parameters(),
        lr=config["optimizer"]["max_lr"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

def load_checkpoint_local(
    load_path: str,
    model: Transformer_LM,
    optimizer: AdamW | None = None,
    device: torch.device | str = "cpu",
) -> int:
    checkpoint = torch.load(load_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        if optimizer is not None:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            elif "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            elif "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])

        if "iteration" in checkpoint:
            return int(checkpoint["iteration"])
        if "iter" in checkpoint:
            return int(checkpoint["iter"])
        if "step" in checkpoint:
            return int(checkpoint["step"])
        if "global_step" in checkpoint:
            return int(checkpoint["global_step"])

        return -1

    raise ValueError("Unsupported checkpoint format.")

def get_eos_id(tokenizer: Tokenizer, eos_token: str = "<|endoftext|>") -> int | None:
    eos_bytes = eos_token.encode("utf-8")

    for token_id, token_bytes in tokenizer.vocab.items():
        if token_bytes == eos_bytes:
            return token_id

    return None


@torch.no_grad()
def evaluate_validation_loss(
    model: Transformer_LM,
    token_ids: list[int],
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, int]:

    model.eval()

    if len(token_ids) <= context_length + 1:
        raise ValueError("Validation data is too short for the configured context_length.")

    usable_sequences = (len(token_ids) - 1) // context_length
    total_batches = usable_sequences // batch_size

    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    if total_batches <= 0:
        raise ValueError("Validation data is too small for the configured batch_size and context_length.")

    total_loss = 0.0
    total_tokens = 0

    for batch_idx in tqdm(range(total_batches), desc="Evaluating validation loss", unit="batch"):
        xs = []
        ys = []
        
        # 需要找到每个序列在token_ids的绝对位置
        start_seq = batch_idx * batch_size

        for b in range(batch_size):
            seq_idx = start_seq + b
            start = seq_idx * context_length
            end = start + context_length

            x = token_ids[start:end]
            y = token_ids[start + 1:end + 1]

            xs.append(x)
            ys.append(y)

        x_tensor = torch.tensor(xs, dtype=torch.long, device=device)
        y_tensor = torch.tensor(ys, dtype=torch.long, device=device)

        logits = model(x_tensor)

        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y_tensor.reshape(-1)

        loss_fn = CrossEntropyLoss(inputs=logits_flat, targets=targets_flat)
        loss = loss_fn.forward()

        batch_tokens = targets_flat.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity, total_tokens


def apply_top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be in the range (0, 1].")

    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    # 计算累计概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    keep_mask = cumulative_probs <= top_p

    # 至少保留概率最大的 token
    keep_mask[..., 0] = True
    
    # torch.where(condition, a, b) 的意思是：如果 condition 为 True，取 a；如果 condition 为 False，取 b
    filtered_sorted_probs = torch.where(
        keep_mask,
        sorted_probs,
        torch.zeros_like(sorted_probs),
    )

    filtered_probs = torch.zeros_like(probs)
    # 把过滤后的概率按照 sorted_indices 放回原始位置
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_probs)

    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return filtered_probs


@torch.no_grad()
def generate(
    model: Transformer_LM,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    context_length: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token: str = "<|endoftext|>",
) -> str:
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    eos_id = get_eos_id(tokenizer, eos_token)

    input_ids = tokenizer.encode(prompt)
    generated_ids = list(input_ids)

    for _ in tqdm(range(max_new_tokens), desc="Generating text", unit="token"):
        # 取最后 context_length 个 token 作为模型输入
        current_context = generated_ids[-context_length:]

        x = torch.tensor([current_context], dtype=torch.long, device=device)

        logits = model(x)
        next_token_logits = logits[:, -1, :]

        next_token_logits = next_token_logits / temperature

        probs = torch.softmax(next_token_logits, dim=-1)
        probs = apply_top_p_filtering(probs, top_p=top_p)

        # 从概率分布中随机采样一个 token
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and generate text with Transformer LM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    checkpoint_path = str(config.get("checkpoint_path", ""))
    prompt = str(config.get("prompt", "Once upon a time"))
    max_new_tokens = int(config.get("max_new_tokens", 256))
    temperature = float(config.get("temperature", 0.8))
    top_p = float(config.get("top_p", 0.9))
    max_eval_batches = config.get("max_eval_batches", None)

    if max_eval_batches is not None:
        max_eval_batches = int(max_eval_batches)

    eval_batch_size = int(config.get("eval_batch_size", config["training"]["batch_size"]))

    if not os.path.exists(config["valid_data_path"]):
        raise FileNotFoundError(f"Validation data file not found: {config['valid_data_path']}")

    if not os.path.exists(config["vocab_path"]):
        raise FileNotFoundError(f"Vocab file not found: {config['vocab_path']}")

    if not os.path.exists(config["merges_path"]):
        raise FileNotFoundError(f"Merges file not found: {config['merges_path']}")

    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided in config file.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device(config["device"])

    print("=" * 80)
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(
        config["vocab_path"],
        config["merges_path"],
        ["<|endoftext|>"],
    )
    print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")

    print("=" * 80)
    print("Building model...")
    model = build_model(config, device)
    optimizer = build_optimizer(config, model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("=" * 80)
    print(f"Loading checkpoint: {checkpoint_path}")
    iteration = load_checkpoint_local(checkpoint_path, model, optimizer, device)
    print(f"Loaded checkpoint at iteration: {iteration}")

    print("=" * 80)
    print(f"Loading validation data: {config['valid_data_path']}")
    valid_ids = load_and_tokenize_data(config["valid_data_path"], tokenizer)
    print(f"Validation tokens: {len(valid_ids):,}")

    print("=" * 80)
    print("Evaluating validation metrics...")

    val_loss, val_ppl, evaluated_tokens = evaluate_validation_loss(
        model=model,
        token_ids=valid_ids,
        batch_size=eval_batch_size,
        context_length=config["model"]["context_length"],
        vocab_size=config["model"]["vocab_size"],
        device=device,
        max_batches=max_eval_batches,
    )

    print("=" * 80)
    print("Validation Results")
    print(f"Checkpoint iteration : {iteration}")
    print(f"Validation loss      : {val_loss:.6f}")
    print(f"Validation perplexity: {val_ppl:.6f}")
    print(f"Evaluated tokens     : {evaluated_tokens:,}")

    print("=" * 80)
    print("Generating text...")

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        context_length=config["model"]["context_length"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    print("=" * 80)
    print("Generation Config")
    print(f"Prompt        : {prompt}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature   : {temperature}")
    print(f"Top-p         : {top_p}")

    print("=" * 80)
    print("Generated Text")
    print(generated_text)
    print("=" * 80)

if __name__ == "__main__":
    main()