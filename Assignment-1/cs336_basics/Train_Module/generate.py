import argparse
import json
import os
import torch
import yaml
from tqdm import tqdm

from cs336_basics.LLM_Module.Transformer_LM import Transformer_LM
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["vocab_path"] = str(config.get("vocab_path", ""))
    config["merges_path"] = str(config.get("merges_path", ""))
    config["checkpoint_path"] = str(config.get("checkpoint_path", ""))
    config["device"] = str(config.get("device", "cpu"))

    model_config = config.get("model", {})
    config["model"] = {
        "vocab_size": int(model_config.get("vocab_size", 10000)),
        "context_length": int(model_config.get("context_length", 256)),
        "d_model": int(model_config.get("d_model", 512)),
        "num_layers": int(model_config.get("num_layers", 4)),
        "num_heads": int(model_config.get("num_heads", 16)),
        "d_ff": int(model_config.get("d_ff", 1344)),
        "rope_theta": float(model_config.get("rope_theta", 10000.0)),
    }

    config["max_new_tokens"] = int(config.get("max_new_tokens", 256))
    config["temperature"] = float(config.get("temperature", 0.8))
    config["top_p"] = float(config.get("top_p", 0.9))
    config["eos_token"] = str(config.get("eos_token", "<|endoftext|>"))
    config["seed"] = int(config.get("seed", 42))

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


def load_checkpoint_local(
    load_path: str,
    model: Transformer_LM,
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


def apply_top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be in the range (0, 1].")

    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    keep_mask = cumulative_probs <= top_p
    keep_mask[..., 0] = True

    filtered_sorted_probs = torch.where(
        keep_mask,
        sorted_probs,
        torch.zeros_like(sorted_probs),
    )

    filtered_probs = torch.zeros_like(probs)
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
    temperature: float = 0.8,
    top_p: float = 0.9,
    eos_token: str = "<|endoftext|>",
) -> str:
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    eos_id = get_eos_id(tokenizer, eos_token)

    input_ids = tokenizer.encode(prompt)
    generated_ids = list(input_ids)

    for _ in tqdm(range(max_new_tokens), desc="Generating", unit="token"):
        current_context = generated_ids[-context_length:]

        x = torch.tensor([current_context], dtype=torch.long, device=device)

        logits = model(x)
        next_token_logits = logits[:, -1, :]

        next_token_logits = next_token_logits / temperature

        probs = torch.softmax(next_token_logits, dim=-1)
        probs = apply_top_p_filtering(probs, top_p=top_p)

        next_id = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Interactive text generation with Transformer LM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    torch.manual_seed(config["seed"])

    if not os.path.exists(config["vocab_path"]):
        raise FileNotFoundError(f"Vocab file not found: {config['vocab_path']}")

    if not os.path.exists(config["merges_path"]):
        raise FileNotFoundError(f"Merges file not found: {config['merges_path']}")

    if not os.path.exists(config["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint file not found: {config['checkpoint_path']}")

    device = torch.device(config["device"])

    print("=" * 80)
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(
        config["vocab_path"],
        config["merges_path"],
        [config["eos_token"]],
    )
    print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")

    print("=" * 80)
    print("Building model...")
    model = build_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("=" * 80)
    print(f"Loading checkpoint: {config['checkpoint_path']}")
    iteration = load_checkpoint_local(config["checkpoint_path"], model, device)
    print(f"Loaded checkpoint at iteration: {iteration}")

    print("=" * 80)
    print("Interactive generation mode")
    print("Type your prompt and press Enter.")
    print("Type q, quit, or exit to stop.")
    print("=" * 80)

    while True:
        prompt = input("\nPrompt> ").strip()

        if prompt.lower() in {"q", "quit", "exit"}:
            break

        if not prompt:
            continue

        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            context_length=config["model"]["context_length"],
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            eos_token=config["eos_token"],
        )

        print("\n" + "=" * 80)
        print("Generated Text")
        print("=" * 80)
        print(generated_text)
        print("=" * 80)


if __name__ == "__main__":
    main()