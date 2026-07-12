import regex
from typing import Iterable, Iterator
import json
import random
import time


PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str],
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 由于需要通过 merges 字典来排序，所以需要一个字典来存储 merges 的优先级
        # enumerate 会遍历列表，同时返回索引和元素本身
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}

        # 将字节转换为 token id，避免直接使用 vocab 字典反复查找
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # 添加一个字典，存储：单词字符串 -> Token IDs 列表
        # 遇到重复的词即可直接取出，不需要再 BPE 合并
        self.cache: dict[str, list[int]] = {}

        if self.special_tokens:
            # 按照长度降序排序，确保更长的符号
            # 例如 "<|eot|><|eot|>" 在更短的符号 "<|eot|>" 之前被匹配
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

            special_token_pattern = "|".join(
                map(regex.escape, sorted_special_tokens)
            )

            # 提前编译好特殊 token 正则对象
            self.special_regex = regex.compile(f"({special_token_pattern})")
        else:
            self.special_regex = None

    def _get_bpe_ids(self, word: str) -> list[int]:
        """
        接收字符串文本片段，直接返回 token ids 列表。结合缓存使用。
        """
        if word in self.cache:
            return self.cache[word]

        # 首先将 word 转换为单字节列表
        parts = [bytes([b]) for b in word.encode("utf-8")]

        while len(parts) > 1:
            # 记录当前轮中优先级最高的合并对
            best_rank = float("inf")
            best_pair = None

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])

                # 如果 pair 在字典里，get 返回对应排名；如果不在，返回 None
                rank = self.merges_priority_map.get(pair)

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # 如果没有任何 pair 在 merges 中，说明无法继续合并
            if best_pair is None:
                break

            # 应用最佳合并对
            merged_token = best_pair[0] + best_pair[1]

            new_parts = []
            i = 0

            while i < len(parts):
                if (
                    i < len(parts) - 1
                    and (parts[i], parts[i + 1]) == best_pair
                ):
                    new_parts.append(merged_token)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1

            parts = new_parts

        ids = [self.bytes_to_id[x] for x in parts]

        self.cache[word] = ids

        return ids

    def encode(self, text: str) -> list[int]:
        """
        将字符串编码为 token ids。
        """
        if not text:
            return []

        if self.special_regex:
            # 按照特殊符号分割 text，保持特殊符号作为分隔符
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]

        final_ids = []

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                # 如果 chunk 是特殊符号，直接编码
                final_ids.append(self.bytes_to_id[chunk.encode("utf-8")])
            else:
                # 如果 chunk 是普通文本，使用 BPE 算法处理
                # 使用 finditer，防止长文本造成 OOM
                for match in PAT.finditer(chunk):
                    word = match.group()

                    # 获取 word 的合并字节片段对应的 ids
                    ids = self._get_bpe_ids(word)
                    final_ids.extend(ids)

        return final_ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """
        批量编码，推荐用于吞吐量测试。
        """
        return [self.encode(text) for text in texts]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式编码，适合处理大文件。
        """
        for text in iterable:
            yield from self.encode(text)

    def compression_stats(self, text: str) -> dict:
        """
        计算单条文本的 BPE 压缩统计信息。

        压缩率定义：
            compression_ratio = 原始 UTF-8 字节数 / BPE token 数

        注意：
            这里的压缩率不是文件压缩率，而是 tokenizer 层面的 token 压缩率。
        """
        encoded = self.encode(text)

        raw_chars = len(text)
        raw_bytes = len(text.encode("utf-8"))
        token_count = len(encoded)

        if token_count == 0:
            return {
                "chars": raw_chars,
                "raw_bytes": raw_bytes,
                "tokens": 0,
                "bytes_per_token": 0.0,
                "chars_per_token": 0.0,
                "compression_ratio": 0.0,
                "tokens_per_byte": 0.0,
            }

        return {
            "chars": raw_chars,
            "raw_bytes": raw_bytes,
            "tokens": token_count,
            "bytes_per_token": raw_bytes / token_count,
            "chars_per_token": raw_chars / token_count,
            "compression_ratio": raw_bytes / token_count,
            "tokens_per_byte": token_count / raw_bytes if raw_bytes > 0 else 0.0,
        }

    def compression_stats_batch(self, texts: list[str]) -> dict:
        """
        计算批量文本的整体 BPE 压缩统计信息。
        """
        total_chars = 0
        total_raw_bytes = 0
        total_tokens = 0

        for text in texts:
            ids = self.encode(text)

            total_chars += len(text)
            total_raw_bytes += len(text.encode("utf-8"))
            total_tokens += len(ids)

        if total_tokens == 0:
            return {
                "total_chars": total_chars,
                "total_raw_bytes": total_raw_bytes,
                "total_tokens": 0,
                "bytes_per_token": 0.0,
                "chars_per_token": 0.0,
                "compression_ratio": 0.0,
                "tokens_per_byte": 0.0,
            }

        return {
            "total_chars": total_chars,
            "total_raw_bytes": total_raw_bytes,
            "total_tokens": total_tokens,
            "bytes_per_token": total_raw_bytes / total_tokens,
            "chars_per_token": total_chars / total_tokens,
            "compression_ratio": total_raw_bytes / total_tokens,
            "tokens_per_byte": (
                total_tokens / total_raw_bytes if total_raw_bytes > 0 else 0.0
            ),
        }

    def decode(self, ids: list[int]) -> str:
        """
        将 token ids 解码回字符串。
        """
        all_bytes = b"".join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # ====================== 准备测试文本 ======================
    try:
        with open(
            "../data/owt-valid.txt",
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            lines = [line.strip() for line in f if line.strip()]
            test_text = (
                random.choice(lines)
                if lines
                else "Hello world! This is a test of the BPE tokenizer."
            )
    except FileNotFoundError:
        test_text = (
            "Hello world! This is a test of the BPE tokenizer. "
            "你好，世界！12345 测试中文和英文混合。"
        )

    # ====================== 加载 vocab ======================
    with open("tyst_vocab.json", "r", encoding="utf-8") as f:
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
                    vocab[token_id] = token_str.encode(
                        "utf-8",
                        errors="surrogateescape",
                    )
            else:
                vocab[token_id] = token_str

    # ====================== 加载 merges ======================
    merges = []

    with open("tyst_merges.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.rstrip("\n").split(" ")

            if len(parts) >= 2:
                merges.append(
                    (
                        parts[0].encode("utf-8"),
                        parts[1].encode("utf-8"),
                    )
                )

    # ====================== 初始化 tokenizer ======================
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"],
    )

    # ====================== 基础测试 ======================
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    stats = tokenizer.compression_stats(test_text)

    print("=" * 70)
    print("基础测试")
    print("原始文本:", test_text)
    print("原始字符数:", stats["chars"])
    print("原始 UTF-8 字节数:", stats["raw_bytes"])
    print("编码长度 tokens:", stats["tokens"])
    print("编码结果:", encoded[:100], "..." if len(encoded) > 100 else "")
    print("解码文本:", decoded)
    print("还原一致:", test_text == decoded)
    print(f"BPE 压缩率: {stats['compression_ratio']:.4f} bytes/token")
    print(f"平均每 token 表示字符数: {stats['chars_per_token']:.4f} chars/token")
    print(f"平均每 token 表示字节数: {stats['bytes_per_token']:.4f} bytes/token")
    print(f"Token 密度: {stats['tokens_per_byte']:.4f} tokens/byte")
    print("=" * 70)

    # ====================== 特殊 token 测试 ======================
    special_test = (
        "Hello world<|endoftext|>This is after special token. "
        "你好<|endoftext|>世界"
    )

    special_encoded = tokenizer.encode(special_test)
    special_stats = tokenizer.compression_stats(special_test)

    print("\n特殊 token 测试:")
    print("输入:", special_test)
    print("编码:", special_encoded)
    print("原始字符数:", special_stats["chars"])
    print("原始 UTF-8 字节数:", special_stats["raw_bytes"])
    print("编码长度 tokens:", special_stats["tokens"])
    print(f"BPE 压缩率: {special_stats['compression_ratio']:.4f} bytes/token")
    print("=" * 70)

    # ====================== 吞吐量测试 ======================
    print("\n开始计算 Tokenizer 吞吐量...")

    # 准备批量测试数据
    if "lines" in locals() and lines:
        test_texts = lines[:2000]
        print(f"使用真实数据集，共 {len(test_texts)} 条文本")
    else:
        base_text = test_text * 20
        test_texts = [base_text] * 500
        print(f"使用合成数据，共 {len(test_texts)} 条文本")

    total_chars = sum(len(text) for text in test_texts)
    total_raw_bytes = sum(len(text.encode("utf-8")) for text in test_texts)

    print(f"总输入字符数: {total_chars:,}")
    print(f"总输入 UTF-8 字节数: {total_raw_bytes:,}")

    # ====================== 单条文本吞吐量测试 ======================
    start_time = time.perf_counter()
    single_encoded = tokenizer.encode(test_texts[0])
    end_time = time.perf_counter()

    single_tokens = len(single_encoded)
    single_time = end_time - start_time
    single_throughput = single_tokens / single_time if single_time > 0 else 0

    single_stats = tokenizer.compression_stats(test_texts[0])

    print(f"\n单条文本测试:")
    print(f"  输入字符数: {len(test_texts[0]):,}")
    print(f"  输入 UTF-8 字节数: {len(test_texts[0].encode('utf-8')):,}")
    print(f"  输出 tokens: {single_tokens:,}")
    print(f"  耗时: {single_time:.4f} 秒")
    print(f"  吞吐量: {single_throughput:,.0f} tokens/second")
    print(f"  BPE 压缩率: {single_stats['compression_ratio']:.4f} bytes/token")
    print(f"  平均每 token 表示字符数: {single_stats['chars_per_token']:.4f} chars/token")
    print(f"  Token 密度: {single_stats['tokens_per_byte']:.4f} tokens/byte")

    # ====================== 批量压缩率测试 ======================
    batch_stats = tokenizer.compression_stats_batch(test_texts)

    print(f"\n批量压缩率测试:")
    print(f"  批量文本条数: {len(test_texts):,}")
    print(f"  总字符数: {batch_stats['total_chars']:,}")
    print(f"  总 UTF-8 字节数: {batch_stats['total_raw_bytes']:,}")
    print(f"  总 tokens: {batch_stats['total_tokens']:,}")
    print(f"  BPE 平均压缩率: {batch_stats['compression_ratio']:.4f} bytes/token")
    print(f"  平均每 token 表示字符数: {batch_stats['chars_per_token']:.4f} chars/token")
    print(f"  平均每 token 表示字节数: {batch_stats['bytes_per_token']:.4f} bytes/token")
    print(f"  Token 密度: {batch_stats['tokens_per_byte']:.4f} tokens/byte")

    # ====================== 批量吞吐量测试 ======================
    num_runs = 100
    total_tokens = 0
    total_time = 0.0

    print(f"\n批量吞吐量测试 ({len(test_texts)} 条文本 × {num_runs} 次运行):")

    for run in range(num_runs):
        start_time = time.perf_counter()
        batch_encoded = tokenizer.encode_batch(test_texts)
        end_time = time.perf_counter()

        run_tokens = sum(len(ids) for ids in batch_encoded)
        run_time = end_time - start_time

        total_tokens += run_tokens
        total_time += run_time

        run_throughput = run_tokens / run_time if run_time > 0 else 0

        print(
            f"  Run {run + 1}: "
            f"{run_tokens:,} tokens in {run_time:.3f}s "
            f"→ {run_throughput:,.0f} tokens/s"
        )

    avg_throughput = total_tokens / total_time if total_time > 0 else 0

    # 由于批量测试重复运行 num_runs 次，因此字符数和字节数也要乘以 num_runs
    total_chars_all_runs = total_chars * num_runs
    total_raw_bytes_all_runs = total_raw_bytes * num_runs

    avg_compression_ratio = (
        total_raw_bytes_all_runs / total_tokens if total_tokens > 0 else 0
    )

    avg_chars_per_token = (
        total_chars_all_runs / total_tokens if total_tokens > 0 else 0
    )

    avg_tokens_per_byte = (
        total_tokens / total_raw_bytes_all_runs
        if total_raw_bytes_all_runs > 0
        else 0
    )

    print(f"\n最终结果:")
    print(f"  总字符数: {total_chars_all_runs:,}")
    print(f"  总 UTF-8 字节数: {total_raw_bytes_all_runs:,}")
    print(f"  总 tokens: {total_tokens:,}")
    print(f"  总耗时: {total_time:.3f} 秒")
    print(f"  **平均吞吐量: {avg_throughput:,.0f} tokens/second**")
    print(f"  平均每条 tokens: {total_tokens / (len(test_texts) * num_runs):.1f}")
    print(f"  BPE 平均压缩率: {avg_compression_ratio:.4f} bytes/token")
    print(f"  平均每 token 表示字符数: {avg_chars_per_token:.4f} chars/token")
    print(f"  Token 密度: {avg_tokens_per_byte:.4f} tokens/byte")
    print("=" * 70)