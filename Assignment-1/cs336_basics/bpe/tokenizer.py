import regex
from typing import Iterable, Iterator
import json
import random

PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 由于需要通过merges字典来排序，所以需要一个字典来存储merges的优先级
        # enumerate会遍历列表，同时返回索引和元素本身
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}
        # 将字节转换为token id，避免直接使用vocab字典
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        # 添加一个字典，存储：单词字符串 -> Token IDs 列表，遇到重复的词即可直接取出，不需要再BPE合并
        self.cache: dict[str, list[int]] = {}
        if self.special_tokens:
            # 按照长度降序排序，确保更长的符号（例如"<|eot|><|eot|>") 在更短的符号（例如"<|eot|>")之前被匹配
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_token_pattern = '|'.join(map(regex.escape, sorted_special_tokens))
            # 提前编译好正则对象
            self.special_regex = regex.compile(f'({special_token_pattern})')
        else:
            self.special_regex = None

    def _get_bpe_ids(self, word: str) -> list[int]:
        """
        接收字符串文本片段，直接返回 token ids 列表。结合缓存使用。
        """
        if word in self.cache:
            return self.cache[word]

        # 首先将word转换为单字节列表
        parts = [bytes([b]) for b in word.encode('utf-8')]

        while len(parts) > 1:
            # 记录所有合并对
            best_rank = float('inf')
            best_pair = None

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                # 如果 pair 在字典里，.get返回对应的排名，如果不在，返回 None
                rank = self.merges_priority_map.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # 如果没有任何 pair 在 merges 中，说明无法继续合并
            if best_pair is None:
                break

            # 应用最佳合并对
            merged_token = best_pair[0] + best_pair[1] # 这里是拼接两个字节，而非直接相加
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
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
        if not text:
            return []

        if self.special_regex:
            # 按照特殊符号分割text，保持特殊符号作为分隔符
            chunks = self.special_regex.split(text)
        else:
            chunks = [text]

        final_ids = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                # 如果chunk是特殊符号，直接编码
                final_ids.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
                # 如果chunk是普通文本，使用BPE算法处理
                # 使用finditer，防止长文本造成OOM
                # finditer 返回一个迭代器，每次返回一个match对象,
                # 常用方法有：match.group()获取匹配到的文本;match.start()获取匹配开始的位置;match.end()获取匹配结束的位置。
                for match in PAT.finditer(chunk):
                    word = match.group()

                    # 获取word的合并字节片段
                    ids = self._get_bpe_ids(word)
                    final_ids.extend(ids)

        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # iterable 是一个“可以迭代得到字符串”的对象
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        all_bytes = b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    try:
        with open("../data/owt_valid.txt", "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
            test_text = random.choice(lines) if lines else "Hello world! This is a test of the BPE tokenizer."
    except FileNotFoundError:
        test_text = "Hello world! This is a test of the BPE tokenizer. 你好，世界！12345"

    # 加载 vocab
    with open("owt_vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    vocab = {}
    for str_id, token_str in vocab_data.items():
        token_id = int(str_id)
        if token_id < 256:
            # 0-255 一定是单个字节
            vocab[token_id] = bytes([token_id])
        else:
            # 高编号的 token
            if isinstance(token_str, str):
                if token_str.startswith(r'\x') and len(token_str) == 4:
                    # 处理 JSON 中 "\\xe2" 这种转义
                    vocab[token_id] = bytes([int(token_str[2:], 16)])
                else:
                    vocab[token_id] = token_str.encode("utf-8", errors="surrogateescape")
            else:
                vocab[token_id] = token_str

    # 加载 merges
    merges = []
    with open("owt_merges.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip('\n').split(" ")
            if len(parts) >= 2:
                merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))

    # 初始化 tokenizer
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )

    # 测试编码与解码
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print("=" * 60)
    print("原始文本:", test_text)
    print("编码长度:", len(encoded))
    print("编码结果:", encoded[:100], "..." if len(encoded) > 100 else "")
    print("解码文本:", decoded)
    print("还原一致:", test_text == decoded)
    print("=" * 60)

    # 额外测试特殊 token
    special_test = "Hello world<|endoftext|>This is after special token."
    print("\n特殊 token 测试:")
    print("输入:", special_test)
    print("编码:", tokenizer.encode(special_test))
