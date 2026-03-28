import os
import regex
from collections import defaultdict
import pickle
import time
from tqdm import tqdm



def _initialize_vocab(vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    # bytes([i])表示：先构造一个只包含一个整数 i 的列表，再把这个列表转成一个单字节内容，bytes(i)表示：创建一个长度为 i 的全 0 字节串
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_next_id = 256
    existing_vocab_values: set[bytes] = set(vocab.values())
    for st in special_tokens:
        if len(vocab) >= vocab_size:
            break
        st_bytes = bytes(st, 'utf-8')
        if st_bytes not in existing_vocab_values:
            vocab[current_next_id] = st_bytes
            current_next_id += 1
            existing_vocab_values.add(st_bytes)
    return vocab, current_next_id


def _get_token_freq(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    token_freq_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except FileNotFoundError:
        return token_freq_table
    if not text:
        return token_freq_table

    if special_tokens:
        pattern = '|'.join(map(regex.escape, special_tokens))
        chunks = regex.split(pattern, text)
    else:
        chunks = [text]
    # 定义微观分割（预分词）正则表达式
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for chunk in chunks:
        words = regex.findall(PAT, chunk)
        for word in words:
            word_bytes = word.encode('utf-8')
            bytes_tuple = tuple(bytes([x]) for x in word_bytes)
            token_freq_table[bytes_tuple] += 1
    return token_freq_table


def _get_initial_pair_counts(token_freq_table: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for token, freq in token_freq_table.items():
        for i in range(len(token) - 1):
            pair_counts[(token[i], token[i + 1])] += freq
    return pair_counts


def merge_token_sequence(token_seq: tuple[bytes, ...],
                         candidate_pair: tuple[bytes, bytes],
                         new_vocab: bytes) -> tuple[bytes, ...]:
    new_token_seq = []
    i = 0
    while i < len(token_seq):
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == candidate_pair:
            new_token_seq.append(new_vocab)
            i = i + 2
        else:
            new_token_seq.append(token_seq[i])
            i = i + 1
    return tuple(new_token_seq)


def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str],
              **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 计时
    start_time = time.time()

    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")

    if input_path is None:
        raise ValueError("input_path cannot be None")

    if not isinstance(special_tokens, list):
        raise ValueError("special_tokens must be a list")

    # 1.初始化词表
    vocab, current_next_id = _initialize_vocab(vocab_size, special_tokens)

    # 2. 读取语料并预分词，获取词频表
    token_freq_table = _get_token_freq(input_path, special_tokens)

    # 3. 初始化 token 对频率表
    pair_counts = _get_initial_pair_counts(token_freq_table)

    merges: list[tuple[bytes, bytes]] = []

    # 打印进度条
    total_merges = vocab_size - len(vocab)
    pbar = tqdm(total=total_merges, desc="Training BPE")

    # 4. BPE 核心训练循环
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        # 合并新词并添加到词表
        max_pair_freq = max(pair_counts.values())
        max_pairs = [k for k, v in pair_counts.items() if v == max_pair_freq]
        candidate_pair = max(max_pairs)

        merges.append(candidate_pair)
        new_vocab = candidate_pair[0] + candidate_pair[1]
        vocab[current_next_id] = new_vocab
        current_next_id += 1

        # 查找受此次合并影响的 token 序列
        affected_token_seqs = []
        for token_seq, freq in token_freq_table.items():
            has_pair = any(
                (token_seq[i], token_seq[i + 1]) == candidate_pair
                for i in range(len(token_seq) - 1)
            )
            if has_pair:
                affected_token_seqs.append((token_seq, freq))

        # 更新全局状态表
        # 减去旧的pair_counts计数
        for token_seq, freq in affected_token_seqs:
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            new_seq = merge_token_sequence(token_seq, candidate_pair, new_vocab)

            # 增加新合并的pair_counts计数
            for i in range(len(new_seq) - 1):
                new_pair = (new_seq[i], new_seq[i + 1])
                pair_counts[new_pair] += freq

            # 更新token_freq_table
            del token_freq_table[token_seq]
            token_freq_table[new_seq] += freq

        pbar.update(1)

    pbar.close()

    # 储存
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    end_time = time.time()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")

    return vocab, merges


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab_result, merges_result = train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 10000, special_tokens)

    print(f"Vocab size: {len(vocab_result)}")
    print(f"Merges count: {len(merges_result)}")