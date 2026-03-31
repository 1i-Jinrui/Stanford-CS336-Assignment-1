import os
import regex
from collections import defaultdict
import pickle
import time
from tqdm import tqdm
import cProfile
import pstats
import io



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


# def _get_token_freq(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
#     token_freq_table: dict[tuple[bytes, ...], int] = defaultdict(int)
#     try:
#         with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
#             text = f.read()
#     except FileNotFoundError:
#         return token_freq_table
#     if not text:
#         return token_freq_table
#
#     if special_tokens:
#         pattern = '|'.join(map(regex.escape, special_tokens))
#         chunks = regex.split(pattern, text)
#     else:
#         chunks = [text]
#     # 定义微观分割（预分词）正则表达式
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#
#     for chunk in chunks:
#         words = regex.findall(PAT, chunk)
#         for word in words:
#             word_bytes = word.encode('utf-8')
#             bytes_tuple = tuple(bytes([x]) for x in word_bytes)
#             token_freq_table[bytes_tuple] += 1
#     return token_freq_table

# 更新版，加入pattern预编译以及流式读取text
def _get_token_freq(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    token_freq_table: dict[tuple[bytes, ...], int] = defaultdict(int)

    # 提前编译正则表达式，大幅提升匹配速度
    PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 获取文件总大小，用于显示读取进度
    try:
        total_size = os.path.getsize(input_path)
    except FileNotFoundError:
        return token_freq_table

    if special_tokens:
        # 编译 special tokens 的正则模式
        st_pattern = regex.compile('|'.join(map(regex.escape, special_tokens)))
    else:
        st_pattern = None

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        # 使用文件大小来驱动进度条
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Reading & Pre-tokenizing") as pbar:
            for line in f:
                line_bytes_len = len(line.encode('utf-8'))

                if st_pattern:
                    chunks = st_pattern.split(line)
                else:
                    chunks = [line]

                for chunk in chunks:
                    if not chunk:
                        continue
                    # 使用预编译的正则进行查找
                    words = PAT.findall(chunk)
                    for word in words:
                        word_bytes = word.encode('utf-8')
                        bytes_tuple = tuple(bytes([x]) for x in word_bytes)
                        token_freq_table[bytes_tuple] += 1

                # 更新读取进度
                pbar.update(line_bytes_len)

    return token_freq_table

# def _get_initial_pair_counts(token_freq_table: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
#     pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
#     for token, freq in token_freq_table.items():
#         for i in range(len(token) - 1):
#             pair_counts[(token[i], token[i + 1])] += freq
#     return pair_counts


# 更新版,同时获得pair_to_words
def _get_initial_pair_counts_and_idx(token_freq_table: dict[tuple[bytes, ...], int]) -> (
        tuple)[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for token_seq, freq in token_freq_table.items():
        for i in range(len(token_seq) - 1):
            pair_counts[(token_seq[i], token_seq[i + 1])] += freq
            pair_to_words[(token_seq[i], token_seq[i + 1])].add(token_seq)
    return pair_counts, pair_to_words


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
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    vocab, current_next_id = _initialize_vocab(vocab_size, sorted_special_tokens)

    # 2. 读取语料并预分词，获取词频表
    token_freq_table = _get_token_freq(input_path, sorted_special_tokens)

    # 3. 初始化 token 对频率表
    pair_counts, pair_to_words = _get_initial_pair_counts_and_idx(token_freq_table)

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
        # affected_token_seqs = []
        # for token_seq, freq in token_freq_table.items():
        #     has_pair = any(
        #         (token_seq[i], token_seq[i + 1]) == candidate_pair
        #         for i in range(len(token_seq) - 1)
        #     )
        #     if has_pair:
        #         affected_token_seqs.append((token_seq, freq))
        affected_token_seqs = list(pair_to_words[candidate_pair])

        # 更新全局状态表
        # 减去旧的pair_counts计数
        for token_seq in affected_token_seqs:
            freq = token_freq_table[token_seq]
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                # 从倒排索引中划掉旧序列的名字
                if token_seq in pair_to_words[pair]:
                    pair_to_words[pair].remove(token_seq)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

            new_seq = merge_token_sequence(token_seq, candidate_pair, new_vocab)

            # 增加新合并的pair_counts计数
            for i in range(len(new_seq) - 1):
                new_pair = (new_seq[i], new_seq[i + 1])
                pair_counts[new_pair] += freq
                pair_to_words[new_pair].add(new_seq)

            # 更新token_freq_table
            del token_freq_table[token_seq]
            token_freq_table[new_seq] += freq

        pbar.update(1)

    pbar.close()

    # ================= 保存为人类可读的格式 =================

    # 1. 保存 Vocab 为 JSON 文件
    import json
    readable_vocab = {}
    for idx, token_bytes in vocab.items():
        # 将 bytes 安全地解码为字符串，遇到非法 utf-8 序列用 \x.. 替代
        safe_string = token_bytes.decode('utf-8', errors='backslashreplace')
        readable_vocab[str(idx)] = safe_string

    with open("vocab.json", "w", encoding="utf-8") as f:
        # indent=2 使 JSON 文件具有缩进和换行，ensure_ascii=False 保证正常显示中文/特殊字符
        json.dump(readable_vocab, f, indent=2, ensure_ascii=False)

    # 2. 保存 Merges 为 TXT 文件
    with open("merges.txt", "w", encoding="utf-8") as f:
        # 第一行通常写一个版本号标识，Hugging Face 的 merges.txt 通常包含这行
        f.write("# version: 0.2\n")
        for pair in merges:
            part1 = pair[0].decode('utf-8', errors='backslashreplace')
            part2 = pair[1].decode('utf-8', errors='backslashreplace')
            # 使用空格分隔合并的两个部分
            f.write(f"{part1} {part2}\n")

    # ========================================================

    end_time = time.time()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")

    return vocab, merges


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab_result, merges_result = train_bpe("../data/owt_train.txt", 32000, special_tokens)

    print(f"Vocab size: {len(vocab_result)}")
    print(f"Merges count: {len(merges_result)}")
