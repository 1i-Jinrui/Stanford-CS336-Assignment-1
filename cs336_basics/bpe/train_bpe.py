import os
import regex
from collections import defaultdict, Counter
import pickle
import time
from tqdm import tqdm
import cProfile
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import BinaryIO


def _initialize_vocab(vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    # bytes([i])表示：先构造一个只包含一个整数 i 的列表，再把这个列表转成一个单字节内容，bytes(i)表示：创建一个长度为 i 的全 0 字节串
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_next_id = 256
    existing_vocab_values: set[bytes] = set(vocab.values())
    for st in special_tokens:
        if len(vocab) >= vocab_size:
            break
        st_bytes = st.encode('utf-8')
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


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    寻找文件安全的字节切分边界，保证不切断 Token
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    # 用指针获取文件大小
    # 接收两个参数（偏移量，基准位置）：把指针移到文件最末尾
    file.seek(0, os.SEEK_END)

    # 获取当前指针的位置（既然在最末尾，这个位置的值就是文件的总字节数）
    file_size = file.tell()

    # 把指针重新移回文件开头
    file.seek(0)

    if file_size == 0:
        return [0, 0]

    chunk_size = file_size // desired_num_chunks

    # 初始化边界猜测位置
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次预读 4KB 寻找边界

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到特殊的 token，将边界定在这个 token 的起始位置
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 可能会找到两个相同的边界，需要去重
    return sorted(set(chunk_boundaries))


def _process_chunk_by_offset(start: int, end: int, input_path: str, special_tokens: list[str]) -> Counter:
    """
    子进程执行的预分词任务：根据起始和结束偏移量，自己去硬盘读取数据
    """
    local_freq = Counter()

    # 子进程独立打开文件并精确读取
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode('utf-8', errors='ignore')

    # 将 Windows 的 CRLF (\r\n) 换行符统一规范化为 LF (\n)
    text = text.replace('\r', '')
    if not text:
        return local_freq

    # 子进程内编译正则
    PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    st_pattern = regex.compile('|'.join(map(regex.escape, special_tokens))) if special_tokens else None

    chunks = st_pattern.split(text) if st_pattern else [text]
    for chunk in chunks:
        if not chunk: continue
        words = PAT.findall(chunk)
        local_freq.update(
            tuple(bytes([x]) for x in word.encode('utf-8'))
            for word in words
        )

    return local_freq


def _get_token_freq_parallel(input_path: str, special_tokens: list[str], num_workers: int = None):
    """
    多进程调度：只负责计算边界并下发坐标，不传输数据
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    total_freq_table = Counter()

    try:
        total_file_size = os.path.getsize(input_path)
    except FileNotFoundError:
        return total_freq_table

    # 确定用于切分的 token（如果是纯文本且无特殊 token，退化为按换行符切分以保证安全）
    split_token = b"<|endoftext|>" if special_tokens else b"\n"

    # 多切分一些 chunk 以实现更好的负载均衡，防止OOM
    desired_chunks = num_workers * 24

    # 计算块的边界
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunks, split_token)

    # 派发坐标任务
    worker_func = partial(_process_chunk_by_offset, input_path=input_path, special_tokens=special_tokens)

    with ProcessPoolExecutor(max_workers=num_workers) as executor, \
            tqdm(total=total_file_size, unit='B', unit_scale=True, desc="Pre-tokenizing") as pbar:

        futures = {}

        # 提交所有的 (start, end) 任务给子进程
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            future = executor.submit(worker_func, start, end)
            futures[future] = end - start

        # 收集结果
        for future in as_completed(futures):
            chunk_bytes = futures[future]
            chunk_table = future.result()

            total_freq_table.update(chunk_table)
            pbar.update(chunk_bytes)

    return total_freq_table


# def _get_initial_pair_counts(token_freq_table: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
#     pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
#     for token, freq in token_freq_table.items():
#         for i in range(len(token) - 1):
#             pair_counts[(token[i], token[i + 1])] += freq
#     return pair_counts


# 更新版,同时获得pair_to_words
def _get_initial_pair_counts_and_idx(token_freq_table: dict[tuple[bytes, ...], int])\
        -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
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
    token_freq_table = _get_token_freq_parallel(input_path, sorted_special_tokens, num_workers=4)

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

        # 更新全局状态表 针对每一条受影响的token序列，删除对应状态，合成新序列，添加新序列的状态
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

    # ================= 保存为可读的格式 =================

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
