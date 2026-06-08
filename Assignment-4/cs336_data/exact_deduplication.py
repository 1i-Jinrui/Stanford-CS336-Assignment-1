import os
import hashlib
from collections import Counter

def hash_line(line: str) -> str:
    return hashlib.md5(line.encode("utf-8")).hexdigest()

def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    lines_count = Counter()

    # 第一遍：统计每一行 hash 出现次数
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line_hash = hash_line(line)
                lines_count[line_hash] += 1

    os.makedirs(output_directory, exist_ok=True)

    # 第二遍：只保留全语料库中唯一出现的行
    for input_path in input_files:
        file_name = os.path.basename(input_path)
        output_path = os.path.join(output_directory, file_name)

        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                line_hash = hash_line(line)
                if lines_count[line_hash] == 1:
                    outfile.write(line)

    return None