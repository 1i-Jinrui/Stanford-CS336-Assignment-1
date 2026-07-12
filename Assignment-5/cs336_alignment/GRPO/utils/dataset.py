import json
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


# -------------------------------------------------------------#
# 加载数据集的函数
# -------------------------------------------------------------#
def load_dataset(
    data_file: str,
    data_type: str = "train",
    prompt_template: str = None,
):
    with open(data_file, "r", encoding="utf-8") as f:
        # prompt 模板直接读取全文
        if data_type == "prompt":
            return f.read()

        if data_type not in ["train", "val"]:
            raise ValueError(f"Invalid data type: {data_type}")

        text = f.read().strip()

    if not text:
        raise ValueError(f"Dataset file is empty: {data_file}")

    # ---------------------------------------------------------#
    # 读取 JSON 数组或 JSONL
    # ---------------------------------------------------------#
    if text.startswith("["):
        # 标准 JSON 数组：
        # [
        #   {...},
        #   {...}
        # ]
        raw_data = json.loads(text)
    else:
        # JSONL：
        # {...}
        # {...}
        raw_data = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    if not isinstance(raw_data, list):
        raise TypeError(
            f"Dataset must contain a list of records, "
            f"but got {type(raw_data).__name__}: {data_file}"
        )

    data = []

    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Record {index} must be a dictionary, "
                f"but got {type(item).__name__}"
            )

        if "problem" not in item:
            raise KeyError(
                f"Record {index} is missing required field 'problem'"
            )

        if "expected_answer" in item:
            answer = item["expected_answer"]
        elif "answer" in item:
            answer = item["answer"]
        else:
            raise KeyError(
                f"Record {index} is missing "
                f"'expected_answer' or 'answer'"
            )

        data.append(
            {
                "problem": str(item["problem"]),
                "answer": str(answer),
            }
        )

    return data

# -------------------------------------------------------------#
# 对 prompt 和 output 进行分词的函数
# -------------------------------------------------------------#
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    """
    对 prompt 和 output 字符串进行分词，并构造一个 mask：
    response token 的位置为 1，prompt 和 padding token 的位置为 0。

    参数：
        prompt_strs (List[str]): prompt 字符串列表。
        output_strs (List[str]): output 字符串列表。
        tokenizer (AutoTokenizer): 用于分词的 tokenizer。

    返回：
        dict[str, torch.Tensor]:
            "input_ids": 形状为 (num_prompts, max_len - 1) 的 torch.Tensor：
                分词后的 prompt 和 output 字符串，并去掉最后一个 token。
            "labels": 形状为 (num_prompts, max_len - 1) 的 torch.Tensor：
                移位后的 input_ids，也就是去掉第一个 token 的 input_ids。
            "response_mask": 形状为 (num_prompts, max_len - 1) 的 torch.Tensor：
                labels 中 response token 位置对应的 mask。
    """
    # 检查 prompts 和 outputs 的数量是否一致
    bz = len(prompt_strs)
    assert bz == len(output_strs), "Number of prompts and outputs must be the same"
    
    # 对 prompts 和 outputs 进行分词 -> dict: input_ids, attention_mask
    prompt_tokens, output_tokens = map(lambda x: tokenizer(x, return_tensors=None, padding=False, truncation=False)["input_ids"], [prompt_strs, output_strs])
    
    # 获取 prompt tokens 和 output tokens 拼接后的最大长度
    max_len = max([len(p_tokens) + len(o_tokens) for p_tokens, o_tokens in zip(prompt_tokens, output_tokens)])
    
    # 创建 input_ids、labels 和 response_mask 张量
    input_ids, labels, response_mask = map(lambda x: torch.zeros((bz, max_len - 1), dtype=x), [torch.long, torch.long, torch.bool])
    
    for i, (p_tokens, o_tokens) in enumerate(zip(prompt_tokens, output_tokens)):
        comb_tokens = torch.tensor(p_tokens + o_tokens)
        concat_len = len(comb_tokens)
        
        # 左侧不填充：0，右侧填充：max_len - concat_len
        padded_comb_tokens = F.pad(comb_tokens, (0, max_len - concat_len), 'constant', value=tokenizer.pad_token_id)
        
        input_ids[i] = padded_comb_tokens[:-1]
        labels[i] = padded_comb_tokens[1:]
        
        # 仅 labels 中 response 部分为 True：从第一个 output token 到最后一个 output token
        response_mask[i,(len(p_tokens)-1):(concat_len-1)] = True
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }