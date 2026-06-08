import json
from datasets import load_dataset

MAX_PAGES = 100_000
OUT = "enwiki_100k.jsonl"

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

written = 0

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for ex in ds:
        if written >= MAX_PAGES:
            break

        text = ex["text"].strip()

        # 保证一行一篇文章
        text = " ".join(text.split())

        if not text:
            continue

        # 每行直接写一个 JSON 字符串，不要 {"text": ...}
        f.write(json.dumps(text, ensure_ascii=False) + "\n")

        written += 1

        if written % 10000 == 0:
            print(f"wrote {written} articles")

print(f"done: wrote {written} articles to {OUT}")