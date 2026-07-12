from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import fasttext
import numpy as np
# Import only the classes needed by the pipeline.
# WarcHeader is intentionally not imported because some FastWARC builds
# do not expose it, while HeaderMap accepts normal string keys.
try:
    from fastwarc import ArchiveIterator, WarcRecordType
except ImportError:
    from fastwarc.warc import ArchiveIterator, WarcRecordType


# ============================================================
# 1. Configuration
# ============================================================


@dataclass(frozen=True)
class PipelineConfig:
    language_model_path: str
    nsfw_model_path: str
    toxic_model_path: str
    quality_model_path: str

    language_threshold: float = 0.80
    nsfw_threshold: float = 0.80
    toxic_threshold: float = 0.80
    quality_threshold: float = 0.50

    # IMPORTANT: set these to the actual labels emitted by your models.
    nsfw_positive_labels: tuple[str, ...] = ("nsfw", "porn", "adult")
    toxic_positive_labels: tuple[str, ...] = (
        "toxic",
        "toxicity",
        "hate",
        "hatespeech",
        "hate_speech",
    )
    # Quality classifier labels: __label__hq = high quality, __label__lq = low quality
    quality_keep_labels: tuple[str, ...] = ("hq",)

    minhash_num_hashes: int = 128
    minhash_num_bands: int = 16
    minhash_ngram_words: int = 5
    minhash_jaccard_threshold: float = 0.80
    random_seed: int = 42


# ============================================================
# 2. FastText model bundle: load once, predict many times
# ============================================================


class ModelBundle:
    def __init__(self, config: PipelineConfig) -> None:
        self.language = fasttext.load_model(config.language_model_path)
        self.nsfw = fasttext.load_model(config.nsfw_model_path)
        self.toxic = fasttext.load_model(config.toxic_model_path)
        self.quality = fasttext.load_model(config.quality_model_path)

    @staticmethod
    def _predict(model, text: str) -> tuple[str, float]:
        # fastText predict expects one logical line for a single-example call.
        one_line = re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()
        if not one_line:
            return "", 0.0

        labels, scores = model.predict(one_line, k=1)
        label = labels[0].replace("__label__", "")
        score = float(scores[0])
        return label, score

    def identify_language(self, text: str) -> tuple[str, float]:
        return self._predict(self.language, text)

    def classify_nsfw(self, text: str) -> tuple[str, float]:
        return self._predict(self.nsfw, text)

    def classify_toxic(self, text: str) -> tuple[str, float]:
        return self._predict(self.toxic, text)

    def classify_quality(self, text: str) -> tuple[str, float]:
        return self._predict(self.quality, text)


# ============================================================
# 3. WET reading
# ============================================================


@dataclass
class WetDocument:
    doc_id: str
    url: str
    text: str


def iter_wet_documents(wet_path: os.PathLike[str] | str) -> Iterator[WetDocument]:
    """Yield conversion records from a Common Crawl WET file.

    WET files already contain extracted plain text, so HTML extraction is not used here.
    """
    wet_path = str(wet_path)

    iterator = ArchiveIterator(
        wet_path,
        record_types=WarcRecordType.conversion,
        parse_http=False,
    )

    for index, record in enumerate(iterator):
        payload = record.reader.read()
        if not payload:
            continue

        text = payload.decode("utf-8", errors="replace").strip()
        if not text:
            continue

        url = record.headers.get("WARC-Target-URI") or ""
        doc_id = f"doc_{index:09d}"
        yield WetDocument(doc_id=doc_id, url=str(url), text=text)


# ============================================================
# 4. Gopher rules
# ============================================================


def run_gopher_quality_filter(text: str) -> bool:
    words = text.split()
    if not words:
        return False

    word_count = len(words)
    if not (50 <= word_count <= 100_000):
        return False

    mean_word_length = sum(len(word) for word in words) / word_count
    if not (3.0 <= mean_word_length <= 10.0):
        return False

    lines = text.splitlines() or [text]
    ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
    if ellipsis_lines / len(lines) > 0.30:
        return False

    alphabetic_words = sum(1 for word in words if any(ch.isalpha() for ch in word))
    if alphabetic_words / word_count < 0.80:
        return False

    return True


# ============================================================
# 5. PII masking
# ============================================================


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

OCTET = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
IP_PATTERN = re.compile(rf"\b{OCTET}\.{OCTET}\.{OCTET}\.{OCTET}\b")

PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)
    (?:\+?1[-.\s]?)?
    (?:
        \(\d{3}\)[-.\s]?
        |
        \d{3}[-.\s]?
    )
    \d{3}[-.\s]?
    \d{4}
    (?:\s?(?:ext|x|extension)\.?\s?\d{2,5})?
    (?!\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def run_mask_emails(text: str) -> tuple[str, int]:
    return EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||", text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_PATTERN.subn("|||PHONE_NUMBER|||", text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return IP_PATTERN.subn("|||IP_ADDRESS|||", text)


def mask_all_pii(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()

    text, count = run_mask_emails(text)
    counts["emails_masked"] += count

    text, count = run_mask_phone_numbers(text)
    counts["phones_masked"] += count

    text, count = run_mask_ips(text)
    counts["ips_masked"] += count

    return text, counts


# ============================================================
# 6. Helpers for labels and files
# ============================================================


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def label_in(label: str, candidates: Sequence[str]) -> bool:
    normalized = normalize_label(label)
    return normalized in {normalize_label(x) for x in candidates}


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_text_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.txt") if p.is_file())


# ============================================================
# 7. Filtering stage
# ============================================================


def filter_wet_to_document_files(
    wet_path: Path,
    output_dir: Path,
    metadata_path: Path,
    models: ModelBundle,
    config: PipelineConfig,
) -> Counter[str]:
    reset_directory(output_dir)
    stats: Counter[str] = Counter()

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for doc in iter_wet_documents(wet_path):
            stats["documents_seen"] += 1

            text = doc.text.strip()
            if not text:
                stats["rejected_empty"] += 1
                continue

            lang_label, lang_score = models.identify_language(text)
            if lang_label != "en" or lang_score < config.language_threshold:
                stats["rejected_language"] += 1
                continue
            stats["passed_language"] += 1

            if not run_gopher_quality_filter(text):
                stats["rejected_gopher"] += 1
                continue
            stats["passed_gopher"] += 1

            nsfw_label, nsfw_score = models.classify_nsfw(text)
            if (
                label_in(nsfw_label, config.nsfw_positive_labels)
                and nsfw_score >= config.nsfw_threshold
            ):
                stats["rejected_nsfw"] += 1
                continue
            stats["passed_nsfw"] += 1

            toxic_label, toxic_score = models.classify_toxic(text)
            if (
                label_in(toxic_label, config.toxic_positive_labels)
                and toxic_score >= config.toxic_threshold
            ):
                stats["rejected_toxic"] += 1
                continue
            stats["passed_toxic"] += 1

            quality_label, quality_score = models.classify_quality(text)
            if not (
                label_in(quality_label, config.quality_keep_labels)
                and quality_score >= config.quality_threshold
            ):
                stats["rejected_quality"] += 1
                continue
            stats["passed_quality"] += 1

            masked_text, pii_counts = mask_all_pii(text)
            stats.update(pii_counts)

            document_path = output_dir / f"{doc.doc_id}.txt"
            document_path.write_text(masked_text.strip() + "\n", encoding="utf-8")

            metadata = {
                "doc_id": doc.doc_id,
                "url": doc.url,
                "language": {"label": lang_label, "score": lang_score},
                "nsfw": {"label": nsfw_label, "score": nsfw_score},
                "toxic": {"label": toxic_label, "score": toxic_score},
                "quality": {"label": quality_label, "score": quality_score},
                "pii": dict(pii_counts),
            }
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            stats["kept_after_filters"] += 1

    return stats


# ============================================================
# 8. Exact line deduplication
# ============================================================


def stable_line_hash(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()


def run_exact_line_deduplication(
    input_files: Sequence[os.PathLike[str] | str],
    output_directory: os.PathLike[str] | str,
) -> dict[str, int]:
    output_dir = Path(output_directory)
    reset_directory(output_dir)

    line_counts: Counter[bytes] = Counter()
    total_lines = 0

    # Pass 1: global exact line frequencies.
    for file_path in input_files:
        with Path(file_path).open("r", encoding="utf-8") as infile:
            for line in infile:
                line_counts[stable_line_hash(line)] += 1
                total_lines += 1

    kept_lines = 0
    removed_lines = 0
    empty_documents_after_dedup = 0

    # Pass 2: retain only lines whose global corpus frequency is exactly 1.
    for file_path in input_files:
        source = Path(file_path)
        destination = output_dir / source.name
        doc_kept_lines = 0

        with source.open("r", encoding="utf-8") as infile, destination.open(
            "w", encoding="utf-8"
        ) as outfile:
            for line in infile:
                if line_counts[stable_line_hash(line)] == 1:
                    outfile.write(line)
                    kept_lines += 1
                    doc_kept_lines += 1
                else:
                    removed_lines += 1

        if doc_kept_lines == 0 or not destination.read_text(encoding="utf-8").strip():
            destination.unlink(missing_ok=True)
            empty_documents_after_dedup += 1

    return {
        "exact_total_lines": total_lines,
        "exact_kept_lines": kept_lines,
        "exact_removed_lines": removed_lines,
        "exact_empty_documents_removed": empty_documents_after_dedup,
    }


# ============================================================
# 9. MinHash + LSH document deduplication
# ============================================================


def normalize_text_for_dedup(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # Replace punctuation/symbols with spaces so tokens do not concatenate.
    text = "".join(
        ch if (ch.isalnum() or ch == "_") else " "
        for ch in text
    )
    return re.sub(r"\s+", " ", text).strip()


def get_word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")

    words = text.split()
    if not words:
        return set()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def stable_shingle_hash(shingle: tuple[str, ...], prime: int) -> int:
    payload = "\x1f".join(shingle).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % prime


def make_minhash_coefficients(
    num_hashes: int,
    seed: int,
    prime: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, prime, size=num_hashes, dtype=np.uint64)
    b = rng.integers(0, prime, size=num_hashes, dtype=np.uint64)
    return a, b


def compute_minhash_signature(
    shingles: set[tuple[str, ...]],
    a: np.ndarray,
    b: np.ndarray,
    prime: int,
) -> np.ndarray:
    if not shingles:
        return np.full(len(a), prime, dtype=np.uint64)

    signature = np.full(len(a), prime, dtype=np.uint64)
    prime_u64 = np.uint64(prime)

    for shingle in shingles:
        base_hash = np.uint64(stable_shingle_hash(shingle, prime))
        values = (a * base_hash + b) % prime_u64
        signature = np.minimum(signature, values)

    return signature


def true_jaccard(
    left: set[tuple[str, ...]],
    right: set[tuple[str, ...]],
) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class UnionFind:
    def __init__(self, items: Iterable[Path]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: Path) -> Path:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: Path, right: Path) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return

        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]

        if rank_left < rank_right:
            self.parent[root_left] = root_right
        elif rank_left > rank_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def run_minhash_deduplication(
    input_files: Sequence[os.PathLike[str] | str],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike[str] | str,
    random_seed: int = 42,
) -> dict[str, int]:
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be divisible by num_bands")
    if not (0.0 <= jaccard_threshold <= 1.0):
        raise ValueError("jaccard_threshold must be in [0, 1]")

    output_dir = Path(output_directory)
    reset_directory(output_dir)

    paths = [Path(p) for p in input_files]
    if not paths:
        return {
            "minhash_input_documents": 0,
            "minhash_candidate_pairs": 0,
            "minhash_duplicate_pairs": 0,
            "minhash_removed_documents": 0,
            "minhash_kept_documents": 0,
        }

    # Largest 32-bit prime. This keeps affine products inside uint64 range.
    prime = 4_294_967_291
    a, b = make_minhash_coefficients(num_hashes, random_seed, prime)

    doc_ngrams: dict[Path, set[tuple[str, ...]]] = {}
    signatures: dict[Path, np.ndarray] = {}

    for path in paths:
        text = path.read_text(encoding="utf-8")
        normalized = normalize_text_for_dedup(text)
        shingles = get_word_ngrams(normalized, ngrams)
        doc_ngrams[path] = shingles
        signatures[path] = compute_minhash_signature(shingles, a, b, prime)

    rows_per_band = num_hashes // num_bands
    candidate_pairs: set[tuple[Path, Path]] = set()

    for band_index in range(num_bands):
        start = band_index * rows_per_band
        end = start + rows_per_band
        buckets: defaultdict[bytes, list[Path]] = defaultdict(list)

        for path, signature in signatures.items():
            band_key = signature[start:end].tobytes()
            buckets[band_key].append(path)

        for bucket_docs in buckets.values():
            if len(bucket_docs) < 2:
                continue
            ordered = sorted(bucket_docs)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    candidate_pairs.add((ordered[i], ordered[j]))

    union_find = UnionFind(paths)
    duplicate_pairs = 0

    # Assignment requirement: verify candidates with TRUE n-gram Jaccard.
    for left, right in candidate_pairs:
        similarity = true_jaccard(doc_ngrams[left], doc_ngrams[right])
        if similarity >= jaccard_threshold:
            union_find.union(left, right)
            duplicate_pairs += 1

    clusters: defaultdict[Path, list[Path]] = defaultdict(list)
    for path in paths:
        clusters[union_find.find(path)].append(path)

    rng = random.Random(random_seed)
    files_to_keep: list[Path] = []
    for cluster in clusters.values():
        if len(cluster) == 1:
            files_to_keep.append(cluster[0])
        else:
            files_to_keep.append(rng.choice(sorted(cluster)))

    for source in files_to_keep:
        shutil.copy2(source, output_dir / source.name)

    return {
        "minhash_input_documents": len(paths),
        "minhash_candidate_pairs": len(candidate_pairs),
        "minhash_duplicate_pairs": duplicate_pairs,
        "minhash_removed_documents": len(paths) - len(files_to_keep),
        "minhash_kept_documents": len(files_to_keep),
    }


# ============================================================
# 10. Final corpus writer
# ============================================================


def write_final_training_corpus(
    input_directory: Path,
    output_path: Path,
) -> int:
    """Write one document per line for downstream GPT-2 tokenization.

    Internal newlines are collapsed to spaces. The tokenizer stage can then append one EOS
    token per line/document.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as outfile:
        for path in list_text_files(input_directory):
            text = path.read_text(encoding="utf-8")
            one_line = re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()
            if not one_line:
                continue
            outfile.write(one_line + "\n")
            count += 1

    return count


# ============================================================
# 11. End-to-end pipeline
# ============================================================


def run_pipeline(
    wet_path: Path,
    work_dir: Path,
    final_output_path: Path,
    config: PipelineConfig,
) -> dict[str, int | float | str | dict]:
    if not wet_path.exists():
        raise FileNotFoundError(f"WET file not found: {wet_path}")

    work_dir.mkdir(parents=True, exist_ok=True)

    filtered_dir = work_dir / "01_filtered_docs"
    exact_dir = work_dir / "02_exact_dedup_docs"
    minhash_dir = work_dir / "03_minhash_dedup_docs"
    metadata_path = work_dir / "document_metadata.jsonl"
    stats_path = work_dir / "pipeline_stats.json"

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    print("[1/5] Loading fastText models...")
    stage_start = time.perf_counter()
    models = ModelBundle(config)
    timings["load_models_seconds"] = time.perf_counter() - stage_start

    print("[2/5] Reading WET and applying filters + PII masking...")
    stage_start = time.perf_counter()
    filter_stats = filter_wet_to_document_files(
        wet_path=wet_path,
        output_dir=filtered_dir,
        metadata_path=metadata_path,
        models=models,
        config=config,
    )
    timings["filter_seconds"] = time.perf_counter() - stage_start

    filtered_files = list_text_files(filtered_dir)
    print(f"      kept after filters: {len(filtered_files)} documents")

    print("[3/5] Exact line deduplication...")
    stage_start = time.perf_counter()
    exact_stats = run_exact_line_deduplication(filtered_files, exact_dir)
    timings["exact_dedup_seconds"] = time.perf_counter() - stage_start
    exact_files = list_text_files(exact_dir)
    print(f"      documents after exact dedup: {len(exact_files)}")

    print("[4/5] MinHash + LSH document deduplication...")
    stage_start = time.perf_counter()
    minhash_stats = run_minhash_deduplication(
        input_files=exact_files,
        num_hashes=config.minhash_num_hashes,
        num_bands=config.minhash_num_bands,
        ngrams=config.minhash_ngram_words,
        jaccard_threshold=config.minhash_jaccard_threshold,
        output_directory=minhash_dir,
        random_seed=config.random_seed,
    )
    timings["minhash_dedup_seconds"] = time.perf_counter() - stage_start

    print("[5/5] Writing final training corpus...")
    stage_start = time.perf_counter()
    final_document_count = write_final_training_corpus(minhash_dir, final_output_path)
    timings["write_final_seconds"] = time.perf_counter() - stage_start
    timings["total_seconds"] = time.perf_counter() - total_start

    seen = int(filter_stats.get("documents_seen", 0))
    rejected_keys = [
        "rejected_empty",
        "rejected_language",
        "rejected_gopher",
        "rejected_nsfw",
        "rejected_toxic",
        "rejected_quality",
    ]
    rejection_breakdown = {
        key: {
            "count": int(filter_stats.get(key, 0)),
            "fraction_of_seen": (float(filter_stats.get(key, 0)) / seen if seen else 0.0),
        }
        for key in rejected_keys
    }

    stats: dict[str, int | float | str | dict] = {
        **dict(filter_stats),
        **exact_stats,
        **minhash_stats,
        "final_documents": final_document_count,
        "wet_path": str(wet_path),
        "final_output_path": str(final_output_path),
        "rejection_breakdown": rejection_breakdown,
        "timings": timings,
        "config": asdict(config),
    }

    with stats_path.open("w", encoding="utf-8") as outfile:
        json.dump(stats, outfile, ensure_ascii=False, indent=2)

    print("\nPipeline complete.")
    print(f"Final corpus: {final_output_path}")
    print(f"Stats:        {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    return stats


# ============================================================
# 12. CLI
# ============================================================


def parse_csv_labels(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter one Common Crawl WET file into a final LM training corpus."
    )

    parser.add_argument("--wet", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("pipeline_work"))
    parser.add_argument("--output", type=Path, default=Path("final_training_corpus.txt"))

    parser.add_argument(
        "--language-model",
        default="../local-shared-data/classifiers/lid.176.bin",
    )
    parser.add_argument(
        "--nsfw-model",
        default="../local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
    )
    parser.add_argument(
        "--toxic-model",
        default="../local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
    )
    parser.add_argument(
        "--quality-model",
        default="quality_classifier.bin",
    )

    parser.add_argument("--language-threshold", type=float, default=0.80)
    parser.add_argument("--nsfw-threshold", type=float, default=0.80)
    parser.add_argument("--toxic-threshold", type=float, default=0.80)
    parser.add_argument("--quality-threshold", type=float, default=0.50)

    parser.add_argument(
        "--nsfw-positive-labels",
        type=parse_csv_labels,
        default=("nsfw", "porn", "adult"),
    )
    parser.add_argument(
        "--toxic-positive-labels",
        type=parse_csv_labels,
        default=("toxic", "toxicity", "hate", "hatespeech", "hate_speech"),
    )
    parser.add_argument(
        "--quality-keep-labels",
        type=parse_csv_labels,
        default=("hq",),
    )

    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument("--num-bands", type=int, default=16)
    parser.add_argument("--ngram-words", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config = PipelineConfig(
        language_model_path=args.language_model,
        nsfw_model_path=args.nsfw_model,
        toxic_model_path=args.toxic_model,
        quality_model_path=args.quality_model,
        language_threshold=args.language_threshold,
        nsfw_threshold=args.nsfw_threshold,
        toxic_threshold=args.toxic_threshold,
        quality_threshold=args.quality_threshold,
        nsfw_positive_labels=tuple(args.nsfw_positive_labels),
        toxic_positive_labels=tuple(args.toxic_positive_labels),
        quality_keep_labels=tuple(args.quality_keep_labels),
        minhash_num_hashes=args.num_hashes,
        minhash_num_bands=args.num_bands,
        minhash_ngram_words=args.ngram_words,
        minhash_jaccard_threshold=args.jaccard_threshold,
        random_seed=args.seed,
    )

    run_pipeline(
        wet_path=args.wet,
        work_dir=args.work_dir,
        final_output_path=args.output,
        config=config,
    )


if __name__ == "__main__":
    main()
