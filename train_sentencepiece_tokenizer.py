import argparse
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import sentencepiece as spm
except ImportError as exc:
    raise ImportError(
        "sentencepiece is required. Install it with: pip install sentencepiece"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer from single text or txt files."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text string used to train tokenizer.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing .txt files used to train tokenizer.",
    )
    parser.add_argument(
        "--en_dir",
        type=str,
        help="Directory containing English .txt files (used with --zh_dir).",
    )
    parser.add_argument(
        "--zh_dir",
        type=str,
        help="Directory containing Chinese .txt files (used with --en_dir).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search .txt files under --input_dir.",
    )
    parser.add_argument(
        "--sample_lines",
        type=int,
        default=0,
        help=(
            "Total sampled lines for bilingual mode. "
            "Must be > 0 when using --en_dir/--zh_dir."
        ),
    )
    parser.add_argument(
        "--en_ratio",
        type=float,
        default=0.5,
        help="English ratio for sampled lines in bilingual mode (0.0 - 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for file and line sampling.",
    )
    parser.add_argument(
        "--max_files_per_lang",
        type=int,
        default=0,
        help=(
            "Randomly select at most this many txt files per language in bilingual mode. "
            "0 means use all txt files."
        ),
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        required=True,
        help="Output model prefix, e.g. ./tokenizer/spm_zh_en .",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
        help="SentencePiece model type.",
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage. For zh/en mixed corpus, 0.9995-1.0 is common.",
    )
    parser.add_argument(
        "--input_sentence_size",
        type=int,
        default=0,
        help="Sample sentence count for training. 0 means use all sentences.",
    )
    parser.add_argument(
        "--unk_id",
        type=int,
        default=0,
        help="UNK token id.",
    )
    parser.add_argument(
        "--bos_id",
        type=int,
        default=1,
        help="BOS token id.",
    )
    parser.add_argument(
        "--eos_id",
        type=int,
        default=2,
        help="EOS token id.",
    )
    parser.add_argument(
        "--pad_id",
        type=int,
        default=-1,
        help="PAD token id. -1 means disabled.",
    )
    parser.add_argument(
        "--hard_vocab_limit",
        action="store_true",
        help="Enable strict vocab size check (default: disabled).",
    )
    args = parser.parse_args()
    validate_args(parser=parser, args=args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    mode_count = 0
    if args.text is not None:
        mode_count += 1
    if args.input_dir is not None:
        mode_count += 1

    bilingual_requested = args.en_dir is not None or args.zh_dir is not None
    if bilingual_requested:
        if args.en_dir is None or args.zh_dir is None:
            parser.error("--en_dir and --zh_dir must be provided together.")
        mode_count += 1

    if mode_count != 1:
        parser.error(
            "Choose exactly one input mode: "
            "--text OR --input_dir OR --en_dir + --zh_dir."
        )
    if args.sample_lines < 0:
        parser.error("--sample_lines must be >= 0.")
    if not (0.0 <= args.en_ratio <= 1.0):
        parser.error("--en_ratio must be in [0.0, 1.0].")
    if args.max_files_per_lang < 0:
        parser.error("--max_files_per_lang must be >= 0.")
    if bilingual_requested and args.sample_lines <= 0:
        parser.error("--sample_lines must be > 0 in bilingual mode.")


def list_txt_files(input_dir: Path, recursive: bool) -> List[Path]:
    pattern_iter: Iterable[Path]
    if recursive:
        pattern_iter = input_dir.rglob("*.txt")
    else:
        pattern_iter = input_dir.glob("*.txt")
    files = sorted(p for p in pattern_iter if p.is_file())
    return files


def write_training_corpus(
    output_path: Path,
    text: Optional[str],
    input_dir: Optional[Path],
    recursive: bool,
) -> Dict[str, int]:
    if text is not None:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("--text is empty after stripping whitespace.")
        with output_path.open("w", encoding="utf-8") as writer:
            writer.write(normalized_text + "\n")
        return {"mode": "single_text", "file_count": 1, "line_count": 1}

    if input_dir is None:
        raise ValueError("input_dir should not be None when text is not provided.")
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir is not a valid directory: {input_dir}")

    txt_files = list_txt_files(input_dir=input_dir, recursive=recursive)
    if not txt_files:
        raise ValueError(f"No .txt files found in directory: {input_dir}")

    print(f"Found {len(txt_files)} txt files. Merging corpus...", flush=True)
    file_iterable: Iterable[Path]
    if tqdm is not None:
        file_iterable = tqdm(txt_files, desc="Merging txt corpus", unit="file")
    else:
        file_iterable = txt_files

    line_count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for txt_file in file_iterable:
            with txt_file.open("r", encoding="utf-8", errors="ignore") as reader:
                for line in reader:
                    line = line.strip()
                    if line:
                        writer.write(line + "\n")
                        line_count += 1
    if line_count == 0:
        raise ValueError(f"No non-empty lines found in directory: {input_dir}")
    return {
        "mode": "single_dir",
        "file_count": len(txt_files),
        "line_count": line_count,
    }


def maybe_sample_files(
    files: List[Path],
    max_files: int,
    rng: random.Random,
    lang_name: str,
) -> List[Path]:
    if max_files <= 0 or len(files) <= max_files:
        return files
    selected = rng.sample(files, k=max_files)
    selected = sorted(selected)
    print(
        f"{lang_name}: randomly selected {len(selected)} files out of {len(files)}.",
        flush=True,
    )
    return selected


def sample_lines_from_files(
    files: List[Path],
    target_lines: int,
    rng: random.Random,
    desc: str,
) -> Tuple[List[str], int]:
    if target_lines == 0:
        return [], 0

    sampled_lines: List[str] = []
    seen_lines = 0
    file_iterable: Iterable[Path]
    if tqdm is not None:
        file_iterable = tqdm(files, desc=desc, unit="file")
    else:
        file_iterable = files

    for txt_file in file_iterable:
        with txt_file.open("r", encoding="utf-8", errors="ignore") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                seen_lines += 1
                if len(sampled_lines) < target_lines:
                    sampled_lines.append(line)
                else:
                    replace_index = rng.randrange(seen_lines)
                    if replace_index < target_lines:
                        sampled_lines[replace_index] = line

    if len(sampled_lines) < target_lines:
        raise ValueError(
            f"Not enough non-empty lines for {desc}: "
            f"required={target_lines}, found={len(sampled_lines)}."
        )
    rng.shuffle(sampled_lines)
    return sampled_lines, seen_lines


def write_bilingual_sampled_corpus(
    output_path: Path,
    en_dir: Path,
    zh_dir: Path,
    recursive: bool,
    sample_lines: int,
    en_ratio: float,
    seed: int,
    max_files_per_lang: int,
) -> Dict[str, int]:
    if not en_dir.exists() or not en_dir.is_dir():
        raise ValueError(f"en_dir is not a valid directory: {en_dir}")
    if not zh_dir.exists() or not zh_dir.is_dir():
        raise ValueError(f"zh_dir is not a valid directory: {zh_dir}")

    rng = random.Random(seed)
    en_files = list_txt_files(en_dir, recursive=recursive)
    zh_files = list_txt_files(zh_dir, recursive=recursive)
    if not en_files:
        raise ValueError(f"No .txt files found in English directory: {en_dir}")
    if not zh_files:
        raise ValueError(f"No .txt files found in Chinese directory: {zh_dir}")

    en_files = maybe_sample_files(
        files=en_files,
        max_files=max_files_per_lang,
        rng=rng,
        lang_name="English",
    )
    zh_files = maybe_sample_files(
        files=zh_files,
        max_files=max_files_per_lang,
        rng=rng,
        lang_name="Chinese",
    )

    en_target = int(round(sample_lines * en_ratio))
    zh_target = sample_lines - en_target
    print(
        f"Bilingual sampling targets -> en: {en_target}, zh: {zh_target}, total: {sample_lines}",
        flush=True,
    )
    en_samples, en_seen = sample_lines_from_files(
        files=en_files,
        target_lines=en_target,
        rng=rng,
        desc="Sampling EN lines",
    )
    zh_samples, zh_seen = sample_lines_from_files(
        files=zh_files,
        target_lines=zh_target,
        rng=rng,
        desc="Sampling ZH lines",
    )

    merged_samples = en_samples + zh_samples
    rng.shuffle(merged_samples)
    with output_path.open("w", encoding="utf-8") as writer:
        for line in merged_samples:
            writer.write(line + "\n")

    return {
        "mode": "bilingual_sampled",
        "file_count_en": len(en_files),
        "file_count_zh": len(zh_files),
        "line_count_en": en_target,
        "line_count_zh": zh_target,
        "line_count": len(merged_samples),
        "seen_lines_en": en_seen,
        "seen_lines_zh": zh_seen,
    }


def main() -> None:
    args = parse_args()

    model_prefix = Path(args.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_corpus_path = Path(temp_file.name)

    try:
        if args.en_dir is not None and args.zh_dir is not None:
            corpus_stats = write_bilingual_sampled_corpus(
                output_path=temp_corpus_path,
                en_dir=Path(args.en_dir),
                zh_dir=Path(args.zh_dir),
                recursive=args.recursive,
                sample_lines=args.sample_lines,
                en_ratio=args.en_ratio,
                seed=args.seed,
                max_files_per_lang=args.max_files_per_lang,
            )
        else:
            input_dir = Path(args.input_dir) if args.input_dir else None
            corpus_stats = write_training_corpus(
                output_path=temp_corpus_path,
                text=args.text,
                input_dir=input_dir,
                recursive=args.recursive,
            )

        train_kwargs = {
            "input": str(temp_corpus_path),
            "model_prefix": str(model_prefix),
            "vocab_size": args.vocab_size,
            "model_type": args.model_type,
            "character_coverage": args.character_coverage,
            "unk_id": args.unk_id,
            "bos_id": args.bos_id,
            "eos_id": args.eos_id,
            "pad_id": args.pad_id,
            "hard_vocab_limit": args.hard_vocab_limit,
        }
        if args.input_sentence_size > 0:
            train_kwargs["input_sentence_size"] = args.input_sentence_size
            train_kwargs["shuffle_input_sentence"] = True

        print("Starting SentencePiece training...", flush=True)
        start_time = time.perf_counter()
        spm.SentencePieceTrainer.train(**train_kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"SentencePiece training completed in {elapsed:.2f} seconds.", flush=True)
    finally:
        temp_corpus_path.unlink(missing_ok=True)

    print("Training finished.")
    if corpus_stats["mode"] == "single_text":
        print(f"Input mode: single text ({corpus_stats['line_count']} line)")
    elif corpus_stats["mode"] == "single_dir":
        print(
            "Input mode: txt directory "
            f"({corpus_stats['file_count']} files, {corpus_stats['line_count']} lines)"
        )
    else:
        print(
            "Input mode: bilingual sampled dirs "
            f"(en files={corpus_stats['file_count_en']}, "
            f"zh files={corpus_stats['file_count_zh']}, "
            f"en lines={corpus_stats['line_count_en']}, "
            f"zh lines={corpus_stats['line_count_zh']})"
        )
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocab saved to: {model_prefix}.vocab")


if __name__ == "__main__":
    main()
