import argparse
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import sentencepiece as spm
except ImportError as exc:
    raise ImportError(
        "sentencepiece is required. Install it with: pip install sentencepiece"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer from single text or txt files."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Single text string used to train tokenizer.",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing .txt files used to train tokenizer.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search .txt files under --input_dir.",
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
    return parser.parse_args()


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
) -> Tuple[int, int]:
    if text is not None:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("--text is empty after stripping whitespace.")
        with output_path.open("w", encoding="utf-8") as writer:
            writer.write(normalized_text + "\n")
        return 1, 1

    if input_dir is None:
        raise ValueError("input_dir should not be None when text is not provided.")
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir is not a valid directory: {input_dir}")

    txt_files = list_txt_files(input_dir=input_dir, recursive=recursive)
    if not txt_files:
        raise ValueError(f"No .txt files found in directory: {input_dir}")

    line_count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for txt_file in txt_files:
            with txt_file.open("r", encoding="utf-8", errors="ignore") as reader:
                for line in reader:
                    line = line.strip()
                    if line:
                        writer.write(line + "\n")
                        line_count += 1
    if line_count == 0:
        raise ValueError(f"No non-empty lines found in directory: {input_dir}")
    return len(txt_files), line_count


def main() -> None:
    args = parse_args()

    model_prefix = Path(args.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_corpus_path = Path(temp_file.name)

    try:
        input_dir = Path(args.input_dir) if args.input_dir else None
        file_count, line_count = write_training_corpus(
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

        spm.SentencePieceTrainer.train(**train_kwargs)
    finally:
        temp_corpus_path.unlink(missing_ok=True)

    print("Training finished.")
    if args.text is not None:
        print(f"Input mode: single text ({line_count} line)")
    else:
        print(f"Input mode: txt directory ({file_count} files, {line_count} lines)")
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocab saved to: {model_prefix}.vocab")


if __name__ == "__main__":
    main()
