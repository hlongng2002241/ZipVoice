"""
Download a capped sample (by character budget, not the whole dataset) from a
HuggingFace streaming dataset and save it as a local JSONL file with a "text"
field per line.

Used to pull modest EN/VI/ZH samples from the FineWeb family for
docs/plans/2026-08-28__multilingual_tts_frontend/2026-08-28__sprint_003__tokenizer_embedding_bakeoff.md,
to extend the real-corpus tokenizer comparison (so far Vietnamese-only, from
the user's own ASR transcripts) to English and Chinese as well.
"""

import argparse
import json

from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=80_000_000,
        help="Stop once this many characters have been collected.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=200_000,
        help="Safety cap in case documents are much shorter than expected.",
    )
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    ds = load_dataset(
        args.dataset, name=args.config_name, split=args.split, streaming=True
    )

    total_chars = 0
    n_docs = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get(args.text_field)
            if not text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total_chars += len(text)
            n_docs += 1
            if n_docs % 2000 == 0:
                print(f"  ... {n_docs} docs, {total_chars} chars so far")
            if total_chars >= args.max_chars or n_docs >= args.max_docs:
                break

    print(f"Done: {n_docs} docs, {total_chars} chars -> {args.output}")


if __name__ == "__main__":
    main()
