"""
Analyze token-frequency statistics of a LanguageModelTokenizer candidate over
a real training-manifest JSONL (one JSON object per line, with a text field).

Sprint 003 deliverable of
docs/plans/2026-08-28__multilingual_tts_frontend/2026-08-28__sprint_003__tokenizer_embedding_bakeoff.md,
run against real Vietnamese data once it became available (see
docs/plans/2026-08-28__multilingual_tts_frontend/eval_sets/README.md for the
corpus-access blocker this resolves).

Reports (not just raw counts):
  - tokens/char fragmentation on real data (extends the hand-written-sentence
    estimate from temp/tokenizer/compare_tokenizers.py with real evidence).
  - [UNK] rate: the real coverage-gap signal -- unlike a skewed-but-nonzero
    frequency distribution, a token that never resolves to anything but [UNK]
    is a genuine tokenizer failure for this data.
  - vocabulary utilization: what fraction of the tokenizer's vocabulary this
    corpus actually exercises.
  - the token-frequency distribution's entropy, normalized against the max
    possible entropy for the observed vocabulary size. This is the correct
    formalization of "how uniform is the distribution" -- see the module
    docstring note below on why a *literally* uniform distribution is neither
    expected nor a real training objective for natural language.
  - singleton-token count/fraction (tokens seen exactly once): the real
    practical risk a skewed distribution poses -- those tokens get very few
    gradient updates on their embedding row.
  - top-K most frequent tokens, for a sanity-check look at what dominates.
"""

import argparse
import json
import math
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL manifest paths (one JSON object per line).",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the JSON field containing the utterance text.",
    )
    parser.add_argument(
        "--pretrained-model-name",
        type=str,
        default="google-bert/bert-base-multilingual-cased",
        help="Tokenizer to analyze, passed to LanguageModelTokenizer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of most frequent tokens to print.",
    )
    parser.add_argument(
        "--extra-tokens",
        type=str,
        nargs="*",
        default=None,
        help="Ordinary vocabulary tokens to add before analysis, to check "
        "whether a discovered coverage gap is actually fixed.",
    )
    parser.add_argument(
        "--stats-out",
        type=str,
        default=None,
        help="If given, write the full per-token frequency table to this "
        "JSONL file (one {token, count} object per line, sorted by count "
        "descending).",
    )
    return parser.parse_args()


def iter_texts(paths, text_field):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field)
                if text:
                    yield text


def entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def main():
    args = get_args()
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    tok = LanguageModelTokenizer(
        pretrained_model_name=args.pretrained_model_name,
        extra_tokens=args.extra_tokens,
    )
    unk_id = tok.hf_tokenizer.unk_token_id

    token_counts: Counter = Counter()
    n_utterances = 0
    n_tokens = 0
    n_chars = 0
    n_unk = 0

    for text in iter_texts(args.manifest, args.text_field):
        ids = tok.texts_to_token_ids([text])[0]
        n_utterances += 1
        n_tokens += len(ids)
        n_chars += len(text.replace(" ", ""))
        for tid in ids:
            token_counts[tid] += 1
            if tid == unk_id:
                n_unk += 1

    vocab_size = tok.vocab_size
    unique_tokens_used = len(token_counts)
    singleton_tokens = sum(1 for c in token_counts.values() if c == 1)
    max_entropy = math.log2(unique_tokens_used) if unique_tokens_used > 1 else 0.0
    observed_entropy = entropy_bits(token_counts)

    print(f"Manifest(s): {args.manifest}")
    print(f"Tokenizer: {args.pretrained_model_name}")
    print(f"Utterances: {n_utterances}")
    print(f"Total tokens: {n_tokens}")
    print(f"Total chars (no space): {n_chars}")
    print(f"Tokens/char (fragmentation, lower = more compression): {n_tokens / n_chars:.4f}")
    print()
    print(f"[UNK] count: {n_unk} ({n_unk / n_tokens:.4%} of all tokens)")
    print(
        f"Vocabulary utilization: {unique_tokens_used}/{vocab_size} "
        f"({unique_tokens_used / vocab_size:.2%}) of the tokenizer's vocabulary "
        "is ever used by this corpus"
    )
    print(
        f"Singleton tokens (seen exactly once): {singleton_tokens} "
        f"({singleton_tokens / unique_tokens_used:.2%} of used vocabulary)"
    )
    print(
        f"Distribution entropy: {observed_entropy:.2f} bits "
        f"(max possible for {unique_tokens_used} used tokens: {max_entropy:.2f} bits, "
        f"i.e. {observed_entropy / max_entropy:.1%} of the theoretical uniform maximum)"
    )
    print()
    print(f"Top-{args.top_k} most frequent tokens:")
    for tid, count in token_counts.most_common(args.top_k):
        piece = tok.hf_tokenizer.convert_ids_to_tokens(tid)
        print(f"  {piece!r:20s} count={count:8d} ({count / n_tokens:.3%} of all tokens)")

    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            for tid, count in token_counts.most_common():
                piece = tok.hf_tokenizer.convert_ids_to_tokens(tid)
                f.write(json.dumps({"token": piece, "token_id": tid, "count": count}, ensure_ascii=False) + "\n")
        print(f"\nWrote full per-token frequency table to {args.stats_out}")


if __name__ == "__main__":
    main()
