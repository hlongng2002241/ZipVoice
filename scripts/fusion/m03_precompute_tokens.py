#!/usr/bin/env python3
# Copyright      2026  LongNH (with Claude Code assistance)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precompute the fusion frontend's phone/lm_token tokenization and
alignment for a manifest. **Compulsory**, not an optimization you can skip:
`tokenize_text_fusion()` (`zipvoice/bin/train_zipvoice.py`) has no live
fallback -- a cut with no precomputed artifact is a hard, immediate error at
training time, not a slow-but-working path.

Why: `tokenize_text_fusion()` used to also do the live computation --
whole-utterance espeak phonemization, Qwen BPE tokenization, and the
phone<->lm_token alignment search (`FusionTokenizer.text_to_artifact()`,
`fusion_tokenizer.py`'s variable-size word-block matching) -- for every cut,
every epoch, since lhotse `CutSet.map()` is lazy and nothing from one epoch
carries over to the next. That was measured as real, substantial extra CPU
work compared to the non-fusion frontend's single G2P call per utterance,
visible as a multi-minute stall at the start of every epoch before the
dataloader workers' prefetch buffer fills up. Making precompute compulsory
(rather than an optional fast path with a live fallback) guarantees that
cost is never silently paid again, and that a manifest that hasn't been
checked by `scripts/fusion/m02_preflight_corpus.py` can't quietly start a
training run at all.

Everything this computes is a pure, deterministic function of
`(text, language, phonemizer, Qwen tokenizer)` -- fusion training runs with
`--lang-auto-prob 0 --lang-wrong-prob 0` specifically so there is no
per-epoch randomness on the phone-branch side to lose by caching. So it is
safe to compute once, here, and have `tokenize_text_fusion()` reuse the
result on every subsequent epoch.

This does NOT precompute `lm_group_features` (the live Qwen hidden states
`TruncatedQwenExtractor.pooled_groups()` produces from `lm_token_ids`) --
those still get computed live, every training step, in the main process.
That's deliberate, not an oversight: under `--train-lm True` those features
have to stay in the autograd graph, so they can never be cached; frozen, the
ADR already decided a live 4-layer forward is cheap enough not to bother
(point 6). This script only removes the phonemization/tokenization/alignment
cost, which is the part `--train-lm` does not touch either way.

Run this after `scripts/fusion/m01_check_corpus_charset.py` and
`scripts/fusion/m02_preflight_corpus.py` (both should already pass clean
before spending time precomputing), and before
`scripts/fusion/m04_train.sh`, which checks for its `--output` and refuses
to start without it -- point `--train-manifest`/`--dev-manifest` there at
this script's `--output`, never at the raw manifest.

Usage:
    python3 scripts/fusion/m03_precompute_tokens.py \
        --manifest data/all/manifests/train.jsonl.gz \
        --token-file scripts/all/pretrained_model_cache/hynt__ZipVoice-Vietnamese-2500h/tokens.txt \
        --output data/all/manifests/train.fusion_precomputed.jsonl.gz
"""

import argparse
import collections
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lhotse import CutSet, load_manifest_lazy  # noqa: E402
from tqdm import tqdm  # noqa: E402

from zipvoice.tokenizer.fusion_tokenizer import (  # noqa: E402
    FusionTokenizer,
    SCHEMA_VERSION,
)
from zipvoice.tokenizer.lm_tokenizer import (  # noqa: E402
    LanguageModelTokenizer,
    normalize_language_name,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pretrained-tokenizer-name",
        default=None,
        help="HuggingFace id for the Qwen tokenizer (must match what "
        "--train-lm training will use). Defaults to LanguageModelTokenizer's "
        "own default (Qwen2.5-0.5B).",
    )
    return parser.parse_args()


def main():
    args = get_args()

    lm_kwargs = {}
    if args.pretrained_tokenizer_name is not None:
        lm_kwargs["pretrained_model_name"] = args.pretrained_tokenizer_name
    lm_tokenizer = LanguageModelTokenizer(**lm_kwargs)
    tokenizers = {
        lang: FusionTokenizer(
            token_file=args.token_file, lang=lang, lm_tokenizer=lm_tokenizer
        )
        for lang in ("en", "vi", "zh")
    }

    cuts = load_manifest_lazy(args.manifest)
    stats = collections.Counter()

    def _precompute(c):
        sup = c.supervisions[0]
        raw_lang = getattr(sup, "language", None)
        lang = normalize_language_name(raw_lang)
        if lang is None:
            raise ValueError(
                f"cut {c.id!r} has a missing or unrecognized language "
                f"({raw_lang!r}). Run scripts/fusion/m02_preflight_corpus.py "
                f"first -- this should already have been caught there."
            )
        if lang not in tokenizers:
            raise ValueError(
                f"cut {c.id!r} is {lang!r}, but no FusionTokenizer was built "
                f"for that language (have: {sorted(tokenizers)})."
            )

        artifact = tokenizers[lang].text_to_artifact(sup.text)
        sup.tokens = artifact.phone_ids
        sup.phone_groups = artifact.phone_groups
        sup.lm_token_ids = artifact.lm_token_ids
        sup.lm_token_groups = artifact.lm_token_groups
        sup.fusion_schema_version = artifact.schema_version
        # Not cached at training time either (see module docstring) -- never
        # meaningful to precompute, so cleared here for the same reason
        # tokenize_text_fusion() clears it after a live computation.
        sup.zero_duration_mask = None

        stats[lang] += 1
        stats["aligned" if artifact.lm_token_groups is not None else "unaligned"] += 1
        return c

    def _precompute_with_progress():
        for c in tqdm(cuts, desc="precomputing fusion tokens"):
            yield _precompute(c)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(_precompute_with_progress()).to_file(out_path)

    print(f"\nWrote precomputed fusion tokens ({SCHEMA_VERSION=}) to {out_path}")
    print(f"By language: {dict(stats)}")
    total = stats["aligned"] + stats["unaligned"]
    if total:
        rate = stats["aligned"] / total
        print(f"Alignment rate: {stats['aligned']}/{total} ({100 * rate:.1f}%)")


if __name__ == "__main__":
    main()
