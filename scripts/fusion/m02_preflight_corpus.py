#!/usr/bin/env python3
# Copyright      2026  LongNH (with Claude Code assistance)
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

"""Corpus preflight for the fusion frontend. Run before committing to a long
training run.

Why this exists, concretely: the first fusion training run spent ~93% of its
batches phone-only and nobody noticed for a full epoch, because
`lm_token_groups=None` is a legal fallback that fires silently (see the
sentence-boundary bug fixed in `_split_into_groups`). Training-time coverage
metrics (`lm_cov_utt`/`lm_cov_grp`) now make that visible per log line,
but they only tell you *after* you have committed the GPU. This tells you
before, in a few minutes, and it reports the losses those metrics cannot
see:

  - phonemizer exceptions and non-empty transcripts yielding zero phones,
    which now raise rather than silently producing an empty artifact;
  - OOV symbols dropped and groups emptied by filtering -- an utterance can
    lose phones, or whole words, and still show 100% conditioning coverage
    over its *surviving* groups, because the denominator is measured after
    the loss;
  - alignment coverage split by language, since one language can fail
    almost completely while the aggregate looks healthy (English is ~0.8%
    of this corpus, so it cannot move the total).

Usage:
    python3 scripts/fusion/m02_preflight_corpus.py \
        --manifest data/all/manifests/train.jsonl.gz \
        --token-file scripts/all/pretrained_model_cache/...\
/hynt__ZipVoice-Vietnamese-2500h/tokens.txt \
        [--limit 3000]
"""

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from zipvoice.tokenizer.fusion_tokenizer import FusionTokenizer  # noqa: E402
from zipvoice.tokenizer.lm_tokenizer import (  # noqa: E402
    LanguageModelTokenizer,
    normalize_language_name,
)

#: Below this, the Qwen branch is disabled for most of a language's data and
#: the run is not testing the fusion architecture. Chosen as a catastrophe
#: alarm, not a quality bar -- measured VI coverage is ~97%.
MIN_ALIGNMENT_RATE = 0.80


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Utterances to scan (0 = all). The default is a few minutes; "
        "the full corpus takes considerably longer and rarely says anything "
        "the sample does not.",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=MIN_ALIGNMENT_RATE,
        help="Exit non-zero if any language with enough samples falls below "
        "this alignment rate.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    lm_tokenizer = LanguageModelTokenizer()
    tokenizers = {}
    stats = collections.defaultdict(collections.Counter)
    failures = []

    with gzip.open(args.manifest, "rt") as stream:
        for i, line in enumerate(stream):
            if args.limit and i >= args.limit:
                break
            sup = json.loads(line)["supervisions"][0]
            lang = normalize_language_name(sup.get("language"))
            if lang is None:
                stats["?"]["unroutable_language"] += 1
                continue
            if lang not in tokenizers:
                tokenizers[lang] = FusionTokenizer(
                    token_file=args.token_file,
                    lang=lang,
                    lm_tokenizer=lm_tokenizer,
                )
            counter = stats[lang]
            counter["utts"] += 1
            text = sup["text"]
            try:
                artifact = tokenizers[lang].text_to_artifact(text)
            except Exception as ex:  # phonemizer failure / zero phones
                counter["EXCEPTION"] += 1
                if len(failures) < 10:
                    failures.append((lang, text[:60], str(ex)[:120]))
                continue

            # Loss that happens BEFORE conditioning coverage is measured, and
            # is therefore invisible to the training-time metrics.
            counter["raw_words"] += len(text.split())
            counter["phone_groups"] += len(artifact.phone_groups)
            if len(artifact.phone_groups) < len(text.split()):
                counter["utts_with_dropped_words"] += 1
            if not artifact.phone_ids:
                counter["zero_phones"] += 1

            if artifact.lm_token_groups is None:
                counter["unaligned"] += 1
            else:
                counter["aligned"] += 1
                counter["groups"] += len(artifact.lm_token_groups)
                counter["empty_groups"] += sum(
                    1 for g in artifact.lm_token_groups if not g
                )

    exit_code = 0
    for lang in sorted(stats):
        c = stats[lang]
        n = c["utts"]
        if not n:
            print(f"[{lang}] {dict(c)}")
            continue
        rate = c["aligned"] / n
        print(
            f"[{lang}] utts={n} aligned={c['aligned']} ({100 * rate:.1f}%) "
            f"unaligned={c['unaligned']} EXCEPTION={c['EXCEPTION']} "
            f"zero_phones={c['zero_phones']} groups={c['groups']} "
            f"empty_groups={c['empty_groups']} "
            f"utts_missing_words={c['utts_with_dropped_words']}"
        )
        if c["EXCEPTION"]:
            print(f"  FAIL [{lang}]: phonemization failed for some utterances")
            exit_code = 1
        if n >= 50 and rate < args.min_rate:
            print(
                f"  FAIL [{lang}]: alignment rate {rate:.1%} is below "
                f"{args.min_rate:.0%} -- the Qwen branch would be disabled "
                f"for most of this language's data, so the run would not be "
                f"testing the fusion architecture."
            )
            exit_code = 1

    for lang, text, err in failures:
        print(f"  [{lang}] {text!r}: {err}")

    print("PREFLIGHT PASSED" if exit_code == 0 else "PREFLIGHT FAILED")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
