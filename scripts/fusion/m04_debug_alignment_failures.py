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

"""Show, per failing utterance, exactly WHERE and WHY phone/lm_token
alignment gave up -- so the failures can be judged rather than guessed at.

`FusionTokenizer` returns ``lm_token_groups=None`` when it cannot put the
phones into one group per whitespace word. That is a legal fallback (the
utterance trains phone-only), but a silent one. This prints the first point
of divergence for each failure: the word, what it phonemizes to on its own,
and what actually appears at that position in the whole-utterance
phonemization.

Reading the output: the two strategies are tried in order, and both must
fail before an utterance is reported here.

  content  each word is phonemized alone and its phones are consumed from
           the utterance sequence, skipping separators. Fails when a word's
           isolated pronunciation differs from its in-context one.
  count    looser: trust the word's group *count* without checking phones.
           Fails when the counts do not add up.

Usage:
    python3 scripts/fusion/m04_debug_alignment_failures.py \
        --manifest data/all/manifests/train.jsonl.gz \
        --token-file .../tokens.txt [--limit 3000] [--max-show 15] \
        [--language Vietnamese]
"""

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import zipvoice.tokenizer.fusion_tokenizer as F  # noqa: E402
from zipvoice.tokenizer.lm_tokenizer import (  # noqa: E402
    LanguageModelTokenizer,
    normalize_language_name,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--max-show", type=int, default=15)
    parser.add_argument("--language", default=None, help="e.g. Vietnamese")
    parser.add_argument("--context-chars", type=int, default=60)
    return parser.parse_args()


def explain(tokenizer, text, locale):
    """Return (word_index, word, phones_alone, phones_in_context, reason)
    for the first place the content walk diverges, or a count summary."""
    skippable = F._SENTENCE_END_PHONES | {" ", ",", "-", "–", "—"}
    flat = tokenizer._espeak_flat_phones(text, locale)
    words = [text[s:e] for s, e in F._whitespace_spans(text)]

    i = 0
    for index, word in enumerate(words):
        alone = tokenizer._word_phones(word, locale)
        content = [p for p in alone if p not in skippable]
        start = i
        matched = 0
        while i < len(flat) and matched < len(content):
            if flat[i] in skippable:
                i += 1
                continue
            if flat[i] != content[matched]:
                return (
                    index,
                    word,
                    "".join(alone),
                    "".join(flat[start : start + len(content) + 4]),
                    "word sounds different in context than on its own",
                )
            i += 1
            matched += 1
        if matched != len(content):
            return (
                index,
                word,
                "".join(alone),
                "".join(flat[start:]),
                "ran out of phones before the word was consumed",
            )
        while i < len(flat) and flat[i] in skippable:
            i += 1
    if i != len(flat):
        return (
            len(words) - 1,
            words[-1] if words else "",
            "",
            "".join(flat[i:]),
            "phones left over after the last word",
        )
    return None


def main():
    args = get_args()
    lm_tokenizer = LanguageModelTokenizer()
    tokenizers = {}
    shown = 0
    scanned = 0
    failed = 0
    reasons = collections.Counter()

    with gzip.open(args.manifest, "rt") as stream:
        for i, line in enumerate(stream):
            if args.limit and i >= args.limit:
                break
            cut = json.loads(line)
            sup = cut["supervisions"][0]
            raw_language = sup.get("language")
            if args.language and raw_language != args.language:
                continue
            lang = normalize_language_name(raw_language)
            if lang is None or lang == "zh":
                continue
            if lang not in tokenizers:
                tokenizers[lang] = F.FusionTokenizer(
                    token_file=args.token_file, lang=lang, lm_tokenizer=lm_tokenizer
                )
            tokenizer = tokenizers[lang]
            text = sup["text"]
            scanned += 1
            try:
                artifact = tokenizer.text_to_artifact(text)
            except Exception as ex:
                print(f"\n=== EXCEPTION {cut.get('id')}: {ex}")
                failed += 1
                continue
            if artifact.lm_token_groups is not None:
                continue
            failed += 1

            locale = F._ESPEAK_LOCALE[lang]
            words = [text[s:e] for s, e in F._whitespace_spans(text)]
            groups = F._split_into_groups(tokenizer._espeak_flat_phones(text, locale))
            counts = [
                len(F._split_into_groups(tokenizer._word_phones(w, locale)))
                for w in words
            ]
            detail = explain(tokenizer, text, locale)
            reason = detail[4] if detail else "counts did not add up"
            reasons[reason] += 1

            if shown >= args.max_show:
                continue
            shown += 1
            print(f"\n=== [{shown}] {cut.get('id')}  lang={raw_language} "
                  f"dur={cut.get('duration', 0):.1f}s")
            print(f"    words={len(words)}  espeak_groups={len(groups)}  "
                  f"sum(per-word counts)={sum(counts)}")
            if detail:
                index, word, alone, in_context, why = detail
                lo = max(0, index - 3)
                print(f"    REASON: {why}")
                print(f"    at word #{index}: {word!r}")
                print(f"      neighbours      : {words[lo:index + 4]}")
                print(f"      phones alone    : {alone!r}")
                print(f"      phones in context: {in_context!r}")
            multi = [(w, c) for w, c in zip(words, counts) if c != 1]
            if multi:
                print(f"    words espeak expands into several: {multi[:8]}")
            at = text.find(words[detail[0]]) if detail else 0
            c = args.context_chars
            print(f"    text: ...{text[max(0, at - c):at + c]}...")

    print(f"\n---\nscanned={scanned}  failed={failed} "
          f"({100 * failed / max(1, scanned):.2f}%)  shown={shown}")
    print("reasons:")
    for reason, count in reasons.most_common():
        print(f"  {count:5d}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
