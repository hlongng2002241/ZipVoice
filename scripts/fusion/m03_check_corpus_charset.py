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

"""Audit corpus text for characters the fusion frontend cannot align well.

The contract this checks: transcripts should contain **letters, whitespace,
and `,.?!-` only**. Everything else is either unpronounceable or expands
unpredictably, and both hurt the phone/lm_token alignment the Qwen branch
depends on.

Why each excluded class matters, measured on this corpus:

  digits      espeak expands one text token into several spoken words --
              "1997" becomes five ("mot nghin chin tram chin muoi bay"),
              "250000" four. The lm_token side still sees one token, so the
              group structure has to be recovered rather than counted.
              Normalizing digits to words upstream removes the problem
              entirely, and Vietnamese already runs with
              use_normalizer=False precisely because that is meant to have
              happened already.
  '/' '(' etc espeak splits inside the token ("ui/ux" became four groups),
              so one whitespace word maps to several spoken words.
  other       symbols with no pronunciation (emoji, currency, quotes) either
              vanish -- taking their group with them -- or get spelled out.

`,.?!-` are allowed because they are ordinary phones in the source
checkpoint's vocabulary and carry duration as silence; they group cleanly.

This is a text-only check: fast, no phonemizer, safe to run on the full
corpus. Run it before m02_preflight_corpus.py -- that one measures the
*consequences* on alignment, this one finds the cause.

Usage:
    python3 scripts/fusion/m03_check_corpus_charset.py \
        --manifest data/all/manifests/train.jsonl.gz [--limit 0] \
        [--max-examples 3]
"""

import argparse
import collections
import gzip
import json
import sys
import unicodedata

#: Punctuation the fusion frontend handles cleanly.
ALLOWED_PUNCTUATION = set(",.?!-")


def classify(ch: str) -> str:
    """Bucket a disallowed character, for a report that says what to fix."""
    if ch.isdigit():
        return "digit"
    category = unicodedata.category(ch)
    if category.startswith("P"):
        return "punctuation/symbol"
    if category.startswith("S"):
        return "symbol"
    if category.startswith("C"):
        return "control/format"
    return f"other[{category}]"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, nargs="+", help="One or more jsonl.gz manifests."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Utterances per manifest (0 = all)."
    )
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument(
        "--allow",
        default="",
        help="Extra characters to treat as allowed, e.g. --allow \"'\"",
    )
    return parser.parse_args()


def main():
    args = get_args()
    allowed = ALLOWED_PUNCTUATION | set(args.allow)

    per_char = collections.Counter()
    per_char_utts = collections.Counter()
    per_class = collections.Counter()
    examples = collections.defaultdict(list)
    utts = bad_utts = 0
    total_secs = bad_secs = 0.0
    secs_by_char = collections.Counter()
    bad_by_language = collections.Counter()
    utts_by_language = collections.Counter()

    for path in args.manifest:
        with gzip.open(path, "rt") as stream:
            for i, line in enumerate(stream):
                if args.limit and i >= args.limit:
                    break
                cut = json.loads(line)
                sup = cut["supervisions"][0]
                duration = float(cut.get("duration", 0.0))
                text = sup.get("text", "")
                lang = sup.get("language", "?")
                utts += 1
                total_secs += duration
                utts_by_language[lang] += 1
                offenders = {
                    ch
                    for ch in text
                    if not (ch.isalpha() or ch.isspace() or ch in allowed)
                }
                if not offenders:
                    continue
                bad_utts += 1
                bad_secs += duration
                bad_by_language[lang] += 1
                for ch in offenders:
                    per_char_utts[ch] += 1
                    secs_by_char[ch] += duration
                    per_class[classify(ch)] += 1
                    if len(examples[ch]) < args.max_examples:
                        at = text.index(ch)
                        examples[ch].append(text[max(0, at - 28) : at + 28])
                for ch in text:
                    if ch in offenders:
                        per_char[ch] += 1

    print(f"utterances scanned : {utts}   ({total_secs / 3600:.1f} h)")
    print(
        f"utterances with disallowed characters: {bad_utts} "
        f"({100 * bad_utts / max(1, utts):.2f}%)   "
        f"{bad_secs / 3600:.2f} h "
        f"({100 * bad_secs / max(1e-9, total_secs):.2f}% of audio)"
    )
    print(f"allowed: letters, whitespace, and {''.join(sorted(allowed))!r}")

    if bad_by_language:
        print("\nby language (affected / total):")
        for lang in sorted(utts_by_language):
            print(
                f"  {lang:<12} {bad_by_language[lang]:>7} / {utts_by_language[lang]:<7}"
                f" ({100 * bad_by_language[lang] / utts_by_language[lang]:.2f}%)"
            )

    if per_class:
        print("\nby class (distinct char occurrences in affected utterances):")
        for cls, count in per_class.most_common():
            print(f"  {cls:<20} {count}")

    if per_char:
        print("\noffending characters, most affected utterances first:")
        for ch, n_utts in per_char_utts.most_common(40):
            name = unicodedata.name(ch, "?")
            print(
                f"  {ch!r:>8} U+{ord(ch):04X} {classify(ch):<18} "
                f"utts={n_utts:<6} {secs_by_char[ch] / 3600:>6.2f}h "
                f"occ={per_char[ch]:<7} {name[:30]}"
            )
            for ex in examples[ch][: args.max_examples]:
                print(f"           ...{ex.strip()}...")

    if bad_utts:
        print("\nCHARSET CHECK FAILED")
        return 1
    print("\nCHARSET CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
