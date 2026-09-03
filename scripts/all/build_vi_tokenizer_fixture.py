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

"""Regenerate tests/fixtures/vi_normalized_samples.jsonl -- the small,
real-and-already-normalized Vietnamese fixture used by
tests/test_fusion_tokenizer.py's VI corpus-equivalence test.

Not tracked in git (see .gitignore's `tests/fixtures/*.jsonl` entry): its
content is real training-utterance text pulled from the training manifest,
which carries the same sensitivity concerns as `data/` (also gitignored).
Run this script locally, from a checkout with `data/all/manifests/train.jsonl.gz`
present, to (re)build the fixture. If the fixture is missing,
test_fusion_tokenizer.py's VI test warns and skips rather than failing --
see that test module's docstring for why the raw, unnormalized
data/corpus/vi/*.jsonl web corpus is not used instead: Vietnamese text is
normalized by a separate, external pipeline before it reaches training, so
only real training-manifest text is representative here.
"""

import argparse
import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(REPO_ROOT / "data" / "all" / "manifests" / "train.jsonl.gz"),
        help="Gzipped lhotse CutSet JSONL manifest to draw samples from.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Vietnamese",
        help="supervision.language value to filter for.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of unique utterance texts to extract.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "tests" / "fixtures" / "vi_normalized_samples.jsonl"),
        help="Output fixture path (one {'text': ...} JSON object per line).",
    )
    return parser.parse_args()


def main():
    args = get_args()

    samples = []
    seen = set()
    with gzip.open(args.manifest, "rt", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= args.num_samples:
                break
            obj = json.loads(line)
            for sup in obj.get("supervisions", []):
                if sup.get("language") != args.language:
                    continue
                text = sup.get("text", "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                samples.append(text)
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in samples:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} unique {args.language!r} samples to {out_path}")


if __name__ == "__main__":
    main()
