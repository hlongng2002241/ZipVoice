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


# --------------------------------------------------------------------------
# The two building blocks.
#
# Both return the SAME shape: a list of (words, units) blocks, where `words`
# is one or more consecutive whitespace words and `units` is what those words
# produce. Pairing them up by walking both lists gives the phone <-> lm_token
# correspondence.
#
# The block is variable-size on purpose. Requiring exactly one word per block
# is what makes alignment fail: espeak decides pronunciation using
# neighbouring words, so "the" alone is Vietnamese 'tˈɛ' but 'ðə' inside
# "book of the", and no per-word reference can reproduce that. Instead of
# giving up on the utterance, `phonemize` widens the block until it matches --
# "of the" becomes a single two-word block. Worst case the whole utterance is
# one block, which is still a valid (if coarse) alignment rather than a
# failure.
# --------------------------------------------------------------------------

_SETUP = {"lang": "vi", "token_file": "exp/fusion/tokens.txt"}
_CACHE = {}


def setup(lang="vi", token_file="exp/fusion/tokens.txt"):
    """Choose the language/vocabulary the two functions below use."""
    _SETUP.update(lang=lang, token_file=token_file)
    _CACHE.clear()


def _tokenizer():
    if "tok" not in _CACHE:
        _CACHE["lm"] = LanguageModelTokenizer()
        _CACHE["tok"] = F.FusionTokenizer(
            token_file=_SETUP["token_file"],
            lang=_SETUP["lang"],
            use_normalizer=False,
            lm_tokenizer=_CACHE["lm"],
        )
    return _CACHE["tok"]


#: Separators and punctuation are ignored when comparing phones, and are
#: kept with the block they trail, so concatenating blocks reproduces the
#: whole-utterance phone sequence exactly.
_SKIPPABLE = F._SENTENCE_END_PHONES | {" ", ",", "-", "\u2013", "\u2014"}

#: How many words a block may span before giving up and lumping the rest
#: together. Merges observed in practice are 2-3 words ("of the"); the bound
#: keeps a pathological utterance from going quadratic.
_MAX_BLOCK_WORDS = 8


def _match_at(flat, cursor, wanted):
    """Consume `wanted`'s phone content from `flat` starting at `cursor`,
    skipping separators. Returns the new cursor, or None if it diverges."""
    content = [p for p in wanted if p not in _SKIPPABLE]
    i = cursor
    matched = 0
    while i < len(flat) and matched < len(content):
        if flat[i] in _SKIPPABLE:
            i += 1
            continue
        if flat[i] != content[matched]:
            return None
        i += 1
        matched += 1
    if matched != len(content):
        return None
    while i < len(flat) and flat[i] in _SKIPPABLE:
        i += 1
    return i


def phonemize(text):
    """text -> [(words, phones), ...]

    `phones` is a slice of the WHOLE-utterance phonemization, never a
    per-word one, so cross-word phonology is preserved and concatenating
    every block reproduces espeak's output for `text` byte for byte. The
    per-word calls are only used to decide where the block boundaries are.

    Blocks widen in BOTH directions. Widening forward alone is not enough:
    the context that changes a word's pronunciation is often to its left --
    "the" reads as Vietnamese 'tˈɛ' on its own and stays that way in "the
    măng", but becomes 'ðə' in "book of the". So when a word will not match,
    previously emitted blocks are absorbed back until the block is wide
    enough to carry its own context.
    """
    tokenizer = _tokenizer()
    locale = F._ESPEAK_LOCALE[_SETUP["lang"]]
    words = text.split()
    flat = tokenizer._espeak_flat_phones(text, locale)

    def probe(lo, hi):
        return tokenizer._word_phones(" ".join(words[lo:hi]), locale)

    blocks = []
    cursor = 0
    i = 0
    while i < len(words):
        must_cover = i          # the block has to reach at least this word
        emitted = False
        while True:
            for j in range(max(i + 1, must_cover + 1),
                           min(i + _MAX_BLOCK_WORDS, len(words)) + 1):
                end = _match_at(flat, cursor, probe(i, j))
                if end is not None:
                    blocks.append((words[i:j], flat[cursor:end]))
                    cursor, i = end, j
                    emitted = True
                    break
            if emitted or not blocks:
                break
            # Absorb the previous block and retry with more left context.
            prev_words, prev_phones = blocks.pop()
            i -= len(prev_words)
            cursor -= len(prev_phones)
        if not emitted:
            blocks.append((words[i:], flat[cursor:]))
            cursor, i = len(flat), len(words)

    assert [p for _, ps in blocks for p in ps] == flat, "blocks must not lose phones"
    assert [w for ws, _ in blocks for w in ws] == words, "blocks must not lose words"
    return blocks


def tokenize(text):
    """text -> [(words, lm_tokens), ...], one word per block.

    Relies on Qwen's BPE being concatenative at whitespace boundaries --
    tokenize(A) + tokenize(B) == tokenize(A + B) -- which was verified on
    300/300 real utterances and on adversarial cases; the vocabulary's only
    multi-word tokens are runs of whitespace, so no merge crosses a space
    between real words. The assert below re-checks it per call, cheaply.

    The leading [LANG:xx] control token is NOT included: it belongs to no
    word, and the caller prepends it.
    """
    hf = _tokenizer().lm_tokenizer.hf_tokenizer
    words = text.split()
    blocks = []
    for index, word in enumerate(words):
        ids = hf.encode(word if index == 0 else " " + word, add_special_tokens=False)
        blocks.append(([word], hf.convert_ids_to_tokens(ids)))

    if " ".join(words) == text:
        whole = hf.convert_ids_to_tokens(hf.encode(text, add_special_tokens=False))
        assert [t for _, ts in blocks for t in ts] == whole, (
            "BPE is not concatenative for this text; per-word tokenization "
            "would not reproduce the real lm_token sequence"
        )
    return blocks


def align(phone_blocks, token_blocks):
    """Merge the two block lists into [(words, phones, lm_tokens), ...].

    Both inputs cover the same word sequence but cut it differently:
    `tokenize` always gives one word per block, `phonemize` occasionally
    merges a few (see its docstring). Two pointers walk the lists, always
    advancing whichever side is behind on words, and a group is emitted the
    moment the two sides have consumed the same words. The result is the
    coarsest-common-refinement of the two segmentations -- as fine as
    possible, only as coarse as the phone side forces.

    Because the emitted group boundaries are word boundaries on both sides,
    this cannot silently misalign: either the words agree at the cut point
    (asserted) or nothing is emitted.
    """
    aligned = []
    i = j = 0
    words_p, phones = [], []
    words_t, tokens = [], []

    while i < len(phone_blocks) or j < len(token_blocks):
        # Advance the side that has consumed fewer words. Ties go to the
        # phone side, which is the one that may merge.
        if len(words_p) <= len(words_t) and i < len(phone_blocks):
            block_words, block_units = phone_blocks[i]
            i += 1
            words_p = words_p + list(block_words)
            phones = phones + list(block_units)
        elif j < len(token_blocks):
            block_words, block_units = token_blocks[j]
            j += 1
            words_t = words_t + list(block_words)
            tokens = tokens + list(block_units)
        else:
            break  # one side ran out while the other still owes words

        if len(words_p) == len(words_t) and words_p:
            assert words_p == words_t, (
                f"the two segmentations disagree on words: "
                f"{words_p} vs {words_t}"
            )
            aligned.append((words_p, phones, tokens))
            words_p, phones, words_t, tokens = [], [], [], []

    assert not words_p and not words_t, (
        f"unconsumed tail: phones side {words_p}, lm_token side {words_t}"
    )
    return aligned


def align_text(text):
    """Convenience: text -> [(words, phones, lm_tokens), ...]."""
    return align(phonemize(text), tokenize(text))


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
                tokenizers[lang] = F.FusionTokenizer(token_file=args.token_file, lang=lang, lm_tokenizer=lm_tokenizer)
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
            # print(text)
            # return 0

            locale = F._ESPEAK_LOCALE[lang]
            words = [text[s:e] for s, e in F._whitespace_spans(text)]
            groups = F._split_into_groups(tokenizer._espeak_flat_phones(text, locale))
            counts = [len(F._split_into_groups(tokenizer._word_phones(w, locale))) for w in words]
            detail = explain(tokenizer, text, locale)
            reason = detail[4] if detail else "counts did not add up"
            reasons[reason] += 1

            if shown >= args.max_show:
                continue
            shown += 1
            print(f"\n=== [{shown}] {cut.get('id')}  lang={raw_language} " f"dur={cut.get('duration', 0):.1f}s")
            print(f"    words={len(words)}  espeak_groups={len(groups)}  " f"sum(per-word counts)={sum(counts)}")
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

    print(f"\n---\nscanned={scanned}  failed={failed} " f"({100 * failed / max(1, scanned):.2f}%)  shown={shown}")
    print("reasons:")
    for reason, count in reasons.most_common():
        print(f"  {count:5d}  {reason}")
    return 0


def attempt(text: str = None):
    text = "Quyển sách thứ hai mà mình đọc trong năm nay là quyển trước lư đồng mắt cua của tác giả nguyễn tuân và quyển này thì là book of the măng của tháng hai thì phải. Cái sách này mình nghĩ là nó được xếp vào cái dạng gọi là tùy bút và nó theo dòng hồi tưởng của nguyễn tuân về một cái thời trai trẻ của ông, cho nên lànó hơi mang hơi hướng là một quyển tùy bút nhưng mà kiểu bán tự truyện vì cái thể loại như vậy cho nên là mình thấy tác giả không chú trọng vào cái kết cấu của câu chuyện, không phải là cố làm cho câu chuyện có một cái tứ, có một cái thắt nút mở nút gì cả, mà nó chỉ là theo những cái dòng suy nghĩ của tác giả vềquá khứ và nó được liên kết với nhau qua cái hình ảnh của chiếc lư đồng mắt cua. Nhân vật tôi trong chuyện thể hiện rất nhiều mâu thuẫn là một người yêu cáiđẹp, yêu cái hào hoa văn hóa, nhưng lại có một cái cuộc sống rất là chán trường, bệ dạc và mình nghĩ là cái điều này có thể không chỉ là thể hiện mỗi cái cánhân đó mà còn phản ánh cái hơi hướng của thời đại lúc đó. Cái giọng văn của nguyễn tuân thì rất là tài hoa, nó mang cái tính ngoài cổ mà nó đẹp nhưng mà một cách kiểu như là rất sầu não ấy. Mình thì lúc mà mình đọc thì có những đoạn mình cảm thấy là wow phê quá kiểu quá hay, nhưng mà cũng có những đoạn thì mìnhkiểu mệt mỏi thực sự kiểu."

    text = "Quyển sách thứ hai mà mình đọc trong năm nay là quyển trước lư đồng mắt cua của tác giả nguyễn tuân và quyển này thì là book of the măng của tháng hai thì phải."

    for t in tokenize(text):
        print(t)
    print()

    for p in phonemize(text):
        print(p)

    return

    lm_tokenizer = LanguageModelTokenizer()
    tokenizer = F.FusionTokenizer(token_file="exp/fusion/tokens.txt", lang="vi", use_normalizer=False, lm_tokenizer=lm_tokenizer)
    output = tokenizer.text_to_artifact(text)
    print(output)

    # return

    for pg, lmg in zip(output.phone_groups, output.lm_token_groups):
        print([output.phones[i] for i in pg], "=", [output.lm_tokens[i] for i in lmg])


if __name__ == "__main__":
    # sys.exit(main())
    attempt()


(['of', 'the'], ['ɒ', 'v', 'ð', 'ə', ' '], ['Ġof', 'Ġthe'])