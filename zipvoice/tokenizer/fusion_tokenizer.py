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

"""Fusion-frontend tokenizer for the Qwen+phoneme fusion text frontend
(see docs/adr/2026-09-02__qwen_phone_fusion_text_frontend.md).

Terminology (see the ADR's Terminology note -- unchanged from inherited
code, not redefined by this module):
  - "phones" / "tokens": the phoneme sequence, exactly as
    ``EmiliaTokenizer``/``EspeakTokenizer`` already produce it (when this
    module's own normalization is left off -- see below).
  - "lm_tokens": ``LanguageModelTokenizer``'s Qwen/BPE output, produced by
    this module (when a ``lm_tokenizer`` is supplied) over the *same*
    canonical text used to derive phones, so that character offsets on both
    sides line up exactly, with the deterministic ``[LANG:xx]`` control
    token for this instance's language prepended as index 0.
  - a "group" is a list of consecutive phones, plus (when available) the
    corresponding list of consecutive ``lm_tokens``, corresponding to one
    word-equivalent span of the canonical text.

By design, and unless told otherwise, this module -- like
``LanguageModelTokenizer`` -- does **not** normalize text: normalization
(number/date/abbreviation expansion etc.) is a data-preparation concern that
must already be done upstream. The one opt-in exception is the
``use_normalizer`` constructor argument (see ``FusionTokenizer.__init__``),
which lets a caller enable ``EmiliaTokenizer``'s existing English/Chinese
normalizers *for this tokenizer's own processing only* -- ``EmiliaTokenizer``
and ``EspeakTokenizer`` themselves are never modified. Since one
``FusionTokenizer`` instance is always locked to a single ``lang`` (see
below), ``use_normalizer`` is a plain per-instance ``bool``; its default
(when left as ``None``) is looked up per-language from
``_DEFAULT_USE_NORMALIZER`` (English/Chinese default on, Vietnamese default
off). When left at that default, flattening this module's phone groups
reproduces *exactly* what the wrapped ``EmiliaTokenizer``/``EspeakTokenizer``
already produces for that same text via ``texts_to_tokens()``/``g2p()`` --
Vietnamese is never normalized here because Vietnamese text is expected to
already be normalized upstream by a separate pipeline before it reaches this
tokenizer.

Duration note: the phone vocabulary already includes literal space and
punctuation characters as ordinary entries (confirmed against the source
checkpoint's tokens.txt: e.g. ``' '`` has its own id, and a real sentence's
phones map to token ids with zero OOV drops). The converged source
checkpoint was trained with these receiving a normal, non-zero share of
duration like any other phone -- punctuation is silence, and silence still
takes time -- so there is no phone-side zero-duration concept here at all.
The only token that legitimately has *no* corresponding acoustic content is
the ``[LANG:xx]`` control tag on the ``lm_tokens`` side (pure metadata, not
a sound), so ``zero_duration_mask`` is entirely an ``lm_tokens``-aligned
field, sourced 100% from ``LanguageModelTokenizer.zero_duration_mask()`` --
this module never computes it itself. Empty (``[]``) when no
``lm_tokenizer`` is supplied, exactly like ``lm_tokens``/``lm_token_ids``.
"""

import logging
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import jieba
from pypinyin import Style, lazy_pinyin

from zipvoice.tokenizer.tokenizer import EmiliaTokenizer

if TYPE_CHECKING:
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f \
            https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

SCHEMA_VERSION = 3

# Canonical short language code -> espeak-ng locale code, for the plain
# (non-Emilia-routed) path. "zh" is handled separately via the Emilia-style
# Han/Latin-script segment routing, not a single espeak locale.
_ESPEAK_LOCALE = {"en": "en-us", "vi": "vi"}

# Default per-language opt-in for FusionTokenizer's own normalization step,
# looked up by `lang` when `use_normalizer=None` is passed to __init__.
# English/Chinese default on (matching EmiliaTokenizer's always-on
# behavior for those languages); Vietnamese defaults off and is not expected
# to ever be turned on here -- Vietnamese normalization happens in a
# separate, external pipeline before data reaches this tokenizer.
_DEFAULT_USE_NORMALIZER: Dict[str, bool] = {"en": True, "zh": True, "vi": False}

_bare_emilia_tokenizer: Optional[EmiliaTokenizer] = None

#: (locale, word) -> phones, for the per-word group-count probe in
#: `FusionTokenizer._blocks_from_words`. Bounded so a long training run
#: cannot grow it without limit; the corpus vocabulary is far smaller.
_WORD_PHONE_CACHE: Dict[Tuple[str, str], List[str]] = {}
_WORD_PHONE_CACHE_MAX = 200_000



def _shared_emilia_tokenizer() -> EmiliaTokenizer:
    """A EmiliaTokenizer instance used only for its text-processing methods
    (get_segment, preprocess_text, tokenize_pinyin, separate_pinyin,
    normalizers) -- never for its token2id mapping, since FusionTokenizer
    has its own (identical-format, but independently loaded) vocabulary.
    Constructed without a token_file, which EmiliaTokenizer supports."""
    global _bare_emilia_tokenizer
    if _bare_emilia_tokenizer is None:
        _bare_emilia_tokenizer = EmiliaTokenizer(token_file=None)
    return _bare_emilia_tokenizer


def _flatten_espeak_output(sentences) -> List[str]:
    return reduce(lambda x, y: x + y, sentences, [])


#: Phone symbols after which espeak may start a new word *without* emitting a
#: separating ' '. See `_split_into_groups`.
#:
#: ',' is deliberately absent, and not because breaking on it would double-
#: break (the lookahead guard already prevents that): espeak emits a space
#: after a comma, so the space rule already closes the group exactly once.
#: Verified -- "alpha, beta" phonemizes to [... 'ə', ',', ' ', 'b', ...].
#:
#: Note this behaviour is **voice-dependent**, which is why the sentence-
#: boundary bug hit Vietnamese and not English: espeak-vi emits '.' with no
#: following space ("nguoi. Rat" -> [... 'j', '.', 'z', ...]), while
#: espeak-en-us drops the '.' entirely and emits a space instead
#: ("beta. gamma" -> [... 'ə', ' ', 'ɡ', ...]). Adding a voice must therefore
#: re-check this set against that voice's real output rather than assume it.
_SENTENCE_END_PHONES = frozenset(".!?;:")


def _split_into_groups(flat_phones: List[str]) -> List[List[str]]:
    """Split a flat phone sequence into word-equivalent groups.

    Groups break on the literal ' ' phone symbol (which `phonemize_espeak`
    emits as an ordinary element -- confirmed empirically), and *also* after
    sentence-final punctuation. The space itself is kept (not dropped) as the
    trailing element of the group before it, so flattening the returned
    groups reproduces `flat_phones` exactly, unchanged.

    The sentence-final rule exists because espeak emits **no space after a
    sentence-final period**: "nguoi. Rat" phonemizes to `... j . z ...` with
    no ' ' between, so a space-only split silently merges the last word of
    one sentence with the first word of the next. Every internal sentence
    boundary then costs exactly one group, the group count stops matching
    the whitespace-word count, and `text_to_artifact` gives up and returns
    `lm_token_groups=None` for the whole utterance -- disabling the Qwen
    branch for it entirely.

    That was not a corner case. Measured on this project's training corpus
    before the fix, alignment succeeded for only **7.3%** of utterances
    (2.7% of those over 50 words, which is most of the corpus at a 28.9s
    mean), because long utterances contain many sentences. The first
    training run was therefore ~93% phone-only.

    A break is suppressed when a ' ' already follows the punctuation, so
    that case still closes the group exactly once and no group is emitted
    that consists solely of a separator.
    """
    groups: List[List[str]] = []
    current: List[str] = []
    last = len(flat_phones) - 1
    for i, symbol in enumerate(flat_phones):
        current.append(symbol)
        if symbol == " " or (
            symbol in _SENTENCE_END_PHONES
            and i < last
            and flat_phones[i + 1] != " "
        ):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _whitespace_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) character spans of each whitespace-delimited run in
    `text`, in order. Used as the word-equivalent span anchor for
    correlating phone groups with `lm_token` character offsets on the
    plain (non-Chinese) path.
    """
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


#: How many words a block may span before the rest of the utterance is taken
#: as one block. Merges observed in practice are 2-3 words ("of the"); the
#: bound stops a pathological utterance from going quadratic.
_MAX_BLOCK_WORDS = 8

#: Ignored when *comparing* phones -- still emitted, and kept with the block
#: they trail, so concatenating blocks reproduces the phone sequence exactly.
#:
#: Includes the IPA stress marks. Stress is sentence-level prosody, not
#: segmental content: espeak gives "how" as 'hˈaʊ' alone but 'hˌaʊ' inside a
#: sentence, and "this" as 'ðˈɪs' alone but 'ðɪs' unstressed. Comparing them
#: would force those words to merge with their neighbours for a difference
#: that cannot change which word a phone belongs to. Measured on English,
#: ignoring stress took a test sentence from 6 groups for 11 words to one
#: group per word.
_SKIPPABLE_PHONES = (
    _SENTENCE_END_PHONES
    | {" ", ",", "-", "\u2013", "\u2014"}
    | {"\u02c8", "\u02cc"}  # primary / secondary stress
)


def _match_phone_content(
    flat: List[str], cursor: int, wanted: List[str]
) -> Optional[int]:
    """Consume `wanted`'s phone content from `flat` at `cursor`, skipping
    separators and punctuation on both sides. Returns the new cursor, or
    None if the content diverges."""
    content = [p for p in wanted if p not in _SKIPPABLE_PHONES]
    i = cursor
    matched = 0
    while i < len(flat) and matched < len(content):
        if flat[i] in _SKIPPABLE_PHONES:
            i += 1
            continue
        if flat[i] != content[matched]:
            return None
        i += 1
        matched += 1
    if matched != len(content):
        return None
    while i < len(flat) and flat[i] in _SKIPPABLE_PHONES:
        i += 1
    return i


@dataclass
class FusionTokenizerArtifact:
    """Versioned per-utterance output of FusionTokenizer.text_to_artifact().

    Attributes:
      language: the canonical short language code used to phonemize this
        utterance ("en", "vi", or "zh").
      phones: flat phone symbol sequence (post-OOV-filtering).
      phone_ids: `phones` mapped through the phone vocabulary, after OOV
        filtering -- may be shorter than the raw phonemizer output if any
        symbol is OOV.
      phone_groups: one entry per word-equivalent group, each a list of indices
        into `phone_ids` belonging to that group. Empty groups (all of
        whose phones were OOV) are dropped.
      lm_tokens: LanguageModelTokenizer's token pieces for this utterance's
        canonical (post-normalization) text, with the deterministic
        ``[LANG:{language}]`` control token prepended as index 0 (this
        fusion architecture always uses the immutable true-language tag,
        never ``[LANG:auto]`` or a wrong tag). Empty if no `lm_tokenizer`
        was supplied to FusionTokenizer.
      lm_token_ids: `lm_tokens` mapped to ids via the same LanguageModelTokenizer.
      lm_token_groups: one entry per group in `phone_groups` (same order,
        same length), each a list of indices into `lm_token_ids` overlapping
        that group's character span in the canonical text -- index 0 (the
        prepended language tag) never appears in any group. `None` if
        lm_tokens were not computed, or if the phone-group/word-span counts
        didn't line up for this utterance (a known residual-risk edge case
        -- see docs/plans/2026-09-02__qwen_phone_fusion_text_frontend/
        2026-09-02__sprint_000__new_tokenizer.md's risk table) -- callers
        must handle `None` rather than assume alignment always succeeds.
      zero_duration_mask: one bool per entry in `lm_token_ids` (**not**
        `phone_ids` -- there is no phone-side zero-duration concept, see
        module docstring). Sourced entirely from
        `LanguageModelTokenizer.zero_duration_mask()`; `True` only for the
        prepended `[LANG:xx]` control token at index 0. Empty (`[]`) when
        no `lm_tokenizer` was supplied, exactly like `lm_tokens`/`lm_token_ids`.
      schema_version: bump when this artifact's structure changes.
        `__post_init__` rejects any value other than the current
        `SCHEMA_VERSION` -- this artifact is never meant to be constructed
        with a stale version at runtime; a cached artifact loaded from disk
        under an older schema must be rebuilt, not accepted as-is.

    `__post_init__` validates the cross-field invariants documented above
    (length agreement, exact/monotonic/non-overlapping `phone_groups`
    coverage of `phone_ids`, in-range `lm_token_groups` indices, schema
    version) -- a malformed artifact raises immediately at construction
    rather than failing confusingly downstream (e.g. in duration allocation
    or dataset collation).
    """

    language: str
    phones: List[str]
    phone_ids: List[int]
    phone_groups: List[List[int]]
    lm_tokens: List[str] = field(default_factory=list)
    lm_token_ids: List[int] = field(default_factory=list)
    lm_token_groups: Optional[List[List[int]]] = None
    zero_duration_mask: List[bool] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        assert self.language in ("en", "vi", "zh"), self.language
        assert self.schema_version == SCHEMA_VERSION, (
            f"FusionTokenizerArtifact schema_version={self.schema_version} "
            f"does not match the current SCHEMA_VERSION={SCHEMA_VERSION} -- "
            "rebuild this artifact rather than using a stale cached one."
        )

        assert len(self.phones) == len(self.phone_ids), (
            "phones/phone_ids length mismatch: "
            f"{len(self.phones)} vs {len(self.phone_ids)}"
        )
        assert len(self.zero_duration_mask) == len(self.lm_token_ids), (
            "zero_duration_mask must have exactly one entry per "
            f"lm_token_id (not phone_id): "
            f"{len(self.zero_duration_mask)} vs {len(self.lm_token_ids)}"
        )
        assert len(self.lm_tokens) == len(self.lm_token_ids), (
            "lm_tokens/lm_token_ids length mismatch: "
            f"{len(self.lm_tokens)} vs {len(self.lm_token_ids)}"
        )

        covered = sorted(idx for group in self.phone_groups for idx in group)
        assert covered == list(range(len(self.phone_ids))), (
            "phone_groups must partition range(len(phone_ids)) exactly, "
            "with no gaps, overlaps, or duplicate indices"
        )
        last_end = -1
        for group in self.phone_groups:
            assert group and group == sorted(group), (
                f"phone_groups entries must be non-empty and sorted: {group}"
            )
            assert group[0] > last_end, (
                f"phone_groups must appear in increasing, non-overlapping "
                f"order: {group} after last_end={last_end}"
            )
            last_end = group[-1]

        if self.lm_token_groups is not None:
            assert len(self.lm_token_groups) == len(self.phone_groups), (
                "lm_token_groups must have exactly one entry per "
                f"phone_groups entry when present: "
                f"{len(self.lm_token_groups)} vs {len(self.phone_groups)}"
            )
            for group in self.lm_token_groups:
                for idx in group:
                    assert 0 <= idx < len(self.lm_token_ids), (
                        f"lm_token_groups index {idx} out of range for "
                        f"{len(self.lm_token_ids)} lm_token_ids"
                    )


def compose_artifacts(
    prompt: "FusionTokenizerArtifact",
    target: "FusionTokenizerArtifact",
) -> "FusionTokenizerArtifact":
    """Combine a prompt artifact and a target artifact into the single
    artifact that inference actually conditions on.

    This is the composition contract sprint 003 deliberately deferred ("how a
    prompt artifact and a target artifact combine without putting two
    ``[LANG:xx]`` tags in one causal Qwen context"). It is resolved the same
    way the pre-fusion multilingual inference path already resolved it, at
    `infer_zipvoice.py`: **only the prompt carries the language tag.**
    `model.sample()` concatenates prompt and target into one sequence, and
    training only ever saw a single utterance with one leading tag, so the
    target's own tag is dropped rather than left mid-sequence.

    Index arithmetic, both sides:

      phones     target index i -> len(prompt.phone_ids) + i
      lm_tokens  target index i -> len(prompt.lm_token_ids) + i - 1

    The ``- 1`` is the dropped tag. It never underflows because index 0 is
    the tag and the tag belongs to no group -- an invariant this artifact
    documents and `__post_init__` does not need to re-derive (verified
    empirically: the lowest index any group uses is 1).

    Alignment failure on *either* side collapses the whole composition to
    phone-only (``lm_token_groups=None``). Partial recovery would mean
    conditioning the target on a prompt whose group structure is unknown,
    which is a different premise than training; the honest fallback is the
    documented one.
    """
    assert prompt.language == target.language, (
        f"cannot compose artifacts from different languages: "
        f"{prompt.language!r} (prompt) vs {target.language!r} (target). "
        f"FusionTokenizer is locked to one language per instance."
    )

    phone_offset = len(prompt.phone_ids)
    phone_groups = list(prompt.phone_groups) + [
        [i + phone_offset for i in group] for group in target.phone_groups
    ]

    # Drop the target's leading [LANG:xx] tag and everything indexed to it.
    lm_tokens = list(prompt.lm_tokens) + list(target.lm_tokens[1:])
    lm_token_ids = list(prompt.lm_token_ids) + list(target.lm_token_ids[1:])
    zero_duration_mask = list(prompt.zero_duration_mask) + list(
        target.zero_duration_mask[1:]
    )

    if prompt.lm_token_groups is None or target.lm_token_groups is None:
        lm_token_groups = None
    else:
        lm_offset = len(prompt.lm_token_ids) - 1
        shifted = []
        for group in target.lm_token_groups:
            assert all(i >= 1 for i in group), (
                f"target lm_token group {group} references index 0, the "
                f"[LANG:xx] tag, which must belong to no group"
            )
            shifted.append([i + lm_offset for i in group])
        lm_token_groups = list(prompt.lm_token_groups) + shifted

    return FusionTokenizerArtifact(
        language=target.language,
        phones=list(prompt.phones) + list(target.phones),
        phone_ids=list(prompt.phone_ids) + list(target.phone_ids),
        phone_groups=phone_groups,
        lm_tokens=lm_tokens,
        lm_token_ids=lm_token_ids,
        lm_token_groups=lm_token_groups,
        zero_duration_mask=zero_duration_mask,
    )


class FusionTokenizer:
    """Wraps EmiliaTokenizer/EspeakTokenizer (never modifies them) to
    additionally recover word-equivalent groups within their phone output,
    and (when a LanguageModelTokenizer is supplied) to produce lm_tokens and
    a phone<->lm_token group correspondence, for the Qwen+phoneme fusion
    text frontend.
    """

    def __init__(
        self,
        token_file: Optional[str] = None,
        lang: str = "en",
        use_normalizer: Optional[bool] = None,
        lm_tokenizer: Optional["LanguageModelTokenizer"] = None,
    ):
        """
        Args:
          token_file: phone vocabulary file ('{token}\t{id}' per line),
            same format/vocabulary as EmiliaTokenizer/EspeakTokenizer use.
          lang: canonical short language code -- "en", "vi", or "zh". "en"/
            "vi" phonemize the whole utterance under one espeak-ng locale
            (matching plain EspeakTokenizer's behavior exactly, modulo
            `use_normalizer`). "zh" uses EmiliaTokenizer's existing
            Han/Latin-script-routed path (English words embedded in Chinese
            text are still correctly phonemized via the English path, not
            pinyin) -- `use_normalizer` applies uniformly to both the
            Chinese-native and embedded-English content in that case.
          use_normalizer: opt-in for this tokenizer's own normalization step
            (EmiliaTokenizer's existing English/Chinese normalizers, reused
            here -- EmiliaTokenizer/EspeakTokenizer themselves are never
            modified). `None` (the default) resolves to
            `_DEFAULT_USE_NORMALIZER[lang]` (``True`` for "en"/"zh", ``False``
            for "vi"). Vietnamese is expected to stay disabled -- Vietnamese
            text is normalized by a separate, external pipeline before it
            reaches this tokenizer; enabling it here is a logged no-op,
            since this module has no Vietnamese normalizer. Whether
            normalization is enabled/disabled is logged at construction
            time.
          lm_tokenizer: an optional, already-constructed
            `LanguageModelTokenizer` (shared across FusionTokenizer
            instances, since constructing one loads a HF tokenizer). Must
            have been constructed with a ``[LANG:{lang}]`` control token
            for this instance's `lang` (true by default -- see
            `LanguageModelTokenizer.DEFAULT_LANG_TAGS`). When given,
            `text_to_artifact()` also produces `lm_tokens`/`lm_token_ids`
            (with that deterministic tag prepended as index 0 -- this
            architecture always uses the immutable true-language tag, never
            `[LANG:auto]` or a wrong tag, so it is not left to a caller to
            add) and a phone<->lm_token group correspondence (group indices
            into `lm_token_ids` are offset by +1 to account for the
            prepended tag), computed over the same canonical
            (post-normalization) text as the phones. When `None` (the
            default), `text_to_artifact()` still works but leaves
            `lm_tokens`/`lm_token_ids` empty and `lm_token_groups` as
            `None`.
        """
        assert lang in ("en", "vi", "zh"), lang
        self.lang = lang
        self.lm_tokenizer = lm_tokenizer

        self.use_normalizer: bool = (
            _DEFAULT_USE_NORMALIZER[lang] if use_normalizer is None else use_normalizer
        )
        self._log_normalizer_status()

        self.has_tokens = False
        if token_file is None:
            logging.debug(
                "Initialize FusionTokenizer without tokens file, "
                "will fail when mapping to ids."
            )
            return
        self.token2id: Dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                token, id_ = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id_
        self.pad_id = self.token2id["_"]
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def _log_normalizer_status(self) -> None:
        if self.use_normalizer:
            logging.info(
                f"FusionTokenizer(lang={self.lang!r}): normalization ENABLED."
            )
        else:
            logging.warning(
                f"FusionTokenizer(lang={self.lang!r}): normalization DISABLED "
                "-- text for this language must already be normalized "
                "upstream before reaching this tokenizer."
            )
        if self.lang == "vi" and self.use_normalizer:
            logging.warning(
                "FusionTokenizer: use_normalizer=True was requested for "
                "lang='vi', but this module has no Vietnamese normalizer to "
                "apply -- this flag is a no-op for 'vi', Vietnamese text is "
                "always phonemized as-is."
            )

    def flat_phones(self, text: str) -> List[str]:
        """The flat phone sequence for `text` (post-normalization, if
        enabled for this tokenizer's language), before vocabulary
        filtering. With `use_normalizer` left at its default, equal to
        EspeakTokenizer(lang="vi").g2p(text) for Vietnamese, or
        EmiliaTokenizer().texts_to_tokens([text])[0] for English/Chinese --
        this equality (for the default config) is the hard requirement
        checked by tests/test_fusion_tokenizer.py.
        """
        groups, _, _ = self._frontend(text)
        return [symbol for group in groups for symbol in group]

    def phone_groups(self, text: str) -> List[List[str]]:
        """Word-equivalent phone groups for `text`. Flattening this list of
        lists reproduces `flat_phones(text)` exactly."""
        groups, _, _ = self._frontend(text)
        return groups

    def text_to_artifact(self, text: str) -> FusionTokenizerArtifact:
        assert (
            self.has_tokens
        ), "Please initialize FusionTokenizer with a tokens file."
        groups, word_blocks, canonical_text = self._frontend(text)

        flat_phones: List[str] = []
        raw_groups: List[List[int]] = []  # indices into flat_phones (pre-OOV)
        for group in groups:
            start = len(flat_phones)
            flat_phones.extend(group)
            raw_groups.append(list(range(start, len(flat_phones))))

        phone_ids: List[int] = []
        raw_to_filtered: Dict[int, int] = {}
        for raw_idx, symbol in enumerate(flat_phones):
            if symbol not in self.token2id:
                logging.debug(f"Skip OOV {symbol}")
                continue
            raw_to_filtered[raw_idx] = len(phone_ids)
            phone_ids.append(self.token2id[symbol])

        # OOV filtering can empty a group; drop it and its words together
        # so the two sides stay in step.
        filtered_groups: List[List[int]] = []
        filtered_words: List[List[str]] = []
        blocks = word_blocks if word_blocks is not None else [None] * len(raw_groups)
        for raw_group, block in zip(raw_groups, blocks):
            filtered = [raw_to_filtered[i] for i in raw_group if i in raw_to_filtered]
            if filtered:
                filtered_groups.append(filtered)
                if block is not None:
                    filtered_words.append(block)

        filtered_phones = [""] * len(phone_ids)
        for raw_idx, filtered_idx in raw_to_filtered.items():
            filtered_phones[filtered_idx] = flat_phones[raw_idx]

        lm_tokens, lm_token_ids, lm_token_groups = self._lm_tokens_and_groups(
            canonical_text, filtered_words if word_blocks is not None else None
        )
        zero_duration_mask = (
            self.lm_tokenizer.zero_duration_mask(lm_token_ids)
            if self.lm_tokenizer is not None
            else []
        )

        return FusionTokenizerArtifact(
            phones=filtered_phones,
            phone_ids=phone_ids,
            phone_groups=filtered_groups,
            language=self.lang,
            lm_tokens=lm_tokens,
            lm_token_ids=lm_token_ids,
            lm_token_groups=lm_token_groups,
            zero_duration_mask=zero_duration_mask,
        )

    def _lm_tokens_and_groups(
        self,
        canonical_text: str,
        word_blocks: Optional[List[List[str]]],
    ) -> Tuple[List[str], List[int], Optional[List[List[int]]]]:
        """lm_tokens for the utterance, grouped to match `word_blocks`.

        Each block's words are tokenized on their own and concatenated,
        which reproduces the whole-text tokenization exactly because Qwen's
        BPE is concatenative at whitespace boundaries -- the vocabulary's
        only multi-word tokens are runs of whitespace, so no merge crosses a
        space between real words. That is asserted below rather than
        trusted, and it is what makes the group boundaries exact instead of
        inferred from character offsets.

        The previous implementation mapped word character-spans onto
        `offset_mapping`, which left standalone space tokens belonging to no
        group at all (measured: 0.30% of lm_tokens). Grouping by
        construction removes that class of gap entirely.
        """
        if self.lm_tokenizer is None:
            return [], [], None

        hf = self.lm_tokenizer.hf_tokenizer
        # This fusion architecture always uses the deterministic, immutable
        # `[LANG:xx]` tag for `self.lang` (never `[LANG:auto]` or a wrong
        # tag -- see the 2026-09-02 ADR's point 2). It occupies lm_token
        # index 0, a control token that belongs to no group, so every
        # group's indices below start at 1.
        tag = f"[LANG:{self.lang}]"
        assert tag in self.lm_tokenizer.lang_tags, (
            f"lm_tokenizer was not constructed with the {tag!r} control "
            f"token (has {self.lm_tokenizer.lang_tags}) -- FusionTokenizer "
            "requires it to prepend the utterance's language tag."
        )
        tag_id = hf.convert_tokens_to_ids(tag)

        if word_blocks is None:
            # No word segmentation for this path (Chinese, see `_frontend`).
            ids = hf.encode(canonical_text, add_special_tokens=False)
            return (
                [tag] + hf.convert_ids_to_tokens(ids),
                [tag_id] + ids,
                None,
            )

        lm_token_ids: List[int] = [tag_id]
        lm_token_groups: List[List[int]] = []
        first = True
        for block in word_blocks:
            group: List[int] = []
            for word in block:
                ids = hf.encode(
                    word if first else " " + word, add_special_tokens=False
                )
                first = False
                group.extend(range(len(lm_token_ids), len(lm_token_ids) + len(ids)))
                lm_token_ids.extend(ids)
            lm_token_groups.append(group)

        lm_tokens = [tag] + hf.convert_ids_to_tokens(lm_token_ids[1:])

        words = [w for block in word_blocks for w in block]
        if words and " ".join(words) == canonical_text:
            whole = hf.encode(canonical_text, add_special_tokens=False)
            assert lm_token_ids[1:] == whole, (
                "per-word tokenization did not reproduce the whole-text "
                "tokenization; Qwen's BPE is not concatenative here, so "
                "these group boundaries would not correspond to the real "
                "lm_token sequence"
            )
        if any(not g for g in lm_token_groups):
            # A word that produces no tokens would give a group with no Qwen
            # evidence, which the extractor marks invalid exactly like
            # padding -- invisible downstream. Refuse instead.
            logging.warning(
                "FusionTokenizer: a word produced no lm_tokens; leaving "
                "lm_token_groups=None rather than emitting an empty group."
            )
            return lm_tokens, lm_token_ids, None
        return lm_tokens, lm_token_ids, lm_token_groups

    # -- internal: unified phone-groups + canonical-text-span frontend --

    def _frontend(
        self, text: str
    ) -> Tuple[List[List[str]], Optional[List[List[str]]], str]:
        """Returns (phone_blocks, word_blocks, canonical_text).

        `word_blocks[i]` are the whitespace words whose phones are
        `phone_blocks[i]` -- usually exactly one word, several only where
        espeak's pronunciation of a word depends on its neighbours. `None`
        means this path provides no word segmentation, and the utterance
        gets no `lm_token_groups` (Chinese; see
        `_zh_phone_groups_and_spans`).

        `canonical_text` is the exact text the phones were derived from,
        i.e. `text` after this tokenizer's own opt-in normalization.
        """
        if self.lang == "zh":
            return self._zh_phone_groups_and_spans(text)
        return self._plain_phone_groups_and_spans(text)

    # -- internal: plain espeak path (en/vi), matching EspeakTokenizer.g2p()
    # when use_normalizer[lang] is False --

    def _plain_phone_groups_and_spans(
        self, text: str
    ) -> Tuple[List[List[str]], Optional[List[List[str]]], str]:
        """Blocks for the plain espeak path (en/vi).

        Returns (phone_blocks, word_blocks, canonical_text) where
        `word_blocks[i]` are the whitespace words whose phones are
        `phone_blocks[i]`. A block is one word wherever possible and several
        only where espeak forces it -- see `_blocks_from_words`.
        """
        if self.lang == "en" and self.use_normalizer:
            canonical_text = _shared_emilia_tokenizer().english_normalizer.normalize(
                text
            )
        else:
            canonical_text = text
        locale = _ESPEAK_LOCALE[self.lang]
        flat = self._espeak_flat_phones(canonical_text, locale)
        words = canonical_text.split()
        phone_blocks, word_blocks = self._blocks_from_words(flat, words, locale)
        return phone_blocks, word_blocks, canonical_text

    def _blocks_from_words(
        self, flat: List[str], words: List[str], locale: str
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Cut `flat` into blocks, one per group of consecutive `words`.

        See docs/adr/2026-09-07__phone_lm_token_group_alignment.md. A block
        is widened until the words in it phonemize, on their own, to the
        phones actually present at that position -- separators and
        punctuation ignored on both sides. Widening goes backward as well as
        forward, because the context that decides a word's pronunciation
        usually sits to its left: "the" is Vietnamese 'tˈɛ' alone and stays
        so in "the mang", but is 'ðə' in "book of the".

        This cannot fail. If nothing matches, the block keeps widening until
        it is the whole remaining utterance -- a coarse alignment, not a
        dropped utterance. The phones emitted are always slices of `flat`,
        never the per-word probes, so cross-word phonology is preserved.
        """
        if not words:
            return ([flat], [[]]) if flat else ([], [])

        phone_blocks: List[List[str]] = []
        word_blocks: List[List[str]] = []
        cursor = 0
        i = 0
        while i < len(words):
            must_cover = i
            emitted = False
            while True:
                upper = min(i + _MAX_BLOCK_WORDS, len(words))
                for j in range(max(i + 1, must_cover + 1), upper + 1):
                    probe = self._word_phones(" ".join(words[i:j]), locale)
                    end = _match_phone_content(flat, cursor, probe)
                    # `end == cursor` means the words consumed no phones at
                    # all -- a punctuation-only token like "-" that espeak
                    # renders as nothing. Emitting it as its own group would
                    # give a group with no phones, which the artifact
                    # forbids and the extractor would mark invalid exactly
                    # like padding. Widen instead, so it joins its neighbour.
                    if end is not None and end > cursor:
                        phone_blocks.append(flat[cursor:end])
                        word_blocks.append(words[i:j])
                        cursor, i = end, j
                        emitted = True
                        break
                if emitted or not phone_blocks:
                    break
                # Absorb the previous block and retry with more left context.
                prev_phones = phone_blocks.pop()
                prev_words = word_blocks.pop()
                cursor -= len(prev_phones)
                i -= len(prev_words)
            if not emitted:
                tail = flat[cursor:]
                if tail or not phone_blocks:
                    phone_blocks.append(tail)
                    word_blocks.append(words[i:])
                else:
                    # No phones left (trailing punctuation-only words):
                    # attach the words to the last real group.
                    word_blocks[-1] = word_blocks[-1] + words[i:]
                cursor, i = len(flat), len(words)

        assert all(phone_blocks), "no phone block may be empty"
        assert [p for b in phone_blocks for p in b] == flat, (
            "block construction must not change the phone sequence"
        )
        assert [w for b in word_blocks for w in b] == words, (
            "block construction must not lose or reorder words"
        )
        return phone_blocks, word_blocks

    def _word_phones(self, word: str, locale: str) -> List[str]:
        """Phones for a single word, cached. Only ever used to count groups,
        never emitted -- see `_plain_phone_groups_and_spans`."""
        key = (locale, word)
        cached = _WORD_PHONE_CACHE.get(key)
        if cached is None:
            try:
                cached = _flatten_espeak_output(phonemize_espeak(word, locale))
            except Exception:
                cached = []
            if len(_WORD_PHONE_CACHE) < _WORD_PHONE_CACHE_MAX:
                _WORD_PHONE_CACHE[key] = cached
        return cached

    def _espeak_flat_phones(self, text: str, locale: str) -> List[str]:
        try:
            sentences = phonemize_espeak(text, locale)
        except Exception as ex:
            # Deliberately not swallowed into an empty list. Returning []
            # here makes `groups` and `spans` BOTH empty, so the alignment
            # check downstream sees equal lengths with no missing spans and
            # returns `lm_token_groups=[]` -- a phonemizer crash then looks
            # exactly like a successful alignment of an utterance that
            # happens to have no words. The empty phone sequence goes on to
            # fail much later in duration allocation
            # (`assert n_acoustic_tokens > 0`), far from the cause and with
            # the original exception long gone. Raise with the reason
            # attached instead.
            raise ValueError(
                f"phonemization failed for locale {locale!r} on text "
                f"{text[:80]!r}: {ex}"
            ) from ex
        phones = _flatten_espeak_output(sentences)
        if text.strip() and not phones:
            raise ValueError(
                f"phonemization of locale {locale!r} produced no phones for "
                f"non-empty text {text[:80]!r}. An empty phone sequence is "
                f"not a usable training sample, and passes the alignment "
                f"check silently because empty groups trivially match empty "
                f"spans."
            )
        return phones

    # -- internal: Emilia-routed path (zh), matching
    # EmiliaTokenizer.texts_to_tokens() when use_normalizer is True
    # (the default for lang="zh") --

    def _zh_phone_groups_and_spans(
        self, text: str
    ) -> Tuple[List[List[str]], Optional[List[List[str]]], str]:
        """Chinese phone groups. Returns `None` for the word blocks, so the
        artifact carries no `lm_token_groups` and Chinese trains phone-only.

        This is deliberate, and stricter than what it replaces. The previous
        character-span mapping produced **overlapping** groups here --
        measured, "你好世界。今天天气很好" gave groups
        [[1],[2],[3],[4,5],[6],[6]], with lm_token 6 assigned to two
        different phone groups and a one-position shift after punctuation.
        Wrong conditioning is worse than none, so it is not emitted.

        The block method used for en/vi does not port directly: it needs a
        word segmentation whose per-piece tokenization concatenates back to
        the whole-text tokenization, and Chinese has no whitespace, so that
        property does not hold at jieba boundaries. Chinese is 0% of this
        project's corpus; enabling it needs its own alignment design plus
        the segment-preservation tests noted in the 2026-09-02 ADR's known
        gaps.
        """
        emilia = _shared_emilia_tokenizer()
        # Full/half-width punctuation mapping required for get_segment()'s
        # own routing to work correctly -- not gated by use_normalizer,
        # since it is a segmentation prerequisite, not the
        # number/abbreviation-expansion normalization use_normalizer
        # controls, and was already unconditionally applied before this
        # option existed.
        text = emilia.preprocess_text(text)

        groups: List[List[str]] = []
        canonical_parts: List[str] = []
        for seg_text, seg_lang in emilia.get_segment(text):
            if seg_lang == "zh":
                seg_groups, _, seg_canonical = self._zh_segment_groups_and_spans(
                    seg_text, emilia, 0
                )
            elif seg_lang == "en":
                normalized = (
                    emilia.english_normalizer.normalize(seg_text)
                    if self.use_normalizer
                    else seg_text
                )
                seg_groups = _split_into_groups(
                    self._espeak_flat_phones(normalized, "en-us")
                )
                seg_canonical = normalized
            elif seg_lang == "pinyin":
                phone = emilia.tokenize_pinyin(seg_text)
                seg_groups = [phone] if phone else []
                seg_canonical = seg_text
            elif seg_lang == "tag":
                seg_groups = [[seg_text]]
                seg_canonical = seg_text
            else:
                logging.warning(
                    "No English or Chinese characters found, "
                    f"skipping segment of unknown language: {(seg_text, seg_lang)}"
                )
                seg_groups, seg_canonical = [], ""

            groups.extend(seg_groups)
            canonical_parts.append(seg_canonical)

        return groups, None, "".join(canonical_parts)

    def _zh_segment_groups_and_spans(
        self, seg_text: str, emilia: EmiliaTokenizer, base_cursor: int
    ) -> Tuple[List[List[str]], List[Optional[Tuple[int, int]]], str]:
        """One group per jieba segment. Phonologically equivalent to
        EmiliaTokenizer.tokenize_ZH()'s single `lazy_pinyin(segs, ...)` call
        on the whole segment list, since pypinyin's tone-sandhi converter
        already processes each supplied list element independently (verified
        against the installed pypinyin source) -- calling it once per
        segment instead of once on the whole list changes nothing about the
        resulting phones, only how boundaries are recovered.

        Group character spans are computed against this segment's own
        canonical (post-normalization, if enabled) text -- `base_cursor` is
        the caller's running offset into the overall canonical text this
        segment's text is appended to.
        """
        if self.use_normalizer:
            try:
                text = emilia.chinese_normalizer.normalize(seg_text)
            except Exception as ex:
                logging.warning(f"Tokenization of Chinese text failed: {ex}")
                return [], [], ""
        else:
            text = seg_text

        groups: List[List[str]] = []
        spans: List[Optional[Tuple[int, int]]] = []
        cursor = 0
        for jieba_seg in jieba.cut(text):
            if not jieba_seg:
                continue
            start = cursor
            cursor += len(jieba_seg)
            full = lazy_pinyin(
                [jieba_seg],
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )
            phones: List[str] = []
            for x in full:
                if not (x[0:-1].isalpha() and x[-1] in ("1", "2", "3", "4", "5")):
                    phones.append(x)
                else:
                    phones.extend(emilia.seperate_pinyin(x))
            if phones:
                groups.append(phones)
                spans.append((base_cursor + start, base_cursor + cursor))
        return groups, spans, text
