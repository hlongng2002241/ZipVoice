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


def _split_into_groups(flat_phones: List[str]) -> List[List[str]]:
    """Split a flat phone sequence into word-equivalent groups on the
    literal ' ' phone symbol (which `phonemize_espeak` emits as an ordinary
    element -- confirmed empirically). The space itself is kept (not
    dropped) as the trailing element of the group before it, so flattening
    the returned groups reproduces `flat_phones` exactly, unchanged.
    """
    groups: List[List[str]] = []
    current: List[str] = []
    for symbol in flat_phones:
        current.append(symbol)
        if symbol == " ":
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
        groups, spans, canonical_text = self._frontend(text)

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

        filtered_groups: List[List[int]] = []
        filtered_spans: List[Optional[Tuple[int, int]]] = []
        for raw_group, span in zip(raw_groups, spans):
            filtered = [raw_to_filtered[i] for i in raw_group if i in raw_to_filtered]
            if filtered:
                filtered_groups.append(filtered)
                filtered_spans.append(span)

        filtered_phones = [""] * len(phone_ids)
        for raw_idx, filtered_idx in raw_to_filtered.items():
            filtered_phones[filtered_idx] = flat_phones[raw_idx]

        lm_tokens, lm_token_ids, lm_token_groups = self._lm_tokens_and_groups(
            canonical_text, filtered_groups, filtered_spans
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
        filtered_groups: List[List[int]],
        filtered_spans: List[Optional[Tuple[int, int]]],
    ) -> Tuple[List[str], List[int], Optional[List[List[int]]]]:
        if self.lm_tokenizer is None:
            return [], [], None

        # This fusion architecture always uses the deterministic, immutable
        # `[LANG:xx]` tag for `self.lang` (never `[LANG:auto]` or a wrong
        # tag -- see the ADR's point 2, which requires --lang-auto-prob and
        # --lang-wrong-prob both be 0 for this architecture), so it is
        # prepended here rather than left to a caller. It occupies
        # lm_token index 0 -- a control token, never part of any group --
        # so every group's lm_token indices are offset by +1 below.
        tag = f"[LANG:{self.lang}]"
        assert tag in self.lm_tokenizer.lang_tags, (
            f"lm_tokenizer was not constructed with the {tag!r} control "
            f"token (has {self.lm_tokenizer.lang_tags}) -- FusionTokenizer "
            "requires it to prepend the utterance's language tag."
        )
        tag_id = self.lm_tokenizer.hf_tokenizer.convert_tokens_to_ids(tag)

        encoding = self.lm_tokenizer.hf_tokenizer(
            canonical_text, add_special_tokens=False, return_offsets_mapping=True
        )
        lm_token_ids: List[int] = [tag_id] + encoding["input_ids"]
        lm_tokens: List[str] = [
            tag
        ] + self.lm_tokenizer.hf_tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets: List[Tuple[int, int]] = encoding["offset_mapping"]

        if len(filtered_spans) != len(filtered_groups) or any(
            s is None for s in filtered_spans
        ):
            logging.debug(
                "FusionTokenizer: could not align phone groups with "
                "lm_token character spans (missing/mismatched word spans) "
                "-- lm_token_groups left as None for this utterance."
            )
            return lm_tokens, lm_token_ids, None

        lm_token_groups: List[List[int]] = []
        for start, end in filtered_spans:
            lm_token_groups.append(
                [
                    i + 1
                    for i, (o_start, o_end) in enumerate(offsets)
                    if o_end > start and o_start < end
                ]
            )
        return lm_tokens, lm_token_ids, lm_token_groups

    # -- internal: unified phone-groups + canonical-text-span frontend --

    def _frontend(
        self, text: str
    ) -> Tuple[List[List[str]], List[Optional[Tuple[int, int]]], str]:
        """Returns (phone_groups, group_spans, canonical_text), where
        `group_spans[i]` is the (start, end) character span of group `i`
        within `canonical_text` (or None if unavailable), and
        `canonical_text` is the exact text phones were derived from --
        i.e. `text` after this tokenizer's own opt-in normalization, if
        any. `lm_tokens` are computed over this same `canonical_text` so
        that offsets on both sides line up.
        """
        if self.lang == "zh":
            return self._zh_phone_groups_and_spans(text)
        return self._plain_phone_groups_and_spans(text)

    # -- internal: plain espeak path (en/vi), matching EspeakTokenizer.g2p()
    # when use_normalizer[lang] is False --

    def _plain_phone_groups_and_spans(
        self, text: str
    ) -> Tuple[List[List[str]], List[Optional[Tuple[int, int]]], str]:
        if self.lang == "en" and self.use_normalizer:
            canonical_text = _shared_emilia_tokenizer().english_normalizer.normalize(
                text
            )
        else:
            canonical_text = text
        groups = _split_into_groups(
            self._espeak_flat_phones(canonical_text, _ESPEAK_LOCALE[self.lang])
        )
        word_spans = _whitespace_spans(canonical_text)
        spans: List[Optional[Tuple[int, int]]] = (
            list(word_spans) if len(word_spans) == len(groups) else [None] * len(groups)
        )
        return groups, spans, canonical_text

    def _espeak_flat_phones(self, text: str, locale: str) -> List[str]:
        try:
            sentences = phonemize_espeak(text, locale)
            return _flatten_espeak_output(sentences)
        except Exception as ex:
            logging.warning(f"Tokenization of {locale} text failed: {ex}")
            return []

    # -- internal: Emilia-routed path (zh), matching
    # EmiliaTokenizer.texts_to_tokens() when use_normalizer is True
    # (the default for lang="zh") --

    def _zh_phone_groups_and_spans(
        self, text: str
    ) -> Tuple[List[List[str]], List[Optional[Tuple[int, int]]], str]:
        emilia = _shared_emilia_tokenizer()
        # Full/half-width punctuation mapping required for get_segment()'s
        # own routing to work correctly -- not gated by use_normalizer,
        # since it is a segmentation prerequisite, not the
        # number/abbreviation-expansion normalization use_normalizer
        # controls, and was already unconditionally applied before this
        # option existed.
        text = emilia.preprocess_text(text)

        groups: List[List[str]] = []
        spans: List[Optional[Tuple[int, int]]] = []
        canonical_parts: List[str] = []
        cursor = 0
        for seg_text, seg_lang in emilia.get_segment(text):
            if seg_lang == "zh":
                seg_groups, seg_spans, seg_canonical = self._zh_segment_groups_and_spans(
                    seg_text, emilia, cursor
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
                word_spans = _whitespace_spans(normalized)
                seg_spans = (
                    [(s + cursor, e + cursor) for s, e in word_spans]
                    if len(word_spans) == len(seg_groups)
                    else [None] * len(seg_groups)
                )
                seg_canonical = normalized
            elif seg_lang == "pinyin":
                phone = emilia.tokenize_pinyin(seg_text)
                seg_groups = [phone] if phone else []
                seg_spans = [(cursor, cursor + len(seg_text))] if phone else []
                seg_canonical = seg_text
            elif seg_lang == "tag":
                seg_groups = [[seg_text]]
                seg_spans = [(cursor, cursor + len(seg_text))]
                seg_canonical = seg_text
            else:
                logging.warning(
                    "No English or Chinese characters found, "
                    f"skipping segment of unknown language: {(seg_text, seg_lang)}"
                )
                seg_groups, seg_spans, seg_canonical = [], [], ""

            groups.extend(seg_groups)
            spans.extend(seg_spans)
            canonical_parts.append(seg_canonical)
            cursor += len(seg_canonical)

        return groups, spans, "".join(canonical_parts)

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
