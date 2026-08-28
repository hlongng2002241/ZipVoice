# Copyright      2026  (authors: LongNH, with Claude Code assistance)
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

"""MultilingualTokenizer: reuse an existing pretrained multilingual
tokenizer's subword vocabulary instead of per-language phonemization.

Deliberately kept in its own module, importing only `transformers` and the
dependency-free Tokenizer interface in `zipvoice.tokenizer.base` -- NOT
`zipvoice.tokenizer.tokenizer`, which hard-requires jieba/piper_phonemize/
pypinyin at import time for its phoneme-based tokenizers. Coupling this class
to that module would force anyone using this pretrained-tokenizer-based front
end to install a phonemizer stack it doesn't need, defeating the point of
docs/adr/2026-08-28__adopt_pretrained_multilingual_tokenizer.md.
"""

import random
from typing import List, Optional

from zipvoice.tokenizer.base import Tokenizer

LANGUAGES = ("en", "vi", "zh")

# Free-form language names/codes (as found in real corpus metadata, e.g. the
# "language": "Vietnamese" field in the user's YouTube-corpus schema) mapped
# to LANGUAGES. Lowercase-matched; extend as new corpora surface new spellings.
_LANGUAGE_ALIASES = {
    "en": "en", "eng": "en", "english": "en",
    "vi": "vi", "vie": "vi", "vietnamese": "vi",
    "zh": "zh", "chi": "zh", "chinese": "zh", "mandarin": "zh", "cmn": "zh",
}


def normalize_language_name(name: Optional[str]) -> Optional[str]:
    """Map a free-form language name/code to one of LANGUAGES, or None if
    `name` is missing/unrecognized. Case-insensitive.
    """
    if not name:
        return None
    return _LANGUAGE_ALIASES.get(name.strip().lower())


def sample_lang_tag(
    true_lang: str,
    auto_prob: float,
    wrong_tag_prob: float,
    rng: random.Random,
) -> str:
    """Sample which ``[LANG:xx]`` tag to prepend to a training utterance.

    Implements the three-condition training/eval design from
    docs/proposals/2026-08-28__primary_language_conditioning.md's "Decided"
    section: most of the time use the ground-truth tag; sometimes replace it
    with ``[LANG:auto]`` (so the model still works when a caller doesn't
    specify a language, mirroring OmniVoice's label dropout); occasionally use
    a *deliberately wrong* tag (to later probe whether the model actually
    uses the tag or ignores it).

    Args:
      true_lang: the utterance's ground-truth language, one of LANGUAGES.
      auto_prob: probability of using ``[LANG:auto]`` instead of the true tag.
      wrong_tag_prob: probability of using a deliberately incorrect tag.
        ``auto_prob + wrong_tag_prob`` must be <= 1.
      rng: a ``random.Random`` instance, passed in explicitly (not the global
        random module) so callers get reproducible, testable sampling.

    Returns:
      One of ``"[LANG:en]"``, ``"[LANG:vi]"``, ``"[LANG:zh]"``,
      ``"[LANG:auto]"``.
    """
    assert true_lang in LANGUAGES, true_lang
    assert 0.0 <= auto_prob <= 1.0, auto_prob
    assert 0.0 <= wrong_tag_prob <= 1.0, wrong_tag_prob
    assert auto_prob + wrong_tag_prob <= 1.0, (auto_prob, wrong_tag_prob)

    draw = rng.random()
    if draw < auto_prob:
        return "[LANG:auto]"
    if draw < auto_prob + wrong_tag_prob:
        wrong_choices = [lang for lang in LANGUAGES if lang != true_lang]
        return f"[LANG:{rng.choice(wrong_choices)}]"
    return f"[LANG:{true_lang}]"


class MultilingualTokenizer(Tokenizer):
    """Tokenizer that reuses an existing pretrained multilingual tokenizer's
    subword vocabulary instead of per-language phonemization, extended with
    reserved ``[LANG:xx]`` control tokens.

    Default base model: **Qwen2.5-0.5B's tokenizer** (byte-level BPE), per
    docs/adr/2026-08-28__choose_qwen25_tokenizer.md. mBERT (WordPiece) was the
    original candidate and remains supported (pass
    ``pretrained_model_name="google-bert/bert-base-multilingual-cased"``), but
    was rejected after real-corpus-at-scale analysis showed a pervasive
    (~0.8-1.0%) `[UNK]` rate across English, Vietnamese, *and* Chinese web
    text, plus meaningfully worse compression for English and (badly) for
    Chinese -- see
    docs/plans/2026-08-28__multilingual_tts_frontend/eval_sets/real_corpus_token_stats.md.
    Qwen2.5-0.5B's byte-level BPE cannot produce `[UNK]` for any input by
    construction, which is what decided it.

    Unlike EmiliaTokenizer, there is no per-language Unicode-range
    segmentation or G2P step here -- the wrapped tokenizer's own vocabulary
    already spans multiple languages. See
    docs/adr/2026-08-28__adopt_pretrained_multilingual_tokenizer.md and
    docs/adr/2026-08-28__language_hint_as_special_tokens.md.

    ``[LANG:xx]`` tags are plain substrings expected to already be present in
    the input text (e.g. ``"[LANG:vi] Xin chào..."``), the same way
    DialogTokenizer expects ``[S1]``/``[S2]`` to already be in the text --
    ``add_special_tokens`` guarantees the wrapped tokenizer treats each tag as
    exactly one token rather than splitting it.
    """

    DEFAULT_LANG_TAGS = ["[LANG:en]", "[LANG:vi]", "[LANG:zh]", "[LANG:auto]"]

    # Historical, mBERT-specific: discovered by analyzing real Vietnamese ASR
    # transcripts (173,461 utterances, see docs/plans/
    # 2026-08-28__multilingual_tts_frontend/eval_sets/real_corpus_token_stats.md)
    # that mBERT's cased vocabulary is missing these capitalized Vietnamese
    # interjections/words as usable WordPiece pieces (their lowercase forms
    # exist). A follow-up systematic check found the gap was much larger than
    # this list (36/68 uppercase Vietnamese diacritic vowels missing
    # entirely), which is part of why mBERT was rejected in favor of
    # Qwen2.5-0.5B -- kept here only for reference/if reverting to mBERT.
    # Not applicable to the default Qwen2.5-0.5B tokenizer, which has zero
    # `[UNK]` by construction and needs no such patch.
    VI_UNK_GAP_TOKENS = [
        "Ờ", "Ừ", "Ồ", "Ơ", "Ồn", "Ổn", "Ơi", "Ổng", "Ạ", "Ổ", "Ùa", "Ớt",
        "Ếch", "Ổi", "Ỷ", "Ầm", "Ụp", "Ửng", "Ới", "VIỆT", "Ừm", "Ế",
        "KIẾM", "TIỀN", "Ơn", "CẦN", "GIỜ",
    ]

    def __init__(
        self,
        pretrained_model_name: str = "Qwen/Qwen2.5-0.5B",
        lang_tags: Optional[List[str]] = None,
        extra_tokens: Optional[List[str]] = None,
    ):
        """
        Args:
          pretrained_model_name: a HuggingFace model id or local path whose
            tokenizer will be loaded via ``AutoTokenizer.from_pretrained``.
          lang_tags: the control tokens to reserve, in addition to the
            wrapped tokenizer's own vocabulary. Defaults to
            ``DEFAULT_LANG_TAGS``.
          extra_tokens: ordinary (non-control) vocabulary words to add via
            ``add_tokens`` -- for filling empirically-discovered coverage
            gaps like ``VI_UNK_GAP_TOKENS``. Unlike `lang_tags`, these
            participate in normal WordPiece matching for any future text,
            not just exact whole-word matches, and are not excluded from
            duration allocation.
        """
        try:
            from transformers import AutoTokenizer
        except Exception as ex:
            raise RuntimeError(f"{ex}\nPlease run\npip install transformers")

        self.hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if extra_tokens:
            self.hf_tokenizer.add_tokens(list(extra_tokens))
        self.lang_tags = (
            list(lang_tags) if lang_tags is not None else list(self.DEFAULT_LANG_TAGS)
        )
        self.hf_tokenizer.add_special_tokens(
            {"additional_special_tokens": self.lang_tags}
        )
        self.control_token_ids = set(
            self.hf_tokenizer.convert_tokens_to_ids(tag) for tag in self.lang_tags
        )
        self.pad_id = self.hf_tokenizer.pad_token_id
        self.vocab_size = len(self.hf_tokenizer)
        self.has_tokens = True

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        # No text normalization here by design -- number/date/abbreviation
        # expansion etc. is a data-preparation concern and must already be
        # done upstream before a corpus reaches training. See
        # docs/plans/2026-08-28__multilingual_tts_frontend/2026-08-28__sprint_003__tokenizer_embedding_bakeoff.md.
        return [
            self.hf_tokenizer.encode(text, add_special_tokens=False)
            for text in texts
        ]

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        return [
            self.hf_tokenizer.convert_ids_to_tokens(ids)
            for ids in self.texts_to_token_ids(texts)
        ]

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        return [
            self.hf_tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list
        ]

    def zero_duration_mask(self, token_ids: List[int]) -> List[bool]:
        """Per-token mask, True where a token is a ``[LANG:xx]`` control
        token that must receive zero acoustic duration -- see
        docs/plans/2026-08-28__multilingual_tts_frontend/
        2026-08-28__sprint_002__special_tokens_and_duration.md.
        """
        return [token_id in self.control_token_ids for token_id in token_ids]
