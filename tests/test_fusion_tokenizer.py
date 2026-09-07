"""Equivalence tests for FusionTokenizer (see
docs/adr/2026-09-02__qwen_phone_fusion_text_frontend.md and
docs/plans/2026-09-02__qwen_phone_fusion_text_frontend/2026-09-02__sprint_000__new_tokenizer.md).

Hard requirement: with `use_normalizer` left at its default (English/Chinese
enabled, Vietnamese disabled), FusionTokenizer's flat phone output must be
*identical* to what EspeakTokenizer/EmiliaTokenizer already produce for the
same (correspondingly normalized) text -- FusionTokenizer only adds grouping
metadata and optional, explicit normalization on top of unchanged
phonemization, it never reimplements it. This is checked at scale against
real corpus/production text (up to 1000 samples per language), since edge
cases in real text (numbers, mixed scripts, unusual punctuation) are exactly
what a small synthetic fixture would miss.

- English/Chinese samples come from data/corpus/{en,zh}/*.jsonl (raw web
  text) -- if not present, the corpus-based tests warn and skip rather than
  fail.
- Vietnamese samples come from tests/fixtures/vi_normalized_samples.jsonl,
  a small fixture of real, *already normalized* Vietnamese utterances
  extracted from data/all/manifests/train.jsonl.gz -- not the raw
  data/corpus/vi/*.jsonl web corpus, which is not representative: Vietnamese
  text is normalized by a separate, external pipeline before it ever reaches
  this tokenizer (hence FusionTokenizer(lang="vi")'s use_normalizer defaults
  to False here), so a fair equivalence test needs text that already
  reflects that pipeline. This fixture is *not* tracked in git (real
  training-utterance text carries the same sensitivity concerns as data/,
  which is also gitignored) -- regenerate it locally with
  scripts/all/build_vi_tokenizer_fixture.py; if it's absent, the VI test
  warns and skips like the EN/ZH corpus-based tests do.
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import List

import pytest

from zipvoice.tokenizer.fusion_tokenizer import (
    SCHEMA_VERSION,
    FusionTokenizer,
    FusionTokenizerArtifact,
)
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer, EspeakTokenizer

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "data" / "corpus"
VI_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vi_normalized_samples.jsonl"
NUM_SAMPLES = 1000


def _load_corpus_samples(lang: str, n: int = NUM_SAMPLES) -> List[str]:
    """Up to `n` short text samples from data/corpus/<lang>/*.jsonl -- the
    first non-empty line of each JSON document's "text" field (documents
    themselves are often multi-paragraph web text; one line per document
    keeps this test fast while still exercising real, diverse corpus text).
    Returns [] if no corpus file is found for `lang`.
    """
    lang_dir = CORPUS_ROOT / lang
    jsonl_files = sorted(lang_dir.glob("*.jsonl")) if lang_dir.is_dir() else []
    if not jsonl_files:
        return []

    samples: List[str] = []
    with open(jsonl_files[0], "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= n:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            first_line = next((l.strip() for l in text.split("\n") if l.strip()), "")
            if first_line:
                samples.append(first_line)
    return samples


def _corpus_samples_or_skip(lang: str) -> List[str]:
    samples = _load_corpus_samples(lang)
    if not samples:
        msg = (
            f"data/corpus/{lang}/*.jsonl not found -- skipping corpus-based "
            f"FusionTokenizer equivalence test for {lang!r}. Run from a "
            "checkout with the corpus data present (see "
            "docs/plans/2026-09-02__qwen_phone_fusion_text_frontend/"
            "2026-09-02__sprint_000__new_tokenizer.md) to exercise this test."
        )
        warnings.warn(msg)
        pytest.skip(msg)
    return samples


def _load_vi_fixture_samples(n: int = NUM_SAMPLES) -> List[str]:
    """Up to `n` real, already-normalized Vietnamese utterances from
    tests/fixtures/vi_normalized_samples.jsonl. Returns [] if the fixture
    is missing.
    """
    if not VI_FIXTURE.is_file():
        return []
    samples: List[str] = []
    with open(VI_FIXTURE, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "").strip()
            if text:
                samples.append(text)
    return samples


def _vi_fixture_samples_or_skip() -> List[str]:
    samples = _load_vi_fixture_samples()
    if not samples:
        msg = (
            f"{VI_FIXTURE} not found -- skipping corpus-based FusionTokenizer "
            "equivalence test for 'vi'. This fixture holds real, "
            "already-normalized Vietnamese text extracted from "
            "data/all/manifests/train.jsonl.gz -- see this test module's "
            "docstring for why the raw data/corpus/vi/*.jsonl web corpus "
            "is not used instead."
        )
        warnings.warn(msg)
        pytest.skip(msg)
    return samples


def _report_mismatches(mismatches, total, label):
    if not mismatches:
        return
    preview = mismatches[:3]
    details = "\n".join(
        f"  [{i}] text={text!r}\n      old={old_phones}\n      new={new_phones}"
        for i, text, old_phones, new_phones in preview
    )
    pytest.fail(
        f"{len(mismatches)}/{total} {label} corpus samples produced "
        f"different phones than the existing tokenizer. First "
        f"{len(preview)} mismatch(es):\n{details}"
    )


def _check_corpus_equivalence(samples, old_phones_fn, new_groups_fn, label):
    """Compute each tokenizer's output in its own clean, uninterrupted pass
    over all samples before comparing -- NOT interleaved per-sample.

    `piper_phonemize`'s underlying espeak-ng engine is stateful across
    calls within a process (confirmed directly: phonemizing the same text
    immediately after a different text can produce different output than
    phonemizing it in isolation). Interleaving old/new calls per-sample
    scrambles that shared engine's state between the two tokenizers and
    produces false mismatches that have nothing to do with tokenizer
    correctness -- each tokenizer must see the same call sequence it would
    see used normally (one tokenizer processing a manifest in sequence, per
    e.g. tokenizer.py's `add_tokens()`), which requires computing all of one
    tokenizer's outputs before starting the other's.
    """
    old_results = [old_phones_fn(text) for text in samples]
    new_group_results = [new_groups_fn(text) for text in samples]

    mismatches = []
    empty_groups_found = []
    for i, (text, old_phones, groups) in enumerate(
        zip(samples, old_results, new_group_results)
    ):
        new_phones = [symbol for group in groups for symbol in group]
        if new_phones != old_phones:
            mismatches.append((i, text, old_phones, new_phones))
        if any(len(g) == 0 for g in groups):
            empty_groups_found.append(i)

    _report_mismatches(mismatches, len(samples), label)
    assert not empty_groups_found, (
        f"{len(empty_groups_found)}/{len(samples)} {label} samples produced "
        f"an empty group, e.g. sample index {empty_groups_found[0]}"
    )


def test_en_corpus_equivalence():
    # FusionTokenizer(lang="en") defaults to use_normalizer["en"]=True, so
    # it normalizes raw text internally before phonemizing. EspeakTokenizer
    # never normalizes, so for a fair equivalence check we normalize the
    # text ourselves (via the same EmiliaTokenizer.english_normalizer
    # FusionTokenizer uses internally) before feeding it to the old
    # tokenizer, and feed the *raw* text to the new one.
    samples = _corpus_samples_or_skip("en")
    old = EspeakTokenizer(lang="en-us")
    new = FusionTokenizer(lang="en")
    emilia = EmiliaTokenizer(token_file=None)
    _check_corpus_equivalence(
        samples,
        lambda text: old.g2p(emilia.english_normalizer.normalize(text)),
        new.phone_groups,
        "en",
    )


def test_en_normalizer_disabled_matches_raw_espeak_exactly():
    # With use_normalizer["en"] explicitly turned off, FusionTokenizer must
    # reproduce EspeakTokenizer's raw (unnormalized) output exactly -- this
    # is the opt-out path real training data (already normalized upstream)
    # would use.
    samples = _corpus_samples_or_skip("en")[:200]
    old = EspeakTokenizer(lang="en-us")
    new = FusionTokenizer(lang="en", use_normalizer=False)
    _check_corpus_equivalence(
        samples, old.g2p, new.phone_groups, "en (normalizer disabled)"
    )


def test_vi_corpus_equivalence():
    # use_normalizer["vi"] defaults to False (and has no effect even if
    # enabled -- see test_vi_normalizer_flag_warns_when_enabled), so this
    # is a plain raw-text equivalence check, same as EspeakTokenizer.
    samples = _vi_fixture_samples_or_skip()
    old = EspeakTokenizer(lang="vi")
    new = FusionTokenizer(lang="vi")
    _check_corpus_equivalence(samples, old.g2p, new.phone_groups, "vi")


def test_zh_corpus_equivalence():
    # use_normalizer defaults to {"en": True, "zh": True}, matching
    # EmiliaTokenizer's own always-on normalization for both Chinese
    # segments and English-embedded-in-Chinese segments -- so this remains
    # a plain raw-text equivalence check.
    samples = _corpus_samples_or_skip("zh")
    old = EmiliaTokenizer()
    new = FusionTokenizer(lang="zh")
    _check_corpus_equivalence(
        samples,
        lambda text: old.texts_to_tokens([text])[0],
        new.phone_groups,
        "zh",
    )


def test_normalizer_status_is_logged_enabled(caplog):
    with caplog.at_level(logging.INFO):
        FusionTokenizer(lang="en")
    messages = " ".join(r.message for r in caplog.records)
    assert "ENABLED" in messages
    assert "'en'" in messages


def test_normalizer_status_is_logged_disabled(caplog):
    with caplog.at_level(logging.WARNING):
        FusionTokenizer(lang="vi")
    messages = " ".join(r.message for r in caplog.records)
    assert "DISABLED" in messages
    assert "'vi'" in messages


def test_vi_normalizer_flag_warns_when_enabled(caplog):
    with caplog.at_level(logging.WARNING):
        FusionTokenizer(lang="vi", use_normalizer=True)
    messages = " ".join(r.message for r in caplog.records)
    assert "no Vietnamese normalizer" in messages


def test_artifact_invariants_with_real_vocab(tmp_path):
    token_file = tmp_path / "tokens.txt"
    old = EspeakTokenizer(lang="en-us")
    text = "hello world, how are you?"
    phones = old.g2p(text)
    # Build a vocab covering every phone symbol that appears, plus pad.
    vocab = ["_"] + sorted(set(phones))
    token_file.write_text(
        "\n".join(f"{tok}\t{i}" for i, tok in enumerate(vocab)), encoding="utf-8"
    )

    new = FusionTokenizer(token_file=str(token_file), lang="en", use_normalizer=False)
    artifact = new.text_to_artifact(text)

    assert artifact.schema_version == 3
    assert artifact.language == "en"
    assert len(artifact.phone_ids) == len(artifact.phones)
    assert artifact.lm_tokens == []
    assert artifact.lm_token_ids == []
    assert artifact.lm_token_groups is None
    # No lm_tokenizer was supplied, so there's no [LANG:xx] tag to flag --
    # zero_duration_mask is lm_token-aligned, not phone-aligned, and empty
    # here exactly like lm_tokens/lm_token_ids.
    assert artifact.zero_duration_mask == []

    # Every phone belongs to exactly one group.
    covered = sorted(idx for group in artifact.phone_groups for idx in group)
    assert covered == list(range(len(artifact.phone_ids)))

    # Groups are monotonic and non-overlapping.
    seen = set()
    last_end = -1
    for group in artifact.phone_groups:
        assert group == sorted(group)
        assert group[0] > last_end
        for idx in group:
            assert idx not in seen
            seen.add(idx)
        last_end = group[-1]

    # phone_ids correctly map through the vocab.
    assert artifact.phone_ids == [new.token2id[p] for p in artifact.phones]


def test_artifact_drops_oov_symbols_consistently(tmp_path):
    # Vocab missing '?' on purpose, to check OOV filtering keeps
    # phone_ids/phone_groups aligned to the post-filtering length.
    token_file = tmp_path / "tokens.txt"
    text = "hello world, how are you?"
    old = EspeakTokenizer(lang="en-us")
    phones = old.g2p(text)
    vocab = ["_"] + sorted(set(phones) - {"?"})
    token_file.write_text(
        "\n".join(f"{tok}\t{i}" for i, tok in enumerate(vocab)), encoding="utf-8"
    )

    new = FusionTokenizer(token_file=str(token_file), lang="en", use_normalizer=False)
    artifact = new.text_to_artifact(text)

    assert "?" not in artifact.phones
    assert len(artifact.phone_ids) == len(phones) - phones.count("?")
    covered = sorted(idx for group in artifact.phone_groups for idx in group)
    assert covered == list(range(len(artifact.phone_ids)))


def test_text_to_artifact_produces_lm_tokens_and_group_correspondence(tmp_path):
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = tmp_path / "tokens.txt"
    text = "hello world, how are you?"
    old = EspeakTokenizer(lang="en-us")
    phones = old.g2p(text)
    vocab = ["_"] + sorted(set(phones))
    token_file.write_text(
        "\n".join(f"{tok}\t{i}" for i, tok in enumerate(vocab)), encoding="utf-8"
    )

    lm_tokenizer = LanguageModelTokenizer()
    new = FusionTokenizer(
        token_file=str(token_file),
        lang="en",
        use_normalizer=False,
        lm_tokenizer=lm_tokenizer,
    )
    artifact = new.text_to_artifact(text)

    assert len(artifact.lm_tokens) == len(artifact.lm_token_ids) > 0
    # The deterministic [LANG:xx] tag is always prepended as index 0.
    assert artifact.lm_tokens[0] == "[LANG:en]"
    assert artifact.lm_token_ids[0] == lm_tokenizer.hf_tokenizer.convert_tokens_to_ids(
        "[LANG:en]"
    )
    # zero_duration_mask is lm_token-aligned and sourced 100% from
    # LanguageModelTokenizer.zero_duration_mask() -- True only for the
    # [LANG:xx] control tag at index 0.
    assert artifact.zero_duration_mask == lm_tokenizer.zero_duration_mask(
        artifact.lm_token_ids
    )
    assert artifact.zero_duration_mask[0] is True
    assert not any(artifact.zero_duration_mask[1:])
    assert artifact.lm_token_groups is not None
    assert len(artifact.lm_token_groups) == len(artifact.phone_groups)
    for group in artifact.lm_token_groups:
        for idx in group:
            assert 0 < idx < len(artifact.lm_token_ids)  # never the tag at index 0


def test_text_to_artifact_rejects_lm_tokenizer_missing_the_lang_tag(tmp_path):
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = tmp_path / "tokens.txt"
    text = "hello world"
    old = EspeakTokenizer(lang="en-us")
    phones = old.g2p(text)
    vocab = ["_"] + sorted(set(phones))
    token_file.write_text(
        "\n".join(f"{tok}\t{i}" for i, tok in enumerate(vocab)), encoding="utf-8"
    )

    # lm_tokenizer constructed without a "[LANG:en]" tag -- FusionTokenizer
    # requires it to prepend this instance's deterministic language tag.
    lm_tokenizer = LanguageModelTokenizer(lang_tags=["[LANG:vi]", "[LANG:zh]"])
    new = FusionTokenizer(
        token_file=str(token_file),
        lang="en",
        use_normalizer=False,
        lm_tokenizer=lm_tokenizer,
    )
    with pytest.raises(AssertionError):
        new.text_to_artifact(text)


def _valid_artifact_kwargs():
    # phones/phone_ids: "ab" split into two single-phone groups. lm_tokens/
    # lm_token_ids/zero_duration_mask default to [] (no lm_tokenizer used).
    return dict(
        language="en",
        phones=["a", "b"],
        phone_ids=[0, 1],
        phone_groups=[[0], [1]],
    )


def test_artifact_post_init_accepts_a_well_formed_artifact():
    artifact = FusionTokenizerArtifact(**_valid_artifact_kwargs())
    assert artifact.schema_version == SCHEMA_VERSION


def test_artifact_post_init_rejects_unknown_language():
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**{**_valid_artifact_kwargs(), "language": "fr"})


def test_artifact_post_init_rejects_stale_schema_version():
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(
            **_valid_artifact_kwargs(), schema_version=SCHEMA_VERSION - 1
        )


def test_artifact_post_init_rejects_phones_phone_ids_length_mismatch():
    kwargs = {**_valid_artifact_kwargs(), "phone_ids": [0]}
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


def test_artifact_post_init_rejects_wrong_zero_duration_mask_length():
    # zero_duration_mask must match lm_token_ids' length (here, default []),
    # not phone_ids' length.
    kwargs = {**_valid_artifact_kwargs(), "zero_duration_mask": [False]}
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


def test_artifact_post_init_rejects_gap_in_phone_groups():
    # Covers index 0 but skips index 1 -- not a full partition of range(2).
    kwargs = {**_valid_artifact_kwargs(), "phone_groups": [[0]]}
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


def test_artifact_post_init_rejects_overlapping_phone_groups():
    kwargs = {**_valid_artifact_kwargs(), "phone_groups": [[0, 1], [1]]}
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


def test_artifact_post_init_rejects_out_of_order_phone_groups():
    kwargs = {**_valid_artifact_kwargs(), "phone_groups": [[1], [0]]}
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


def test_artifact_post_init_rejects_lm_tokens_lm_token_ids_length_mismatch():
    kwargs = {
        **_valid_artifact_kwargs(),
        "lm_tokens": ["a", "b"],
        "lm_token_ids": [0],
    }
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


def test_artifact_post_init_rejects_lm_token_groups_count_mismatch():
    # phone_groups has 2 entries; lm_token_groups only has 1.
    kwargs = {
        **_valid_artifact_kwargs(),
        "lm_tokens": ["a"],
        "lm_token_ids": [0],
        "lm_token_groups": [[0]],
    }
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


def test_artifact_post_init_rejects_out_of_range_lm_token_groups_index():
    kwargs = {
        **_valid_artifact_kwargs(),
        "lm_tokens": ["a"],
        "lm_token_ids": [0],
        "lm_token_groups": [[0], [5]],
    }
    with pytest.raises(AssertionError):
        FusionTokenizerArtifact(**kwargs)


# --- Sentence-boundary grouping ------------------------------------------
#
# These exist because a space-only group split silently merged the last word
# of every sentence with the first word of the next: espeak emits NO ' '
# after a sentence-final period. Each internal sentence boundary cost one
# group, the group count stopped matching the whitespace-word count, and the
# artifact fell back to `lm_token_groups=None` -- disabling the Qwen branch.
# On this project's corpus that fallback fired for 92.7% of utterances, and
# an entire training run was ~93% phone-only before anyone noticed.
#
# The old tests could not have caught it: the only positive alignment test
# used the single-sentence string "hello world, how are you?", and the tests
# that did use real multi-sentence corpus text only compared *phones*, which
# the bug does not affect. See test_vi_corpus_alignment_rate for the
# structural fix -- assert the fallback's RATE on real data, not just its
# correctness on one handpicked string.


def test_split_into_groups_breaks_after_sentence_final_punctuation():
    from zipvoice.tokenizer.fusion_tokenizer import _split_into_groups

    # espeak's real shape at a sentence boundary: '.' with no following ' '.
    flat = ["a", " ", "b", ".", "c", " ", "d"]
    groups = _split_into_groups(flat)
    assert ["".join(g) for g in groups] == ["a ", "b.", "c ", "d"], (
        "the word after a sentence-final period must start a new group"
    )


def test_split_into_groups_does_not_double_break_on_punct_then_space():
    from zipvoice.tokenizer.fusion_tokenizer import _split_into_groups

    # When a space *does* follow the punctuation, the space rule closes the
    # group; breaking on both would emit a group consisting only of ' '.
    groups = _split_into_groups(["a", ".", " ", "b"])
    assert ["".join(g) for g in groups] == ["a. ", "b"]
    assert all(g and "".join(g).strip(" .!?;:") for g in groups), (
        "no group may consist solely of separators"
    )


def test_split_into_groups_flattening_is_lossless():
    """The phone sequence must survive regrouping byte-for-byte -- sprint
    000's hard requirement is that this tokenizer changes grouping only,
    never phonemization.
    """
    from zipvoice.tokenizer.fusion_tokenizer import _split_into_groups

    for flat in (
        ["a", " ", "b", ".", "c"],
        ["x", ".", " ", "y", "!", "z", " "],
        ["solo"],
        ["end", "."],
        [],
    ):
        assert [s for g in _split_into_groups(flat) for s in g] == flat


def test_multi_sentence_text_aligns_en(tmp_path):
    """Multi-sentence English must produce lm_token_groups.

    NOTE: this does NOT exercise the sentence-boundary bug, and must not be
    mistaken for its regression test. espeak-en-us drops the sentence period
    and emits a space, so English never had the merge -- verified: this test
    passes unchanged against the old space-only splitter. The real
    regression is test_multi_sentence_text_aligns_vi below. Kept because
    multi-sentence alignment is worth covering in both voices.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    text = "hello world. how are you today. this is a third sentence."
    old = EspeakTokenizer(lang="en-us")
    phones = old.g2p(text)
    token_file = tmp_path / "tokens.txt"
    vocab = ["_"] + sorted(set(phones))
    token_file.write_text(
        "\n".join(f"{tok}\t{i}" for i, tok in enumerate(vocab)), encoding="utf-8"
    )
    tok = FusionTokenizer(
        token_file=str(token_file),
        lang="en",
        use_normalizer=False,
        lm_tokenizer=LanguageModelTokenizer(),
    )
    artifact = tok.text_to_artifact(text)
    assert artifact.lm_token_groups is not None, (
        "multi-sentence text fell back to phone-only -- the sentence-boundary "
        "merge has regressed"
    )
    assert len(artifact.lm_token_groups) == len(artifact.phone_groups)
    assert len(artifact.phone_groups) == len(text.split())


def test_vi_corpus_alignment_rate():
    """Assert the alignment *success rate* on real corpus text.

    This is the test whose absence let the sentence-boundary bug ship.
    `lm_token_groups=None` is a legal, documented fallback, so no
    correctness assertion anywhere could fail when it fired -- the only
    observable symptom was its frequency, and nothing measured that. The
    threshold is deliberately far below the ~94% currently measured: this
    guards against a collapse of the mechanism, not against normal drift.

    The residual failures are dominated by digits in the text (espeak
    expands "1997" into five spoken words, so one whitespace token yields
    five groups). Vietnamese runs with use_normalizer=False and text is
    required to be normalized upstream, so digit-bearing text is
    out-of-contract rather than a tokenizer defect.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")
    samples = _vi_fixture_samples_or_skip()[:200]
    tok = FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=LanguageModelTokenizer()
    )
    aligned = sum(
        1 for text in samples if tok.text_to_artifact(text).lm_token_groups is not None
    )
    rate = aligned / len(samples)
    assert rate >= 0.80, (
        f"only {aligned}/{len(samples)} ({rate:.1%}) of real Vietnamese "
        f"utterances produced lm_token_groups. Below this threshold the Qwen "
        f"branch is disabled for most of training and the fusion "
        f"architecture silently degrades to phone-only."
    )


def test_vi_corpus_group_correspondence_is_correct_not_just_present():
    """Check the span -> lm_token half of the mapping.

    SCOPE, precisely: word span i is assigned to phone group i *by position*
    once the counts match, so decoding LM group i back to word i validates
    that the character-offset overlap picked the right lm_tokens for that
    span. It does NOT independently establish what phone group i contains --
    a compensating merge and split would still decode correctly here. The
    phone side is covered separately, against a reference that never touches
    group indices, by
    test_vi_phone_groups_match_independent_per_word_phonemization.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer
    import unicodedata

    from zipvoice.tokenizer.fusion_tokenizer import _whitespace_spans

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")

    lm_tokenizer = LanguageModelTokenizer()
    tok = FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=lm_tokenizer
    )

    def canon(s):
        return unicodedata.normalize("NFC", s.strip().strip(".,!?;:\"'()")).lower()

    samples = _vi_fixture_samples_or_skip()[:60]
    checked = 0
    mismatches = []
    for text in samples:
        artifact = tok.text_to_artifact(text)
        if artifact.lm_token_groups is None:
            continue
        words = [text[s:e] for s, e in _whitespace_spans(text)]
        assert len(words) == len(artifact.lm_token_groups), (
            "an aligned artifact must have exactly one group per whitespace word"
        )
        for i, group in enumerate(artifact.lm_token_groups):
            decoded = lm_tokenizer.hf_tokenizer.decode(
                [artifact.lm_token_ids[j] for j in group]
            )
            checked += 1
            if canon(decoded) != canon(words[i]):
                mismatches.append((i, words[i], decoded))

    assert checked > 500, f"too few groups checked ({checked}) to be meaningful"
    assert not mismatches, (
        f"{len(mismatches)}/{checked} groups do not correspond to their word "
        f"-- alignment is accepted but wrong. First 5: {mismatches[:5]}"
    )


def test_multi_sentence_text_aligns_vi():
    """THE regression for the sentence-boundary bug.

    Must be Vietnamese, and must have a capitalised word after the period:
    that is the exact shape espeak-vi emits as `... j '.' z ...` with no
    separating space. Verified discriminating -- against the old space-only
    splitter this text yields 12 groups for 14 words and falls back to
    lm_token_groups=None; with the fix it yields 14 and aligns.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")

    text = "toi di hoc moi ngay. Ban di lam moi tuan. Chung ta gap nhau"
    tok = FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=LanguageModelTokenizer()
    )
    artifact = tok.text_to_artifact(text)
    assert len(artifact.phone_groups) == len(text.split()), (
        f"expected one phone group per word, got {len(artifact.phone_groups)} "
        f"for {len(text.split())} words -- sentence boundaries are merging again"
    )
    assert artifact.lm_token_groups is not None, (
        "multi-sentence Vietnamese fell back to phone-only -- the "
        "sentence-boundary merge has regressed"
    )


def test_vi_phone_groups_match_independent_per_word_phonemization():
    """Independent phone-to-word check.

    The span/LM-token test above cannot establish this on its own: word span
    i is assigned to phone group i *by position* after count matching, so
    decoding LM group i back to word i is partly circular -- a compensating
    merge and split would still decode correctly. This reference never
    touches group indices: it phonemizes each word alone and compares.

    Scoped to Vietnamese, where per-word phonemization was verified to
    reproduce whole-utterance output exactly (4521/4521 groups). Other
    voices have contextual pronunciation that this reference would not
    reproduce, so do not widen it without re-establishing that.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer
    from zipvoice.tokenizer.tokenizer import phonemize_espeak

    from zipvoice.tokenizer.fusion_tokenizer import _whitespace_spans

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")

    tok = FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=LanguageModelTokenizer()
    )
    samples = _vi_fixture_samples_or_skip()[:15]
    # espeak is stateful across calls, so do one clean pass per side.
    artifacts = [tok.text_to_artifact(t) for t in samples]
    per_word = {}
    for text in samples:
        for word in text.split():
            if word not in per_word:
                per_word[word] = "".join(
                    p for sent in phonemize_espeak(word, "vi") for p in sent
                )

    strip = lambda s: s.strip(" .,!?;:")  # noqa: E731  separators only
    checked, mismatches = 0, []
    for text, artifact in zip(samples, artifacts):
        if artifact.lm_token_groups is None:
            continue
        words = [text[s:e] for s, e in _whitespace_spans(text)]
        phone_groups = [
            [artifact.phones[i] for i in g] for g in artifact.phone_groups
        ]
        if len(phone_groups) != len(words):
            continue
        for word, group in zip(words, phone_groups):
            checked += 1
            if strip("".join(group)) != strip(per_word[word]):
                mismatches.append((word, "".join(group), per_word[word]))

    assert checked > 300, f"too few groups checked ({checked}) to be meaningful"
    assert not mismatches, (
        f"{len(mismatches)}/{checked} phone groups do not match the word "
        f"phonemized on its own -- groups are misassigned even though the "
        f"counts line up. First 5: {mismatches[:5]}"
    )
