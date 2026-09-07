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


def _check_corpus_equivalence(
    samples, old_phones_fn, new_groups_fn, label, script="latin"
):
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
    # Two classes of sample are excluded, both deliberately.
    #
    # 1. Text containing Han characters. Since 2026-09-07 the tokenizer
    #    routes Han-script segments through the Chinese pinyin path whatever
    #    `lang` is, because the alternative was espeak announcing "chinese
    #    letter" once per character. Byte-identity with a single-voice
    #    EspeakTokenizer therefore holds only for non-Han text, and that is
    #    the point rather than a regression.
    # 2. Non-empty text that phonemizes to nothing at all -- raw web corpus
    #    lines like '[' or '-------'. The tokenizer now raises on those
    #    rather than emitting an empty artifact, so they cannot be
    #    equivalence cases.
    from zipvoice.tokenizer.fusion_tokenizer import _has_han

    # `script` says which half of the routing this call is checking.
    #   "latin": non-Han text only. Han text now goes through the Chinese
    #            path whatever `lang` is, so it is deliberately no longer
    #            byte-identical to a single-voice EspeakTokenizer.
    #   "han":   Han text only. Latin text under lang="zh" now follows the
    #            Vietnamese fallback rather than EmiliaTokenizer's en-us, so
    #            it is deliberately no longer Emilia-identical.
    usable = []
    for text in samples:
        if (script == "latin") == _has_han(text):
            continue
        if script == "han" and any(c.isascii() and c.isalpha() for c in text):
            # Mixed Han+Latin. Under lang="zh" a Latin segment follows the
            # Vietnamese fallback by the author's routing rule, where
            # EmiliaTokenizer used en-us -- so "Amazon.co.uk" inside Chinese
            # is deliberately no longer Emilia-identical. What must stay
            # identical is the Chinese phonemization itself, which Han-only
            # samples test directly.
            continue
        try:
            old_phones_fn(text)
            new_groups_fn(text)
        except ValueError:
            continue
        usable.append(text)
    samples = usable

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
        script="han",
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


def test_flatten_reports_where_sentences_joined():
    """espeak returns ONE LIST PER SENTENCE, and that boundary is real
    information.

    The original code discarded it in `reduce(x + y)` and then guessed
    boundaries back from punctuation, on the mistaken belief that espeak
    "emits no space after a period". The concatenation itself must stay --
    upstream `EspeakTokenizer.g2p` does the same and the source checkpoint
    was trained on the concatenated sequence -- so the boundary is reported
    alongside instead.
    """
    from zipvoice.tokenizer.fusion_tokenizer import _flatten_espeak_output
    from zipvoice.tokenizer.tokenizer import phonemize_espeak

    sentences = phonemize_espeak("mọi người. Rất nhiều", "vi")
    assert len(sentences) == 2, "espeak should split this into two sentences"

    flat, ends = _flatten_espeak_output(sentences)
    assert flat == [p for s in sentences for p in s], "concatenation unchanged"
    assert ends == [len(sentences[0]), len(flat)]
    # the join really is separator-free -- that is why it had to be reported
    assert flat[ends[0] - 1] == "." and flat[ends[0]] != " "


def test_sentence_end_bounds_trailing_attachment():
    """A separator emitted after a sentence end belongs to the NEXT sentence.

    Trailing-separator attachment must stop there, or a block absorbs the
    following sentence's leading whitespace. This is the only thing sentence
    metadata is used for.

    It is deliberately NOT a group boundary. Two sentences being phonemized
    independently means widening across one is unnecessary to recover
    pronunciation context; it does not make a shared conditioning group
    invalid, and a conditioning group is not required to be a phonological
    unit. Enforcing it as a hard cut made results strictly worse -- a
    rejected match fell through to the whole-remainder fallback, producing a
    coarser block that crossed the boundary anyway.
    """
    from zipvoice.tokenizer.fusion_tokenizer import _match_phone_content

    flat = ["a", ".", " ", "b"]
    # sentence 1 ends at index 2, so the space at index 2 is sentence 2's
    assert _match_phone_content(flat, 0, ["a", "."], limit=2) == 2
    # without the limit the space is absorbed into the preceding block
    assert _match_phone_content(flat, 0, ["a", "."]) == 3


def test_sentence_metadata_never_makes_blocks_coarser(monkeypatch):
    """Supplying sentence ends must never produce a worse segmentation than
    omitting them. The first attempt did exactly that.
    """
    import zipvoice.tokenizer.fusion_tokenizer as module

    tok = _vi_tokenizer_or_skip()
    probes = {"X.": ["a", "."], "Y": ["b"], "X. Y": ["a", ".", "b"]}
    monkeypatch.setattr(
        module.FusionTokenizer,
        "_word_phones",
        lambda self, w, l: probes.get(w, list(w)),
    )
    flat = ["a", ".", " ", "b"]
    without = tok._blocks_from_words(flat, ["X.", "Y"], "vi")[0]
    with_ends = tok._blocks_from_words(
        flat, ["X.", "Y"], "vi", sentence_ends=[2, 4]
    )[0]
    assert len(with_ends) >= len(without), (
        f"sentence metadata coarsened the result: {without} -> {with_ends}"
    )
    assert [p for b in with_ends for p in b] == flat


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
    """Multi-sentence English aligns, and stays close to one group per word.

    NOTE this does not exercise the sentence-boundary merge that the
    2026-09-02 work fixed -- espeak-en-us drops the period and emits a
    space, so English never had it. The Vietnamese test below is that
    regression. This one guards granularity: a group MAY span several words
    (see the 2026-09-07 ADR) but should rarely need to, and a slide toward
    coarse groups would silently weaken conditioning while alignment still
    reported success.
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

    assert artifact.lm_token_groups is not None
    assert len(artifact.lm_token_groups) == len(artifact.phone_groups)
    n_words = len(text.split())
    assert len(artifact.phone_groups) >= n_words - 2, (
        f"{len(artifact.phone_groups)} groups for {n_words} words -- groups "
        f"are merging far more than expected; check that stress marks are "
        f"still ignored when matching phone content"
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


def test_vi_single_word_groups_match_independent_phonemization():
    """Independent phone-to-word check, for the groups that span one word.

    The span/lm_token test cannot establish this: word span i is assigned to
    group i by position, so decoding lm group i back to word i is partly
    circular. This reference never consults group indices -- it phonemizes
    the word alone and compares.

    Scoped to single-word groups on purpose. A multi-word group exists
    precisely because the words in it do not phonemize the same way alone as
    together (the 2026-09-07 ADR), so comparing them against an isolated
    reference would be testing the thing the design gives up on. Stress
    marks are ignored, being prosody rather than segmental content.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer
    from zipvoice.tokenizer.tokenizer import phonemize_espeak

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
    artifacts = [tok.text_to_artifact(t) for t in samples]
    per_word = {}
    for text in samples:
        for word in text.split():
            if word not in per_word:
                per_word[word] = "".join(
                    p for sent in phonemize_espeak(word, "vi") for p in sent
                )

    ignore = " .,!?;:\u02c8\u02cc"
    strip = lambda s: "".join(c for c in s if c not in ignore)  # noqa: E731
    checked, single, mismatches = 0, 0, []
    for text, artifact in zip(samples, artifacts):
        assert artifact.lm_token_groups is not None, "alignment must not fail"
        blocks, words, _ = tok._plain_phone_groups_and_spans(text)
        for block, block_words in zip(blocks, words):
            checked += 1
            if len(block_words) != 1:
                continue
            single += 1
            if strip("".join(block)) != strip(per_word[block_words[0]]):
                mismatches.append((block_words[0], "".join(block)))

    assert single > 300, f"too few single-word groups checked ({single})"
    assert single / checked > 0.95, (
        f"only {100 * single / checked:.1f}% of groups span a single word; "
        f"granularity has degraded"
    )
    assert not mismatches, (
        f"{len(mismatches)}/{single} single-word groups do not match the "
        f"word phonemized on its own. First 5: {mismatches[:5]}"
    )


# --- The 2026-09-07 contract: groups may span several words ---------------
#
# The old construction required exactly one group per whitespace word and
# dropped the whole utterance to phone-only when espeak would not divide
# that way. Four rounds of boundary rules got that from 7.3% to 99.9% and
# stalled. See docs/adr/2026-09-07__phone_lm_token_group_alignment.md for
# why the frame was wrong. These tests pin the new contract, which is
# strictly weaker per-group and strictly stronger overall: a group is one or
# more consecutive words, and alignment cannot fail.


def _vi_tokenizer_or_skip():
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")
    return FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=LanguageModelTokenizer()
    )


def test_alignment_never_fails_on_real_corpus():
    """The property the whole redesign buys: `lm_token_groups` is never None.

    Previously this was a rate to be maximized, and the last 0.1% was
    written off as inherent to espeak. It is not inherent -- a block that
    will not match simply widens, in the limit to the whole utterance. A
    coarse group is a valid alignment; a dropped utterance is not.
    """
    tok = _vi_tokenizer_or_skip()
    samples = _vi_fixture_samples_or_skip()[:150]
    failures = [t for t in samples if tok.text_to_artifact(t).lm_token_groups is None]
    assert not failures, (
        f"{len(failures)}/{len(samples)} utterances produced no alignment; "
        f"block widening should make that impossible. First: {failures[0][:120]!r}"
    )


def test_groups_stay_word_sized_on_real_corpus():
    """Coverage must not be bought with granularity.

    Alignment can always succeed by making one enormous group, so a 100%
    rate alone would not detect a collapse. Assert the distribution too:
    almost every group should still be a single word.
    """
    tok = _vi_tokenizer_or_skip()
    samples = _vi_fixture_samples_or_skip()[:100]
    single = total = 0
    largest = 0
    for text in samples:
        _, word_blocks, _ = tok._plain_phone_groups_and_spans(text)
        for block in word_blocks:
            total += 1
            single += len(block) == 1
            largest = max(largest, len(block))
    assert total > 5000, f"too few groups ({total}) to be meaningful"
    assert single / total > 0.99, (
        f"only {100 * single / total:.2f}% of groups span one word "
        f"(largest {largest}); conditioning granularity has degraded"
    )


def test_artifact_invariants_hold_on_real_corpus():
    """The three properties that make the construction safe, checked
    end-to-end rather than only inside the builder's own asserts."""
    from zipvoice.tokenizer.tokenizer import phonemize_espeak

    tok = _vi_tokenizer_or_skip()
    hf = tok.lm_tokenizer.hf_tokenizer
    for text in _vi_fixture_samples_or_skip()[:40]:
        artifact = tok.text_to_artifact(text)
        assert artifact.lm_token_groups is not None

        # 1. groups partition the phones, in order, without gaps or overlap
        covered = [i for group in artifact.phone_groups for i in group]
        assert covered == list(range(len(artifact.phone_ids)))

        # 2. the lm_tokens are exactly the whole-text tokenization
        if " ".join(text.split()) == text:
            whole = hf.encode(text, add_special_tokens=False)
            assert artifact.lm_token_ids[1:] == whole

        # 3. one lm group per phone group, none empty, tag never in a group
        assert len(artifact.lm_token_groups) == len(artifact.phone_groups)
        assert all(artifact.lm_token_groups)
        assert min(i for g in artifact.lm_token_groups for i in g) >= 1


def test_a_fused_pair_is_kept_in_one_group():
    """The case that motivated the redesign.

    espeak-vi decides pronunciation from neighbouring words: "the" is
    Vietnamese 'tˈɛ' alone but English 'ðə' inside "book of the", and it
    fuses "of the" into one run with no separator. The old code failed the
    whole utterance; the new one puts those two words in one group.
    """
    tok = _vi_tokenizer_or_skip()
    text = "Quyển sách này là book of the măng của tháng hai"
    artifact = tok.text_to_artifact(text)
    assert artifact.lm_token_groups is not None

    _, word_blocks, _ = tok._plain_phone_groups_and_spans(text)
    merged = [b for b in word_blocks if len(b) > 1]
    assert merged == [["of", "the"]], f"expected only 'of the' merged, got {merged}"

    # and that group must carry BOTH words' lm_tokens
    index = next(i for i, b in enumerate(word_blocks) if len(b) > 1)
    tokens = [artifact.lm_tokens[i] for i in artifact.lm_token_groups[index]]
    assert hf_decode(tok, artifact, index).strip() == "of the", tokens


def hf_decode(tok, artifact, group_index):
    ids = [artifact.lm_token_ids[i] for i in artifact.lm_token_groups[group_index]]
    return tok.lm_tokenizer.hf_tokenizer.decode(ids)


def test_punctuation_only_word_joins_its_neighbour():
    """A token espeak renders as nothing must not become an empty group.

    An empty group carries no Qwen evidence, and the extractor marks it
    invalid exactly like padding -- so it would silently disable
    conditioning for that position rather than fail.
    """
    tok = _vi_tokenizer_or_skip()
    blocks, word_blocks, _ = tok._plain_phone_groups_and_spans("xin chao - cac ban")
    assert all(blocks), "no phone block may be empty"
    assert [w for b in word_blocks for w in b] == "xin chao - cac ban".split()


def test_chinese_aligns_and_does_not_overlap():
    """Chinese used to emit OVERLAPPING lm groups -- measured, lm_token 6
    assigned to two different phone groups. It briefly returned no alignment
    at all rather than a wrong one; with script routing plus boundary
    coarsening it now aligns correctly, so assert the correctness rather
    than the old containment.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")
    tok = FusionTokenizer(
        token_file=token_file, lang="zh", lm_tokenizer=LanguageModelTokenizer()
    )
    for text in ("你好世界", "今天天气很好", "我喜欢学习中文"):
        artifact = tok.text_to_artifact(text)
        assert artifact.phone_groups, text
        assert artifact.lm_token_groups is not None, text
        flat = [i for group in artifact.lm_token_groups for i in group]
        assert flat == list(range(1, len(artifact.lm_token_ids))), (
            f"{text}: every lm_token must be in exactly one group -- "
            f"duplicates were the original Chinese defect"
        )
        assert len(artifact.lm_token_groups) == len(artifact.phone_groups)


def test_token_crossing_a_boundary_coarsens_it():
    """A boundary an lm_token straddles is not a usable group boundary.

    jieba and BPE disagree: measured, jieba cuts "今天|天气|很好" while BPE
    emits "很好" across the 天气/很好 boundary, and "我喜欢" across two
    boundaries in "我喜欢学习中文". Assigning such a token to one side gives
    that group a token containing the other group's text -- silent, since
    every group is still non-empty and every token still used exactly once.
    Routing decides which phonemizer runs; it need not survive as a
    conditioning boundary, so the contradicted boundary is dropped.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")
    tok = FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=LanguageModelTokenizer()
    )
    hf = tok.lm_tokenizer.hf_tokenizer

    for text in ("今天天气很好", "我喜欢学习中文"):
        artifact = tok.text_to_artifact(text)
        assert artifact.lm_token_groups is not None
        offsets = hf(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        # No surviving boundary may be straddled by a token. Reconstruct the
        # boundaries from group sizes over the token offsets.
        cuts = []
        consumed = 0
        for group in artifact.lm_token_groups[:-1]:
            consumed += len(group)
            cuts.append(offsets[consumed - 1][1])
        for start, end in offsets:
            for cut in cuts:
                assert not (start < cut < end), (
                    f"{text}: token ({start},{end}) still straddles cut {cut}"
                )


def test_all_oov_returns_no_alignment_instead_of_crashing():
    """When every phone group is filtered away there is no destination group
    for the offset walk. That used to raise IndexError; the full lm_tokens
    are kept and alignment is reported as unavailable.
    """
    import tempfile

    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    directory = tempfile.mkdtemp()
    token_file = os.path.join(directory, "tokens.txt")
    with open(token_file, "w", encoding="utf-8") as handle:
        handle.write("_\t0\n")  # nothing but the pad symbol
    tok = FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=LanguageModelTokenizer()
    )
    artifact = tok.text_to_artifact("xin chao")
    assert artifact.lm_token_groups is None
    assert len(artifact.lm_token_ids) > 1, "the lm_tokens must still be kept"


# --- Defects found by external review of the 2026-09-07 redesign ----------
#
# All five were confirmed by measurement before being fixed. Each passed the
# partition and byte-identity checks that were already in place, which is
# why they need cases of their own: internally consistent bookkeeping is not
# evidence of correct correspondence.


def test_unusual_whitespace_does_not_change_qwen_input():
    """The lm_token sequence must be the whole-text tokenization, always.

    An earlier version tokenized each block separately and relied on Qwen's
    BPE being concatenative at whitespace. It is not, for *runs* of
    whitespace -- the vocabulary contains multi-whitespace tokens -- so
    "xin  chao" (double space) tokenized differently whole than in pieces,
    silently rewriting the text Qwen sees. Worse, the verification only ran
    when the rebuilt text already matched, i.e. never on the inputs that
    could break it.
    """
    tok = _vi_tokenizer_or_skip()
    hf = tok.lm_tokenizer.hf_tokenizer
    for text in ("xin chao cac ban", "xin  chao\tcac ban", " xin chao ", "xin\n chao"):
        artifact = tok.text_to_artifact(text)
        assert artifact.lm_token_ids[1:] == hf.encode(
            text, add_special_tokens=False
        ), f"lm_tokens differ from the whole-text tokenization for {text!r}"


def test_every_lm_token_belongs_to_exactly_one_group():
    """Total coverage, not just in-range indices.

    A minimum-index check permits gaps and duplicates. The original
    character-overlap rule left standalone space tokens in no group at all
    (0.30% of lm_tokens, measured), and the Chinese path assigned one token
    to two groups.
    """
    tok = _vi_tokenizer_or_skip()
    for text in _vi_fixture_samples_or_skip()[:25]:
        artifact = tok.text_to_artifact(text)
        assert artifact.lm_token_groups is not None
        flat = [i for group in artifact.lm_token_groups for i in group]
        assert flat == list(range(1, len(artifact.lm_token_ids))), (
            "lm_token_groups must partition every token exactly once, "
            "excluding the [LANG:xx] tag at index 0"
        )


def test_oov_word_keeps_its_place_in_the_qwen_input():
    """A word whose phones are all OOV must not vanish from the LM input.

    Dropping it removes context from every token after it. Measured before
    the fix: "xin chao ban" reached Qwen as "xin ban". The word now merges
    into a neighbouring group -- the phone side loses what the vocabulary
    cannot represent, the text side stays whole.
    """
    import tempfile

    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    full = _vi_tokenizer_or_skip()
    text = "xin chao ban"
    blocks, words, _ = full._plain_phone_groups_and_spans(text)
    middle = set(blocks[next(i for i, w in enumerate(words) if w == ["chao"])])
    vocab = ["_"] + sorted({p for b in blocks for p in b} - middle)

    directory = tempfile.mkdtemp()
    token_file = os.path.join(directory, "tokens.txt")
    with open(token_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(f"{s}\t{i}" for i, s in enumerate(vocab)))

    tok = FusionTokenizer(
        token_file=token_file, lang="vi", lm_tokenizer=LanguageModelTokenizer()
    )
    artifact = tok.text_to_artifact(text)
    seen = tok.lm_tokenizer.hf_tokenizer.decode(artifact.lm_token_ids[1:])
    assert "chao" in seen, f"the OOV word was deleted from Qwen's input: {seen!r}"
    assert len(artifact.lm_token_groups) == len(artifact.phone_groups)


def test_stress_mark_is_not_absorbed_by_the_preceding_group():
    """Ignoring a symbol when comparing and deciding who owns it are
    different operations.

    Stress belongs to the syllable it precedes, i.e. the next word. Treating
    every ignorable symbol as a trailing separator moved it backwards, and
    both the partition and byte-identity checks still passed.
    """
    from zipvoice.tokenizer.fusion_tokenizer import _match_phone_content

    flat = ["a", " ", "ˈ", "b"]
    end = _match_phone_content(flat, 0, ["a"])
    assert end == 2, (
        f"consumed {flat[0:end]} -- the next syllable's stress mark was "
        f"absorbed into the previous group"
    )
    assert _match_phone_content(flat, end, ["ˈ", "b"]) == 4


def test_leftover_phones_are_attached_not_asserted_away():
    """A greedy prefix match can leave phones behind.

    With P("X")=[a], P("Y")=[b] but P("X Y")=[a,b,c], the walk accepts X then
    Y, exhausts the words, and never tries the span that would consume 'c'.
    That used to fail the partition assert -- a crash, killing the batch.
    The words are exhausted, so the leftovers belong to the final block.
    """
    import zipvoice.tokenizer.fusion_tokenizer as module

    tok = _vi_tokenizer_or_skip()
    original = module.FusionTokenizer._word_phones
    probes = {"X": ["a"], "Y": ["b"], "X Y": ["a", "b", "c"]}
    module.FusionTokenizer._word_phones = lambda self, w, l: probes.get(w, list(w))
    try:
        blocks, words = tok._blocks_from_words(["a", "b", "c"], ["X", "Y"], "vi")
    finally:
        module.FusionTokenizer._word_phones = original

    assert [p for b in blocks for p in b] == ["a", "b", "c"]
    assert [w for b in words for w in b] == ["X", "Y"]
    assert all(blocks)


def test_text_with_no_phones_returns_nothing_rather_than_asserting():
    """`words=["-"]` with no phones must not crash the partition assert.

    Callers reject non-empty text that yields zero phones upstream, so this
    is only reachable directly -- but an assert here would be a crash, and
    the honest answer is "no blocks".
    """
    tok = _vi_tokenizer_or_skip()
    assert tok._blocks_from_words([], ["-"], "vi") == ([], [])


# --- Script routing (2026-09-07, author's rule) ---------------------------
#
# A Han segment is always phonemized as Chinese whatever `lang` is; a Latin
# segment follows `lang`, with lang="zh" falling back to Vietnamese. Before
# this, a VI-primary run rendered 你好 as espeak announcing "chinese letter"
# once per character.


def test_han_is_phonemized_as_chinese_whatever_the_lang():
    """The rule's first half. Under lang="vi" espeak has no reading for Han
    characters and announced their character class instead -- measured,
    "你好世界" became 'tʃˈaɪniːzlˈetə' four times.
    """
    tok = _vi_tokenizer_or_skip()
    artifact = tok.text_to_artifact("Tôi thích 你好世界 rồi")
    phones = "".join(artifact.phones)
    assert "n0i2h0ao3" in phones, f"Chinese not routed to pinyin: {phones}"
    assert "aɪniːz" not in phones, f"espeak announced the character class: {phones}"
    assert "t̪ˈoj" in phones, "Vietnamese must still be Vietnamese"


def test_latin_follows_lang_and_zh_falls_back_to_vietnamese():
    """The rule's second half, including the author's explicit choice that
    lang="zh" reads Latin as Vietnamese rather than English. That trades
    English-inside-Chinese ("Amazon" reads Vietnamese) for consistency with
    a Vietnamese-primary project; it was raised and confirmed deliberately.
    """
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    token_file = (
        "scripts/all/pretrained_model_cache/"
        "hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
    )
    if not os.path.exists(token_file):
        pytest.skip(f"{token_file} not available")

    lm = LanguageModelTokenizer()
    text = "你好 phở"
    phones = {}
    for lang in ("vi", "en", "zh"):
        tok = FusionTokenizer(token_file=token_file, lang=lang, lm_tokenizer=lm)
        phones[lang] = "".join(tok.text_to_artifact(text).phones)

    # Chinese identical in all three -- it never follows `lang`.
    assert "n0i2h0ao3" in phones["vi"]
    assert "n0i2h0ao3" in phones["en"]
    assert "n0i2h0ao3" in phones["zh"]
    # Latin follows lang: en spells the Vietnamese word out, vi does not.
    assert phones["vi"] != phones["en"]
    # lang="zh" falls back to the Vietnamese voice for Latin.
    assert phones["zh"] == phones["vi"]


def test_mixed_script_utterance_aligns_completely():
    """Routing must not cost alignment: every lm_token still belongs to
    exactly one group, across a three-language utterance."""
    tok = _vi_tokenizer_or_skip()
    text = "Tôi thích ăn phở và 你好世界 cùng machine learning"
    artifact = tok.text_to_artifact(text)
    assert artifact.lm_token_groups is not None
    flat = [i for group in artifact.lm_token_groups for i in group]
    assert flat == list(range(1, len(artifact.lm_token_ids)))
    assert len(artifact.lm_token_groups) == len(artifact.phone_groups)
    assert all(artifact.lm_token_groups)
