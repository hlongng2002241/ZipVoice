import random
from collections import Counter

from zipvoice.tokenizer.multilingual_tokenizer import (
    MultilingualTokenizer,
    normalize_language_name,
    sample_lang_tag,
)


def _make_tokenizer():
    return MultilingualTokenizer(
        pretrained_model_name="google-bert/bert-base-multilingual-cased"
    )


def test_normalize_language_name_handles_real_corpus_spellings():
    assert normalize_language_name("Vietnamese") == "vi"
    assert normalize_language_name("vi") == "vi"
    assert normalize_language_name("ENGLISH") == "en"
    assert normalize_language_name("Chinese") == "zh"
    assert normalize_language_name("Mandarin") == "zh"


def test_normalize_language_name_none_for_missing_or_unrecognized():
    assert normalize_language_name(None) is None
    assert normalize_language_name("") is None
    assert normalize_language_name("Klingon") is None


def test_default_pretrained_model_is_qwen25():
    # Final decision per docs/adr/2026-08-28__choose_qwen25_tokenizer.md.
    tok = MultilingualTokenizer()
    assert tok.hf_tokenizer.name_or_path == "Qwen/Qwen2.5-0.5B"
    # Byte-level BPE: the whole point of the switch was zero [UNK] risk.
    assert tok.hf_tokenizer.unk_token_id is None
    ids = tok.texts_to_token_ids(["Xin chào 你好, hello world!"])[0]
    assert len(ids) > 0


def test_lang_tags_are_single_tokens_not_split():
    tok = _make_tokenizer()
    tokens = tok.texts_to_tokens(["[LANG:vi] Xin chào, rất vui được gặp bạn."])[0]
    assert tokens[0] == "[LANG:vi]"
    # The tag must not have been split into subword pieces.
    assert tokens.count("[LANG:vi]") == 1


def test_lang_tag_ids_are_appended_beyond_base_vocab():
    tok = _make_tokenizer()
    base_vocab_size = tok.hf_tokenizer.vocab_size
    for tag in MultilingualTokenizer.DEFAULT_LANG_TAGS:
        tag_id = tok.hf_tokenizer.convert_tokens_to_ids(tag)
        assert tag_id >= base_vocab_size
        assert tag_id in tok.control_token_ids


def test_zero_duration_mask_flags_only_control_tokens():
    tok = _make_tokenizer()
    ids = tok.texts_to_token_ids(["[LANG:en] Hello world"])[0]
    mask = tok.zero_duration_mask(ids)
    assert mask[0] is True
    assert not any(mask[1:])


def test_mixed_en_vi_zh_tokenizes_without_a_language_router():
    tok = _make_tokenizer()
    text = "[LANG:auto] Mr King, xin chào 你好, rất vui gặp bạn 我们."
    ids = tok.texts_to_token_ids([text])[0]
    tokens = tok.texts_to_tokens([text])[0]
    assert len(ids) == len(tokens) > 0
    assert tokens[0] == "[LANG:auto]"


def test_sample_lang_tag_zero_dropout_always_true_tag():
    rng = random.Random(0)
    for _ in range(50):
        assert sample_lang_tag("vi", auto_prob=0.0, wrong_tag_prob=0.0, rng=rng) == "[LANG:vi]"


def test_sample_lang_tag_wrong_tag_never_matches_true_lang():
    rng = random.Random(1)
    for _ in range(200):
        tag = sample_lang_tag("en", auto_prob=0.0, wrong_tag_prob=1.0, rng=rng)
        assert tag in ("[LANG:vi]", "[LANG:zh]")


def test_extra_tokens_fill_the_discovered_vi_unk_gap():
    tok_without_fix = _make_tokenizer()
    unk_id = tok_without_fix.hf_tokenizer.unk_token_id
    ids_before = tok_without_fix.hf_tokenizer.encode("Ừ", add_special_tokens=False)
    assert ids_before == [unk_id]

    tok_with_fix = MultilingualTokenizer(
        pretrained_model_name="google-bert/bert-base-multilingual-cased",
        extra_tokens=MultilingualTokenizer.VI_UNK_GAP_TOKENS,
    )
    ids_after = tok_with_fix.hf_tokenizer.encode("Ừ", add_special_tokens=False)
    assert unk_id not in ids_after
    assert tok_with_fix.hf_tokenizer.convert_ids_to_tokens(ids_after) == ["Ừ"]


def test_sample_lang_tag_distribution_matches_configured_probabilities():
    rng = random.Random(42)
    n = 20000
    counts = Counter(
        sample_lang_tag("zh", auto_prob=0.2, wrong_tag_prob=0.1, rng=rng)
        for _ in range(n)
    )
    auto_rate = counts["[LANG:auto]"] / n
    true_rate = counts["[LANG:zh]"] / n
    wrong_rate = (counts["[LANG:en]"] + counts["[LANG:vi]"]) / n
    assert abs(auto_rate - 0.2) < 0.02
    assert abs(wrong_rate - 0.1) < 0.02
    assert abs(true_rate - 0.7) < 0.02
