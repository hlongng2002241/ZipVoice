"""End-to-end plumbing tests for the fusion frontend (sprint 003, step 5).

Covers the whole path a training step actually takes, which no single unit
test does: real text -> `tokenize_text_fusion` -> supervision fields ->
`SpeechSynthesisDataset` collation -> `prepare_input` -> the live Qwen
extractor -> `compute_fbank_loss` -> a real loss with gradients.

That path crosses four files, and the failure mode it guards against is
boring but expensive: a field written under one name and read under another
would sail through every unit test and only surface as a confusing crash
(or, worse, a silently missing Qwen branch) at real training time.
"""

import pytest
import torch
from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from lhotse.dataset import PrecomputedFeatures

from zipvoice.bin.train_zipvoice import (
    check_resume_text_frontend,
    compute_fbank_loss,
    tokenize_text,
    tokenize_text_fusion,
)
from zipvoice.dataset.dataset import SpeechSynthesisDataset
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import AttributeDict, prepare_input

TOKEN_FILE = (
    "scripts/all/pretrained_model_cache/hynt__ZipVoice-Vietnamese-2500h/tokens.txt"
)

_TINY_KWARGS = dict(
    feat_dim=100,
    fm_decoder_dim=32,
    fm_decoder_num_layers=[1, 1, 1, 1, 1],
    fm_decoder_downsampling_factor=[1, 2, 4, 2, 1],
    fm_decoder_cnn_module_kernel=[31, 15, 7, 15, 31],
    text_encoder_num_layers=1,
    text_encoder_dim=32,
)


def _cut(cut_id, text, language):
    rec = Recording(
        id=cut_id,
        sources=[AudioSource(type="file", channels=[0], source="/dev/null")],
        sampling_rate=24000,
        num_samples=24000,
        duration=1.0,
    )
    sup = SupervisionSegment(
        id=cut_id,
        recording_id=cut_id,
        start=0,
        duration=1.0,
        text=text,
        language=language,
    )
    cut = rec.to_cut()
    cut.supervisions = [sup]
    return cut


class _FakeFeatureInputStrategy(PrecomputedFeatures):
    def __call__(self, cuts):
        n = len(cuts)
        return torch.zeros(n, 40, 100), torch.tensor([40] * n)


@pytest.fixture(scope="module")
def fusion_tokenizers():
    from zipvoice.tokenizer.fusion_tokenizer import FusionTokenizer
    from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

    lm = LanguageModelTokenizer()
    return {
        lang: FusionTokenizer(token_file=TOKEN_FILE, lang=lang, lm_tokenizer=lm)
        for lang in ("en", "vi")
    }


@pytest.fixture(scope="module")
def extractor():
    from zipvoice.models.modules.qwen_extractor import TruncatedQwenExtractor

    return TruncatedQwenExtractor(num_layers=4, device=torch.device("cpu"))


def _tokenized_cuts(fusion_tokenizers):
    cuts = [
        _cut("a", "Sau khi shopping xong, chloe di an pho.", "Vietnamese"),
        _cut("b", "Hello everyone, welcome back.", "English"),
    ]
    return CutSet.from_cuts([tokenize_text_fusion(c, fusion_tokenizers) for c in cuts])


def test_tokenize_attaches_all_artifact_fields(fusion_tokenizers):
    cut = tokenize_text_fusion(
        _cut("a", "Sau khi shopping xong.", "Vietnamese"), fusion_tokenizers
    )
    sup = cut.supervisions[0]
    assert len(sup.tokens) > 0, "phone ids land on the usual `tokens` field"
    assert all(isinstance(t, int) for t in sup.tokens)
    # The group structure must cover exactly the phones, or the model's
    # group index would not line up with its phone sequence.
    covered = sorted(i for g in sup.phone_groups for i in g)
    assert covered == list(range(len(sup.tokens)))
    hf = fusion_tokenizers["vi"].lm_tokenizer.hf_tokenizer
    assert sup.lm_token_ids[0] == hf.convert_tokens_to_ids("[LANG:vi]"), (
        "the deterministic language tag must lead the lm_token sequence"
    )
    assert len(sup.lm_token_groups) == len(sup.phone_groups)


def test_language_selects_the_right_tokenizer(fusion_tokenizers):
    """Each FusionTokenizer is locked to one language, so the cut's own
    `supervision.language` has to pick between them -- picking wrong would
    silently phonemize under the wrong rules.
    """
    vi = tokenize_text_fusion(_cut("a", "xin chao cac ban", "Vietnamese"), fusion_tokenizers)
    en = tokenize_text_fusion(_cut("b", "xin chao cac ban", "English"), fusion_tokenizers)
    assert vi.supervisions[0].tokens != en.supervisions[0].tokens, (
        "the same text under different languages must phonemize differently"
    )


def test_unknown_language_is_rejected(fusion_tokenizers):
    with pytest.raises(ValueError, match="unrecognized language"):
        tokenize_text_fusion(_cut("a", "hello", "Klingon"), fusion_tokenizers)
    with pytest.raises(ValueError, match="no FusionTokenizer"):
        tokenize_text_fusion(_cut("a", "ni hao", "Chinese"), fusion_tokenizers)


def test_fusion_clears_a_stale_zero_duration_mask(fusion_tokenizers):
    """A supervision that already carries a `zero_duration_mask` (from a
    previous pass with a lm_token-based tokenizer) must not keep it: that
    mask counts lm_tokens, but everything downstream applies it to phones.
    With a length mismatch it trips an assert; with a coincidental match it
    silently gives real phones zero duration, which is worse.
    """
    cut = _cut("a", "Sau khi shopping xong.", "Vietnamese")
    cut.supervisions[0].zero_duration_mask = [True, False, False]
    cut = tokenize_text_fusion(cut, fusion_tokenizers)

    assert getattr(cut.supervisions[0], "zero_duration_mask", None) is None
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[CutSet.from_cuts([cut])]
    assert "zero_duration_mask" not in batch, (
        "a cleared mask must leave the key absent, matching the schema a "
        "fusion run is supposed to produce"
    )


def test_non_fusion_tokenizer_clears_stale_fusion_fields():
    """The mirror image: switching back to a non-fusion frontend must drop
    the group fields, or the dataset collates them for a run whose
    `qwen_extractor` is None and `compute_fbank_loss` rejects the batch.
    """

    class _FakeTokenizer:
        def texts_to_token_ids(self, texts):
            return [[1, 2, 3] for _ in texts]

    cut = _cut("a", "hello", "English")
    cut.supervisions[0].phone_groups = [[0], [1, 2]]
    cut.supervisions[0].lm_token_ids = [7, 8]
    cut.supervisions[0].lm_token_groups = [[0], [1]]
    cut.supervisions[0].zero_duration_mask = [True, False]
    cut = tokenize_text(cut, _FakeTokenizer())

    sup = cut.supervisions[0]
    for field in (
        "phone_groups",
        "lm_token_ids",
        "lm_token_groups",
        "zero_duration_mask",
    ):
        assert getattr(sup, field, None) is None, f"{field} survived"
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[CutSet.from_cuts([cut])]
    assert set(batch) == {"features", "features_lens", "text", "tokens"}


def test_fused_cut_is_rejected_by_a_string_tokenizer():
    """Fusion writes integer ids into `supervision.tokens`; the pre-tokenized
    branch of `tokenize_text` expects symbol strings and looks them up in a
    str-keyed vocabulary that silently skips misses. Without this guard a
    fused cut would come out with `tokens=[]` and fail far away from the
    cause, in duration allocation.
    """

    class _FakeTokenizer:
        def tokens_to_token_ids(self, tokens_list):
            return [[] for _ in tokens_list]  # what really happens: all miss

    cut = _cut("a", "hello", "English")
    cut.supervisions[0].tokens = [1, 2, 3]
    with pytest.raises(ValueError, match="integer `tokens`"):
        tokenize_text(cut, _FakeTokenizer())


def test_extractor_rejects_unimplemented_lang_tag_removal():
    """`include_lang_tag=False` was accepted and then ignored, running the
    identical computation while claiming the tag was gone -- which would
    silently invalidate a tag-removal ablation rather than fail it.
    """
    from zipvoice.models.modules.qwen_extractor import TruncatedQwenExtractor

    with pytest.raises(NotImplementedError, match="include_lang_tag"):
        TruncatedQwenExtractor(num_layers=1, include_lang_tag=False)


def test_resume_rejects_a_changed_text_frontend():
    """A resume that omits `--qwen-layers` falls back to the argparse default
    and would train against a different Qwen layer than the checkpoint
    learned -- the pooled width is 896 either way, so nothing else catches it.
    """
    params = AttributeDict(
        {"tokenizer": "fusion", "qwen_layers": 4, "text_frontend": "fusion"}
    )
    # Matching checkpoint: fine.
    check_resume_text_frontend(
        params, {"tokenizer": "fusion", "qwen_layers": 4, "text_frontend": "fusion"}
    )
    # No checkpoint at all, or a checkpoint predating these keys: fine.
    check_resume_text_frontend(params, None)
    check_resume_text_frontend(params, {"batch_idx_train": 100})

    with pytest.raises(ValueError, match="qwen_layers"):
        check_resume_text_frontend(params, {"qwen_layers": 8})
    with pytest.raises(ValueError, match="tokenizer"):
        check_resume_text_frontend(params, {"tokenizer": "multilingual"})
    # Which frozen Qwen supplied the conditioning matters too: a different
    # checkpoint at the same hidden width loads without complaint.
    with pytest.raises(ValueError, match="pretrained_tokenizer_name"):
        check_resume_text_frontend(
            AttributeDict({**params, "pretrained_tokenizer_name": "Qwen/Qwen2.5-0.5B"}),
            {"pretrained_tokenizer_name": "some/other-qwen"},
        )

    # Non-fusion runs are unaffected.
    check_resume_text_frontend(
        AttributeDict({"tokenizer": "emilia", "qwen_layers": 4}), {"qwen_layers": 8}
    )


def test_resume_guard_handles_the_default_qwen_correctly():
    """`pretrained_tokenizer_name=None` is a real value meaning the default
    model, not absent metadata. Treating None as "nothing saved" would wave
    through a swap to a different same-width Qwen; comparing it literally
    against the explicit default name would reject an identical setup.
    """
    base = {"tokenizer": "fusion", "text_frontend": "fusion", "qwen_layers": 4}

    # Saved the default implicitly, now resuming with a *different* model.
    with pytest.raises(ValueError, match="pretrained_tokenizer_name"):
        check_resume_text_frontend(
            AttributeDict({**base, "pretrained_tokenizer_name": "other/qwen"}),
            {**base, "pretrained_tokenizer_name": None},
        )

    # The same model named two different ways must be accepted.
    check_resume_text_frontend(
        AttributeDict({**base, "pretrained_tokenizer_name": None}),
        {**base, "pretrained_tokenizer_name": "Qwen/Qwen2.5-0.5B"},
    )
    check_resume_text_frontend(
        AttributeDict({**base, "pretrained_tokenizer_name": "Qwen/Qwen2.5-0.5B"}),
        {**base, "pretrained_tokenizer_name": None},
    )


def test_dataset_collates_the_fusion_fields(fusion_tokenizers):
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[_tokenized_cuts(fusion_tokenizers)]
    for key in ("tokens", "phone_groups", "lm_token_ids", "lm_token_groups"):
        assert key in batch, f"{key} did not survive dataset collation"
        assert len(batch[key]) == 2
    # Ragged by design: group structure differs per utterance.
    assert batch["phone_groups"][0] != batch["phone_groups"][1]


def test_non_fusion_batches_keep_the_old_schema():
    """A tokenizer that produces no fusion fields must yield a batch dict
    without the keys at all -- not present-but-None.
    """
    cut = _cut("a", "x", "English")
    cut.supervisions[0].tokens = [1, 2, 3]
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[CutSet.from_cuts([cut])]
    assert "tokens" in batch
    for key in ("phone_groups", "lm_token_ids", "lm_token_groups"):
        assert key not in batch


def test_prepare_input_returns_fusion_fields(fusion_tokenizers):
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[_tokenized_cuts(fusion_tokenizers)]
    params = AttributeDict({"feat_scale": 1.0})

    tokens, mask, fusion_fields, features, features_lens = prepare_input(
        params=params,
        batch=batch,
        device=torch.device("cpu"),
        return_tokens=True,
        return_feature=True,
        return_zero_duration_mask=True,
        return_fusion_fields=True,
    )
    assert fusion_fields is not None
    assert set(fusion_fields) == {"phone_groups", "lm_token_ids", "lm_token_groups"}
    assert fusion_fields["phone_groups"] == batch["phone_groups"]
    assert mask is None, "phones carry no control tokens, so no mask"


def test_prepare_input_returns_none_without_fusion_fields():
    params = AttributeDict({"feat_scale": 1.0})
    batch = {
        "tokens": [[1, 2]],
        "features": torch.zeros(1, 4, 100),
        "features_lens": torch.tensor([4]),
    }
    tokens, fusion_fields, features, features_lens = prepare_input(
        params=params,
        batch=batch,
        device=torch.device("cpu"),
        return_tokens=True,
        return_feature=True,
        return_fusion_fields=True,
    )
    assert fusion_fields is None


def test_full_training_step(fusion_tokenizers, extractor):
    """The real thing: a loss computed through the whole fused path."""
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[_tokenized_cuts(fusion_tokenizers)]
    params = AttributeDict({"feat_scale": 1.0, "condition_drop_ratio": 0.0})

    tokens, mask, fusion_fields, features, features_lens = prepare_input(
        params=params,
        batch=batch,
        device=torch.device("cpu"),
        return_tokens=True,
        return_feature=True,
        return_zero_duration_mask=True,
        return_fusion_fields=True,
    )

    torch.manual_seed(0)
    model = ZipVoice(
        vocab_size=fusion_tokenizers["vi"].vocab_size,
        pad_id=fusion_tokenizers["vi"].pad_id,
        text_embed_dim=192,
        text_frontend="fusion",
        qwen_hidden_size=extractor.hidden_size,
        **_TINY_KWARGS,
    )

    loss, info = compute_fbank_loss(
        params=params,
        model=model,
        features=torch.randn_like(features),
        features_lens=features_lens,
        tokens=tokens,
        is_training=True,
        zero_duration_mask=mask,
        fusion_fields=fusion_fields,
        qwen_extractor=extractor,
    )
    assert torch.isfinite(loss) and loss.requires_grad
    assert info["frames"] > 0

    loss.backward()
    assert model.fusion.qwen_proj.weight.grad.abs().sum() > 0, (
        "no gradient reached the Qwen projection -- the branch is not wired in"
    )
    assert model.fusion.gate_proj.weight.grad is not None


def test_fusion_fields_without_extractor_is_rejected(fusion_tokenizers):
    params = AttributeDict({"feat_scale": 1.0, "condition_drop_ratio": 0.0})
    model = ZipVoice(
        vocab_size=fusion_tokenizers["vi"].vocab_size,
        text_embed_dim=192,
        text_frontend="fusion",
        **_TINY_KWARGS,
    )
    with pytest.raises(AssertionError, match="qwen_extractor"):
        compute_fbank_loss(
            params=params,
            model=model,
            features=torch.randn(1, 40, 100),
            features_lens=torch.tensor([40]),
            tokens=[[1, 2, 3]],
            is_training=True,
            fusion_fields={
                "phone_groups": [[[0], [1, 2]]],
                "lm_token_ids": [[1, 2]],
                "lm_token_groups": [[[0], [1]]],
            },
            qwen_extractor=None,
        )


def test_configured_fusion_run_rejects_a_batch_with_no_fusion_fields(
    fusion_tokenizers, extractor
):
    """The dangerous converse of the assert above.

    A fusion-configured run whose batch carries no fusion fields is
    indistinguishable, at the model, from an intentional phone-only run: the
    model falls back to the phone embedding and trains happily. That is
    exactly how an entire training run was spent ~93% phone-only after the
    `_split_into_groups` sentence-boundary bug, with nothing failing and
    nothing logged above debug level. Missing metadata is a pipeline error
    and must be loud, unlike the per-utterance alignment fallback (which
    keeps the fields and sets lm_token_groups=None only for the affected
    utterances).
    """
    params = AttributeDict({"feat_scale": 1.0, "condition_drop_ratio": 0.0})
    model = ZipVoice(
        vocab_size=fusion_tokenizers["vi"].vocab_size,
        text_embed_dim=192,
        text_frontend="fusion",
        **_TINY_KWARGS,
    )
    with pytest.raises(AssertionError, match="no fusion fields"):
        compute_fbank_loss(
            params=params,
            model=model,
            features=torch.randn(1, 40, 100),
            features_lens=torch.tensor([40]),
            tokens=[[1, 2, 3]],
            is_training=True,
            fusion_fields=None,
            qwen_extractor=extractor,
        )


def test_training_step_reports_qwen_coverage(fusion_tokenizers, extractor):
    """Coverage must be observable in the training log.

    The sentence-boundary bug was invisible for a whole run because nothing
    measured how often conditioning actually reached the model. These two
    metrics make that a number on every log line: one utterance aligned,
    one not, must report 50% utterance coverage.
    """
    params = AttributeDict({"feat_scale": 1.0, "condition_drop_ratio": 0.0})
    model = ZipVoice(
        vocab_size=fusion_tokenizers["vi"].vocab_size,
        text_embed_dim=192,
        text_frontend="fusion",
        **_TINY_KWARGS,
    )
    features_lens = torch.tensor([40, 40])
    _, info = compute_fbank_loss(
        params=params,
        model=model,
        features=torch.randn(2, 40, 100),
        features_lens=features_lens,
        tokens=[[1, 2, 3], [4, 5, 6]],
        is_training=True,
        fusion_fields={
            "phone_groups": [[[0], [1, 2]], [[0], [1, 2]]],
            "lm_token_ids": [[1, 2], [3, 4]],
            # Second utterance failed alignment -- the legal fallback.
            "lm_token_groups": [[[0], [1]], None],
        },
        qwen_extractor=extractor,
    )
    frames = int(features_lens.sum().item())
    assert info["qwen_cov_utt"] / frames == pytest.approx(0.5), (
        "one of two utterances aligned -> 50% utterance coverage"
    )
    # Both groups of the aligned utterance carry real lm_tokens.
    assert info["qwen_cov_grp"] / frames == pytest.approx(1.0)
