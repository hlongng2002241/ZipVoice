"""End-to-end tests for `ZipVoice(text_frontend="fusion")` -- the model-side
wiring of the Qwen+phoneme fusion frontend (sprint 003, step 3).

Deliberately covers the things a unit test of `PhoneQwenFusion` alone can't:
that the fused representation survives the whole
`forward_text_embed -> text_encoder -> duration allocation -> gather` path
(which is where the inherited sentinel-padding contract actually bites), and
that the pre-fusion "embedding" frontend is untouched.
"""

import pytest
import torch

from zipvoice.models.zipvoice import ZipVoice

_TINY_KWARGS = dict(
    feat_dim=100,
    fm_decoder_dim=32,
    fm_decoder_num_layers=[1, 1, 1, 1, 1],
    fm_decoder_downsampling_factor=[1, 2, 4, 2, 1],
    fm_decoder_cnn_module_kernel=[31, 15, 7, 15, 31],
    text_encoder_num_layers=1,
    text_encoder_dim=32,
)

PHONE_VOCAB = 360  # the real source-checkpoint phone vocabulary size
QWEN_DIM = 896
EMBED_DIM = 192


def _model(**overrides):
    kwargs = dict(
        vocab_size=PHONE_VOCAB,
        text_embed_dim=EMBED_DIM,
        text_frontend="fusion",
        qwen_hidden_size=QWEN_DIM,
        **_TINY_KWARGS,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return ZipVoice(**kwargs)


def _batch():
    """Two utterances, deliberately unequal length so the batch's longest
    sequence is the one that exercises the sentinel position.
    """
    tokens = [[5, 6, 7, 8, 9], [10, 11, 12]]
    phone_groups = [[[0, 1], [2, 3, 4]], [[0], [1, 2]]]
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)
    return tokens, phone_groups, qwen, valid


def test_fusion_model_shape_and_no_separate_embedding():
    model = _model()
    assert model.embed is None, "fusion owns its phone embedding; no dead table"
    assert model.fusion.phone_embed.weight.shape == (PHONE_VOCAB, EMBED_DIM)
    assert model.fusion.qwen_proj.in_features == QWEN_DIM
    assert model.fusion.qwen_proj.out_features == EMBED_DIM
    # The whole point of the repositioned in_proj: it is 192-wide again, so
    # the source checkpoint's own in_proj is shape-compatible.
    assert model.text_encoder.in_proj.in_features == EMBED_DIM


def test_forward_text_embed_matches_padded_width():
    model = _model()
    tokens, phone_groups, qwen, valid = _batch()
    embed, lens = model.forward_text_embed(tokens, phone_groups, qwen, valid)
    # pad_labels appends one sentinel beyond the longest real sequence.
    assert embed.shape == (2, 6, 100)  # (B, max_len + 1, feat_dim out)
    assert torch.equal(lens, torch.tensor([5, 3]))


def test_forward_runs_and_backprops():
    model = _model()
    tokens, phone_groups, qwen, valid = _batch()
    features_lens = torch.tensor([40, 24])
    features = torch.randn(2, 40, 100)
    noise = torch.randn(2, 40, 100)
    t = torch.rand(2, 1, 1)

    loss = model(
        tokens=tokens,
        features=features,
        features_lens=features_lens,
        noise=noise,
        t=t,
        phone_groups=phone_groups,
        qwen_group_features=qwen,
        qwen_group_valid=valid,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()
    assert model.fusion.phone_embed.weight.grad is not None
    assert model.fusion.qwen_proj.weight.grad.abs().sum() > 0
    assert model.fusion.gate_proj.weight.grad is not None


def test_duration_gather_reaches_the_sentinel_position():
    """The inherited contract that a review round flagged as easy to miss:
    `get_tokens_index()` appends a residual-duration entry, so it can return
    an index equal to an utterance's phone count. If the fused sequence had
    exactly one vector per real phone, that gather would go out of range --
    and only for the batch's *longest* sequence, which shorter-sequence
    smoke tests would never catch.
    """
    from zipvoice.utils.common import get_tokens_index, prepare_avg_tokens_durations

    model = _model()
    model.eval()
    tokens, phone_groups, qwen, valid = _batch()
    # The residual entry only carries frames when the feature length doesn't
    # divide evenly by the phone count (durations are `utt_duration //
    # n_tokens` each). 41/5 leaves a remainder, so index 5 is really emitted;
    # a tidy 40/5 would silently never touch the sentinel and this test would
    # pass even against a model that had no slot for it.
    features_lens = torch.tensor([41, 24])

    embed, tokens_lens = model.forward_text_embed(tokens, phone_groups, qwen, valid)
    durations = prepare_avg_tokens_durations(features_lens, tokens_lens)
    index = get_tokens_index(durations, int(features_lens.max()))

    longest = int(tokens_lens.argmax())
    assert int(index[longest].max()) == int(tokens_lens[longest]), (
        "expected the residual index to land exactly on the sentinel slot"
    )
    assert int(index.max()) < embed.shape[1], (
        "the fused sequence must have a vector at the sentinel index"
    )
    # And the full path a training step takes must not raise.
    text_condition, _ = model.forward_text_condition(
        embed, tokens_lens, features_lens
    )
    assert text_condition.shape == (2, 41, 100)


def test_starts_near_phone_only():
    """At init the fused conditioning must be ~what a phone-only frontend
    would give, which is what makes transplanting the source checkpoint's
    `in_proj` and phone embedding meaningful.
    """
    model = _model()
    model.eval()  # the Zipformer is stochastic in train mode
    tokens, phone_groups, qwen, valid = _batch()

    fused, _ = model.forward_text_embed(tokens, phone_groups, qwen, valid)
    phone_only, _ = model.forward_text_embed(tokens, phone_groups, None, None)

    rel = (fused - phone_only).norm() / phone_only.norm()
    assert rel < 0.1, f"expected a near-phone-only start, got {rel:.3f}"


def test_none_groups_falls_back_to_phone_only():
    model = _model()
    model.eval()
    tokens, phone_groups, qwen, valid = _batch()
    # Utterance 1's lm_token alignment failed (sprint 000's documented case).
    valid[1] = False

    # Exact at the fusion module's own output -- the actual guarantee.
    from zipvoice.models.modules.fusion import build_phone_group_index
    from zipvoice.utils.common import pad_labels

    padded = pad_labels(tokens, pad_id=model.pad_id, device=torch.device("cpu"))
    group_ids, has_group = build_phone_group_index(
        phone_groups, padded_len=padded.shape[1], device=torch.device("cpu")
    )
    fused_raw = model.fusion(padded, group_ids, has_group, qwen, valid)
    torch.testing.assert_close(fused_raw[1], model.fusion.phone_embed(padded)[1])

    # And end-to-end through the text encoder.
    fused, _ = model.forward_text_embed(tokens, phone_groups, qwen, valid)
    phone_only, _ = model.forward_text_embed(tokens, phone_groups, None, None)
    torch.testing.assert_close(fused[1], phone_only[1])


def test_fusion_requires_phone_groups():
    model = _model()
    tokens, _, qwen, valid = _batch()
    with pytest.raises(AssertionError, match="phone_groups"):
        model.forward_text_embed(tokens, None, qwen, valid)


def test_inference_paths_refuse_until_the_contract_is_decided():
    """Generation must not silently invent a prompt/target composition
    contract while that decision is still open -- see sprint 003's
    "Deferred, not decided here".
    """
    model = _model()
    tokens, _, _, _ = _batch()
    with pytest.raises(NotImplementedError, match="composition"):
        model.forward_text_inference_ratio_duration(
            tokens=tokens,
            prompt_tokens=[[1, 2], [3, 4]],
            prompt_features_lens=torch.tensor([10, 10]),
            speed=1.0,
        )
    with pytest.raises(NotImplementedError, match="composition"):
        model.forward_text_inference_gt_duration(
            tokens=tokens,
            features_lens=torch.tensor([20, 20]),
            prompt_tokens=[[1, 2], [3, 4]],
            prompt_features_lens=torch.tensor([10, 10]),
        )


def test_fusion_rejects_pretrained_embed_source():
    with pytest.raises(AssertionError, match="embed_source"):
        _model(embed_source="pretrained", pretrained_embed_model="Qwen/Qwen2.5-0.5B")


def test_embedding_frontend_is_unchanged():
    """The pre-fusion path must behave exactly as before -- same module tree,
    same call signature, no fusion module built.
    """
    torch.manual_seed(0)
    model = ZipVoice(vocab_size=1000, text_embed_dim=64, **_TINY_KWARGS)
    assert model.text_frontend == "embedding"
    assert model.fusion is None
    assert model.embed.weight.shape == (1000, 64)
    embed, lens = model.forward_text_embed([[1, 2, 3, 4], [5, 6]])
    assert embed.shape == (2, 5, 100)
    assert torch.equal(lens, torch.tensor([4, 2]))
