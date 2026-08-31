"""Regression tests for the inference-time zero_duration_mask gap: before
this fix, [LANG:xx] control tokens got zero acoustic duration during
training but a real (equal) share at inference, since
forward_text_inference_gt_duration/forward_text_inference_ratio_duration
never accepted a mask. See docs/proposals/2026-08-31__warmstart_from_pretrained_checkpoint.md
and this session's diagnosis of near-silent generated audio.
"""

import torch

from zipvoice.models.zipvoice import ZipVoice, _concat_zero_duration_masks

TINY_KWARGS = dict(
    feat_dim=100,
    fm_decoder_dim=32,
    fm_decoder_num_layers=[1, 1, 1, 1, 1],
    fm_decoder_downsampling_factor=[1, 2, 4, 2, 1],
    fm_decoder_cnn_module_kernel=[31, 15, 7, 15, 31],
    text_encoder_num_layers=1,
    text_encoder_dim=32,
)


def _tiny_model():
    model = ZipVoice(vocab_size=20, text_embed_dim=64, pad_id=0, **TINY_KWARGS)
    model.eval()
    return model


def test_concat_zero_duration_masks_both_none():
    assert _concat_zero_duration_masks(None, None, [[1, 2, 3]], [[4, 5]]) is None


def test_concat_zero_duration_masks_only_prompt():
    r = _concat_zero_duration_masks([[True, False, False]], None, [[1, 2, 3]], [[4, 5]])
    assert r == [[True, False, False, False, False]]


def test_concat_zero_duration_masks_only_target():
    r = _concat_zero_duration_masks(None, [[True, False]], [[1, 2, 3]], [[4, 5]])
    assert r == [[False, False, False, True, False]]


def test_concat_zero_duration_masks_both_given():
    r = _concat_zero_duration_masks(
        [[True, False, False]], [[False, True]], [[1, 2, 3]], [[4, 5]]
    )
    assert r == [[True, False, False, False, True]]


def test_forward_text_inference_gt_duration_respects_prompt_mask():
    model = _tiny_model()
    prompt_tokens = [[1, 2, 3]]
    tokens = [[4, 5]]
    prompt_features_lens = torch.tensor([9])
    features_lens = torch.tensor([6])

    with torch.no_grad():
        tc_masked, _ = model.forward_text_inference_gt_duration(
            tokens=tokens,
            features_lens=features_lens,
            prompt_tokens=prompt_tokens,
            prompt_features_lens=prompt_features_lens,
            prompt_zero_duration_mask=[[True, False, False]],
        )
        tc_unmasked, _ = model.forward_text_inference_gt_duration(
            tokens=tokens,
            features_lens=features_lens,
            prompt_tokens=prompt_tokens,
            prompt_features_lens=prompt_features_lens,
        )
    assert not torch.allclose(tc_masked, tc_unmasked)


def test_forward_text_inference_ratio_duration_accepts_prompt_mask():
    model = _tiny_model()
    prompt_tokens = [[1, 2, 3]]
    tokens = [[4, 5]]
    prompt_features_lens = torch.tensor([9])

    with torch.no_grad():
        tc, padding_mask = model.forward_text_inference_ratio_duration(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features_lens=prompt_features_lens,
            speed=1.0,
            prompt_zero_duration_mask=[[True, False, False]],
        )
    assert tc.shape[0] == 1 and tc.shape[-1] == 100
    assert padding_mask.shape[0] == 1
