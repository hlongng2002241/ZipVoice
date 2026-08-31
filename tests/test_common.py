import torch

from zipvoice.utils.common import (
    AttributeDict,
    get_tokens_index,
    prepare_avg_tokens_durations,
    prepare_input,
)


def test_prepare_avg_tokens_durations_no_mask_matches_original_behaviour():
    features_lens = [10]
    tokens_lens = [5]
    durations = prepare_avg_tokens_durations(features_lens, tokens_lens)
    assert durations == [[2, 2, 2, 2, 2]]


def test_prepare_avg_tokens_durations_zero_duration_mask():
    # 4 tokens, the first one is a [LANG:xx] control token -> 0 frames.
    # The remaining 3 acoustic tokens split the 9 frames evenly.
    features_lens = [9]
    tokens_lens = [4]
    zero_duration_mask = [[True, False, False, False]]
    durations = prepare_avg_tokens_durations(
        features_lens, tokens_lens, zero_duration_mask=zero_duration_mask
    )
    assert durations == [[0, 3, 3, 3]]


def test_prepare_avg_tokens_durations_all_zero_raises():
    features_lens = [9]
    tokens_lens = [2]
    zero_duration_mask = [[True, True]]
    try:
        prepare_avg_tokens_durations(
            features_lens, tokens_lens, zero_duration_mask=zero_duration_mask
        )
        assert False, "expected an AssertionError"
    except AssertionError:
        pass


def test_zero_duration_token_gets_no_frames_in_tokens_index():
    # 1 utterance, 4 tokens (first is a zero-duration control token),
    # 9 total frames.
    durations = prepare_avg_tokens_durations(
        features_lens=[9],
        tokens_lens=[4],
        zero_duration_mask=[[True, False, False, False]],
    )
    num_frames = 9
    tokens_index = get_tokens_index(durations, num_frames)
    # The control token (index 0) should never be selected for any frame.
    assert 0 not in tokens_index[0].tolist()
    # The 3 acoustic tokens (indices 1, 2, 3) should evenly cover all 9 frames.
    expected = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert torch.equal(tokens_index[0], expected)


def _batch(tokens, zero_duration_mask=None, n_frames=5):
    batch = {
        "tokens": tokens,
        "features": torch.randn(len(tokens), n_frames, 8),
        "features_lens": torch.tensor([n_frames] * len(tokens)),
    }
    if zero_duration_mask is not None:
        batch["zero_duration_mask"] = zero_duration_mask
    return batch


def test_prepare_input_default_arity_matches_master_contract():
    # return_zero_duration_mask defaults to False -- return shape when
    # return_tokens=True must be [tokens, features, features_lens], exactly
    # master's original arity, not [tokens, mask, features, features_lens].
    params = AttributeDict({"feat_scale": 1.0})
    result = prepare_input(params, _batch([[1, 2]]), device="cpu")
    assert len(result) == 3
    tokens, features, features_lens = result
    assert tokens == [[1, 2]]


def test_prepare_input_explicit_mask_request_returns_four_tuple():
    params = AttributeDict({"feat_scale": 1.0})
    result = prepare_input(
        params,
        _batch([[1, 2]], zero_duration_mask=[[True, False]]),
        device="cpu",
        return_zero_duration_mask=True,
    )
    assert len(result) == 4
    tokens, mask, features, features_lens = result
    assert mask == [[True, False]]


def test_prepare_input_all_none_mask_normalizes_to_none():
    params = AttributeDict({"feat_scale": 1.0})
    _, mask, _, _ = prepare_input(
        params,
        _batch([[1, 2], [3, 4]], zero_duration_mask=[None, None]),
        device="cpu",
        return_zero_duration_mask=True,
    )
    assert mask is None


def test_prepare_input_mixed_mask_normalizes_untagged_to_all_false():
    params = AttributeDict({"feat_scale": 1.0})
    _, mask, _, _ = prepare_input(
        params,
        _batch([[1, 2], [3, 4, 5]], zero_duration_mask=[None, [True, False, False]]),
        device="cpu",
        return_zero_duration_mask=True,
    )
    assert mask == [[False, False], [True, False, False]]


def test_prepare_input_mask_requires_tokens():
    params = AttributeDict({"feat_scale": 1.0})
    try:
        prepare_input(
            params,
            _batch([[1, 2]]),
            device="cpu",
            return_tokens=False,
            return_zero_duration_mask=True,
        )
        assert False, "expected an AssertionError"
    except AssertionError:
        pass
