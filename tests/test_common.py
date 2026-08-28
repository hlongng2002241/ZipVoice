import torch

from zipvoice.utils.common import get_tokens_index, prepare_avg_tokens_durations


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
