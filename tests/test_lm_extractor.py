"""Contract tests for the Qwen branch's truncated extractor (see
zipvoice/models/modules/lm_extractor.py and the ADR's point 6).

The load-bearing test here is `test_truncated_matches_full_forward`: it
proves the extractor's shortened layer stack produces *exactly* the same
hidden state a full 24-layer forward would report at layer N, i.e. that
"genuinely truncated" and "sliced from a full forward" agree numerically
while only the truncated one pays for N layers of compute.

Memory note: these tests run on CPU and load Qwen2.5-0.5B (~2GB in fp32).
The parity test loads a second, full copy and frees it immediately -- this
machine is shared, so nothing here holds two copies longer than it must.
"""

import gc

import pytest
import torch

from zipvoice.models.modules.lm_extractor import TruncatedQwenExtractor

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
NUM_LAYERS = 4


@pytest.fixture(scope="module")
def extractor():
    ex = TruncatedQwenExtractor(
        model_name=MODEL_NAME, num_layers=NUM_LAYERS, device=torch.device("cpu")
    )
    yield ex
    del ex
    gc.collect()


@pytest.fixture(scope="module")
def token_ids():
    """Two short, unequal-length id sequences (so padding is exercised)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    a = tok.encode("Sau khi shopping xong, chloe di an pho.", add_special_tokens=False)
    b = tok.encode("Hello world.", add_special_tokens=False)
    assert len(a) != len(b), "fixture should have unequal lengths to test padding"
    return [a, b]


def test_only_n_layers_are_kept(extractor):
    assert len(extractor.model.layers) == NUM_LAYERS
    assert extractor.total_layers > NUM_LAYERS, "fixture assumes a deeper base model"
    assert isinstance(extractor.model.norm, torch.nn.Identity), (
        "Qwen's own final RMSNorm must be skipped -- it is calibrated for "
        "full-depth statistics, not layer-N statistics (ADR point 6)"
    )
    assert extractor.model.config.use_cache is False
    assert not extractor.model.training
    assert all(not p.requires_grad for p in extractor.model.parameters())


def test_truncated_matches_full_forward(extractor, token_ids):
    """The contract: truncating the stack == reading hidden_states[N] of a
    full forward. If these ever diverge, the extraction contract is broken.
    """
    from transformers import AutoModel

    full = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    full.eval()
    try:
        ids = torch.tensor([token_ids[1]], dtype=torch.int64)
        with torch.no_grad():
            ref = full(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[NUM_LAYERS]
    finally:
        del full
        gc.collect()

    got, lens = extractor.hidden_states([token_ids[1]])
    assert lens.tolist() == [len(token_ids[1])]
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_batched_matches_single(extractor, token_ids):
    """Right-padding must not perturb real positions -- otherwise batch
    composition would silently change the features a given utterance gets.
    """
    batched, lens = extractor.hidden_states(token_ids)
    for i, ids in enumerate(token_ids):
        single, _ = extractor.hidden_states([ids])
        torch.testing.assert_close(
            batched[i, : len(ids)], single[0, : len(ids)], rtol=1e-4, atol=1e-4
        )
    assert lens.tolist() == [len(ids) for ids in token_ids]


def test_pooled_groups_takes_the_last_subtoken(extractor, token_ids):
    hidden, _ = extractor.hidden_states(token_ids)
    # Group the first utterance's tokens into two groups; leave the last
    # token out of any group entirely (mirroring how the [LANG:xx] tag at
    # index 0 belongs to no group in a real artifact).
    n = len(token_ids[0])
    groups_a = [[1, 2], [3, 4, 5]]
    groups_b = [[0]]
    assert n > 6, "fixture too short for this grouping"

    pooled, valid = extractor.pooled_groups(token_ids, [groups_a, groups_b])
    assert pooled.shape == (2, 2, extractor.hidden_size)
    assert valid.tolist() == [[True, True], [True, False]]

    torch.testing.assert_close(pooled[0, 0], hidden[0, 2])  # last of [1,2]
    torch.testing.assert_close(pooled[0, 1], hidden[0, 5])  # last of [3,4,5]
    torch.testing.assert_close(pooled[1, 0], hidden[1, 0])  # last of [0]
    # Padded group slot stays zero and invalid.
    assert torch.count_nonzero(pooled[1, 1]) == 0


def test_none_groups_marks_whole_utterance_invalid(extractor, token_ids):
    """`lm_token_groups=None` (sprint 000's alignment-failure case) must be
    distinguishable from "a group whose vector happens to be zero", so the
    caller can fall back to phone-only for that utterance.
    """
    pooled, valid = extractor.pooled_groups(token_ids, [[[1, 2]], None])
    assert valid[0].any(), "utterance 0 aligned fine"
    assert not valid[1].any(), "utterance 1 failed alignment -> nothing valid"
    assert torch.count_nonzero(pooled[1]) == 0


def test_empty_group_is_invalid_not_zero_vector(extractor, token_ids):
    pooled, valid = extractor.pooled_groups(token_ids, [[[1], []], [[0]]])
    assert valid[0].tolist() == [True, False]
    assert torch.count_nonzero(pooled[0, 1]) == 0


def test_extraction_builds_no_autograd_graph(extractor, token_ids):
    hidden, _ = extractor.hidden_states(token_ids)
    assert not hidden.requires_grad
    assert hidden.grad_fn is None
