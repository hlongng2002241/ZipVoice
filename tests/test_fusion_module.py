"""Tests for the phone+Qwen fusion module (zipvoice/models/modules/fusion.py).

These pin the properties sprint 003's step 1 committed to, several of which
came out of a review round that caught them being ambiguous or missing:

  - the phone branch stays *per-phone* (only the Qwen branch is collapsed to
    group granularity and broadcast);
  - at initialization the output is near-identical to the plain phone
    embedding, which is what makes transplanting `in_proj`/the phone
    embedding from a source checkpoint meaningful;
  - the trailing sentinel position `pad_labels()` adds comes out as a plain
    pad embedding (a missing vector there makes `get_tokens_index()` index
    out of range, and only for the longest sequence in a batch);
  - an utterance whose `lm_token` alignment failed falls back to phone-only
    rather than crashing or silently misaligning.
"""

import math

import pytest
import torch

from zipvoice.models.modules.fusion import (
    NO_GROUP,
    PhoneQwenFusion,
    build_phone_group_index,
)
from zipvoice.utils.common import pad_labels

PHONE_VOCAB = 32
EMBED_DIM = 8
QWEN_DIM = 16
PAD_ID = 0


@pytest.fixture
def fusion():
    torch.manual_seed(0)
    return PhoneQwenFusion(
        phone_vocab_size=PHONE_VOCAB, embed_dim=EMBED_DIM, qwen_hidden_size=QWEN_DIM
    )


def _batch(device=torch.device("cpu")):
    """Two utterances: 5 phones in 2 groups, and 3 phones in 2 groups."""
    phone_ids = [[3, 4, 5, 6, 7], [8, 9, 10]]
    phone_groups = [[[0, 1], [2, 3, 4]], [[0], [1, 2]]]
    padded = pad_labels(phone_ids, pad_id=PAD_ID, device=device)  # (2, 6)
    group_ids, in_group = build_phone_group_index(
        phone_groups, padded_len=padded.shape[1], device=device
    )
    return phone_ids, phone_groups, padded, group_ids, in_group


def test_build_phone_group_index_matches_groups():
    _, _, padded, group_ids, in_group = _batch()
    # Utterance 0: phones 0,1 -> group 0; phones 2,3,4 -> group 1; then the
    # trailing sentinel (and any padding) belong to no group.
    assert group_ids[0].tolist() == [0, 0, 1, 1, 1, NO_GROUP]
    # Utterance 1 is shorter, so it has both its sentinel and real padding.
    assert group_ids[1].tolist() == [0, 1, 1, NO_GROUP, NO_GROUP, NO_GROUP]
    assert in_group.tolist() == [
        [True, True, True, True, True, False],
        [True, True, True, False, False, False],
    ]


def test_build_phone_group_index_handles_none_utterance():
    group_ids, in_group = build_phone_group_index(
        [[[0, 1]], None], padded_len=3, device=torch.device("cpu")
    )
    assert group_ids[1].tolist() == [NO_GROUP] * 3
    assert not in_group[1].any(), "a None-grouped utterance claims no positions"


def test_pad_labels_appends_the_sentinel_position():
    """Guards the inherited contract the fusion path must preserve: the
    padded width is max_len + 1, so `get_tokens_index()`'s residual index
    (which can equal an utterance's phone count) always has a slot.
    """
    padded = pad_labels([[1, 2, 3], [4]], pad_id=PAD_ID, device=torch.device("cpu"))
    assert padded.shape == (2, 4), "expected max real length (3) + 1 sentinel"
    assert padded[0].tolist() == [1, 2, 3, PAD_ID]


def test_gate_starts_near_eps(fusion):
    """The gate must start at ~eps so the model begins near phone-only.

    Not *exactly* eps: the weights are deliberately small-random rather than
    zero (see `test_gate_weights_are_trainable_under_scaled_adam`), so the
    per-group values scatter slightly around eps. The tolerance below is far
    tighter than anything that would disturb the near-phone-only start.
    """
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)
    gates = fusion.current_gate_values(padded, group_ids, in_group, qwen, valid)
    assert torch.allclose(
        gates, torch.full_like(gates, fusion.gate_init_eps), atol=5e-3
    ), f"gate must start near eps, got {gates}"


def test_gate_weights_are_trainable_under_scaled_adam(fusion):
    """Regression: the gate weights must not be initialized to exactly zero.

    ScaledAdam scales each non-scalar tensor's update by that tensor's own
    RMS, floored at `param_min_rms`. A zero-initialized tensor therefore
    takes steps 1e-5 of normal size and never escapes zero -- measured, a
    zero init moves to RMS 1e-5 after 10k steps of constant unit gradient.
    The gate would silently degrade to a single global constant (its scalar
    bias) instead of the per-group input-dependent gate the ADR specifies.
    """
    from zipvoice.utils.optim import ScaledAdam

    param_min_rms = ScaledAdam([torch.nn.Parameter(torch.zeros(2, 2))]).param_groups[
        0
    ]["param_min_rms"]
    rms = fusion.gate_proj.weight.pow(2).mean().sqrt().item()
    assert rms > 100 * param_min_rms, (
        f"gate weight RMS {rms:.2e} is too close to ScaledAdam's floor "
        f"{param_min_rms:.0e} -- it will not train"
    )


def test_output_is_near_phone_only_at_init(fusion):
    """At init the fused output must be ~the plain phone embedding -- that's
    what makes the source-checkpoint transplant meaningful.
    """
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)

    fused = fusion(padded, group_ids, in_group, qwen, valid)
    phone_only = fusion.phone_embed(padded)

    assert fused.shape == phone_only.shape
    rel = (fused - phone_only).norm() / phone_only.norm()
    assert rel < 0.05, f"expected a near-phone-only start, got relative diff {rel:.3f}"


def test_phone_identity_is_preserved_within_a_group(fusion):
    """The Qwen contribution is shared across a group, but each phone keeps
    its own embedding -- two different phones in the same group must not
    collapse to the same vector.
    """
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)
    fused = fusion(padded, group_ids, in_group, qwen, valid)

    # Positions 0 and 1 are distinct phones in the same group (group 0).
    assert group_ids[0, 0] == group_ids[0, 1]
    assert not torch.allclose(fused[0, 0], fused[0, 1]), (
        "phones sharing a group collapsed to one vector -- the phone branch "
        "must stay per-phone"
    )
    # And their difference should still be exactly the phone-embedding
    # difference scaled by (1 - gate), since the Qwen part is identical.
    phone_embed = fusion.phone_embed(padded)
    # The gate's actual value, not the nominal eps: the weights are small
    # random, so group 0's gate is near eps but not identical to it.
    gate = fusion.current_gate_values(padded, group_ids, in_group, qwen, valid)[0, 0]
    torch.testing.assert_close(
        fused[0, 0] - fused[0, 1],
        (1.0 - gate) * (phone_embed[0, 0] - phone_embed[0, 1]),
        rtol=1e-4,
        atol=1e-6,
    )


def test_qwen_contribution_is_shared_within_a_group(fusion):
    """The flip side: changing one group's Qwen vector must move every phone
    of that group and nothing else.
    """
    _, _, padded, group_ids, in_group = _batch()
    valid = torch.ones(2, 2, dtype=torch.bool)
    qwen_a = torch.zeros(2, 2, QWEN_DIM)
    qwen_b = qwen_a.clone()
    qwen_b[0, 1] = torch.randn(QWEN_DIM)  # perturb utterance 0's group 1 only

    out_a = fusion(padded, group_ids, in_group, qwen_a, valid)
    out_b = fusion(padded, group_ids, in_group, qwen_b, valid)
    moved = (out_a - out_b).abs().sum(dim=-1) > 1e-8

    assert moved[0].tolist() == [False, False, True, True, True, False], (
        "only group 1's phones (positions 2-4) should move"
    )
    assert not moved[1].any(), "the other utterance must be untouched"


def test_sentinel_and_padding_get_plain_phone_embeddings(fusion):
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)
    fused = fusion(padded, group_ids, in_group, qwen, valid)
    phone_only = fusion.phone_embed(padded)

    # Utterance 0's sentinel is position 5; utterance 1's are 3..5.
    torch.testing.assert_close(fused[0, 5], phone_only[0, 5])
    torch.testing.assert_close(fused[1, 3:], phone_only[1, 3:])


def test_invalid_groups_fall_back_to_phone_only(fusion):
    """`lm_token_groups=None` for an utterance (sprint 000's documented
    alignment-failure case) must yield exactly the phone-only path.
    """
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)
    valid[1] = False  # utterance 1's alignment failed

    fused = fusion(padded, group_ids, in_group, qwen, valid)
    phone_only = fusion.phone_embed(padded)
    torch.testing.assert_close(fused[1], phone_only[1])
    assert not torch.allclose(fused[0], phone_only[0]), "utterance 0 still fuses"


def test_failed_alignment_with_more_phone_groups_than_qwen_groups(fusion):
    """Regression: a batch mixing a failed alignment with a *larger* phone
    group count than any aligned utterance used to abort the whole batch.

    `pooled_groups()` sizes its group dimension from the utterances that
    aligned, so an utterance with `lm_token_groups=None` contributes none --
    while still having phone groups of its own, whose indices then run past
    that dimension. The promise for a failed alignment is phone-only, not a
    crash, and this composition is expected to occur in real training.
    """
    phone_ids = [[3, 4, 5, 6, 7], [8, 9]]
    phone_groups = [[[0], [1], [2], [3], [4]], [[0], [1]]]  # 5 groups vs 2
    padded = pad_labels(phone_ids, pad_id=PAD_ID, device=torch.device("cpu"))
    group_ids, in_group = build_phone_group_index(
        phone_groups, padded_len=padded.shape[1], device=torch.device("cpu")
    )
    assert int(group_ids.max()) == 4

    # What the extractor returns for [None, [[..],[..]]]: only 2 group slots.
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.tensor([[False, False], [True, True]])

    fused = fusion(padded, group_ids, in_group, qwen, valid)
    phone_only = fusion.phone_embed(padded)
    assert fused.shape == phone_only.shape
    # The unaligned utterance degrades to phone-only...
    torch.testing.assert_close(fused[0], phone_only[0])
    # ...and the aligned one still fuses.
    assert not torch.allclose(fused[1, :2], phone_only[1, :2])


def test_all_alignments_failed_in_a_batch(fusion):
    """The degenerate version: nothing aligned, so the Qwen tensor collapses
    to a single group slot while phones still have several groups.
    """
    phone_ids = [[3, 4, 5], [6, 7]]
    phone_groups = [[[0], [1], [2]], [[0], [1]]]
    padded = pad_labels(phone_ids, pad_id=PAD_ID, device=torch.device("cpu"))
    group_ids, in_group = build_phone_group_index(
        phone_groups, padded_len=padded.shape[1], device=torch.device("cpu")
    )
    qwen = torch.zeros(2, 1, QWEN_DIM)
    valid = torch.zeros(2, 1, dtype=torch.bool)

    fused = fusion(padded, group_ids, in_group, qwen, valid)
    torch.testing.assert_close(fused, fusion.phone_embed(padded))


def test_diverged_groups_on_an_aligned_utterance_still_raises(fusion):
    """The guard must not become a blanket "ignore out-of-range": if an
    utterance whose alignment *succeeded* has a stray index, phone and
    lm_token groups have genuinely diverged and that should fail loudly.
    """
    phone_ids = [[3, 4, 5]]
    phone_groups = [[[0], [1], [2]]]  # 3 phone groups...
    padded = pad_labels(phone_ids, pad_id=PAD_ID, device=torch.device("cpu"))
    group_ids, in_group = build_phone_group_index(
        phone_groups, padded_len=padded.shape[1], device=torch.device("cpu")
    )
    qwen = torch.randn(1, 2, QWEN_DIM)  # ...but only 2 Qwen groups
    valid = torch.ones(1, 2, dtype=torch.bool)  # and it claims to have aligned
    with pytest.raises(AssertionError, match="diverged"):
        fusion(padded, group_ids, in_group, qwen, valid)


def test_no_qwen_features_is_pure_phone_path(fusion):
    _, _, padded, group_ids, in_group = _batch()
    fused = fusion(padded, group_ids, in_group, None, None)
    torch.testing.assert_close(fused, fusion.phone_embed(padded))


def test_qwen_scale_tracks_phone_embedding_rms(fusion):
    """`eps` only means "1% of a phone-sized vector" if the Qwen branch is
    actually rescaled to the phone embedding's magnitude -- and that has to
    be re-derived after a transplant replaces the random init.
    """
    with torch.no_grad():
        fusion.phone_embed.weight.mul_(7.0)  # stand-in for a transplant
    before = float(fusion.qwen_scale.detach())
    after = fusion.calibrate_qwen_scale()
    assert after == pytest.approx(before * 7.0, rel=1e-5)
    expected = float(fusion.phone_embed.weight.pow(2).mean().sqrt())
    assert float(fusion.qwen_scale.detach()) == pytest.approx(expected, rel=1e-6)


def test_gate_is_learnable_from_init(fusion):
    """Zero-initialized gate weights must still receive gradient, or the
    gate could never become input-dependent.
    """
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)
    fusion(padded, group_ids, in_group, qwen, valid).sum().backward()
    assert fusion.gate_proj.weight.grad is not None
    assert fusion.gate_proj.weight.grad.abs().sum() > 0, (
        "zero-init gate weights got no gradient -- the gate could never learn"
    )
    assert fusion.qwen_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("aligned", [True, False])
def test_forward_under_autocast(fusion, aligned):
    """Training runs with `--use-fp16 True`, so the fused path must survive
    autocast -- and nothing else in this suite exercises it.

    The specific hazard: `qwen_proj` is autocast to half precision while
    `phone_embed` (an `nn.Embedding`) is not, and the two meet in a
    `scatter_add_`, which requires matching dtypes. Both the aligned and the
    alignment-failed path reach that reduction, so both are checked.
    """
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.full((2, 2), aligned, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        fused = fusion(padded, group_ids, in_group, qwen, valid)
    assert fused.shape == (2, padded.shape[1], EMBED_DIM)
    assert torch.isfinite(fused).all()

    fused.float().sum().backward()
    assert fusion.phone_embed.weight.grad is not None
    if aligned:
        assert fusion.qwen_proj.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for fp16")
def test_forward_under_cuda_fp16_autocast():
    """The precision training actually uses (fp16 on CUDA), not just the CPU
    bfloat16 stand-in.
    """
    torch.manual_seed(0)
    fusion = PhoneQwenFusion(
        phone_vocab_size=PHONE_VOCAB, embed_dim=EMBED_DIM, qwen_hidden_size=QWEN_DIM
    ).cuda()
    _, _, padded, group_ids, in_group = _batch(device=torch.device("cuda"))
    qwen = torch.randn(2, 2, QWEN_DIM, device="cuda")
    valid = torch.ones(2, 2, dtype=torch.bool, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fused = fusion(padded, group_ids, in_group, qwen, valid)
    assert torch.isfinite(fused).all()
    fused.float().sum().backward()
    assert fusion.qwen_proj.weight.grad.abs().sum() > 0


def test_gate_can_reach_qwen_only(fusion):
    """Sanity: with the gate forced to 1, the output is the broadcast Qwen
    vector -- confirming the two ends of the mixing range are reachable.
    """
    _, _, padded, group_ids, in_group = _batch()
    qwen = torch.randn(2, 2, QWEN_DIM)
    valid = torch.ones(2, 2, dtype=torch.bool)
    with torch.no_grad():
        fusion.gate_proj.bias.fill_(20.0)  # sigmoid(20) ~= 1
    fused = fusion(padded, group_ids, in_group, qwen, valid)
    # Positions 0 and 1 share group 0, so at gate~1 they must now coincide.
    torch.testing.assert_close(fused[0, 0], fused[0, 1], rtol=1e-4, atol=1e-5)
