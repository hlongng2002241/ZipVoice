from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from lhotse.dataset import PrecomputedFeatures

from zipvoice.dataset.dataset import SpeechSynthesisDataset


def _cut(cut_id, tokens, zero_duration_mask=None):
    rec = Recording(
        id=cut_id,
        sources=[AudioSource(type="file", channels=[0], source="/dev/null")],
        sampling_rate=24000,
        num_samples=24000,
        duration=1.0,
    )
    sup = SupervisionSegment(id=cut_id, recording_id=cut_id, start=0, duration=1.0, text="x")
    cut = rec.to_cut()
    sup.tokens = tokens
    if zero_duration_mask is not None:
        sup.zero_duration_mask = zero_duration_mask
    cut.supervisions = [sup]
    return cut


class _FakeFeatureInputStrategy(PrecomputedFeatures):
    """Bypasses real feature extraction -- these tests only care about the
    non-feature batch keys."""

    def __call__(self, cuts):
        import torch

        n = len(cuts)
        return torch.zeros(n, 5, 8), torch.tensor([5] * n)


def test_legacy_tokenizer_batch_has_no_zero_duration_mask_key():
    # No cut has a zero_duration_mask attribute at all -- the batch dict
    # must not have the key, matching master's schema exactly.
    cuts = CutSet.from_cuts([_cut("a", [1, 2]), _cut("b", [3, 4])])
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[cuts]
    assert "zero_duration_mask" not in batch


def test_multilingual_tokenizer_batch_has_zero_duration_mask_key():
    cuts = CutSet.from_cuts(
        [_cut("a", [1, 2], zero_duration_mask=[True, False]), _cut("b", [3, 4])]
    )
    ds = SpeechSynthesisDataset(
        feature_input_strategy=_FakeFeatureInputStrategy(), return_tokens=True
    )
    batch = ds[cuts]
    assert batch["zero_duration_mask"] == [[True, False], None]
