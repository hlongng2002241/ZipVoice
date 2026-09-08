# Copyright      2026  LongNH (with Claude Code assistance)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phone + Qwen fusion, gated per group.

See docs/adr/2026-09-02__qwen_phone_fusion_text_frontend.md (points 3-5) and
sprint 003's steps 1 and 3.

The shape of the design, in one place:

  - The **phone branch stays per-phone**. Every phone position keeps its own
    embedding; nothing is averaged away. At the gate's initial value this
    module's output is (approximately) just the source checkpoint's own
    per-phone conditioning, which is the whole reason `in_proj` and the phone
    embedding can be transplanted at all.
  - The **Qwen branch is necessarily per-group**: Qwen runs over `lm_tokens`,
    which have no 1:1 correspondence with phones, so one pooled vector per
    group is the finest granularity available. That single vector is
    broadcast to every phone in its group.
  - A **learned per-group gate** decides how much of each to use:

        fused[i] = (1 - g[grp(i)]) * phone_embed[i] + g[grp(i)] * lm_vec[grp(i)]

    initialized at `g ≈ eps` (0.01), i.e. near-phone-only, with no floor and
    no entropy regularization (per the ADR, gate collapse is an accepted,
    monitored risk rather than an architecturally prevented one).

Two things the ADR left open and this module pins down:

  - **Gate input**: `[projected Qwen group vector ; mean phone embedding of
    the group]`. The gate's real job is "is the phone branch trustworthy for
    this group?", which is a *mismatch* question -- the phone branch is wrong
    exactly where the group's true language differs from the language the
    whole utterance was phonemized under. Seeing only the Qwen vector would
    tell the gate what language the group looks like, but not what the phones
    were built as; seeing both lets it compare. (The group's phone mean is
    used *only* to compute the gate scalar -- the output path still uses each
    phone's own embedding.) A documented alternative, if this proves weak: add
    an explicit primary-language embedding to the gate input instead of
    inferring the phonemization language from the phones themselves.
  - **Gate shape**: one scalar per group (not per-channel). Simplest thing
    that matches the ADR's "eps=0.01 mixing weight" language; `gate_dim` can
    be widened to `embed_dim` later for a per-channel variant.

Branch-magnitude normalization (the ADR flags this as must-resolve, since
`eps=0.01` is meaningless without it): the Qwen branch is RMS-normalized and
then rescaled by a learnable scalar initialized to the phone embedding's own
RMS. That makes "0.01" mean "1% of a phone-embedding-sized vector" rather
than 1% of an arbitrary, freshly-initialized projection's output scale.
`calibrate_lm_scale()` re-derives that scalar after the phone embedding is
transplanted from a source checkpoint (its RMS at construction time is only
the random init's).
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

NO_GROUP = -1

# Standard deviation of the gate weights at initialization. Must stay well
# above ScaledAdam's `param_min_rms` (1e-5) or the tensor cannot train at
# all -- see the note in `PhoneQwenFusion.__init__`.
GATE_WEIGHT_INIT_STD = 0.01


def build_phone_group_index(
    phone_groups: Sequence[Optional[Sequence[Sequence[int]]]],
    padded_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn per-utterance phone groups into a per-phone-position group index.

    Args:
      phone_groups: per utterance, `FusionTokenizerArtifact.phone_groups`
        (each group a list of indices into that utterance's `phone_ids`), or
        `None` for an utterance that should fall back to phone-only.
      padded_len: width of the padded phone sequence -- i.e. `max phone len
        + 1`, including the trailing sentinel position `pad_labels()` adds
        (see `PhoneQwenFusion.forward`'s note on why that position exists).
      device: device for the returned tensors.

    Returns:
      (group_ids, in_group): both (B, padded_len). `group_ids` holds the
      group index for each real phone position and `NO_GROUP` (-1) elsewhere;
      `in_group` is the corresponding bool mask. Positions without a group --
      padding, the trailing sentinel, and every position of a `None`-grouped
      utterance -- get pure phone embeddings.

    Uses `repeat_interleave` over group sizes rather than scattering each
    phone index individually, which is valid because
    `FusionTokenizerArtifact.__post_init__` guarantees `phone_groups`
    partitions `range(len(phone_ids))` exactly, in monotonically increasing,
    non-overlapping order -- so the group ids for positions `0..n_phones-1`
    are just each group's index repeated by that group's size.
    """
    batch_size = len(phone_groups)
    group_ids = torch.full(
        (batch_size, padded_len), NO_GROUP, dtype=torch.int64, device=device
    )
    for i, groups in enumerate(phone_groups):  # ragged input: B iterations
        if not groups:
            continue
        sizes = torch.tensor([len(m) for m in groups], dtype=torch.int64)
        row = torch.repeat_interleave(torch.arange(len(groups)), sizes)
        n = int(row.numel())
        assert n <= padded_len, (n, padded_len)
        group_ids[i, :n] = row.to(device)
    return group_ids, group_ids != NO_GROUP


class PhoneQwenFusion(nn.Module):
    """Fuses a per-phone phone embedding with a per-group Qwen vector."""

    def __init__(
        self,
        phone_vocab_size: int,
        embed_dim: int = 192,
        lm_hidden_size: int = 896,
        gate_init_eps: float = 0.01,
    ):
        super().__init__()
        assert 0.0 < gate_init_eps < 1.0, gate_init_eps

        self.phone_embed = nn.Embedding(phone_vocab_size, embed_dim)
        self.lm_proj = nn.Linear(lm_hidden_size, embed_dim)
        # Gate sees both branches; see the module docstring.
        self.gate_proj = nn.Linear(2 * embed_dim, 1)
        self.lm_scale = nn.Parameter(torch.ones(()))

        self.embed_dim = embed_dim
        self.lm_hidden_size = lm_hidden_size
        self.gate_init_eps = gate_init_eps

        # Start the gate near `eps` and input-*almost*-independent. The bias
        # carries the whole intended value, logit(eps); the weights are small
        # random rather than exactly zero.
        #
        # Zero would be the obvious choice -- the logit would then be the bias
        # alone, exactly eps -- and it is wrong here, because ScaledAdam scales
        # every non-scalar tensor's update by that tensor's own RMS
        # (`optim.py:198`, `delta *= param_rms.clamp(min=param_min_rms)`, with
        # `param_min_rms=1e-5`). A zero tensor has RMS 0, so its steps are
        # floored at 1e-5 of the normal size, and the scale-growth path cannot
        # rescue it either: `scale_grads = (p * grad).sum(...)` is identically
        # zero when p is zero, and `is_too_small = param_rms < param_min_rms`
        # explicitly masks the scale step to 0 below the floor. Gradients
        # flow either way; the update *scale* is what collapses.
        #
        # So zero initialization imposes a severe initial update-scale
        # bottleneck. Measured with a constant unit gradient, RMS by step:
        # 1e-5 at 10k, 5.2e-5 at 25k, 8.1e-4 at 50k, 0.198 at 100k -- roughly
        # 10k steps to reach the floor, after which growth accelerates. It is
        # a bottleneck, not a permanent freeze, and the real escape time
        # depends on the actual gradient sequence, which this probe does not
        # model. A 0.02 init takes ~2000x larger steps from step 1 and needs
        # no escape at all, which is reason enough not to start at zero for a
        # warm-start fine-tune whose early steps are the ones that matter.
        #
        # GATE_WEIGHT_INIT_STD is small enough that it barely perturbs the
        # initial gate and large enough to be far above the RMS floor. As an
        # order-of-magnitude guide the logit spread is about
        # std * lm_scale * sqrt(2 * embed_dim) ~= 0.1, against a bias of
        # logit(0.01) = -4.60 -- taking both halves of the gate input to have
        # per-element RMS ~= lm_scale. That holds by construction for the
        # Qwen half (RMS-normalized, then scaled by it) but only loosely for
        # the phone half, which is a per-group *mean* of embeddings and so can
        # be considerably smaller than the table's RMS. The estimate is
        # therefore an upper bound; what actually pins this down is the
        # measurement below. Measured on the real config, loading the
        # transplanted warm-start checkpoint and running real VI/EN text
        # through the extractor: gate weight RMS 9.5e-3 (~950x the floor),
        # gate 0.0086..0.0109 across real groups (mean 0.0095, std 5e-4),
        # and the fused output within 1.3% of plain phone-only.
        with torch.no_grad():
            self.gate_proj.weight.normal_(0.0, GATE_WEIGHT_INIT_STD)
            self.gate_proj.bias.fill_(math.log(gate_init_eps / (1.0 - gate_init_eps)))
            self.calibrate_lm_scale()

    @torch.no_grad()
    def calibrate_lm_scale(self) -> float:
        """Set the Qwen branch's output scale to the phone embedding's RMS.

        Call this again after transplanting a source checkpoint's phone
        embedding -- at construction the RMS measured here is only the random
        initialization's, which is not what the model will actually train
        with. Without this, `gate_init_eps` mixes in 1% of a vector whose
        magnitude is arbitrary relative to the phone branch, and "near
        phone-only initialization" stops meaning anything specific.
        """
        rms = self.phone_embed.weight.pow(2).mean().sqrt()
        self.lm_scale.fill_(float(rms))
        return float(rms)

    def _lm_and_gate(
        self,
        phone_embed: torch.Tensor,
        phone_group_ids: torch.Tensor,
        phone_in_group: torch.Tensor,
        lm_group_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared by `forward()` and `current_gate_values()` so the two can
        never drift apart.

        Returns (lm_vec, gate, safe_ids): the scaled per-group Qwen vectors
        (B, G, D), the per-group gate (B, G, 1), and the clamped group index
        (B, S) both callers reuse for gathering.
        """
        batch_size, num_groups, _ = lm_group_features.shape

        # Project, then put the Qwen branch on the phone branch's scale so
        # `gate_init_eps` has a defined functional meaning.
        lm = self.lm_proj(lm_group_features)  # (B, G, D)
        lm = lm * torch.rsqrt(
            lm.pow(2).mean(dim=-1, keepdim=True) + 1e-8
        )  # unit RMS per group vector
        lm = lm * self.lm_scale

        # Keep this block dtype-coherent by construction. Under autocast,
        # `lm_proj` returns fp16 while `phone_embed` -- an `nn.Embedding`,
        # which autocast does not cast -- stays fp32. As written above, the
        # RMS normalization happens to promote `lm_vec` back to fp32 (autocast
        # runs `mean`/`rsqrt` in fp32) so the two agree, but that is
        # incidental to operation order: `scatter_add_` requires matching
        # dtypes and would raise the moment that ordering changed. Cast
        # explicitly rather than depend on autocast's promotion rules.
        phones = phone_embed.to(lm.dtype)

        # Gate input: this group's Qwen vector alongside the mean embedding
        # of its own phones (mean used for gating only -- see docstring).
        # Clamped at both ends: `min` handles NO_GROUP, `max` handles group
        # ids past the Qwen tensor's group dimension (see `forward`'s note on
        # why those legitimately occur). Both kinds of position are masked
        # out of `member` below, so a clamped index contributes nothing to
        # any group's mean.
        safe_ids = phone_group_ids.clamp(min=0, max=num_groups - 1)  # (B, S)
        member = phone_in_group.unsqueeze(-1).to(lm.dtype)  # (B, S, 1)
        index = safe_ids.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        sums = torch.zeros_like(lm).scatter_add_(1, index, phones * member)
        counts = torch.zeros(
            batch_size, num_groups, device=lm.device, dtype=lm.dtype
        ).scatter_add_(1, safe_ids, member.squeeze(-1))
        phone_group_mean = sums / counts.clamp(min=1.0).unsqueeze(-1)  # (B, G, D)

        gate = torch.sigmoid(
            self.gate_proj(torch.cat([lm, phone_group_mean], dim=-1))
        )  # (B, G, 1)
        return lm, gate, safe_ids

    def forward(
        self,
        phone_ids_padded: torch.Tensor,
        phone_group_ids: torch.Tensor,
        phone_in_group: torch.Tensor,
        lm_group_features: Optional[torch.Tensor] = None,
        lm_group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          phone_ids_padded: (B, S) phone ids from `pad_labels()`. Note `S`
            already includes the one trailing sentinel position `pad_labels()`
            appends beyond every utterance's real length: `get_tokens_index()`
            can return an index equal to an utterance's phone count (it
            appends a residual-duration entry), so a vector must exist there
            to gather from. Those positions belong to no group and come out
            of this module as plain pad embeddings, exactly as they would
            from `self.embed(...)` in the pre-fusion model.
          phone_group_ids: (B, S) group index per position, `NO_GROUP` where
            none -- from `build_phone_group_index()`.
          phone_in_group: (B, S) bool -- does this position map to a group
            slot? Initially `phone_group_ids != NO_GROUP`, then narrowed
            below by `& in_range`, after which it is no longer derivable
            from `phone_group_ids`, which is why it travels separately.

            It is NOT "is this a real phone". Within one aligned utterance
            those coincide, but this tensor spans the padded batch, where
            three other kinds of position exist: batch padding, the trailing
            sentinel (present even for the longest utterance), and every
            position of an utterance whose alignment returned None.

            Distinguish it from `phone_has_conditioning` below: this asks
            whether a slot is addressed, that one whether the slot holds
            real Qwen evidence.
          lm_group_features: (B, G, lm_hidden_size) pooled Qwen vectors,
            or None to run phone-only (equivalent to an all-invalid batch --
            used by tests, and by any caller without a Qwen branch).
          lm_group_mask: (B, G) bool marking real pooled groups.

        Returns:
          (B, S, embed_dim) fused text embedding.
        """
        phone_embed = self.phone_embed(phone_ids_padded)  # (B, S, D)

        if lm_group_features is None:
            return phone_embed

        assert lm_group_mask is not None, (
            "lm_group_mask must accompany lm_group_features"
        )
        batch_size, num_groups, _ = lm_group_features.shape
        assert lm_group_mask.shape == (batch_size, num_groups)

        # A phone group index can legitimately exceed the Qwen tensor's group
        # dimension. `TruncatedQwenExtractor.pooled_groups()` sizes that
        # dimension from the utterances that actually aligned, so an
        # utterance whose lm_token alignment failed (`lm_token_groups=None`,
        # a documented outcome) contributes zero groups to it while still
        # having phone groups of its own. Those positions must fall back to
        # phone-only -- which is exactly what this frontend promises for a
        # failed alignment -- rather than abort the batch. Treating them as
        # "no group" does that; asserting here would instead kill training on
        # a batch composition that is expected to occur.
        in_range = phone_group_ids < num_groups
        if not bool(in_range.all()):
            # Out-of-range is only legitimate for an utterance that aligned
            # nothing at all. If one that *did* align has a stray index, the
            # phone/lm_token group counts have genuinely diverged, which
            # `FusionTokenizerArtifact.__post_init__` is supposed to prevent
            # -- fail loudly rather than silently drop its Qwen branch.
            offending = (~in_range).any(dim=1)
            assert not bool((lm_group_mask.any(dim=1) & offending).any()), (
                "phone group index out of range for an utterance whose "
                "lm_token alignment succeeded -- phone_groups and "
                "lm_token_groups have diverged"
            )
            phone_in_group = phone_in_group & in_range

        lm_vec, gate, safe_ids = self._lm_and_gate(
            phone_embed, phone_group_ids, phone_in_group, lm_group_features
        )
        index = safe_ids.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        # Broadcast group-level Qwen contribution and gate onto phones.
        lm_at_phone = torch.gather(lm_vec, 1, index)  # (B, S, D)
        gate_at_phone = torch.gather(gate, 1, safe_ids.unsqueeze(-1))  # (B, S, 1)
        phone_has_conditioning = phone_in_group & torch.gather(
            lm_group_mask, 1, safe_ids
        )  # (B, S)

        fused = (1.0 - gate_at_phone) * phone_embed + gate_at_phone * lm_at_phone
        # Anywhere without a real, valid group -- padding, the trailing
        # sentinel, and every position of an utterance whose lm_token
        # alignment failed -- falls back to the plain phone embedding.
        return torch.where(phone_has_conditioning.unsqueeze(-1), fused, phone_embed)

    @torch.no_grad()
    def current_gate_values(
        self,
        phone_ids_padded: torch.Tensor,
        phone_group_ids: torch.Tensor,
        phone_in_group: torch.Tensor,
        lm_group_features: torch.Tensor,
        lm_group_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-group gate values for logging/monitoring (sprint 004 watches
        these for collapse). Returns (B, G), zero where the group isn't real.
        """
        phone_embed = self.phone_embed(phone_ids_padded)
        _, gate, _ = self._lm_and_gate(
            phone_embed, phone_group_ids, phone_in_group, lm_group_features
        )
        gate = gate.squeeze(-1)
        return gate * lm_group_mask.to(gate.dtype)
