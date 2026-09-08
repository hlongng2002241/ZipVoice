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

"""Qwen branch of the Qwen+phoneme fusion text frontend: a genuinely
truncated Qwen forward plus per-group pooling.

See docs/adr/2026-09-02__qwen_phone_fusion_text_frontend.md (point 6, "Qwen
extraction contract") and sprint 003's step 1/step 3.

**Deliberately not an `nn.Module` member of `ZipVoice`.** Qwen's weights are
frozen and fully reproducible from the HuggingFace id, so registering them as
submodules would add ~196M frozen parameters (embedding table + N layers) to
every saved checkpoint and to `model_avg`'s float64 copy, for no benefit --
and would force sprint 003's warm-start builder to reason about tensors that
never change. Instead the extractor lives beside the model: a caller runs it
and passes the resulting per-group vectors into `ZipVoice`, exactly like
`tokens` are already computed outside the model today. This also keeps the
"run live vs. cache offline" decision a caller-side concern (v1 runs live --
see the ADR), swappable later without touching model code.

Extraction contract implemented here (all four points pinned by the ADR):
  1. **Genuinely truncated**, not sliced: the decoder layer stack is
     physically shortened to the first N layers, so only N layers' compute
     ever runs. Verified against a full forward's `hidden_states[N]` by
     tests/test_qwen_extractor.py.
  2. **Qwen's own final RMSNorm is skipped** (replaced with `Identity`) --
     it is calibrated for 24-layer-deep statistics, not layer-N statistics;
     the freshly-trainable `linear(896->192)` downstream absorbs the scale.
  3. `use_cache=False` explicitly (Qwen2 defaults to True, building an
     unused KV-cache on every forward).
  4. `eval()` + `torch.no_grad()`: frozen, never trained, no graph built.
"""

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_NUM_LAYERS = 4  # empirically validated -- see the ADR's point 6.


class TruncatedQwenExtractor:
    """Runs the first `num_layers` of a frozen Qwen model and pools one
    vector per phone/`lm_token` group.

    Args:
      model_name: HuggingFace id, matching what `LanguageModelTokenizer`
        tokenized with (mismatched tokenizer/model ids would silently pair
        token ids with the wrong embedding rows).
      num_layers: how many decoder layers to actually run (N in the ADR).
      device / dtype: where to run and in what precision. `float32` by
        default -- these hidden states feed a freshly-initialized projection,
        so there is no source-checkpoint precision to match.
      include_lang_tag: whether the artifact's leading `[LANG:xx]` control
        token stays in Qwen's causal input. Default True, which keeps group
        indices consistent with `FusionTokenizerArtifact` exactly as built
        (its `lm_token_groups` index into the full sequence, tag included).
        Recorded explicitly because it changes every downstream hidden state.
        Caveat worth knowing: `[LANG:xx]` ids land in Qwen2.5-0.5B's *padded*
        embedding rows (151665-151668, inside the 151936-row table but past
        the 151665 real-vocab entries), so their embedding vectors are
        untrained -- and Qwen is frozen here, so it can never learn to use
        them. The tag therefore contributes an arbitrary-but-consistent
        vector at position 0 rather than a meaningful language signal; the
        language hint that actually matters reaches the model through the
        phone branch's phonemization instead.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_layers: int = DEFAULT_NUM_LAYERS,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        include_lang_tag: bool = True,
    ):
        try:
            from transformers import AutoModel
        except Exception as ex:
            raise RuntimeError(f"{ex}\nPlease run\npip install transformers")

        if not include_lang_tag:
            # Nothing below reads this flag: `hidden_states()` feeds the
            # artifact's `lm_token_ids` to Qwen verbatim, tag included, and
            # `pooled_groups()` indexes into that same sequence. Accepting
            # False would therefore run the *identical* computation while
            # claiming the tag had been removed -- silently invalidating any
            # tag-removal ablation built on it. Implementing it properly means
            # dropping index 0 from `lm_token_ids` and shifting every
            # `lm_token_groups` index down by one, which belongs in the
            # artifact, not here. Note the group slots all survive that shift:
            # index 0 belongs to no group (verified -- the lowest index any
            # group uses is 1), so removing the tag cannot empty a group, and
            # the lm_token groups must stay 1:1 with the phone groups.
            raise NotImplementedError(
                "include_lang_tag=False is not implemented: the leading "
                "[LANG:xx] token is part of the artifact's lm_token_ids and "
                "its group indices. Strip it when building the artifact."
            )
        model = AutoModel.from_pretrained(model_name, dtype=dtype)
        total_layers = len(model.layers)
        assert 1 <= num_layers <= total_layers, (num_layers, total_layers)

        # (1) Physically truncate, so only `num_layers` ever run -- not a
        # full forward with an intermediate state read out afterwards.
        model.layers = nn.ModuleList(list(model.layers[:num_layers]))
        # (2) Skip Qwen's own final RMSNorm (24-layer-calibrated).
        model.norm = nn.Identity()
        # (3)/(4) frozen, eval, no cache.
        model.config.use_cache = False
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if device is not None:
            model.to(device)

        self.model = model
        self.model_name = model_name
        self.num_layers = num_layers
        self.total_layers = total_layers
        self.dtype = dtype
        self.include_lang_tag = include_lang_tag
        self.hidden_size = model.config.hidden_size

        logging.info(
            f"TruncatedQwenExtractor: {model_name}, running {num_layers}/"
            f"{total_layers} layers, hidden_size={self.hidden_size}, "
            f"dtype={dtype}, device={self.device}, "
            f"include_lang_tag={include_lang_tag}, "
            f"final RMSNorm skipped, frozen."
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: torch.device) -> "TruncatedQwenExtractor":
        self.model.to(device)
        return self

    @torch.no_grad()
    def hidden_states(
        self, lm_token_ids: Sequence[Sequence[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Layer-N hidden states for a batch of `lm_token` id sequences.

        Right-padded with an attention mask. Right padding (rather than
        left) is correct here because the model is causal: real tokens sit
        at positions `0..len-1`, so the default `arange` position ids are
        right for every real token, and no real token can attend to a
        later pad position. Padded positions' outputs are garbage and are
        never read (`pooled_groups()` only gathers real indices) --
        tests/test_qwen_extractor.py checks batched output matches
        one-at-a-time output for every real position.

        Returns:
          (hidden, lens): hidden is (B, L_max, hidden_size); lens is (B,).
        """
        assert len(lm_token_ids) > 0, "empty batch"
        device = self.device
        lens = torch.tensor(
            [len(ids) for ids in lm_token_ids], dtype=torch.int64, device=device
        )
        max_len = int(lens.max())
        assert max_len > 0, "every utterance has zero lm_tokens"

        # Build the padded batch in one tensor construction (a rectangular
        # Python list -> one `torch.tensor`), then derive the attention mask
        # by comparison rather than filling rows one at a time.
        padded = [list(ids) + [0] * (max_len - len(ids)) for ids in lm_token_ids]
        input_ids = torch.tensor(padded, dtype=torch.int64, device=device)
        attention_mask = (
            torch.arange(max_len, device=device).unsqueeze(0) < lens.unsqueeze(1)
        ).to(torch.int64)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state, lens

    @torch.no_grad()
    def pooled_groups(
        self,
        lm_token_ids: Sequence[Sequence[int]],
        lm_token_groups: Sequence[Optional[Sequence[Sequence[int]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One pooled Qwen vector per group, via **last-subtoken pooling**.

        The last `lm_token` of a group has attended to that whole group plus
        everything before it (Qwen2 is causal), making it the
        most-context-complete single position to represent the group -- see
        the ADR's point 3, which also records mean pooling as a documented
        alternative to try later.

        Args:
          lm_token_ids: per utterance, the artifact's `lm_token_ids`.
          lm_token_groups: per utterance, the artifact's `lm_token_groups`,
            or `None` where sprint 000's alignment failed for that
            utterance.

        Returns:
          (pooled, valid): pooled is (B, G_max, hidden_size), zero-filled
          for padded group slots and for whole utterances whose
          `lm_token_groups` was `None`; valid is a (B, G_max) bool mask,
          True only where a real pooled group vector exists. An utterance
          with `lm_token_groups=None` yields an all-False row, which the
          caller must treat as phone-only for that utterance (sprint 003's
          step 1 fallback), *not* as "a group whose Qwen vector is zero".
        """
        assert len(lm_token_ids) == len(lm_token_groups), (
            len(lm_token_ids),
            len(lm_token_groups),
        )
        hidden, _ = self.hidden_states(lm_token_ids)
        batch_size, seq_len, hidden_size = hidden.shape

        # Ragged Python input -> one rectangular index tensor -> one gather,
        # rather than copying each pooled vector individually.
        # An empty group carries no Qwen evidence, so it stays invalid (index
        # 0 is a placeholder that the validity mask zeroes out afterwards)
        # rather than getting an invented vector.
        last_index_rows: List[List[int]] = []
        valid_rows: List[List[bool]] = []
        for groups in lm_token_groups:  # ragged input: B iterations
            groups = groups or []
            last_index_rows.append([int(g[-1]) if g else 0 for g in groups])
            valid_rows.append([bool(g) for g in groups])

        # A batch where nothing aligned at all still needs a well-formed
        # (B, 1, H) result rather than a zero-width tensor downstream.
        max_groups = max([len(r) for r in last_index_rows] + [1])
        padded_index = [r + [0] * (max_groups - len(r)) for r in last_index_rows]
        padded_valid = [r + [False] * (max_groups - len(r)) for r in valid_rows]

        index = torch.tensor(padded_index, dtype=torch.int64, device=hidden.device)
        valid = torch.tensor(padded_valid, dtype=torch.bool, device=hidden.device)
        if index.numel():
            assert int(index.max()) < seq_len, (int(index.max()), seq_len)

        pooled = torch.gather(
            hidden, 1, index.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        pooled = pooled * valid.unsqueeze(-1).to(pooled.dtype)
        return pooled, valid
