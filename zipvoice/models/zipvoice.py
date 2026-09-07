# Copyright    2024    Xiaomi Corp.        (authors:  Wei Kang
#                                                     Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from zipvoice.models.modules.fusion import PhoneQwenFusion, build_phone_group_index
from zipvoice.models.modules.solver import EulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.utils.common import (
    condition_time_mask,
    get_tokens_index,
    make_pad_mask,
    pad_labels,
    prepare_avg_tokens_durations,
)

# `text_frontend` values. "embedding" is every pre-fusion behaviour
# (master's scratch embedding and the Qwen-embedding-table variant),
# unchanged. "fusion" is the Qwen+phoneme fusion frontend -- see
# docs/adr/2026-09-02__qwen_phone_fusion_text_frontend.md.
TEXT_FRONTEND_EMBEDDING = "embedding"
TEXT_FRONTEND_FUSION = "fusion"


def _make_pretrained_embedding(
    pretrained_embed_model: str,
    vocab_size: int,
):
    """Build the `nn.Embedding` for `embed_source="pretrained"`.

    Loads `pretrained_embed_model`'s input embedding table and uses it to
    initialize a `vocab_size`-row `nn.Embedding`, fine-tuned end-to-end
    afterward (not frozen). Two cases, both real (found via an actual
    training smoke test, not assumed):

    - `vocab_size <= source_vocab_size`: the source checkpoint's embedding
      table already has enough rows. This is the common case in practice --
      e.g. Qwen2.5-0.5B's real tokenizer vocab is 151,665, but its embedding
      table has 151,936 rows (padded for hardware alignment / future vocab
      growth); `[LANG:xx]` tokens added by LanguageModelTokenizer land
      contiguously at 151665-151668, still inside that padding, giving
      `vocab_size=151669 < source_vocab_size=151936`. Just slice the first
      `vocab_size` rows -- every row used is a real pretrained one (the
      unused padding rows were never trained on by the source model either,
      so they're not meaningfully different from a random init for our new
      tokens, but they're free -- no separate random init needed).
    - `vocab_size > source_vocab_size`: our vocabulary is bigger than what
      the source model provides. Copy what exists into the first
      `source_vocab_size` rows and leave the remainder at `nn.Embedding`'s
      default random initialization.

    No projection layer is added here: ``TTSZipformer`` (used for
    ``text_encoder``) already applies ``self.in_proj = nn.Linear(in_dim,
    encoder_dim)`` as the first step of its forward pass, so constructing
    ``text_encoder`` with ``in_dim`` set to this embedding's actual
    dimension (the source model's hidden size, e.g. 896 for Qwen2.5-0.5B)
    reuses that existing projection instead of adding a redundant one.

    Returns:
      (embedding, source_dim) -- the caller uses source_dim as
      text_encoder's in_dim.
    """
    try:
        from transformers import AutoModel
    except Exception as ex:
        raise RuntimeError(f"{ex}\nPlease run\npip install transformers")

    source_model = AutoModel.from_pretrained(pretrained_embed_model)
    source_weight = source_model.get_input_embeddings().weight.detach()
    source_vocab_size, source_dim = source_weight.shape

    embed = nn.Embedding(vocab_size, source_dim)
    with torch.no_grad():
        if vocab_size <= source_vocab_size:
            embed.weight[:] = source_weight[:vocab_size]
        else:
            embed.weight[:source_vocab_size] = source_weight
    return embed, source_dim


def _concat_zero_duration_masks(
    prompt_mask: Optional[List[List[bool]]],
    mask: Optional[List[List[bool]]],
    prompt_tokens: List[List[int]],
    tokens: List[List[int]],
) -> Optional[List[List[bool]]]:
    """Concatenate per-utterance zero_duration_mask lists in the same order
    prompt_tokens/tokens themselves get concatenated for inference (prompt
    first, then the target). Missing prompt or target masks are treated as
    all-False (no control tokens) so a caller only needs to supply a mask for
    whichever side actually has one -- e.g. only the prompt is tagged with
    `[LANG:xx]` at inference (see infer_zipvoice.py), so only
    `prompt_mask` is normally given.
    """
    if prompt_mask is None and mask is None:
        return None
    if prompt_mask is None:
        prompt_mask = [[False] * len(t) for t in prompt_tokens]
    if mask is None:
        mask = [[False] * len(t) for t in tokens]
    return [p + m for p, m in zip(prompt_mask, mask)]


class ZipVoice(nn.Module):
    """The ZipVoice model."""

    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 100,
        vocab_size: int = 26,
        pad_id: int = 0,
        embed_source: str = "scratch",
        pretrained_embed_model: Optional[str] = None,
        text_frontend: str = TEXT_FRONTEND_EMBEDDING,
        qwen_hidden_size: int = 896,
        gate_init_eps: float = 0.01,
    ):
        """
        Initialize the model with specified configuration parameters.

        Args:
            fm_decoder_downsampling_factor: List of downsampling factors for each layer
                in the flow-matching decoder.
            fm_decoder_num_layers: List of the number of layers for each block in the
                flow-matching decoder.
            fm_decoder_cnn_module_kernel: List of kernel sizes for CNN modules in the
                flow-matching decoder.
            fm_decoder_feedforward_dim: Dimension of the feedforward network in the
                flow-matching decoder.
            fm_decoder_num_heads: Number of attention heads in the flow-matching
                decoder.
            fm_decoder_dim: Hidden dimension of the flow-matching decoder.
            text_encoder_num_layers: Number of layers in the text encoder.
            text_encoder_feedforward_dim: Dimension of the feedforward network in the
                text encoder.
            text_encoder_cnn_module_kernel: Kernel size for the CNN module in the
                text encoder.
            text_encoder_num_heads: Number of attention heads in the text encoder.
            text_encoder_dim: Hidden dimension of the text encoder.
            time_embed_dim: Dimension of the time embedding.
            text_embed_dim: Dimension of the text embedding.
            query_head_dim: Dimension of the query attention head.
            value_head_dim: Dimension of the value attention head.
            pos_head_dim: Dimension of the position attention head.
            pos_dim: Dimension of the positional encoding.
            feat_dim: Dimension of the acoustic features.
            vocab_size: Size of the vocabulary.
            pad_id: ID used for padding tokens.
            embed_source: "scratch" (default) trains a randomly-initialized
                `nn.Embedding(vocab_size, text_embed_dim)` jointly with the
                rest of the model. "pretrained" loads the input embedding
                table of `pretrained_embed_model` (a HuggingFace model id)
                as-is, at its native hidden size -- **`text_embed_dim` is
                then ignored**; the effective dimension is the pretrained
                model's hidden size (e.g. 896 for Qwen2.5-0.5B), and
                `text_encoder`'s existing `in_proj` layer (see
                `TTSZipformer`) handles projecting down to
                `text_encoder_dim`, so no separate projection is added here.
                Both the embedding and `text_encoder` are fine-tuned
                end-to-end. See docs/adr/2026-08-28__choose_qwen25_tokenizer.md
                and docs/plans/2026-08-28__multilingual_tts_frontend/
                eval_sets/embedding_structure_check.md for why pretrained
                loading was found to carry real relational structure worth
                starting from.
            pretrained_embed_model: HuggingFace model id to load the
                embedding table from when `embed_source="pretrained"` (e.g.
                "Qwen/Qwen2.5-0.5B", matching LanguageModelTokenizer's
                default). Required, and unused, when `embed_source="scratch"`.
            text_frontend: "embedding" (default) keeps every pre-fusion
                behaviour exactly as-is -- one embedding table looked up per
                token, `embed_source` deciding where its weights come from.
                "fusion" builds the Qwen+phoneme fusion frontend instead
                (`PhoneQwenFusion`: a per-phone phone embedding, a projected
                per-group Qwen vector, and a learned per-group gate), in
                which case `vocab_size` is the **phone** vocabulary,
                `embed_source`/`pretrained_embed_model` are unused (Qwen
                enters as pooled features computed outside the model, not as
                an embedding table), and `text_encoder` keeps its native
                `text_embed_dim` input width so the source checkpoint's
                `in_proj` stays transplantable. See
                docs/adr/2026-09-02__qwen_phone_fusion_text_frontend.md.
            qwen_hidden_size: width of the pooled Qwen vectors the fusion
                frontend consumes (896 for Qwen2.5-0.5B). Unused unless
                `text_frontend="fusion"`.
            gate_init_eps: initial Qwen mixing weight for the fusion gate
                (near-phone-only start, no floor -- see the ADR's point 4).
                Unused unless `text_frontend="fusion"`.
        """
        super().__init__()
        assert text_frontend in (
            TEXT_FRONTEND_EMBEDDING,
            TEXT_FRONTEND_FUSION,
        ), text_frontend

        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim * 3,
            out_dim=feat_dim,
            downsampling_factor=fm_decoder_downsampling_factor,
            num_encoder_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            encoder_dim=fm_decoder_dim,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
        )

        assert embed_source in ("scratch", "pretrained"), embed_source
        # `self.embed`'s construction/assignment is deliberately deferred
        # until after `self.text_encoder` below, to match master's original
        # parameter registration order (needed for optimizer-state-dict
        # compatibility, which is keyed by position, not name) and -- for the
        # scratch path specifically -- master's original RNG consumption
        # order (nn.Embedding draws its random init at construction time).
        # See docs/plans/2026-08-29__master_backward_compatibility.md.
        # The fusion frontend registers `self.fusion` at that same point, for
        # the same reason, so the "embedding" paths keep byte-identical
        # ordering.
        pretrained_embed = None
        if text_frontend == TEXT_FRONTEND_FUSION:
            assert embed_source == "scratch", (
                "text_frontend='fusion' owns its own phone embedding; the "
                "Qwen side enters as pooled features computed outside the "
                f"model, so embed_source must stay 'scratch' (got {embed_source!r})"
            )
        elif embed_source == "pretrained":
            assert pretrained_embed_model is not None, (
                "pretrained_embed_model is required when embed_source='pretrained'"
            )
            # text_embed_dim can't be known without loading the source model
            # (to read its hidden size), so this must happen before
            # text_encoder is sized -- unlike the scratch path below, master
            # has no pretrained path to preserve RNG-order compatibility
            # with here.
            pretrained_embed, text_embed_dim = _make_pretrained_embedding(
                pretrained_embed_model=pretrained_embed_model,
                vocab_size=vocab_size,
            )

        self.text_encoder = TTSZipformer(
            in_dim=text_embed_dim,
            out_dim=feat_dim,
            downsampling_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            encoder_dim=text_encoder_dim,
            feedforward_dim=text_encoder_feedforward_dim,
            num_heads=text_encoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=False,
        )

        if text_frontend == TEXT_FRONTEND_FUSION:
            # The phone embedding lives inside the fusion module, so no
            # separate `self.embed` is built at all -- an unused table would
            # be dead parameters in every checkpoint.
            self.embed = None
            self.fusion = PhoneQwenFusion(
                phone_vocab_size=vocab_size,
                embed_dim=text_embed_dim,
                qwen_hidden_size=qwen_hidden_size,
                gate_init_eps=gate_init_eps,
            )
        else:
            self.embed = (
                pretrained_embed
                if pretrained_embed is not None
                else nn.Embedding(vocab_size, text_embed_dim)
            )
            self.fusion = None

        self.text_frontend = text_frontend
        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.pad_id = pad_id

        self.solver = EulerSolver(self, func_name="forward_fm_decoder")

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity.
        Args:
            t:  A tensor of shape (N, 1, 1) or a tensor of a float,
                in the range of (0, 1).
            xt: the input of the current timestep, including condition
                embeddings and noisy acoustic features.
            text_condition: the text condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            speech_condition: the speech condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding, True means masked
                position, with the shape (N, T).
            guidance_scale: The guidance scale in classifier-free guidance,
                which is a tensor of shape (N, 1, 1) or a tensor of a float.

        Returns:
            predicted velocity, with the shape (batch, seq_len, emb_dim).
        """

        xt = torch.cat([xt, text_condition, speech_condition], dim=2)

        assert t.dim() in (0, 3)
        # Handle t with the shape (N, 1, 1):
        # squeeze the last dimension if it's size is 1.
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        # Handle t with a single value: expand to the size of batch size.
        if t.dim() == 0:
            t = t.repeat(xt.shape[0])

        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(xt.shape[0])

            vt = self.fm_decoder(
                x=xt, t=t, padding_mask=padding_mask, guidance_scale=guidance_scale
            )
        else:
            vt = self.fm_decoder(x=xt, t=t, padding_mask=padding_mask)
        return vt

    def forward_text_embed(
        self,
        tokens: List[List[int]],
        phone_groups: Optional[List[Optional[List[List[int]]]]] = None,
        qwen_group_features: Optional[torch.Tensor] = None,
        qwen_group_valid: Optional[torch.Tensor] = None,
    ):
        """
        Get the text embeddings.
        Args:
            tokens: a list of list of token ids. Under
                `text_frontend="fusion"` these are **phone** ids.
            phone_groups: fusion only -- per utterance,
                `FusionTokenizerArtifact.phone_groups` (or None for an
                utterance whose lm_token alignment failed, which falls back
                to phone-only).
            qwen_group_features: fusion only -- (B, G, qwen_hidden_size)
                pooled Qwen vectors from `TruncatedQwenExtractor`. None runs
                the phone branch alone, which is what an oracle/ablation
                "phone-only" configuration wants.
            qwen_group_valid: fusion only -- (B, G) bool marking real groups.
        Returns:
            embed: the text embeddings, shape (batch, seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )
        # Note `pad_labels` appends one position beyond every utterance's real
        # length; `get_tokens_index()` can return an index equal to a token
        # count (it appends a residual-duration entry), so a vector must exist
        # there to gather from. Both frontends below preserve that: the
        # embedding path embeds the extra pad id, and the fusion path leaves
        # that position (which belongs to no group) as a plain pad embedding.
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B, S)
        if self.text_frontend == TEXT_FRONTEND_FUSION:
            assert phone_groups is not None, (
                "text_frontend='fusion' needs phone_groups (from "
                "FusionTokenizerArtifact) to know which phones share a group"
            )
            group_ids, has_group = build_phone_group_index(
                phone_groups, padded_len=tokens_padded.shape[1], device=device
            )
            embed = self.fusion(
                tokens_padded,
                group_ids,
                has_group,
                qwen_group_features,
                qwen_group_valid,
            )  # (B, S, C)
        else:
            embed = self.embed(tokens_padded)  # (B, S, C)
        tokens_lens = torch.tensor(
            [len(token) for token in tokens], dtype=torch.int64, device=device
        )
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B, S)

        embed = self.text_encoder(
            x=embed, t=None, padding_mask=tokens_padding_mask
        )  # (B, S, C)
        return embed, tokens_lens

    def forward_text_condition(
        self,
        embed: torch.Tensor,
        tokens_lens: torch.Tensor,
        features_lens: torch.Tensor,
        zero_duration_mask: Optional[List[List[bool]]] = None,
    ):
        """
        Get the text condition with the same length of the acoustic feature.
        Args:
            embed: the text embeddings, shape (batch, token_seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
            features_lens: the length of each acoustic feature sequence,
                shape (batch,).
        Returns:
            text_condition: the text condition, shape
                (batch, feature_seq_len, emb_dim).
            padding_mask: the padding mask of text condition, shape
                (batch, feature_seq_len).
        """

        num_frames = int(features_lens.max())

        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B, T)

        tokens_durations = prepare_avg_tokens_durations(
            features_lens, tokens_lens, zero_duration_mask=zero_duration_mask
        )

        tokens_index = get_tokens_index(tokens_durations, num_frames).to(
            embed.device
        )  # (B, T)

        text_condition = torch.gather(
            embed,
            dim=1,
            index=tokens_index.unsqueeze(-1).expand(
                embed.size(0), num_frames, embed.size(-1)
            ),
        )  # (B, T, F)
        return text_condition, padding_mask

    def _require_inference_fusion_fields(
        self,
        phone_groups,
        qwen_group_features,
    ) -> None:
        """Under `text_frontend="fusion"`, inference must be given the fusion
        fields for the **already-composed** prompt+target sequence.

        Sprint 003 left this path raising NotImplementedError because the
        prompt/target composition contract was undecided. It is decided now
        (`fusion_tokenizer.compose_artifacts`): only the prompt carries the
        `[LANG:xx]` tag, matching what the pre-fusion multilingual inference
        path already did, and matching what training saw -- one utterance,
        one leading tag.

        What remains worth guarding is the same thing the training path
        guards: a fusion-configured model handed no fusion fields would
        silently fall back to the phone embedding and generate audio from a
        *different architecture* than the checkpoint was trained as, with
        nothing to indicate it. That is how a whole training run went ~93%
        phone-only unnoticed. Missing fields are a caller error, not a
        fallback.
        """
        if self.text_frontend != TEXT_FRONTEND_FUSION:
            return
        if phone_groups is None or qwen_group_features is None:
            raise ValueError(
                "text_frontend='fusion' requires phone_groups and "
                "qwen_group_features at inference, for the composed "
                "prompt+target sequence (see "
                "zipvoice.tokenizer.fusion_tokenizer.compose_artifacts and "
                "TruncatedQwenExtractor.pooled_groups). Without them the "
                "model would silently generate as a phone-only model."
            )

    def forward_text_train(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
        zero_duration_mask: Optional[List[List[bool]]] = None,
        phone_groups: Optional[List[Optional[List[List[int]]]]] = None,
        qwen_group_features: Optional[torch.Tensor] = None,
        qwen_group_valid: Optional[torch.Tensor] = None,
    ):
        """
        Process text for training, given text tokens and real feature lengths.
        """
        embed, tokens_lens = self.forward_text_embed(
            tokens,
            phone_groups=phone_groups,
            qwen_group_features=qwen_group_features,
            qwen_group_valid=qwen_group_valid,
        )
        text_condition, padding_mask = self.forward_text_condition(
            embed, tokens_lens, features_lens, zero_duration_mask=zero_duration_mask
        )
        return (
            text_condition,
            padding_mask,
        )

    def forward_text_inference_gt_duration(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
        zero_duration_mask: Optional[List[List[bool]]] = None,
        prompt_zero_duration_mask: Optional[List[List[bool]]] = None,
        phone_groups: Optional[List[Optional[List[List[int]]]]] = None,
        qwen_group_features: Optional[torch.Tensor] = None,
        qwen_group_valid: Optional[torch.Tensor] = None,
    ):
        """
        Process text for inference, given text tokens, real feature lengths and prompts.

        `phone_groups`/`qwen_group_features`/`qwen_group_valid` describe the
        **concatenated** prompt+target sequence, matching `cat_tokens` below.
        """
        self._require_inference_fusion_fields(phone_groups, qwen_group_features)
        cat_zero_duration_mask = _concat_zero_duration_masks(
            prompt_zero_duration_mask, zero_duration_mask, prompt_tokens, tokens
        )
        tokens = [
            prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
        ]
        features_lens = prompt_features_lens + features_lens
        embed, tokens_lens = self.forward_text_embed(
            tokens,
            phone_groups=phone_groups,
            qwen_group_features=qwen_group_features,
            qwen_group_valid=qwen_group_valid,
        )
        text_condition, padding_mask = self.forward_text_condition(
            embed, tokens_lens, features_lens, zero_duration_mask=cat_zero_duration_mask
        )
        return text_condition, padding_mask

    def forward_text_inference_ratio_duration(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
        speed: float,
        zero_duration_mask: Optional[List[List[bool]]] = None,
        prompt_zero_duration_mask: Optional[List[List[bool]]] = None,
        phone_groups: Optional[List[Optional[List[List[int]]]]] = None,
        qwen_group_features: Optional[torch.Tensor] = None,
        qwen_group_valid: Optional[torch.Tensor] = None,
    ):
        """
        Process text for inference, given text tokens and prompts,
        feature lengths are predicted with the ratio of token numbers.

        `phone_groups`/`qwen_group_features`/`qwen_group_valid` describe the
        **concatenated** prompt+target sequence, matching `cat_tokens` below.
        """
        self._require_inference_fusion_fields(phone_groups, qwen_group_features)
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )

        cat_zero_duration_mask = _concat_zero_duration_masks(
            prompt_zero_duration_mask, zero_duration_mask, prompt_tokens, tokens
        )
        cat_tokens = [
            prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
        ]

        prompt_tokens_lens = torch.tensor(
            [len(token) for token in prompt_tokens],
            dtype=torch.int64,
            device=device,
        )

        tokens_lens = torch.tensor(
            [len(token) for token in tokens],
            dtype=torch.int64,
            device=device,
        )

        cat_embed, cat_tokens_lens = self.forward_text_embed(
            cat_tokens,
            phone_groups=phone_groups,
            qwen_group_features=qwen_group_features,
            qwen_group_valid=qwen_group_valid,
        )

        features_lens = prompt_features_lens + torch.ceil(
            (prompt_features_lens / prompt_tokens_lens * tokens_lens / speed)
        ).to(dtype=torch.int64)

        text_condition, padding_mask = self.forward_text_condition(
            cat_embed, cat_tokens_lens, features_lens, zero_duration_mask=cat_zero_duration_mask
        )
        return text_condition, padding_mask

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition_drop_ratio: float = 0.0,
        zero_duration_mask: Optional[List[List[bool]]] = None,
        phone_groups: Optional[List[Optional[List[List[int]]]]] = None,
        qwen_group_features: Optional[torch.Tensor] = None,
        qwen_group_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model for training.
        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: the length of each acoustic feature sequence, shape (batch,).
            noise: the intitial noise, with the shape (batch, seq_len, feat_dim).
            t: the time step, with the shape (batch, 1, 1).
            condition_drop_ratio: the ratio of dropped text condition.
            zero_duration_mask: per-utterance, per-token booleans marking
                control tokens (e.g. LanguageModelTokenizer's [LANG:xx] tags)
                that must receive zero acoustic duration. None (default)
                preserves the original behaviour of giving every token an
                equal share of the utterance's duration. Note this is an
                `lm_tokens`-side concept: under `text_frontend="fusion"`,
                `tokens` are phones, which never contain a control token, so
                this stays None there (see the ADR's "Artifact invariants").
            phone_groups: fusion only -- see `forward_text_embed`.
            qwen_group_features: fusion only -- see `forward_text_embed`.
            qwen_group_valid: fusion only -- see `forward_text_embed`.
        Returns:
            fm_loss: the flow-matching loss.
        """

        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
            zero_duration_mask=zero_duration_mask,
            phone_groups=phone_groups,
            qwen_group_features=qwen_group_features,
            qwen_group_valid=qwen_group_valid,
        )

        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        if condition_drop_ratio > 0.0:
            drop_mask = (
                torch.rand(text_condition.size(0), 1, 1).to(text_condition.device)
                > condition_drop_ratio
            )
            text_condition = text_condition * drop_mask

        xt = features * t + noise * (1 - t)
        ut = features - noise  # (B, T, F)

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        return fm_loss

    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features: torch.Tensor,
        prompt_features_lens: torch.Tensor,
        features_lens: Optional[torch.Tensor] = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
        zero_duration_mask: Optional[List[List[bool]]] = None,
        prompt_zero_duration_mask: Optional[List[List[bool]]] = None,
        phone_groups: Optional[List[Optional[List[List[int]]]]] = None,
        qwen_group_features: Optional[torch.Tensor] = None,
        qwen_group_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate acoustic features, given text tokens, prompts feature
            and prompt transcription's text tokens.
        Args:
            tokens: a list of list of text tokens.
            prompt_tokens: a list of list of prompt tokens.
            prompt_features: the prompt feature with the shape
                (batch_size, seq_len, feat_dim).
            prompt_features_lens: the length of each prompt feature,
                with the shape (batch_size,).
            features_lens: the length of the predicted eature, with the
                shape (batch_size,). It is used only when duration is "real".
            duration: "real" or "predict". If "real", the predicted
                feature length is given by features_lens.
            num_step: the number of steps to use in the ODE solver.
            guidance_scale: the guidance scale for classifier-free guidance.
            zero_duration_mask: per-utterance, per-token booleans marking
                control tokens (e.g. [LANG:xx] tags) in `tokens` that must
                receive zero acoustic duration, matching training. None
                (default) means `tokens` has no control tokens.
            prompt_zero_duration_mask: same, for `prompt_tokens`.
            phone_groups / qwen_group_features / qwen_group_valid: required
                under text_frontend="fusion", and describing the
                **concatenated** prompt+target sequence -- build them with
                `fusion_tokenizer.compose_artifacts` and
                `TruncatedQwenExtractor.pooled_groups`, not from the target
                alone.
        """

        assert duration in ["real", "predict"]

        if duration == "predict":
            (
                text_condition,
                padding_mask,
            ) = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
                zero_duration_mask=zero_duration_mask,
                prompt_zero_duration_mask=prompt_zero_duration_mask,
                phone_groups=phone_groups,
                qwen_group_features=qwen_group_features,
                qwen_group_valid=qwen_group_valid,
            )
        else:
            assert features_lens is not None
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                zero_duration_mask=zero_duration_mask,
                prompt_zero_duration_mask=prompt_zero_duration_mask,
                phone_groups=phone_groups,
                qwen_group_features=qwen_group_features,
                qwen_group_valid=qwen_group_valid,
            )
        batch_size, num_frames, _ = text_condition.shape

        speech_condition = torch.nn.functional.pad(
            prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
        )  # (B, T, F)

        # False means speech condition positions.
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        x0 = torch.randn(
            batch_size,
            num_frames,
            prompt_features.size(-1),
            device=text_condition.device,
        )

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
        x1_wo_prompt_lens = (~padding_mask).sum(-1) - prompt_features_lens
        x1_prompt = torch.zeros(
            x1.size(0), prompt_features_lens.max(), x1.size(2), device=x1.device
        )
        x1_wo_prompt = torch.zeros(
            x1.size(0), x1_wo_prompt_lens.max(), x1.size(2), device=x1.device
        )
        for i in range(x1.size(0)):
            x1_wo_prompt[i, : x1_wo_prompt_lens[i], :] = x1[
                i,
                prompt_features_lens[i] : prompt_features_lens[i]
                + x1_wo_prompt_lens[i],
            ]
            x1_prompt[i, : prompt_features_lens[i], :] = x1[
                i, : prompt_features_lens[i]
            ]

        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens

    def sample_intermediate(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        speech_condition_mask: torch.Tensor,
        t_start: float,
        t_end: float,
        num_step: int = 1,
        guidance_scale: torch.Tensor = None,
        phone_groups: Optional[List[Optional[List[List[int]]]]] = None,
        qwen_group_features: Optional[torch.Tensor] = None,
        qwen_group_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate acoustic features in intermediate timesteps.
        Args:
            tokens: List of list of token ids.
            features: The acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: The length of each acoustic feature sequence,
                with the shape (batch,).
            noise: The initial noise, with the shape (batch, seq_len, feat_dim).
            speech_condition_mask: The mask for speech condition, True means
                non-condition positions, with the shape (batch, seq_len).
            t_start: The start timestep.
            t_end: The end timestep.
            num_step: The number of steps for sampling.
            guidance_scale: The scale for classifier-free guidance inference,
                with the shape (batch, 1, 1).
            phone_groups: fusion only -- see `forward_text_embed`.
            qwen_group_features: fusion only -- see `forward_text_embed`.
            qwen_group_valid: fusion only -- see `forward_text_embed`.
        """
        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
            phone_groups=phone_groups,
            qwen_group_features=qwen_group_features,
            qwen_group_valid=qwen_group_valid,
        )

        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        x_t_end = self.solver.sample(
            x=noise,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_start=t_start,
            t_end=t_end,
        )
        x_t_end_lens = (~padding_mask).sum(-1)
        return x_t_end, x_t_end_lens
