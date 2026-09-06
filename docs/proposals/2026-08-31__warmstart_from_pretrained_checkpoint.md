# Warm-start the multilingual model from a converged pretrained checkpoint instead of training from scratch

- **Status**: Accepted — implemented and smoke-tested; full training run and quality evaluation still pending
- **Author**: LongNH (with Claude Code assistance)
- **Date**: 2026-08-31
- **Related**: [2026-08-28__mbert_multilingual_tokenizer.md](2026-08-28__mbert_multilingual_tokenizer.md)
  (the pretrained-embedding + multilingual-tokenizer architecture this warm-start
  targets), [2026-08-28__primary_language_conditioning.md](2026-08-28__primary_language_conditioning.md).

## Summary

Our from-scratch training run of the multilingual (Qwen2.5-0.5B embedding +
`MultilingualTokenizer`) ZipVoice architecture produces near-unintelligible,
mostly-silent audio at ~64,000-96,000 training batches, while a public
Vietnamese fine-tune checkpoint using the original master-branch architecture
(scratch embedding, espeak-based phoneme tokenizer, 525,000 training updates)
sounds normal on the same reference voice and text. Instead of continuing to
train our architecture from scratch — which the ZipVoice paper itself trains
for **1,000,000 updates on 100,000 hours** of data, a scale we are nowhere
near — this proposes warm-starting our model from that converged checkpoint's
weights: 887 of 889 parameter tensors are shape-identical between the two
architectures (everything except the embedding table and one projection
layer, both of which differ only because of the embedding-dimension change),
so the entire `fm_decoder` and nearly all of `text_encoder` can be
transplanted directly, leaving only the new embedding + projection to learn.

## Motivation

### Diagnosis that led here

This session investigated why our trained checkpoint's output "pronounces
nothing" despite healthy-looking training curves (loss converging smoothly,
normal LR schedule):

1. **CFG training was never disabled.** `--condition-drop-ratio` is `0.2` in
   both `master` and our training script, unchanged — and this exactly
   matches the ZipVoice paper's own stated setup ("the text condition is
   dropped with a probability of 20% for CFG"), confirmed via web search of
   the paper (arXiv [2506.13053](https://arxiv.org/abs/2506.13053)).
2. **Inference-time CFG extrapolation was ruled out empirically.** Codex
   traced `zipvoice/models/modules/solver.py`'s actual guidance formula:
   `guidance_scale=1.0` (the default) means `(1+scale)*conditional -
   scale*unconditional`, doubled again for `t<=0.5` — i.e. up to
   `3*conditional - 2*unconditional` for most sampling steps, not
   "conditional-only" as originally assumed. Testing `--guidance-scale 0.0`
   (true conditional-only) on our checkpoint did **not** fix the poor
   quality — ruling this out as the dominant cause.
3. **Raw vs. averaged (`model_avg`) checkpoint was ruled out empirically.**
   The public baseline checkpoint's filename indicated an averaged
   checkpoint; ours was loading the raw `model` weights. Building a derived
   checkpoint from our own `model_avg` and testing it produced audio that
   was **quieter and more silent** than the raw model (RMS 0.0066 vs. 0.011,
   90.8% vs. 86.8% silence) — the opposite of a fix.
4. **A real but modest train/inference inconsistency was found and
   confirmed** (fixed 2026-08-31): `zero_duration_mask` (which correctly
   excludes `[LANG:xx]` control tokens from acoustic duration during
   training) was never threaded through the inference-time duration
   functions
   (`forward_text_inference_gt_duration`/`forward_text_inference_ratio_duration`
   in `zipvoice/models/zipvoice.py`), and `--lang` (see
   [2026-08-28__primary_language_conditioning.md](2026-08-28__primary_language_conditioning.md))
   tagged both the prompt and the target text, producing two control tokens
   in the concatenated sequence at inference vs. one per utterance in
   training. Codex's quantitative trace put this at roughly 2% of frames
   misallocated for a representative sentence — real, but not commensurate
   with 80%+ silence on its own.
5. **The decisive finding**: the ZipVoice paper trains its base model for
   **1,000,000 updates on the 100,000-hour Emilia dataset**. Our run was at
   ~64,000-96,000 updates on a **1,380-hour** corpus — roughly 6-10% of the
   paper's total updates, on a corpus ~70x smaller. Even a full planned
   15-epoch run of our corpus would only reach ~355,000 total updates, about
   a third of the paper's figure. This strongly reframes items 1-4 as real
   but secondary issues, with **undertraining as the dominant explanation** —
   consistent with both negative experiments above (a genuinely broken
   conditioning pathway would likely respond differently to CFG/model_avg
   changes; a simply-immature model looks bad under all of them).

### Why warm-start instead of just training longer

Reaching anywhere near 1M updates from scratch on our current corpus/hardware
is not a practical near-term path. Directly comparing our from-scratch
checkpoint's `state_dict` against the public converged checkpoint's revealed
that **887 of 889 tensors have identical shapes** — the two architectures
differ only in `embed.weight` (ours: `[151669, 896]` Qwen2.5-0.5B-dimensioned
vs. the source's: `[360, 192]` scratch-dimensioned) and
`text_encoder.in_proj.weight` (`896->192` vs. `192->192`, same root cause).
Everything else — the entire `fm_decoder` (the bulk of the model's capacity,
converged over the source checkpoint's 525,000 updates) and the rest of
`text_encoder` — is architecturally identical and can be copied directly.

## Proposed Design

1. **`scripts/all/m04_build_warmstart_checkpoint.py`** (implemented, general
   purpose — takes any source repo/checkpoint as CLI arguments, not tied to
   one specific pretrained model): downloads the source checkpoint,
   constructs our architecture (Qwen2.5-0.5B embedding +
   `MultilingualTokenizer` vocab), copies all shape-matching tensors from the
   source state dict unchanged, leaves `embed.weight` (already initialized
   from Qwen's real pretrained embedding values, per
   `_make_pretrained_embedding`) and `text_encoder.in_proj.weight` at their
   freshly-constructed values, sanity-checks the result loads with
   `strict=True` into a fresh instance of our architecture, and saves it to
   `exp/all_warmstart/warmstart_from_pretrained.pt` in the plain `{"model":
   state_dict}` format `--checkpoint` expects. Run and verified against a
   real public checkpoint: prints "Copied 887 tensors... Kept our own... for
   2 tensors" and passes the strict-load sanity check.
2. **`scripts/all/m05_train_warmstart.sh`** (implemented): fine-tunes from
   that checkpoint via `train_zipvoice.py --checkpoint
   exp/all_warmstart/warmstart_from_pretrained.pt` (a fresh-optimizer
   pretrained-weights load — not `--resume-from-checkpoint`, which would try
   to resume the source checkpoint's optimizer state, itself
   shape-incompatible with our embedding/projection anyway) into a separate
   `exp/all_warmstart/` directory, so it doesn't touch the existing
   from-scratch run in `exp/all/`. Otherwise mirrors
   `scripts/all/m01_train.sh`'s validated settings (same manifests,
   `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`/`PYTORCH_ALLOC_CONF` env vars, LR
   schedule). Smoke-tested end-to-end on real GPU: loaded the warm-start
   checkpoint (258,626,276 total parameters, matching expectations) and
   completed 5 real training steps without error.
3. **`scripts/all/m02_infer_cpu.py`** (implemented, general purpose): CPU-only
   inference, either against our own local checkpoints (default) or, when
   `--hf-repo` is given, against any master-architecture HuggingFace
   checkpoint (`--checkpoint-name`/`--config-file`/`--tokens-file`). The
   HuggingFace mode (originally a separate `m03_infer_cpu_pretrained.py`,
   merged into this script) was used during this session's diagnosis to
   establish the working baseline this proposal warm-starts from.

The bet: since `fm_decoder` and most of `text_encoder` already know how to
turn a text-conditioned latent into good Vietnamese speech, only the new
embedding + one projection layer need to learn to produce a compatible
conditioning signal — a much smaller adaptation problem than learning the
whole pipeline from random initialization, and plausibly reachable in far
fewer than 1M updates.

## Alternatives Considered

- **Continue the from-scratch run much longer.** Not rejected outright, but
  deprioritized: reaching a training scale comparable to the paper's 1M
  updates is impractical on our current corpus/timeline/hardware, and every
  cheap inference-time fix attempted (CFG scale, checkpoint averaging) failed
  to close the gap, consistent with needing an order of magnitude more
  training rather than a settings change.
- **Fix the known `zero_duration_mask`/duplicate-tag inference bugs first,
  then re-evaluate the from-scratch checkpoint.** Still worth doing
  eventually (see Open Questions), but Codex's own quantitative estimate
  (~2% of frames) makes it unlikely to close an 80%+ silence gap by itself,
  so it doesn't change the priority of addressing the training-scale problem
  first.

## Open Questions

- What learning rate and epoch count are actually appropriate for this
  fine-tuning scenario? `m05_train_warmstart.sh` reuses the from-scratch run's
  `--base-lr 0.0001` unchanged; `--num-epochs` was since raised from 15 to 50
  (intentional, not a bug -- a mostly-pretrained network converging faster
  per-epoch on acoustic quality doesn't mean it needs fewer total epochs if
  content-correctness turns out to be the slow part), with no evidence yet on
  whether the LR itself is right for a mostly-pretrained network.
- Will `fm_decoder`'s statistics, learned from a phoneme-based
  `text_encoder`'s output distribution, transfer well to a BPE/Qwen-embedding-
  based `text_encoder`'s output once the latter adapts? This is the core
  unproven bet of this proposal — plausible given `fm_decoder`'s role is
  largely architecture-agnostic (turn a temporally-aligned conditioning
  signal + noise into mel), but not guaranteed.
- ~~Should the `zero_duration_mask`-at-inference and duplicate-`[LANG:xx]`-tag
  issues (motivation items 4) be fixed before evaluating warm-start quality~~
  — **resolved 2026-08-31**: both fixed (see motivation item 4 above), so a
  still-poor warm-start result can no longer be attributed to these known
  bugs.
- No formal success criterion is defined yet for "the warm-start worked" —
  this proposal doesn't include a plan/ADR per the user's request; a
  follow-up plan should define concrete checkpoints-to-check and an
  evaluation method (subjective listening, or an ASR/WER-based check) once
  the fine-tuning run is underway.

## Next steps

Run `scripts/all/m05_train_warmstart.sh` (after building the warm-start
checkpoint with `scripts/all/m04_build_warmstart_checkpoint.py`, pointing it
at a suitable converged master-architecture checkpoint) and periodically
check intelligibility (e.g. every 4,000-8,000 batches) rather than waiting
for a fixed epoch count, given the training-scale requirement here is
fundamentally different from (and likely much smaller than) the from-scratch
run's. No plan or ADR is being written for this yet, per the user's request —
revisit if/when this approach is validated and needs a fuller implementation
plan.
