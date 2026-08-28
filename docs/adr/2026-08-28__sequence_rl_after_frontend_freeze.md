# Sequence FlowTTS-GRPO RL training after the text front-end is frozen, start with LoRA/decoder-only

- **Date:** 2026-08-28
- **Status:** Accepted
- **Related:** [../proposals/2026-08-28__flowtts_grpo_rl_finetuning.md](../proposals/2026-08-28__flowtts_grpo_rl_finetuning.md)

## Context

The FlowTTS-GRPO proposal originally argued that RL post-training is orthogonal
to the text front-end work (tokenizer/embedding, language-hint tokens) because
RL fine-tunes an already-trained checkpoint regardless of which front end
produced it, and that ZipVoice's flow-matching decoder is small enough
(118.5M params) to skip LoRA and fine-tune all parameters directly during RL,
unlike the reference paper which used LoRA specifically to make RL affordable on
much larger base models (F5-TTS-Base ~330M+, CosyVoice 3.0's 0.5B/1.5B LM).

An external review (Codex, gpt-5.6-sol, high reasoning effort, run 2026-08-28)
pushed back on both points:

1. "Orthogonal" is true mathematically (RL can run on any checkpoint) but
   misleading operationally — RL fine-tunes a *specific* checkpoint, and
   changing the tokenizer or adding language-conditioning tokens afterward
   (different vocabulary, different embedding space, different duration
   behavior) invalidates that RL investment.
2. RL rollout/training cost is dominated by group sampling, vocoding, and
   reward-model inference across a denoising window, not by base-model
   parameter count — "the model is small" doesn't actually justify
   full-parameter updates being safe or necessary. Full-parameter RL updates
   also risk damaging pronunciation/multilingual generalization established by
   supervised training, a real concern once the front-end proposals land.

The author accepted both corrections.

## Decision

We will run FlowTTS-GRPO RL training **only after** the text front-end
(tokenizer/vocabulary and language-hint tokens, per the two sibling ADRs) is
frozen on a stable supervised checkpoint — not in parallel with front-end
development. Within the RL training itself, we will start with LoRA or
decoder-only adaptation (freezing the embedding and `text_encoder`, tuning only
`fm_decoder`) and only compare against full-parameter fine-tuning as a follow-up
ablation once the constrained version is shown to be stable.

What *does* run in parallel with front-end development, because it doesn't
depend on which checkpoint it eventually targets: the SDE-rollout correctness
gates (see the linked proposal), and building/validating batched reward-function
wrappers around the existing evaluation models.

## Alternatives Considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Run RL in parallel with front-end work (original plan) | Maximizes parallelism, no idle time waiting on the front-end track | Any front-end change invalidates the RL-tuned checkpoint, wasting the compute already spent; encourages tuning against a moving target | Rejected per Codex's correctness argument — the two tracks share a checkpoint dependency even though the RL *method* itself is front-end-agnostic |
| Full-parameter fine-tuning from the start (original plan) | Simpler to implement (no LoRA/adapter plumbing); no capacity ceiling from a low-rank adapter | Cost is dominated by rollout/reward-inference overhead regardless of trainable-parameter count, so the size argument for skipping LoRA doesn't hold; higher risk of catastrophic forgetting of pronunciation/multilingual behavior | Rejected per Codex's cost-driver and forgetting-risk argument |
| LoRA/decoder-only first, full fine-tune as follow-up ablation (chosen) | Establishes training stability on a smaller, safer parameter surface first; full fine-tuning remains available as a later, explicitly-compared option rather than a default | Slightly more implementation surface (adapter plumbing) before the first result | Matches the reference paper's own justification for LoRA and Codex's forgetting-risk concern, while keeping the door open to full fine-tuning if the constrained version proves insufficient |

## Consequences

- Easier: the RL track has a stable, unmoving target checkpoint to tune once it
  starts, avoiding repeated invalidation as the front-end proposals evolve.
- Easier: LoRA/decoder-only tuning bounds the blast radius of a bad RL update on
  pronunciation/multilingual quality established by supervised training.
- Harder: the RL track's final training run is now gated on the front-end plan's
  completion — see
  [../plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/)
  and [../plans/2026-08-28__flowtts_grpo_rl/](../plans/2026-08-28__flowtts_grpo_rl/)
  for the explicit dependency between the two plans.
- Neutral: the correctness-gate and reward-wrapper infrastructure work is
  unaffected by this sequencing decision and proceeds independently.
