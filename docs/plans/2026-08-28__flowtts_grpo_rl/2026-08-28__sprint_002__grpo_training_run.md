# Sprint 002 — GRPO training run

- **Date:** 2026-08-28
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Draft

## Goal

Implement the GRPO loss and run LoRA/decoder-only RL training against the
frozen front-end checkpoint, with reward-hacking monitoring and mandatory human
evaluation, before ever attempting a full-parameter fine-tune comparison.

## Background

**Blocking dependency**: this sprint must not start until
[../2026-08-28__multilingual_tts_frontend/2026-08-28__sprint_004__frozen_frontend_training_run.md](../2026-08-28__multilingual_tts_frontend/2026-08-28__sprint_004__frozen_frontend_training_run.md)
is Done, per
[../../adr/2026-08-28__sequence_rl_after_frontend_freeze.md](../../adr/2026-08-28__sequence_rl_after_frontend_freeze.md).
Depends on sprint 001's correctness-gated SDE rollout and reward wrappers.

## Scope

- **In scope:** the GRPO loss (group sampling, group-relative advantage
  normalization, clipped surrogate objective, KL penalty against a frozen
  reference copy, plus a supervised flow-matching replay/anchor loss as a second
  guard against reward-driven regression); LoRA or decoder-only adaptation
  (freezing the embedding and `text_encoder`, tuning only `fm_decoder`) as the
  starting configuration; reward-weight (`λ1, λ2, λ3`) and noise-level (`a`)
  sweeps, since the paper's values are dataset/model-specific and won't transfer
  as-is; hard-case curriculum integration from sprint 001; checkpoint selection
  via validation reward **and** human A/B evaluation; full-parameter fine-tuning
  as a follow-up ablation, attempted only after the constrained version is
  stable.
- **Out of scope:** any change to the text front end.

## Approach

1. Confirm the blocking dependency is actually Done before starting any of this
   sprint's work.
2. Implement the GRPO loss on top of sprint 001's correctness-gated SDE rollout.
3. Start training with LoRA/decoder-only adaptation; monitor for reward hacking
   per the risk list below throughout, not just at the end.
4. Sweep reward weights and noise level on a small scale (2-GPU, following the
   reference paper's own ablation methodology) before committing to a full run.
5. Select checkpoints using validation reward trends on a dev-easy/dev-hard-style
   split **and** periodic human A/B evaluation — not reward trends alone.
6. Only after the LoRA/decoder-only run is shown stable, run a full-parameter
   fine-tuning comparison as an explicit ablation, not a default.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Reward hacking: ASR-favorable-not-human-favorable acoustics, prompt-copying for speaker similarity, MOS-predictor exploitation, std-normalization amplifying near-zero-variance rewards | Reward increases while real quality doesn't (or degrades) | Supervised replay/anchor loss alongside KL; mandatory human A/B eval before any rollout decision |
| Circular evaluation (selecting checkpoints by the same reward-model family used to train them) | Best-scoring checkpoint isn't actually the best-sounding one | Human evaluation is a required gate, not optional, before checkpoint selection is final |
| Cross-lingual generalization to Vietnamese is unverified for ZipVoice (the paper's own cross-lingual generalization result was for CV3, trained only on EN+ZH) | RL gains may not transfer to VI without VI-specific reward signal | Use the user's existing Vietnamese reward models in the sweep/evaluation, don't assume EN+ZH training generalizes |
| Starting before the front-end checkpoint is truly frozen | Wastes RL compute on an invalidated checkpoint | Hard gate on the blocking dependency above |

## Testing & Validation

The dev-easy/dev-hard-style validation protocol from the reference paper;
reward-weight and noise-level sweep results; mandatory human A/B preference
testing before any checkpoint is proposed for rollout; explicit comparison
between LoRA/decoder-only and full-parameter fine-tuning results.

## Rollout

The resulting RL-tuned checkpoint is offered as an **opt-in alternative**
inference option pending human evaluation sign-off — not an automatic
replacement of the supervised checkpoint from the front-end plan.
