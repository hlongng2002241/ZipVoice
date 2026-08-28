# Sprint 004 — Frozen front-end training run

- **Date:** 2026-08-28
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Draft

## Goal

Train ZipVoice end-to-end with the winning tokenizer/embedding/`[LANG:xx]`
configuration from sprint 003 frozen, validate against the full eval suite, and
produce the versioned checkpoint that the FlowTTS-GRPO RL plan depends on.

## Background

Per [../../adr/2026-08-28__sequence_rl_after_frontend_freeze.md](../../adr/2026-08-28__sequence_rl_after_frontend_freeze.md),
the checkpoint produced here is a hard dependency for
[../2026-08-28__flowtts_grpo_rl/2026-08-28__overview.md](../2026-08-28__flowtts_grpo_rl/2026-08-28__overview.md) —
no further front-end changes should land against it afterward without
re-opening that dependency.

## Scope

- **In scope:** full training run with sprint 003's winning configuration;
  regression testing against current EN/ZH quality (must not regress);
  Vietnamese-specific validation using the user's existing VI reward/eval
  models; go/no-go decision to freeze and version the checkpoint.
- **Out of scope:** any RL post-training (separate, gated plan); further
  tokenizer/embedding experimentation (loops back to sprint 003 if this sprint's
  gate fails, rather than being re-litigated here).

## Approach

1. Run the full training job with the configuration sprint 003 selected.
2. Evaluate against sprint 001's full eval suite: per-language and code-switch
   WER/CER, pronunciation spot checks, normalization coverage.
3. Run an explicit EN/ZH regression check against ZipVoice's current
   (pre-change) behavior — this is a net-new front end, not an incremental
   tweak, so an accidental regression on already-working languages is a real
   risk, not a formality.
4. Run Vietnamese-specific validation using the user's existing VI reward
   models (ASR/speaker/quality) mentioned in the FlowTTS-GRPO proposal.
5. Go/no-go: if regression or VI-quality gates fail, loop back to sprint 003
   with the failure evidence rather than shipping a weaker front end.
6. On success, version and freeze the checkpoint as the RL plan's input.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Regression on existing EN/ZH quality from dropping phonemes | Ships a front end that's worse where ZipVoice already worked | Explicit regression gate against pre-change behavior, not just "new languages work" |
| Vietnamese quality doesn't meet bar despite winning the sprint 003 bakeoff | Full-scale training exposes issues small-scale runs didn't | Loop back to sprint 003 with the specific failure mode, rather than accepting a weak result to stay on schedule |
| Front-end changes after this sprint silently invalidate the RL plan's dependency | RL work wasted, per ADR-3's whole rationale | Treat this checkpoint as frozen once versioned; any further front-end change requires explicitly re-opening the RL plan's dependency |

## Testing & Validation

Full eval suite from sprint 001; EN/ZH regression comparison against
pre-change ZipVoice; Vietnamese validation via the user's existing reward
models; human spot-check on a sample of outputs across all three languages and
the code-switched set.

## Rollout

On passing all gates, the checkpoint is versioned and frozen as the input to
`2026-08-28__flowtts_grpo_rl`. The prior phonemizer-based checkpoint/path
remains available as a fallback until the new front end has been validated in
downstream use, not deleted immediately.
