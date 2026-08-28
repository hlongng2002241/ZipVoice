# Sprint 001 — Correctness gates & reward-function infrastructure

- **Date:** 2026-08-28
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Draft

## Goal

Implement the SDE-rollout extension to `EulerSolver` and pass all 7 correctness
gates from the proposal before it's trusted in a training loop, and build
batched, GPU-resident reward-function wrappers around the existing (and the
user's Vietnamese-specific) evaluation models — all independent of which
checkpoint sprint 002 eventually targets, so this work is not wasted regardless
of front-end-plan timing.

## Background

Per [../../proposals/2026-08-28__flowtts_grpo_rl_finetuning.md](../../proposals/2026-08-28__flowtts_grpo_rl_finetuning.md)'s
"Correctness gates" section (added after Codex review): matching ZipVoice's and
F5-TTS's time-direction convention is necessary but not sufficient for the SDE
formula to be trustworthy — the stochastic transition kernel, discretization,
and endpoint handling all need independent verification.

## Scope

- **In scope:**
  - SDE rollout mode for `EulerSolver`/`DiffusionModel` (`zipvoice/models/modules/solver.py`),
    windowed to a step subset (`S_min`, window size) per the paper's approach.
  - All 7 correctness gates, each as an automated test on synthetic/small-scale
    tensors:
    1. Noise-scale limit (`σ_t → 0` reduces to the existing deterministic Euler
       step).
    2. Marginal-distribution check.
    3. Log-probability agreement (closed-form vs. implementation).
    4. Ratio sanity (`r_t(θ) ≈ 1` before the first update).
    5. KL sanity (exactly 0 when policy = reference).
    6. CFG-off still conditional (verify `guidance_scale=0.0`'s exact branching
       in `DiffusionModel.forward`).
    7. No supervised regression (SDE/GRPO machinery wired up but reward
       replaced by the original flow-matching loss shouldn't change quality).
  - Batched, GPU-resident reward-function wrappers: speaker similarity
    (`zipvoice/eval/speaker_similarity`), WER/CER (`zipvoice/eval/wer`), a
    UTMOS/DNSMOS-equivalent perceptual-quality score
    (`zipvoice/eval/mos/utmos.py`), and the user's existing Vietnamese-specific
    reward models — adapted for mid-rollout, per-completed-sample use (rewards
    are terminal, evaluated once on fully generated/vocoded audio, not
    stepwise).
  - Hard-case text augmentation (word/sentence repetition) with its own,
    separately-weighted curriculum and a normal-text regression suite guarding
    against over-learning repetition-heavy behavior.
- **Out of scope:** the GRPO loss itself and any real RL training run against a
  live checkpoint (sprint 002).

## Approach

1. Implement the SDE rollout mode, deriving the exact update rule from
   ZipVoice's own `x_t`/`v_t` convention (verified in the proposal) rather than
   copying the paper's equation as a black box.
2. Implement and pass each of the 7 gates in order, on synthetic tensors first.
3. Adapt each existing eval model into a batched, GPU-resident callable reward
   function; benchmark throughput/latency to confirm it won't bottleneck rollout
   generation or contend for GPU memory with it.
4. Implement hard-case augmentation with a separate curriculum weight and a
   held-out normal-text set to detect repetition-behavior regressions.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| A gate fails and reveals a fundamental issue with the SDE derivation | Discovered before wasting real training compute (the point of this sprint) | This is exactly what the gates are for — treat a failure as a design finding, not a bug to patch around |
| Existing eval models are file-oriented/unbatched/slow | Reward orchestration bottlenecks rollout throughput | Explicit batching/GPU-residency work is in scope here, not assumed free |
| UTMOS/DNSMOS-equivalent behavior differs meaningfully across languages, especially Vietnamese | Reward signal could be miscalibrated for VI specifically | Validate the perceptual-quality reward's behavior per language before trusting it in sprint 002, using the user's Vietnamese-specific models as a cross-check |

## Testing & Validation

The 7 correctness gates as automated tests; reward-wrapper throughput/latency
benchmarks; hard-case curriculum's regression-suite results on held-out normal
text.

## Rollout

This sprint's code is reusable regardless of which checkpoint sprint 002
eventually targets — per the overview, it proceeds independently of the
front-end plan's timeline.
