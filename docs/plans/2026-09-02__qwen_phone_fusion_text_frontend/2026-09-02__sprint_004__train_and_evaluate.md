# Sprint 004: Train the fused architecture

- **Date:** 2026-09-02
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Draft
- **Related:** [2026-09-02__qwen_phone_fusion_overview.md](2026-09-02__qwen_phone_fusion_overview.md),
  [2026-09-02__sprint_003__build_fusion_frontend.md](2026-09-02__sprint_003__build_fusion_frontend.md)
  (must complete first)

## Goal

Train the fused architecture to produce checkpoints that can be handed off
for evaluation. **Evaluation itself (ASR/CER, listening comparisons, and the
go/no-go judgment on whether this fixes "natural voice, wrong words") is
handled in a separate project and is out of scope here.**

## Background

This session's own diagnosis already showed flow-matching loss can plateau at
a stable value without indicating whether pronunciation content is correct
(the original warm-start reached a comparable loss level almost immediately
via decoder-transfer alone, well before content correctness was ever
established) -- so this repo's job is to produce well-formed, periodically
checkpointed training runs; judging their quality happens downstream, in the
separate evaluation project.

## Scope

- **In scope:** training the fused architecture from its new warm-start
  checkpoint; `--finetune True` set correctly this time (per sprint 001's
  finding, if applicable) to avoid re-introducing that confound; periodic
  checkpointing at a cadence useful for external evaluation; lightweight
  gate-weight logging during training.
- **Out of scope:** ASR/CER computation, listening comparisons, and any other
  evaluation logic or tooling -- these live in a separate project and are not
  built or run from this repo.

## Approach

1. **Train** from the sprint-003 warm-start checkpoint, mirroring
   `m05_train_warmstart.sh`'s validated settings otherwise (manifests, env
   vars, LR schedule), explicitly setting `--finetune True` this time.
   Training itself is unmodified -- no evaluation logic added to the training
   loop.
2. **Monitor gate behavior during training**: log gate-weight distributions
   periodically (a lightweight logging addition, not an evaluation pipeline)
   to catch gate collapse or majority-language bias early, since this is
   directly observable from training-time tensors without needing any
   external evaluation.
3. **Checkpoint at a cadence useful for external evaluation** (e.g. every
   4k-8k batches, per the original warm-start proposal's cadence), so the
   separate evaluation project has comparable checkpoints to work with
   against the from-scratch run (`exp/all/`), the original warm-start run
   (`exp/all_warmstart/`), and the oracle artifacts from sprint 003's
   fixed-gate comparison (step 9 -- sprint 002 was dropped; that comparison
   now lives in sprint 003, see its doc and the ADR's note on why).

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Gate ends up ignoring one branch entirely (collapse) -- accepted risk per the ADR, not mitigated by a floor or regularization | Effectively reduces to single-branch behavior, same limitations as before; the Qwen-correction bet (ADR point 1) fails to activate | Gate-weight monitoring (step 2) should catch this early, without needing the external evaluation project; a collapsed gate is a valid (negative) experiment result, not evidence more training is needed |
| Whether content correctness actually improves | Determines whether this architecture is worth keeping | Judged entirely in the separate evaluation project -- not something this sprint or repo determines |

## Testing & Validation

Training-side validation only: confirm training runs without error, loss
behaves sensibly, checkpoints save correctly, and gate-weight logs show no
early collapse. Quality/correctness validation is out of scope -- handled in
the separate evaluation project against the checkpoints this sprint produces.

## Run log

**Launched 2026-09-06 12:07** via `scripts/fusion/m01_train.sh`, in a tmux
session named `ziptrain`, from `exp/fusion/warmstart_fusion.pt`. Console
output is tee'd to `exp/fusion/train_console.log` in addition to the
trainer's own `exp/fusion/log/`.

Config as launched: `--tokenizer fusion`, `--qwen-layers 4`,
`--gate-init-eps 0.01`, `--lang-auto-prob 0`, `--lang-wrong-prob 0`,
`--finetune True`, `--base-lr 0.0001`, `--use-fp16 True`,
`--max-duration 190`, `--num-buckets 100`, `--num-workers 8`,
`--save-every-n 4000`, `--num-epochs 50`, single GPU (L40S 46GB).
122,837,414 trainable parameters -- the frozen Qwen is correctly outside
that count. Extractor confirmed on `cuda:0`.

**Measured cost of `--num-epochs 50`.** Steady state is 2.39 s/batch at
37.4GB of 46GB GPU. The corpus is 1380 audio-hours (171,961 cuts, mean
28.9s), which at `--max-duration 190` is ~26,149 batches per epoch:

| | |
|---|---|
| one epoch | ~17.4 h |
| 50 epochs as configured | ~36 days |
| checkpoint every 4000 batches | ~2.7 h |

The 50 is a deliberate choice, not an inherited default: the warm-start
proposal's Open Questions record it being raised from 15 specifically
because "a mostly-pretrained network converging faster per-epoch on
acoustic quality doesn't mean it needs fewer total epochs if
content-correctness turns out to be the slow part" -- which is exactly this
project's thesis. The measurement above does not overturn that reasoning;
it prices it. What is new is that the price is ~36 days of a single L40S on
a shared machine, which was not known when the number was set.

`--num-epochs` sets only a ceiling, and `--save-every-n 4000` lands a
usable checkpoint every ~2.7h, so the run can be stopped and evaluated at
any point without losing work. The decision this actually forces is
therefore not "50 or fewer" up front, but *when to first evaluate* -- and
that is gated on the eval set (sprint 003 step 2), not on training time.

One startup artifact worth not misreading: the **first batch took 5:01**
with the GPU idle. That is `DynamicBucketingSampler` filling its buffers
(`--num-buckets 100` gives a 500k-cut shuffle buffer against a 172k-cut
corpus), not a stall. Batches 2 onward run at 2.4 s.

**Expected gate trajectory.** The gate weight moves ~3.7e-7 per step
(measured, see sprint 003's status), so at ~26k steps/epoch its RMS shifts
on the order of 0.01 per epoch. Fusion behaviour therefore becomes
measurable within the first day or two rather than at epoch 50, which is an
argument for evaluating early checkpoints rather than waiting.

### Run 1 (2026-09-06): stopped, invalid

Stopped at epoch 2 / batch ~32000 (19.6 h/epoch, 30,092 batches/epoch --
both measured, not estimated) on discovering the run was **~93%
phone-only**: the `_split_into_groups` sentence-boundary merge disabled the
Qwen branch for all but 7.3% of Vietnamese utterances. See the ADR's
"sentence-boundary incident" section. Validation loss fell normally
throughout (0.03889 -> 0.03804 over 24k batches) and stayed below the
pre-fusion baseline, which is exactly why loss alone could not be trusted as
evidence the architecture was working -- it was measuring a phone-only model.

The checkpoints in `exp/fusion/` are from this invalid run.

### Before restarting

1. `python3 scripts/fusion/m02_preflight_corpus.py --manifest ... --token-file ...`
   must pass. It reports per-language alignment, phonemizer failures, and
   the OOV/emptied-group losses that training-time coverage cannot see.
   Expect Vietnamese ~97%; expect **English to fail** at ~0%, which is the
   known mislabelling (95.5% of "English"-labelled utterances contain
   Vietnamese diacritics), not an alignment defect.
2. Watch `qwen_cov_utt` / `qwen_cov_grp` on the first log lines. They should
   be ~0.97 and ~1.0. If either is near zero the conditioning is not
   reaching the model and the run is not worth continuing -- this is the
   check whose absence cost run 1.

## Rollout

Checkpoints from this sprint are handed off to the separate evaluation
project. If that project judges the architecture successful, it becomes the
new primary checkpoint/architecture, superseding `exp/all_warmstart/`; the
from-scratch and original warm-start runs remain available for historical
comparison. If not, findings feed back into a new proposal/plan.
