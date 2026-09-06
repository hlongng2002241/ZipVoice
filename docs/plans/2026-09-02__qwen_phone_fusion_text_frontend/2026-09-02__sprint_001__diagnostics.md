# Sprint 001: Diagnose the current plateau before building anything new

- **Date:** 2026-09-02
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Draft
- **Related:** [2026-09-02__qwen_phone_fusion_overview.md](2026-09-02__qwen_phone_fusion_overview.md),
  [../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md](../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md),
  [2026-09-02__sprint_000__new_tokenizer.md](2026-09-02__sprint_000__new_tokenizer.md)
  (independent of this sprint -- may run in parallel, neither depends on the
  other)

## Goal

Rule out (or find and fix) a simpler, non-architectural explanation for
"natural voice, wrong words" before investing in the fusion frontend --
using only the *existing* checkpoint and codebase, no new model code.

## Background

The 76k-batch plateau (no improvement from 48k) is consistent with several
different root causes, not just "the text frontend needs fusion": a
data/tokenizer correctness bug, a training-script misconfiguration
(`--finetune True` missing from `m05_train_warmstart.sh`), or genuinely weak
but non-zero text conditioning that hasn't yet converged. Building the fusion
architecture on top of an undiagnosed simpler bug would waste the engineering
effort and make the real cause harder to isolate afterward.

## Scope

- **In scope:** manifest/data-pairing audit, tokenizer consistency checks,
  at-scale text-conditioning ablations (shuffle/replace/reverse/zero),
  checking whether `--finetune True` explains any of the plateau.
- **Out of scope:** any new model architecture code; the G2P/Qwen-fusion
  pipeline (sprint 003).

## Approach

1. **Data/tokenizer correctness audit.**
   - Sample N utterances from the training manifest; decode their token IDs
     back to text via `LanguageModelTokenizer` and cross-check against the
     manifest's own transcript field.
   - Confirm training-time and inference-time tokenizer construction produce
     identical vocab/`token2id` mappings (same `pretrained_tokenizer_name`,
     same saved tokenizer directory).
   - Confirm `[LANG:xx]` ground-truth tags match each utterance's actual
     `supervision.language` field, not a stale or mismatched value.
   - Confirm no stale phoneme-token manifest fields are being read instead of
     the current multilingual tokenization path.
2. **`--finetune True` check.** Compare `train_zipvoice.py`'s behavior with
   and without `--finetune True` on a short run from the warm-start
   checkpoint (LR schedule, dropout/model-schedule phase per
   `train_zipvoice.py:218`, batch-count offset at lines 620-623). Determine
   whether this alone shifts early-training behavior meaningfully.
3. **At-scale text-conditioning ablations**, on the existing 76k-batch
   checkpoint (GPU now available for this):
   - Correct text vs. text shuffled across utterances vs. text with words
     reversed vs. zeroed text-conditioning, measured via loss (stratified by
     flow time `t`). Audio can be generated for a fixed sample set here, but
     judging it is out of scope (see note below).
   - This extends (doesn't replace) the manual word-order test already done
     -- need to characterize *degree* and *locality* of text use across many
     examples, not just confirm it's non-zero on one.

**Evaluation note**: perceptual/ASR-based judgment of generated audio is
handled in a separate project, out of scope here. This sprint produces loss
numbers and (optionally) generated audio files as artifacts; interpreting
audio quality happens elsewhere.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| A real data-pairing or tokenizer bug is found | Would explain the plateau independent of any architecture question | Fix and re-evaluate the *existing* checkpoint's behavior before deciding whether fusion is still needed |
| `--finetune True` meaningfully changes training dynamics | Any later before/after fusion comparison would be confounded unless controlled for | Decide whether to fix this on the baseline before sprint 004's training comparison, or explicitly hold it constant across both |
| At-scale ablations show text conditioning is weaker/more local than the single manual test suggested | Changes how much the fusion architecture alone can be expected to help | Feed findings into sprint 003's fixed-gate oracle-phone experiment design |

## Testing & Validation

- Audit findings documented with concrete evidence (decoded token/text pairs,
  tokenizer config diffs, ablation loss/audio comparisons) -- not just
  pass/fail.
- No architecture decision changes as a result of this sprint alone; findings
  feed into sprint 003's design and the overall go/no-go on sprints 003-004.

## Rollout

N/A (diagnostic sprint, no code shipped).
