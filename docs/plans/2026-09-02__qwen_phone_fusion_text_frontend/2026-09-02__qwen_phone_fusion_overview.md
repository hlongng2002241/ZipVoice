# Qwen+phoneme fusion text frontend -- implementation overview

- **Date:** 2026-09-02
- **Author:** LongNH (with Claude Code assistance)
- **Status:** In Progress
- **Related:** [../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md](../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md)
  (the accepted architecture this plan implements),
  [../../proposals/2026-09-01__embedding_interface_warmup_curriculum.md](../../proposals/2026-09-01__embedding_interface_warmup_curriculum.md)
  (full analysis)

## Goal

Determine whether the warm-started multilingual model's "natural voice, wrong
words" plateau is fixable by the accepted Qwen+phoneme fusion text frontend --
and if the cheap diagnostics below don't first reveal a simpler root cause --
build and train it.

## Background

See the ADR for the full architecture and rationale, and its Terminology note
for the `tokens` (= phones, unchanged from inherited code) / `lm_tokens` (=
Qwen/BPE output, new) convention used throughout this plan.

In short: the current model only uses Qwen's static embedding table (no
context), while OmniVoice runs text through Qwen's full transformer stack;
espeak-style G2P can't do per-word language-ID on code-switched Latin-script
text, and a from-scratch classifier for this already failed on the hard
cases. The accepted fix fuses a context-aware Qwen branch (N transformer
layers, not just embeddings) with an explicit phoneme branch, gated together
per group, feeding a repositioned `in_proj` that becomes transplantable from
the source checkpoint again.

**Critical constraint driving this plan's sequencing**: the fusion
architecture is real, non-trivial engineering (new tokenizer, new
preprocessing pipeline, new model code, duration-allocation changes). Before
investing in the model-architecture parts of it, cheap diagnostics must rule
out simpler explanations for the current plateau -- otherwise we risk
building real complexity on top of, say, an undiagnosed data-pairing bug or a
training-script misconfiguration, either of which the fusion architecture
would not fix and would make harder to isolate afterward.

**Explicit constraint on execution order**: sprint 000 (the new tokenizer)
and sprint 001 (diagnostics on the *existing* checkpoint) do not depend on
each other and may proceed in parallel -- neither touches `zipvoice/models/`.
**No changes to model architecture code (`zipvoice/models/zipvoice.py`,
`zipvoice/models/modules/zipformer.py`) happen until sprints 000-001 are
complete and reviewed.** Sprint 003 is the first point model code is touched.

**Sprint 002 (a dedicated pre-architecture oracle-validation sprint) was
dropped** -- see the ADR's "Why a separate pre-architecture oracle-validation
sprint was dropped" note. In short: the only checkpoint available to validate
against right now is not the one this project will actually warm-start the
fused architecture from, so a dedicated validation step run against it now
would not inform the real go/no-go decision. The oracle-validation idea
survives as sprint 003's fixed-gate two-branch comparison instead, which runs
against whatever checkpoint is actually current at that point.

## Scope

- **In scope:** building the new wrapping tokenizer (phones + `lm_tokens` +
  group mapping, with a strict equivalence test against existing tokenizers);
  diagnosing the current plateau; validating the fusion hypothesis cheaply
  before building any model-architecture changes; Qwen N-layer feature
  extraction; the new model code (phone embedding, Qwen-branch projection,
  per-group fusion gate, `text_encoder.in_dim` change, `in_proj`/phone-embedding
  source-transplant); duration allocation moving to phone-count; training and
  handing off checkpoints for external evaluation.
- **Out of scope:** multi-candidate (per-language) phoneme gating (explicitly
  rejected, see ADR); cross-attention-based alignment (explicitly rejected);
  the embedding-only curriculum and embedding-distillation ideas (deprioritized
  background, not built here); fixing `--finetune True` on the *existing*
  from-scratch/warm-start scripts (tracked as a separate, independent
  confound -- worth fixing but not part of this plan's deliverable); ASR/CER
  and listening evaluation (handled in a separate project the author owns).

## Approach

Four sprints (numbered 000, 001, 003, 004 -- 002 was dropped, see above and
the ADR; numbering is left as-is rather than renumbering the remaining
sprints, to avoid rewriting file names/cross-references for no functional
reason). 000 and 001 may run in parallel; 003-004 are strictly gated behind
sprint 001's findings and touch model architecture code for the first time:

0. **[Sprint 000: Build the new tokenizer](2026-09-02__sprint_000__new_tokenizer.md)**
   -- a new tokenizer class wrapping (not modifying) `EmiliaTokenizer`/
   `EspeakTokenizer`, producing phones + groups + `lm_tokens` +
   an `lm_tokens`-aligned zero-duration mask (not phone-aligned -- corrected
   during implementation, see that sprint's doc), with an automated
   equivalence test proving its phone output matches the existing
   tokenizers' exactly. No model-architecture code touched.
1. **[Sprint 001: Diagnostics](2026-09-02__sprint_001__diagnostics.md)** --
   data/tokenizer correctness audit and at-scale text-conditioning tests on
   the *existing* checkpoint, before touching any new architecture. If this
   finds a structural bug (data pairing, tokenizer mismatch, the missing
   `--finetune` flag) that plausibly explains "natural voice, wrong words" on
   its own, fix and re-evaluate *that* first -- fusion work would be wasted
   effort layered on top of an unrelated bug.
3. **[Sprint 003: Build the fusion frontend](2026-09-02__sprint_003__build_fusion_frontend.md)**
   -- Qwen N-layer offline feature extraction, the new model code (phone
   embedding table, Qwen-branch projection, per-group fusion gate,
   `text_encoder.in_dim=192`, source-transplant for `in_proj` and the phone
   embedding), duration allocation moving to phone-count, the manifest/dataset
   plumbing changes needed to consume sprint 000's tokenizer artifact safely,
   and (as its first step) the fixed-gate (oracle/uniform/wrong) two-branch
   comparison that used to be sprint 002's job -- run here instead, against
   whichever checkpoint is the actual current warm-start source at the time,
   since it needs this sprint's own model code to exist anyway.
4. **[Sprint 004: Train](2026-09-02__sprint_004__train_and_evaluate.md)** --
   train the fused architecture (from a fresh warm-start checkpoint, per the
   ADR's transplant plan) and hand off checkpoints. Evaluation (ASR/phone-error
   metrics, listening samples) is handled in a separate project, out of scope
   for this plan.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Diagnostics (sprint 001) find the real cause is unrelated to fusion (e.g. a data bug) | Sprints 003-004 become unnecessary or need to be re-scoped | Sprint 001 is a hard gate; do not proceed to sprint 003 until its findings are reviewed |
| Oracle phones still can't recover pronunciation, once tested (now sprint 003's fixed-gate comparison, not a dedicated pre-architecture sprint -- see the ADR) | Fusion architecture (sprint 003-004) would not fix the underlying problem | Sprint 003's fixed-gate step is a hard gate; if it fails, escalate for a fresh diagnosis rather than continuing to a learned gate/training anyway |
| N (Qwen layers) chosen speculatively rather than validated | Wasted compute/param budget, or insufficient context signal | **Resolved**: a linear-probe experiment validated N=4 (see the ADR's point 6) -- N=2/4/8/12 all scored ~96-98%, N=24 slightly worse; N=4 kept for margin, not because the data required it over N=2 |
| Gate collapse toward phone-only re-emerges (same failure mode as the abandoned BERT classifier, relocated) | The Qwen-correction bet (ADR point 1) fails to activate | **Accepted risk, not mitigated** (no floor/entropy/dropout, per ADR point 4) -- monitored via gate-weight logging during sprint 004's training; a failed gate is a failed experiment result, not evidence more training is needed |
| Word-count/group mismatches between phone output and `lm_token` output on real (not synthetic) data | Broken group correspondence for some fraction of utterances | Sprint 000's equivalence test plus validation on a real corpus sample (not just synthetic test cases) before sprint 003 relies on it |
| `--finetune True` missing from the current warm-start script confounds any before/after comparison | Hard to attribute improvement to fusion vs. this unrelated fix | Track as a separate, independently-fixable item; note explicitly in sprint 001's findings whether it needs fixing before comparisons are meaningful |
| Chinese-specific tokenizer decisions (segment routing, punctuation handling) were made by Claude Code, not the author, who does not read Chinese | Decisions may miss a linguistic consideration a Chinese speaker would catch | Flagged explicitly in the ADR and sprint 000; worth a native-speaker review pass if one becomes available before sprint 003 |
| The warm-start source checkpoint is not fixed: the current one (`hynt/ZipVoice-Vietnamese-2500h`) was fine-tuned Vietnamese-only and has lost English pronunciation; the author intends to switch to the original upstream checkpoint once the fusion architecture exists | Any validation/comparison run against the current checkpoint (including sprint 003's fixed-gate step) risks being confounded by that checkpoint's own training history rather than testing backbone capacity in general -- this is exactly why sprint 002 was dropped, see the ADR | Re-run sprint 003's fixed-gate comparison against whichever checkpoint is actually the current warm-start source at that time, not an assumption fixed when this plan was written |
| `piper_phonemize`'s espeak-ng engine is stateful across calls within a process (found during sprint 000: phonemizing text A differently depending on what text was phonemized immediately before it) -- a pre-existing property, not introduced by this plan's code | Could plausibly affect existing tokenization of large manifests too (e.g. `tokenizer.py`'s `add_tokens()`), not just this sprint's new tokenizer | Sprint 000's equivalence test computes each tokenizer's output in one uninterrupted pass to avoid it; worth a dedicated, separate investigation into whether it affects existing training-data tokenization, independent of this plan |

## Testing & Validation

Each sprint defines its own success criteria (see sprint files). Overall:
**loss is not a valid success signal** for this work (already shown to
plateau without indicating content-correctness). Quality/correctness
evaluation (ASR/phone-error metrics, listening checks) is handled in a
separate project, out of scope for this plan -- this plan's deliverables are
a validated new tokenizer, diagnostic findings, validated components, and
trained checkpoints for that external project to judge.

## Rollout

No production rollout concept applies (research/training work, not a
user-facing service change). The relevant "rollout" is: each sprint's
findings must be reviewed before starting the next, model architecture code
is not touched before sprint 003, and the fused architecture (if built)
trains in a separate experiment directory from the existing warm-start run,
so the current checkpoint remains available as a baseline throughout.
