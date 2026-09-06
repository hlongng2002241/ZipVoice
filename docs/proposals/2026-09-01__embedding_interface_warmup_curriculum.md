# Fixing text-conditioning correctness in the warm-started multilingual model

- **Status**: Accepted (architecture decided 2026-09-02; see
  [../adr/2026-09-02__qwen_phone_fusion_text_frontend.md](../adr/2026-09-02__qwen_phone_fusion_text_frontend.md)
  for the formal decision record and
  [../plans/2026-09-02__qwen_phone_fusion_text_frontend/](../plans/2026-09-02__qwen_phone_fusion_text_frontend/)
  for the implementation plan). **Implemented 2026-09-06** (commit
  `1ac4b28`); first training run in progress, see sprint 004's run log.
- **Author**: LongNH (with Claude Code assistance; reviewed by Codex, gpt-5.6-sol,
  high reasoning effort, in two separate sessions)
- **Date**: 2026-09-01, updated 2026-09-02
- **Related**: [2026-08-31__warmstart_from_pretrained_checkpoint.md](2026-08-31__warmstart_from_pretrained_checkpoint.md)
  (the warm-start this proposal applies to).

## Summary

The warm-started multilingual model (887/889 tensors transplanted from a
converged master-architecture checkpoint; only `embed.weight` and
`text_encoder.in_proj` started fresh) produces natural-sounding human speech but
pronounces the wrong words, unchanged from 48k to 76k training batches.
Investigation (see Motivation and Design Evolution) ruled out simple
conditioning collapse (word order provably affects output), found that a
from-scratch per-word language-ID classifier fails on exactly the hard
code-switching cases that matter, and confirmed our current architecture only
uses Qwen's *static* embedding table -- none of its actual pretrained
transformer reasoning, unlike OmniVoice, which runs text through Qwen's full
stack. The accepted direction: **fuse a context-aware Qwen branch (N of Qwen's
own transformer layers, not just the embedding table) with an explicit
phoneme branch (word-mapped via whole-sentence G2P), combined by a learned
gate, feeding into a repositioned `in_proj` that becomes transplantable from
the source checkpoint again.** See the ADR for the precise decision and
rationale, and the plan for the implementation sequencing (which starts with
cheap diagnostics before any of this is built).

## Motivation

### Original observation (2026-09-01)

After the warm-start run reached ~48k batches (~2 epochs), the qualitative
result changed from the from-scratch run's near-total silence to: natural
prosody/timbre/voice quality, but unintelligible or incorrect word content.
Loss (flow-matching MSE) plateaued around 0.07 since roughly batch 2700 and
stayed flat.

Architecture: `embed` is an `nn.Embedding(vocab_size, 896)` initialized from
Qwen2.5-0.5B's real pretrained embedding table; `text_encoder.in_proj` is
`nn.Linear(896, 192)` bridging that into `text_encoder`'s internal 192-dim
space; the rest of `text_encoder` and all of `fm_decoder` are copied directly
from the converged source checkpoint (`EspeakTokenizer`, phoneme-level tokens).
All parameters have been jointly trainable from step 1 (no freezing).

### Empirical findings (2026-09-02, from direct testing)

1. **Word-order sensitivity confirmed.** "A B C" vs. "A C B" produce genuinely
   different speech/pronunciation on the current checkpoint. Rules out simple
   conditioning collapse -- the `embed`->`in_proj`->`text_encoder` pathway
   demonstrably carries and uses positional/sequential information, but isn't
   converging to *correct* content.
2. **Still 100% wrong at 76k batches**, no improvement from 48k. Argues against
   "just needs more training time" on the current setup.
3. **Total audio duration is comparable** between the source checkpoint's
   output and the new model's output for comparable text -- real evidence
   against a *gross* duration bug, but doesn't rule out *per-token*
   misallocation, since ZipVoice's duration-prediction formula
   (`forward_text_inference_ratio_duration`, `zipvoice.py:428-473`) is
   tokenizer-agnostic by construction. Accepted as real but secondary to
   content-correctness.
4. **Prior hands-on experience with per-word language-ID classification.** A
   BERT-based token-classification model (trained by randomly mixing sentences
   from monolingual corpora into synthetic paragraphs) worked in normal cases
   but reliably failed on (a) surface-form-ambiguous EN/VI words, and (b) short
   spans of one language embedded between long spans of the other -- likely
   because the synthetic training mixture underrepresented that specific
   pattern, biasing the classifier toward a document-majority-language prior.
   Abandoned. This directly motivates *not* repeating an isolated upstream
   hard-classification step.
5. **The real training corpus already contains genuine EN/VI code-switched
   utterances** (not just synthetic text mixing) with per-utterance primary-
   language labels from existing dataset metadata -- meaning real code-switch
   *acoustic* supervision already exists, addressing (not eliminating) one of
   the risks raised during review (see ADR).

## Design Evolution (condensed)

This section summarizes how the accepted architecture was reached; see git
history of this file for the full turn-by-turn exploration if needed.

1. **Embedding-only training curriculum** (freeze all but `embed`+`in_proj`,
   train under `fm_loss` alone, then unfreeze) -- sound in principle (real
   precedent: Artetxe et al. 2020, de Vries & Nissim), but real implementation
   gaps exist in this codebase's freeze/unfreeze machinery (see "Other
   Considerations"). Superseded as the primary direction, retained as
   background.
2. **Embedding-space alignment/distillation** (align `embed`+`in_proj`'s
   output toward the source checkpoint's converged phoneme embeddings) --
   real TTS precedent (Tu et al. 2019, Huang et al. 2022), shares the
   word-correspondence machinery the fusion idea needs. Retained as
   background/complementary, not the primary direction.
3. **Multi-candidate phoneme fusion** (run G2P under every candidate
   language's rules per span, gate among them) -- reviewed by Codex, found
   sound with real precedent (closest: Dict-TTS), but the "free word-level
   grouping" assumption was wrong (`phonemize_espeak` groups by *sentence*,
   not word) and multi-language candidate gating added complexity the author
   didn't actually want.
4. **Simplified to a single phone branch + Qwen branch**, fused by one
   two-way gate (author correction, 2026-09-02): no per-language candidate
   gating; the phoneme branch uses whichever language a span is in (see ADR
   for how this connects to the "add language information to the
   phonemizer" idea below), and the "invalid mixture" risk is about blending
   *modalities* (token-derived vs. phone-derived), not blending across
   languages.
5. **Word-boundary G2P mapping confirmed cheap, not hard.** Empirically
   verified: `phonemize_espeak()`'s sentence-level output already contains
   literal space characters marking word boundaries within the phone list --
   splitting on them recovers word-to-phone groups from a *single*
   whole-sentence call, preserving all cross-word phonological rules (no need
   to call G2P per word, which would have lost that context). Word-count
   mismatches (numbers, abbreviations) disappear once text is normalized
   before G2P -- confirmed empirically, and matches this project's actual
   pipeline (normalize once, upstream, before either the BPE tokenizer or the
   phonemizer sees the text).
6. **Static Qwen embedding cannot supply language context; Qwen's actual
   transformer layers can.** `embed` only imports Qwen's embedding *table*
   (context-free, static lookup) -- not analogous to what a full pretrained
   LLM offers. Checked `OmniVoice` (`../OmniVoice/omnivoice/`) directly:
   it loads Qwen via `AutoModel.from_pretrained` (full pretrained transformer,
   28 layers for Qwen3-0.6B) and runs the *entire* stack every forward pass
   (`omnivoice.py:418-424`) -- genuinely contextual. This motivated routing
   our own token branch through N of Qwen2.5-0.5B's 24 transformer layers
   (run once, offline, as a frozen feature extractor) instead of just its
   embedding table.
7. **Parameter cost checked directly** (Qwen2.5-0.5B): embedding table
   135.9M params (52.5% of the current 258.6M-param ZipVoice model already);
   each transformer layer 14.9M params; 2/4/8 layers = 29.8M/59.6M/119.3M.
   Recommendation: start with 4 layers (enough for real cross-attention
   context integration, given language-ID is a relatively shallow property in
   transformer LMs per multilingual-probing literature; validate empirically
   against known hard examples rather than committing to a number
   speculatively) -- accepted by the author as a fixed, one-time offline cost.
8. **Fusion point iterated three times** before settling: (a) fuse-then-project
   (rejected: forces one projection to do two jobs); (b) fuse after two
   independent 192-dim projections, feeding into `text_encoder`'s *existing*
   `in_proj` unchanged (workable, but needs `TTSZipformer.forward()` modified
   to skip its own internal `in_proj` call); (c) **accepted**: reposition
   `text_encoder`'s `in_dim` to 192 and feed the *fused* 192-dim
   representation as `text_encoder`'s input -- no changes needed inside
   `TTSZipformer`/`text_encoder` at all, and as a bonus, `in_proj` becomes
   shape-compatible with the source checkpoint's original `in_proj` (192->192)
   again, making it transplantable -- a third piece of pretrained capability
   recovered (previously only `fm_decoder` and `text_encoder`'s Zipformer
   layers were transplanted).

See the ADR for the final, precise architecture and rationale.

## Other Considerations

Earlier analysis and secondary findings, kept brief since the primary focus is
the fusion architecture (see ADR/plan).

- **Embedding-only warm-up curriculum**: real implementation gaps found --
  `get_parameter_groups_with_lrs()`'s `freeze_modules`/`unfreeze_modules`
  (`zipvoice/utils/common.py:650`) isn't wired to `train_zipvoice.py`'s CLI,
  only matches top-level module names, and doesn't actually set
  `requires_grad=False` (only excludes params from optimizer groups).
  Optimizer-state handling across a phase transition is undefined. Superseded
  by the fusion direction; not currently planned.
- **Duration/pacing**: matching *total* audio duration does not rule out
  *per-token* duration misallocation (the prediction formula is
  tokenizer-agnostic by construction). The accepted architecture's plan
  includes reverting duration allocation to phone-count (not BPE-token-count)
  once phones are available per word -- see ADR/plan.
- **`--finetune True` missing from `m05_train_warmstart.sh`**: selects a fixed
  LR scheduler and skips an early high-dropout/schedule phase
  (`train_zipvoice.py:218`, offset at lines 620-623) intended for fine-tuning
  runs. Should be checked/fixed independently of the fusion work, since it
  could independently explain slow or poor adaptation on the *current*
  checkpoint.
- **Chinese within-language ambiguity is already solved in this codebase**:
  `EmiliaTokenizer.tokenize_ZH()` (`tokenizer.py:279-297`) uses `jieba.cut()`
  for word segmentation feeding `pypinyin.lazy_pinyin(..., tone_sandhi=True)`,
  which uses that word-level context to disambiguate polyphones via its
  phrase dictionary -- real, working, reusable for the phone branch's Chinese
  handling. **Correction**: `EmiliaTokenizer.tokenize_EN()` is plain
  `phonemize_espeak(text, "en-us")`, identical to `EspeakTokenizer` -- no
  special English disambiguation exists anywhere in this codebase, and
  `EmiliaTokenizer` has no Vietnamese path at all (English and Vietnamese
  both need plain espeak regardless of which tokenizer class is used). Once
  decomposed per-language, "which G2P to trust" isn't a genuine three-way
  tradeoff: only Chinese has an actual choice, and pypinyin+jieba is a
  strict improvement there.
- **Whether the source `text_encoder` is language-/frontend-agnostic at all**
  remains the central unproven bet of the original warm-start proposal --
  "887/889 tensor shapes match" proves architectural, not functional,
  compatibility.
- **Documentation drift, resolved**: the epoch-count mismatch (15 vs. 50
  epochs) between this proposal and `m05_train_warmstart.sh` was an
  intentional change by the user, not a bug; corrected in the related
  proposal doc.

## Alternatives Considered

- **Just keep joint-training longer** (status quo). Rejected -- the
  76k-batch plateau with no improvement from 48k argues against this being
  sufficient on its own.
- **Per-word language-ID classifier, then language-aware phonemization**
  (repeat the author's earlier BERT-classifier approach inside the TTS
  pipeline). Rejected: already tried and found insufficiently accurate on
  exactly the hard cases that matter most.
- **Multi-candidate (per-language) phoneme gating.** Considered and
  simplified away -- see Design Evolution point 4.
- **Cross-attention learned alignment between the phone and Qwen-token
  sequences** (instead of explicit word-level mapping). Rejected: would make
  the model re-learn a correspondence that's actually deterministic and
  computable, burning limited training data on a solved problem instead of
  the genuinely uncertain part (how much to trust each branch). See ADR.
- **Embedding-only curriculum and embedding-alignment/distillation as the
  primary direction.** Not rejected outright, but deprioritized behind
  fusion -- see Design Evolution points 1-2 and the ADR's Alternatives table.

## Open Questions

- Does a data/tokenizer correctness audit turn up a structural bug that
  explains "natural voice, wrong words" on its own, independent of the
  fusion architecture entirely? (First step of the plan, precisely because
  this needs ruling out before the fusion work would be worth trusting.)
- At scale (not just one manual test), how *local* and how *strong* is the
  model's current use of text conditioning?
- Can a source-compatible oracle phone path recover correct pronunciation at
  all? If not, none of the fusion/alignment work is worth building yet.
- How many Qwen layers actually separate the known hard examples well in
  practice (2 vs. 4 vs. 8), rather than the speculative "start with 4"
  recommendation?
- Is `--finetune True` missing from `m05_train_warmstart.sh` a significant
  independent confound on the *current* (pre-fusion) checkpoint?

## Next steps

See [../adr/2026-09-02__qwen_phone_fusion_text_frontend.md](../adr/2026-09-02__qwen_phone_fusion_text_frontend.md)
for the formal architecture decision, and
[../plans/2026-09-02__qwen_phone_fusion_text_frontend/](../plans/2026-09-02__qwen_phone_fusion_text_frontend/)
for the implementation plan.
