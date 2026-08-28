# Sprint 003 — Tokenizer + embedding bakeoff

- **Date:** 2026-08-28
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Tokenizer decision Done (Qwen2.5-0.5B, see Progress) — embedding
  form/duration-scheme ablations and real WER-based validation still need
  sprint 004's training run

## Goal

Run Codex's revised selection methodology — real pronunciation/WER outcomes, not
tokens-per-character alone — across tokenizer candidates and embedding-init
strategies, and pick the specific configuration to freeze for sprint 004.

## Background

Per [../../proposals/2026-08-28__mbert_multilingual_tokenizer.md](../../proposals/2026-08-28__mbert_multilingual_tokenizer.md)'s
"Selection methodology" section (revised after Codex review), the original
tokens-per-character comparison correctly filtered out clearly bad candidates
(e.g. LFM2.5-350M's Vietnamese fragmentation) but is insufficient to pick a
final answer — it measures sequence-length efficiency, not phonetic adequacy.
This sprint runs the actual bakeoff. Uses sprint 001's eval sets and sprint
002's `[LANG:xx]` mechanism (both are inputs, not outputs, of this sprint).

## Scope

- **In scope:**
  - Small-scale training runs (not just static tokenization stats) for each
    candidate: mBERT (leading candidate), Qwen2.5-0.5B (next-best trade-off per
    the tokens/char pre-filter), a corrected-phonemizer-with-language-routing
    baseline, and a character-level baseline — Codex's "essential ablations"
    list.
  - Embedding-init ablation: pretrained-checkpoint-loaded vs. trained-from-scratch,
    for at least the leading tokenizer candidate.
  - Duration-scheme ablation: uniform vs. script-weighted split, independent of
    tokenizer choice.
  - A learned duration-predictor comparison point against the zero-parameter
    script-weight heuristic (Codex's suggestion, not to be dismissed as
    automatically out of scope).
  - **Correction (2026-08-28)**: text normalization is not actually a
    per-candidate axis. `EnglishTextNormalizer`/`ChineseTextNormalizer`
    (`zipvoice/tokenizer/normalizer.py`) operate on raw text *before* any
    tokenizer sees it (number/abbreviation expansion via regex + `inflect`/
    `cn2an`), so they're reusable as-is regardless of which subword tokenizer
    wins this bakeoff — there's nothing to compare across candidates here.
  - **Decided (2026-08-29), supersedes the above "Vietnamese normalizer
    blocker"**: `MultilingualTokenizer` does not call any text normalizer
    (`EnglishTextNormalizer`, `ChineseTextNormalizer`, or a Vietnamese
    equivalent) at all, by design. The author's decision: all normalization
    (number/date/abbreviation expansion, etc.) must already be done upstream,
    before a corpus reaches `train_zipvoice.py` — it is a data-preparation
    concern, not a tokenizer concern, for this multilingual path. There is no
    "missing Vietnamese normalizer" gap to fill here; it is out of scope for
    this sprint and for `MultilingualTokenizer` generally.
  - Latency/memory measurement per candidate's embedding-table size.
- **Out of scope:** the full-scale training run (sprint 004); anything specific
  to FlowTTS-GRPO RL.

## Approach

1. Stand up a small-scale training harness capable of producing comparable WER
   results per candidate (this is new — the original draft only had static
   tokenizer stats).
2. Run the candidate matrix: {tokenizer} × {embedding init} × {duration scheme},
   prioritized using the existing tokens/char pre-filter to avoid spending
   compute on already-disqualified candidates (e.g. skip LFM2.5-350M given its
   measured VI/ZH fragmentation).
3. Evaluate each run against sprint 001's held-out sets: per-language and
   code-switch WER/CER, pronunciation spot checks on the ambiguous-word subset,
   normalization coverage, latency/memory.
4. Write up results and update
   [../../proposals/2026-08-28__mbert_multilingual_tokenizer.md](../../proposals/2026-08-28__mbert_multilingual_tokenizer.md)'s
   "mBERT vs. Qwen2.5-0.5B: final decision" section with the actual outcome —
   **done**: Qwen2.5-0.5B, recorded in
   [../../adr/2026-08-28__choose_qwen25_tokenizer.md](../../adr/2026-08-28__choose_qwen25_tokenizer.md).

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Small-scale runs may not predict full-scale training behavior | Wrong choice only discovered after sprint 004's expensive run | Sprint 004's regression suite is the final gate; treat sprint 003's pick as strong evidence, not certainty |
| Compute budget for the full candidate × embedding × duration matrix may be large | Bakeoff stalls or gets cut short | Use the existing tokens/char table to prune obviously-inferior candidates before running full WER evaluations on all of them |
| mBERT's pretrained input embeddings may not actually help once its transformer is dropped (Codex's point: they were trained jointly with contextual layers) | Loading pretrained weights could be no better than random init | This is exactly what the embedding-init ablation in this sprint tests — don't assume the answer |

## Testing & Validation

Per-candidate WER/CER report against sprint 001's eval sets; pronunciation
spot-check results on the ambiguous-word subset; normalization-coverage
checklist; latency/memory table.

## Progress (2026-08-28)

**Done, corpus-independent:**
- Tokenization-efficiency comparison across 7 candidates (tokens/char per
  language) — already in
  [../../proposals/2026-08-28__mbert_multilingual_tokenizer.md](../../proposals/2026-08-28__mbert_multilingual_tokenizer.md)'s
  "Model selection analysis" — used here as the pre-filter Codex recommended.
- [eval_sets/embedding_structure_check.md](eval_sets/embedding_structure_check.md):
  directly tested Codex's objection to loading mBERT's pretrained embedding
  table (that it "may not be obviously useful" without its transformer).
  Result: mBERT's raw embeddings show real relational structure for English
  (0.429 vs. 0.056 mean cosine similarity, related vs. unrelated word pairs)
  and, more weakly, for Vietnamese (0.324 vs. 0.169) — a random-init baseline
  confirms the test is well-calibrated (~0 either way with no training). This
  is real evidence in favor of loading pretrained weights over training from
  scratch, not just an assumption.
- Corrected the normalization-per-candidate scope item (see Scope above) —
  it isn't actually a differentiator between tokenizer candidates.

**Provisional recommendation (2026-08-28), pending real validation:**
- **Tokenizer: mBERT** (`bert-base-multilingual-cased`). Smallest embedding
  footprint (91.8M) among candidates with good measured EN/VI/ZH
  tokenization, real TTS-integration precedent (BERT-VITS2-style), and now
  confirmed non-trivial embedding structure even without its transformer.
- **Embedding form: load pretrained weights** (`get_input_embeddings().weight`,
  `119547×768`) into ZipVoice's `nn.Embedding`, projected down via
  `Linear(768, 192)`, fine-tuned end-to-end (not frozen) — per the embedding
  structure check above. The weaker (but still positive) Vietnamese-specific
  signal is a documented caveat, not a reason to prefer random init instead.

**Update (2026-08-28, real data)**: the user provided access to two real
Vietnamese ASR-transcript corpora (173,461 utterances total — see
[eval_sets/real_corpus_token_stats.md](eval_sets/real_corpus_token_stats.md)).
This partially lifts the corpus-access blocker for **text-based tokenizer
analysis** specifically:
- Real fragmentation is worse than the earlier synthetic estimate but
  essentially ties mBERT and Qwen2.5-0.5B (0.372 vs. 0.368 tokens/char) —
  doesn't change the recommendation, given mBERT's smaller embedding table.
- Found and fixed a real, narrow mBERT vocabulary gap: 27 capitalized
  Vietnamese interjection/emphasis words (e.g. `Ừ`, `Ờ`, `Ồ`) were hitting true
  `[UNK]` (0.0058% of tokens) because only their lowercase forms exist in
  mBERT's vocab. Fixed via `MultilingualTokenizer`'s new `extra_tokens`
  parameter — confirmed `[UNK]` count drops to exactly 0 on the same corpus.
- Directly answered the plan's original "should token distribution be as
  uniform as possible" question with real measurement: no — natural-language
  token frequency is inherently Zipfian (66.1% of theoretical maximum
  entropy here), and that's expected, not fixable by tokenizer choice. The
  real actionable risk is singleton-token undertraining (21.1% of used
  vocabulary seen only once), which the already-planned pretrained-embedding
  loading mitigates directly.

**Still not done — genuinely blocked, not skipped:** everything that needs
actual **paired text-audio training** (small-scale training runs producing
real per-language/code-switch WER/CER, pronunciation spot-checks on trained
output, duration-scheme ablations) is unaffected by the text-only corpus
access above — those need a running training pipeline against these corpora's
audio, which is a separate, larger undertaking not attempted here. The
recommendation is now backed by real text-corpus evidence in addition to the
earlier synthetic/structural checks, but is still not a substitute for that
training-based validation.

**Final update (2026-08-28) — tokenizer decision finalized: Qwen2.5-0.5B, not
mBERT.** The Vietnamese-only ASR-corpus result above (mBERT/Qwen2.5-0.5B
essentially tied) did not hold once tested at 10x scale (~690M characters
each) on real web text across **all three** target languages
(FineWeb-edu/FineWeb-2, not just Vietnamese):

| Metric | mBERT EN | Qwen2.5 EN | mBERT VI | Qwen2.5 VI | mBERT ZH | Qwen2.5 ZH |
|---|---|---|---|---|---|---|
| Tokens/char | 0.272 | **0.257** | **0.367** | 0.374 | 0.938 | **0.692** |
| `[UNK]` rate | 0.969% | **0%** | 0.791% | **0%** | 0.899% | **0%** |

mBERT's `[UNK]` issue turned out to be pervasive across all three languages
(~0.8-1.0%), not the narrow, Vietnamese-specific, patchable gap the smaller
ASR-corpus test suggested — confirmed by a systematic check finding 36/68
possible uppercase Vietnamese diacritic characters missing from mBERT's
vocabulary entirely. mBERT was also meaningfully worse at English compression
and dramatically worse at Chinese (0.938 vs. 0.692 tokens/char). Qwen2.5-0.5B
had zero `[UNK]` across every corpus and language tested.

**Decision recorded in
[../../adr/2026-08-28__choose_qwen25_tokenizer.md](../../adr/2026-08-28__choose_qwen25_tokenizer.md).**
`MultilingualTokenizer` (`zipvoice/tokenizer/multilingual_tokenizer.py`) now
defaults to Qwen2.5-0.5B (vocab 151,669, hidden 896, ~136.1M embedding
params) rather than mBERT. This sprint's tokenizer-selection work is
**done** — the remaining open item (embedding pretrained-vs-scratch,
duration-scheme ablations, real WER validation) still needs the sprint 004
training run to resolve.

## Rollout

**Qwen2.5-0.5B is the frozen tokenizer choice** for sprint 004. Embedding
form (load pretrained vs. train from scratch) and duration scheme
(uniform vs. script-weighted) remain to be confirmed by that training run,
not further text-only analysis. mBERT's full investigation is kept in
[../../proposals/2026-08-28__mbert_multilingual_tokenizer.md](../../proposals/2026-08-28__mbert_multilingual_tokenizer.md)
and this document for future reference, not discarded.
