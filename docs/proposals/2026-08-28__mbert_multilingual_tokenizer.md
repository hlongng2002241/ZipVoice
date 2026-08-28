# Multilingual text front-end for mixed EN/VI/ZH: pretrained tokenizer + embedding

- **Status**: Accepted — **final tokenizer: Qwen2.5-0.5B**, not mBERT (the
  original leading candidate throughout most of this document — see
  [../adr/2026-08-28__choose_qwen25_tokenizer.md](../adr/2026-08-28__choose_qwen25_tokenizer.md)
  for why it changed). The mBERT-specific investigation is kept throughout as
  the historical record of how this conclusion was reached, per this
  project's convention of preserving rejected-alternative analysis rather
  than deleting it.
- **Author**: LongNH (with Claude Code assistance)
- **Date**: 2026-08-28
- **Related**: [2026-08-28__primary_language_conditioning.md](2026-08-28__primary_language_conditioning.md)
  (the `[LANG:xx]` special tokens described there are now part of this proposal's
  vocabulary — see "Language tags as special tokens" below — with proposal 2
  covering the training-time dropout/conditioning mechanics around them)
- **External review**: discussed with Codex (gpt-5.6-sol, high reasoning effort)
  on 2026-08-28; its critique and the author's resulting decisions are folded into
  this revision — see "Revisions after Codex review" at the end of each section
  it touched, and [docs/plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/)
  for the implementation plan.

## Summary

Replace ZipVoice's phonemizer-based text front-end with the tokenizer of an
**existing, already-calibrated multilingual tokenizer** — **Qwen2.5-0.5B**,
the final choice after a real-data bakeoff (mBERT was the leading candidate
for most of this investigation; see "mBERT vs. Qwen2.5-0.5B: final decision"
below for why it was rejected) — no pretrained transformer blocks, no
contextual encoding. Its subwords become the token sequence that feeds directly
into ZipVoice's existing small `text_encoder` (already present in
`zipvoice/models/zipvoice.py`). Per-token duration for the flow-matching model
switches from a uniform average split to a rule-based, per-character script-weighted
split (borrowed from a sibling project, OmniVoice), with special tokens (see below)
explicitly excluded from duration allocation. `[LANG:en]`/`[LANG:vi]`/`[LANG:zh]`
special tokens (see "Language tags as special tokens") are added to the same
vocabulary so a per-utterance language hint can ride alongside the text tokens.

This fixes the original problem — ZipVoice cannot correctly handle mixed
English + Vietnamese text — without adding a pretrained transformer at inference
time. **Correction from the original draft**: this does *not* keep the footprint
"close to today's model" — see "Known risks" for the corrected parameter count.

## Problem

`EmiliaTokenizer` (`zipvoice/tokenizer/tokenizer.py:201-499`) segments input text
into language spans using raw Unicode ranges:

```python
def is_chinese(self, char): return "一" <= char <= "龥"
def is_alphabet(self, char): return ("A"-"Z") or ("a"-"z")   # ASCII only
```

Vietnamese diacritic characters (ư, ệ, ơ, …) fall outside both ranges, land in the
`"other"` bucket, and never get routed to `phonemize_espeak(text, "vi")`. So mixed
English+Vietnamese sentences are mis-segmented and mis-phonemized. (Vietnamese-only
utterances already work today via a separate path — `scripts/vi/m03_perpare_tokens.sh`
uses `EspeakTokenizer(lang="vi")` directly on whole utterances — but that path
doesn't help once English and Vietnamese are mixed in the same sentence, which is
the case this proposal targets.)

## Alternatives considered and rejected

### 1. Fix segmentation only, keep phonemizer

Extending `get_segment()`/`is_alphabet()` to detect Vietnamese Latin+diacritic
spans and route them to `phonemize_espeak(text, "vi")` would fix the immediate bug
with a two-line change. Rejected as the *sole* fix because the user wants to move
away from phonemizer-based G2P entirely (phonemizer cannot express the fact that a
"vi" span and an "en" span use different pronunciation rules for the same Latin
letters without external language tagging, which is brittle for genuinely
code-switched sentences).

### 2. PL-BERT (`temp/PL-BERT/`)

Investigated in detail. Rejected because:
- It is **not a replacement for phonemization** — `temp/PL-BERT/phonemize.py` still
  calls `phonemizer` (espeak backend) to produce the phoneme sequence PL-BERT
  consumes. It adds a contextual encoder *on top of* phonemes, it doesn't remove
  the need for per-language G2P.
- No pretrained checkpoint exists for Vietnamese or Chinese — the public release is
  English-only (Wikipedia, 1M steps). Vietnamese/Chinese support would require
  pretraining from scratch: per-language word tokenizer/vocab, large text corpora,
  and non-trivial compute.
- ZipVoice's architecture has no equivalent splice point. In StyleTTS, PL-BERT
  embeddings feed a separate duration-predictor module. ZipVoice
  (`zipvoice/models/zipvoice.py:132`) is a single flow-matching model with
  `nn.Embedding(vocab_size, text_embed_dim) → text_encoder` and no
  duration-predictor submodule — there's nowhere to plug PL-BERT features in
  without designing a new fusion point from scratch.

### 3. Full pretrained multilingual transformer as text encoder (BERT-VITS2 style)

BERT-VITS2 keeps per-language phonemization for the duration-bearing sequence and
adds a full pretrained BERT forward pass, duplicating each word's contextual
embedding across its phones (`word2ph` alignment) as auxiliary conditioning.
Technically the most proven pattern, but rejected here because it requires
**running the full transformer at inference** (e.g. mBERT-base ~178M total, or
larger), which conflicts with the explicit goal of keeping the model small.

### 4. Replace phonemes with raw BPE subwords, uniform duration split (as originally proposed)

ZipVoice does not have an explicit duration predictor. Per-token duration is
computed by uniform division:

```python
# zipvoice/utils/common.py:252-258
def prepare_avg_tokens_durations(features_lens, tokens_lens):
    avg_token_duration = utt_duration // tokens_lens[i]
    tokens_durations.append([avg_token_duration] * tokens_lens[i])
```

Every token gets an equal-width slice of the frame sequence, then the flow-matching
decoder refines from that crude prior (an E2-TTS/F5-TTS-style trick). This works
because **phonemes have roughly comparable average duration**. It breaks badly for
BPE/WordPiece subwords, whose length in seconds varies enormously by word
("a" vs. "internationalization") — uniform splitting has no way to account for that,
and would need a real learned duration predictor to fix properly (an architecture
change beyond the scope wanted here).

### 5. OmniVoice's duration-estimation approach (`../OmniVoice/omnivoice/utils/duration.py`)

Investigated at the user's request. OmniVoice is architecturally a different
paradigm — an autoregressive LM (Qwen3-0.6B backbone) predicting discrete audio
codec tokens, where duration *emerges* from how many audio tokens the model
generates. Its `RuleDurationEstimator` is only used to guess the **total**
utterance duration up front (for buffering/chunking at inference), via a
per-character phonetic weight table keyed by Unicode script (Latin ≈ 1.0,
CJK ≈ 3.0, digits ≈ 3.5, punctuation ≈ 0.5, combining marks ≈ 0.0, etc. — see
`omnivoice/utils/duration.py:39-162`). It is not a per-token training target and
doesn't directly transfer to ZipVoice's flow-matching architecture.

**What we borrow from it anyway**: the *idea* of a zero-parameter, per-character
script-weight table generalizes cleanly to fixing alternative (4)'s problem. Instead
of `frames_i = total_frames / num_tokens` (uniform), use:

```
weight_i = sum(script_weight(c) for c in characters spanned by token_i)
frames_i = total_frames * weight_i / sum(weight_j for all tokens j)
```

This gives long subword tokens proportionally more frames than short ones, with no
added parameters and no learned duration-predictor module — same spirit as
OmniVoice's estimator, adapted from "estimate total duration from a reference rate"
to "distribute a known total duration across uneven tokens."

## Proposed solution

1. **Tokenizer**: swap ZipVoice's phoneme-based segmentation for an **existing,
   already-calibrated multilingual tokenizer** — the author's explicit preference
   is to reuse a proven pretrained tokenizer rather than train a custom one from
   scratch, since existing tokenizers (mBERT, XLM-R, Qwen, etc.) were carefully
   calibrated on large multilingual corpora. **Final choice: Qwen2.5-0.5B's
   `AutoTokenizer`** (byte-level BPE, vocab 151,669, hidden dim 896) — see
   "mBERT vs. Qwen2.5-0.5B: final decision" below for the real-data comparison
   that decided this over mBERT (WordPiece, vocab 119,547), the leading
   candidate for most of this document's history. Removes the unicode-range
   language-detection logic entirely — the chosen tokenizer's own vocabulary
   already handles EN/VI/ZH (and more) natively, so mixed code-switched text needs
   no external language tagging for segmentation (a separate, coarser
   *pronunciation* hint is still useful — see point 2).
2. **Language tags as special tokens**: add `[LANG:en]`, `[LANG:vi]`, `[LANG:zh]`
   (extend as more languages are added) as reserved vocabulary entries, same
   mechanism as ZipVoice's existing `[S1]`/`[S2]` speaker tags
   (`DialogTokenizer`, `tokenizer.py:502-515`). This merges what was proposal 2's
   design directly into this proposal's vocabulary, since both are decided by the
   same tokenizer-construction step. See
   [2026-08-28__primary_language_conditioning.md](2026-08-28__primary_language_conditioning.md)
   for the training-time dropout mechanics and inference API around these tokens.
3. **Embedding — (a) load pretrained, final choice**:
   - **(a) Load from the pretrained checkpoint (final choice)**: initialize
     ZipVoice's `nn.Embedding(vocab_size, text_embed_dim)`
     (`zipvoice/models/zipvoice.py`) from Qwen2.5-0.5B's
     `get_input_embeddings().weight` (`151669 × 896`), used **at its native
     896-dim, not projected down to 192**. `text_embed_dim` is overridden to
     896 in this mode: `text_encoder` (a `TTSZipformer`) already applies its
     own `in_proj = nn.Linear(in_dim, encoder_dim)` as the first step of its
     forward pass, so setting `text_encoder`'s `in_dim` to 896 reuses that
     existing projection down to `text_encoder_dim` (192) instead of adding a
     second, redundant `nn.Linear` — caught after an initial implementation
     that did add a separate projection layer, once the user pointed out
     `TTSZipformer` already had one. Both the embedding and `text_encoder`
     fine-tune end-to-end. Codex had questioned whether pretrained
     loading offers any real benefit over (b) for a base model whose input
     embeddings were trained jointly with its transformer layers. Sprint 003's
     embedding-structure check
     ([docs/plans/2026-08-28__multilingual_tts_frontend/eval_sets/embedding_structure_check.md](../plans/2026-08-28__multilingual_tts_frontend/eval_sets/embedding_structure_check.md))
     tested this directly for both candidates rather than leaving it as a
     guess: both mBERT's and Qwen2.5-0.5B's raw embeddings carry real
     relational structure without their transformers (mBERT: 0.429 vs. 0.056
     cosine similarity for related vs. unrelated English pairs, 0.324 vs.
     0.169 for Vietnamese; Qwen2.5-0.5B: 0.510 vs. 0.125 for English, 0.199
     vs. 0.147 for Vietnamese) — evidence in favor of (a) regardless of which
     tokenizer won, not a foregone conclusion assumed away.
   - **(b) Train from scratch**: a randomly-initialized `nn.Embedding` sized to
     the chosen tokenizer's vocabulary, trained jointly with the rest of the
     model as ZipVoice does today — no pretrained weights, no projection layer
     needed. Still the fallback if (a)'s advantage doesn't materialize once
     real training is possible.
   - **This remains provisional pending actual training**: the structure check
     shows the pretrained embedding space isn't noise for either candidate,
     but it doesn't replace an actual trained-model WER/quality comparison,
     which needs a full training run this proposal doesn't cover. The new
     `[LANG:xx]` tokens from point 2 have no pretrained counterpart, and in
     the pretrained-embedding case are **not** separately randomly
     initialized as originally planned here: `ZipVoice._make_pretrained_embedding`
     (`zipvoice/models/zipvoice.py`) instead slices them into Qwen2.5-0.5B's
     embedding table's unused padding rows (its table has 151,936 rows but
     the real tokenizer vocab is only 151,665, leaving room for the 4 new
     tags). Those padding rows were never trained on by Qwen either, so
     they're not meaningfully different from a fresh random init for our
     purposes — but they're free (no separate init step needed). Trained
     from scratch (`embed_source="scratch"`), the new tag rows *are* part of
     the single random `nn.Embedding` init, matching the original plan.
4. **No transformer blocks loaded** — ZipVoice's existing `text_encoder` (the small
   transformer already sitting right after the embedding lookup) is left
   unchanged and does the contextualizing work over the initialized vectors.
5. **Duration**: replace `prepare_avg_tokens_durations`'s uniform split with the
   script-weighted split described above, with two corrections from the Codex
   review:
   - **Special tokens (`[LANG:xx]`, `[S1]`/`[S2]`) get zero duration** — they carry
     no acoustic content and must be explicitly excluded from the per-token frame
     allocation, not given a share of frames like a real subword would.
     **Implemented**: `MultilingualTokenizer.zero_duration_mask()` plus
     `prepare_avg_tokens_durations`/`get_tokens_index`
     (`zipvoice/utils/common.py`) now zero out those positions and redistribute
     their frames across the remaining real tokens.
   - The per-character, **script-weighted** split itself (as opposed to zero-ing
     out special tokens) is **not implemented** — real tokens still get an equal
     share of the remaining frames. Per Sprint 003, uniform-vs-weighted duration
     for real tokens remains an open training ablation, not a settled decision.
   - The per-character weight lookup must have defined behavior for WordPiece
     continuation pieces (`##foo`), `[UNK]`, and punctuation — left unspecified in
     the original draft. Moot for now since the weighted split isn't implemented.

## Model selection analysis

Candidates were compared on two axes: **embedding-table parameter count**
(`vocab_size × hidden_dim` — the only cost that matters once transformer blocks are
dropped) and **empirical tokenization efficiency** (tokens produced per character,
lower is better/more compressed) on representative EN, VI, ZH, and mixed
EN+VI+ZH sentences. Prototype scripts and raw output are in `temp/tokenizer/`
(`test_lfm25_tokenizer.py`, `compare_tokenizers.py`, and their JSON reports).

| Model | Embed params (vocab×dim) | EN tok/char | VI tok/char | ZH tok/char | mixed tok/char |
|---|---|---|---|---|---|
| LFM2.5-350M | 67.1M | 0.28 | 0.90 | 1.08 | 0.77 |
| mBERT (leading candidate, later rejected) | 91.8M | 0.36 | 0.31 | 1.00 | 0.45 |
| **Qwen2.5-0.5B (final choice)** | **136.1M** | 0.28 | 0.29 | 0.73 | 0.64 |
| Sailor2-1B | 136.1M | *(same tokenizer as Qwen — no new info)* |
| Qwen3-0.6B | 155.6M | 0.28 | 0.29 | 0.73 | 0.64 |
| Gemma 3 270M | ~170M | untested — gated on HF, needs license acceptance |
| XLM-RoBERTa base | 192.0M | 0.36 | 0.29 | 0.73 | 0.42 |
| BLOOMZ-560m | 256.7M | 0.28 | 0.29 | 0.62 | 0.38 |
| Llama 3.2-1B | 262.7M | untested — ruled out by size alone (see below) |
| Llama 3.1-8B | 525.3M | untested — ruled out by size and overall model scale |

**Llama 3.1/3.2, added 2026-08-28** (user asked to check them): same trap as
Gemma-3-270M/BLOOMZ-560m — a large shared vocab (128,256, tiktoken-style BPE)
multiplied by each model's hidden dim produces a huge embedding table even for
the *smallest* variant (Llama 3.2-1B: 2048-dim → 262.7M, already ~2x
Qwen2.5-0.5B; Llama 3.1-8B: 4096-dim → 525.3M, plus the model itself is far
outside the "small" goal). Ruled out by embedding size alone, without needing
to test tokenization quality — the same reasoning already applied to Gemma/
BLOOMZ. Also gated on HuggingFace (requires accepting Meta's license, unlike
Qwen2.5-0.5B/mBERT which are fully open), confirmed by three failed
unauthenticated download attempts — a real practical friction cost on top of
the size problem.

These numbers are from small hand-written sentences, superseded by the
real-corpus-at-scale numbers in "mBERT vs. Qwen2.5-0.5B: final decision"
below — kept here as the original first-pass filter that ruled out LFM2.5,
XLM-R, BLOOMZ, and Gemma before the real bakeoff.

Key findings from running these tokenizers, not just reading their model cards:

- **LFM2.5-350M's official "16 languages incl. Vietnamese" claim does not hold up
  empirically.** Its byte-level BPE falls back to raw UTF-8 byte fragments for
  Vietnamese diacritics (e.g. `ệ` → `['á»', 'ĩ']`, two meaningless byte-halves) and
  for less-common Chinese characters, giving VI/ZH tokens/char ratios ≥ 0.90 —
  barely better than no compression at all. This tracks with its technical report,
  which lists the tokenizer as optimized for EN/AR/ZH/FR/DE/JA/KO/ES specifically —
  Vietnamese wasn't in that core set even in the "2.5" generation.
- **mBERT's Vietnamese handling is better than its general reputation suggests.**
  Prior assumption (mBERT VI support is "known mediocre") did not survive contact
  with the actual tokenizer output: 0.31 tokens/char, essentially as clean as
  English (0.36), and it segments Chinese per-character, which matches how
  ZipVoice already handles ZH via pinyin-per-character elsewhere in the codebase.
- **BLOOMZ-560m gives the best raw compression (0.38 mixed)** but has the largest
  embedding table of anything tested (256.7M) — like Gemma 3 270M (whose own
  announcement blog states 170M of its 270M total parameters *is* the embedding
  table), almost the entire size budget of these "small" models is vocabulary, not
  transformer capacity — exactly the part kept in this proposal's design. Both are
  poor fits for a size-constrained embedding-only extraction.
- **XLM-R is only marginally better than mBERT** (0.29 vs 0.31 VI, 0.42 vs 0.45
  mixed) at **2.1x the embedding parameters** (192.0M vs 91.8M) — not a good trade.
- **Qwen2.5-0.5B dominates Qwen3-0.6B for this use case**: identical tokenizer and
  fragmentation numbers, but 20M fewer embedding parameters (896 vs. 1024 hidden
  dim), since the extra transformer capacity Qwen3 was built for is discarded
  anyway.
- **Sailor2-1B reuses Qwen2.5's tokenizer verbatim**, so it produces identical
  fragmentation numbers to Qwen2.5/Qwen3 — it offers no new tokenization
  information, and its only theoretical edge (SEA-adapted embedding *weights*,
  not vocab) can't be evaluated without downloading the full 1B model, which is
  also over the stated size budget.

### mBERT vs. Qwen2.5-0.5B: final decision

This section tracks how the recommendation evolved as real evidence
accumulated — kept in full rather than only showing the final answer, since
each reversal was driven by a specific, real finding, not a change of mind.

**Round 1 (embedding structure, mBERT only)**: a check of whether mBERT's raw
input embeddings carry useful structure without their transformer layers
(Codex's specific objection) found yes, clearly for English (0.429 vs. 0.056
cosine similarity, related vs. unrelated word pairs) and, more weakly, for
Vietnamese (0.324 vs. 0.169) — see
[docs/plans/2026-08-28__multilingual_tts_frontend/eval_sets/embedding_structure_check.md](../plans/2026-08-28__multilingual_tts_frontend/eval_sets/embedding_structure_check.md).
This favored loading pretrained weights over training from scratch, but
didn't yet compare against Qwen2.5-0.5B on the same test.

**Round 2 (real Vietnamese ASR transcripts, 173,461 utterances)**: on real
data, mBERT's fragmentation was worse than the earlier synthetic estimate
(0.372 vs. 0.31 tokens/char) and essentially tied Qwen2.5-0.5B (0.368). A
first pass found and patched 27 capitalized Vietnamese words hitting true
`[UNK]` via a new `extra_tokens` parameter on `MultilingualTokenizer` — then a
systematic check showed this was a **structural** gap, not a narrow one: 36 of
68 possible uppercase Vietnamese diacritic-vowel characters are missing from
mBERT's vocabulary entirely, so the 27-word patch only covered what this one
corpus happened to contain. Re-running the embedding-structure check on
Qwen2.5-0.5B found mBERT's Vietnamese-specific structure 3x stronger (0.155
vs. 0.052 gap), while Qwen2.5-0.5B had a structural zero-`[UNK]` guarantee and
slightly better English structure (0.385 vs. 0.373 gap) — a genuine trade-off
at this point, not a clean win either way. Full detail in
[docs/plans/2026-08-28__multilingual_tts_frontend/eval_sets/real_corpus_token_stats.md](../plans/2026-08-28__multilingual_tts_frontend/eval_sets/real_corpus_token_stats.md).

**Round 3 (decisive) — real web-scale text, 10x larger, all three languages**:
the user supplied ~690M-character samples of FineWeb-edu (English) and
FineWeb-2 (Vietnamese, Chinese) — 10x the size of the Vietnamese ASR corpus,
and critically, the **first real test of English and Chinese**, not just
Vietnamese. Results settled the trade-off:

| Metric | mBERT EN | Qwen2.5 EN | mBERT VI | Qwen2.5 VI | mBERT ZH | Qwen2.5 ZH |
|---|---|---|---|---|---|---|
| Tokens/char | 0.272 | **0.257** | **0.367** | 0.374 | 0.938 | **0.692** |
| `[UNK]` rate | 0.969% | **0%** | 0.791% | **0%** | 0.899% | **0%** |
| Vocab utilization | 53.2% | 59.2% | 44.8% | 51.1% | 44.7% | 68.2% |

**mBERT's `[UNK]` problem turned out to be pervasive, not Vietnamese-specific**
— 0.8-1.0% of all tokens across *every* language tested, on real web text.
mBERT is also worse at English compression and **dramatically** worse at
Chinese (0.938 tokens/char — essentially one token per character, almost no
compression — vs. Qwen2.5-0.5B's 0.692). mBERT's one remaining edge (Vietnamese
fragmentation) shrank to a 2% difference, no longer meaningful. Qwen2.5-0.5B
had zero `[UNK]` tokens across every corpus tested, in every language —
confirming its byte-level BPE's structural guarantee, not a lucky coincidence.

**Final decision: Qwen2.5-0.5B**, recorded in
[../adr/2026-08-28__choose_qwen25_tokenizer.md](../adr/2026-08-28__choose_qwen25_tokenizer.md).
The embedding table is 44.3M (48%) larger than mBERT's (136.1M vs. 91.8M) —
a real cost against the size goal, accepted because a structural `[UNK]` risk
across all three target languages, plus meaningfully worse compression for
two of them, outweighs it.

**Revision after Codex review** (kept for context): tokens-per-character was
originally the *only* metric used in this section, and Codex correctly pushed
back on that — it measures sequence-length efficiency, not phonetic adequacy.
The rounds above are the direct result of following that push-back through:
`[UNK]` rate and real-corpus-scale testing turned out to matter more than the
original tokens/char comparison suggested. Codex also pointed out that calling
BERT-VITS2 "precedent" overstated the case: BERT-VITS2 uses BERT's *contextual
outputs* as phone-aligned auxiliary conditioning, not bare input embeddings as
the primary duration-bearing sequence, which is what this proposal actually
does — that similarity was weaker than the original draft implied, for
whichever base tokenizer is used.

### Selection methodology (revised per Codex's suggestion)

Codex's suggested bakeoff methodology, and what actually happened against
each item:

1. **Per-language and code-switch WER/CER** on held-out EN, VI, ZH, and mixed
   EN+VI+ZH utterances — **not done**, requires an actual training run
   (paired audio-text), which is outside this proposal's scope (the training
   run is sprint 004 of the implementation plan). What *was* achievable
   without training — `[UNK]` rate and tokens/char at real corpus scale
   across all three languages — turned out to be decisive on its own (see
   "mBERT vs. Qwen2.5-0.5B: final decision" above).
2. **Pronunciation error analysis on ambiguous words** — the spot-check list
   exists ([docs/plans/2026-08-28__multilingual_tts_frontend/eval_sets/ambiguous_words_en_vi.jsonl](../plans/2026-08-28__multilingual_tts_frontend/eval_sets/ambiguous_words_en_vi.jsonl))
   but pronunciation itself can't be checked without a trained model.
3. **Text-normalization coverage** — corrected, not actually a per-candidate
   axis: `EnglishTextNormalizer`/`ChineseTextNormalizer`
   (`zipvoice/tokenizer/normalizer.py`) run on raw text before any tokenizer
   sees it, so they're reusable regardless of which tokenizer won. The real
   gap (no Vietnamese normalizer exists yet) is a shared prerequisite, not a
   differentiator — see sprint 003's plan doc.
4. **Latency/memory** — done via the embedding-table parameter counts
   throughout this document (91.8M mBERT vs. 136.1M Qwen2.5-0.5B).
5. **Uniform vs. script-weighted duration** — the script-weighted design is
   specified in "Proposed solution" above; a controlled ablation against
   uniform splitting still needs a training run.
6. **Learned duration/alignment component** — not attempted; remains a future
   escalation path if the zero-parameter script-weight heuristic proves
   insufficient once training is possible.

The tokens/char table above (from small hand-written sentences) correctly
served as a first-pass filter (it already disqualified LFM2.5-350M and, on
size alone, Gemma-3-270M/BLOOMZ-560m/Llama-3.x), but the actual mBERT vs.
Qwen2.5-0.5B decision was made on real-corpus-scale `[UNK]` rate and
compression (steps 1 and 4 combined), not the original synthetic tokens/char
numbers alone.

## Known risks / open tradeoffs

- **Corrected size claim (Codex review, updated for the final Qwen2.5-0.5B
  choice).** The original draft claimed this "keeps the size/latency
  footprint close to today's model." That's wrong, and more so now than with
  mBERT: ZipVoice's existing `fm_decoder` + `text_encoder` is 122.8M params
  (measured directly, at `text_embed_dim=192`); adding Qwen2.5-0.5B's 136.1M
  embedding table brings the total to **roughly 259M — more than double**,
  not "close" (mBERT would have been ~215M, about 75% larger — still not
  close, but less so than Qwen2.5-0.5B's actual cost). No separate projection
  layer is added: `text_encoder`'s own existing `in_proj` (present regardless
  of `embed_source`) does the 896→192 step for free once `text_embed_dim` is
  set to the pretrained model's native 896, adding only a negligible ~135K
  params over its scratch-mode size (896×192 vs. 192×192). Inference-time cost
  is still cheap (an embedding lookup is O(1) per token, no transformer forward
  pass), but the parameter count and optimizer-state memory during training are
  materially larger. This size increase was accepted explicitly in
  [../adr/2026-08-28__choose_qwen25_tokenizer.md](../adr/2026-08-28__choose_qwen25_tokenizer.md)
  as the right trade against mBERT's structural `[UNK]` risk. Option 3(b)
  (train from scratch, sized to the chosen tokenizer's vocab at ZipVoice's
  native 192-dim, no projection) would avoid the size cost entirely but gives
  up the embedding-structure advantage confirmed in the structure check.
- **Special/control tokens need explicit duration handling.** Covered under
  "Proposed solution" above — `[LANG:xx]` and speaker tags must get zero duration,
  which the current duration-allocation code doesn't support yet.
- **Script weights can't distinguish EN from VI.** Both use Latin script, so a
  per-character script-weight table (Latin ≈ 1.0) fixes the long-token-vs-short-token
  duration problem but says nothing about which language a Latin-script span
  should sound like — that's what the `[LANG:xx]` tag is for, not the duration
  heuristic.
- **Loses contextualization.** A bare embedding table gives one static vector per
  subword regardless of sentence context — the actual prosody/homograph
  disambiguation gains PL-BERT/BERT-VITS2 report come from their transformer
  layers, which this proposal explicitly drops for size reasons. Any
  disambiguation ZipVoice achieves will have to come from its own small
  `text_encoder` learning it from paired TTS data, not from mBERT's pretraining.
- **Implicit G2P.** Dropping phonemes means the model must learn pronunciation
  purely from paired text-audio data, with no explicit linguistic supervision.
  This is a bigger risk for English specifically (highly irregular
  spelling-to-sound mapping) if the English training set isn't large. Should be
  watched in evaluation, particularly on rare/irregular words and proper nouns.
- **Duration weights are hand-tuned, not learned.** The script-weight table is a
  reasonable heuristic (and it's zero-parameter, matching the size goal) but it is
  not adapted to this project's actual training data the way a learned duration
  predictor would be. If mispronunciation/timing artifacts show up in practice,
  a small learned duration predictor is the natural escalation path.

## Next steps

Decisions from this proposal are recorded in
[../adr/2026-08-28__adopt_pretrained_multilingual_tokenizer.md](../adr/2026-08-28__adopt_pretrained_multilingual_tokenizer.md)
(reuse an existing pretrained tokenizer),
[../adr/2026-08-28__language_hint_as_special_tokens.md](../adr/2026-08-28__language_hint_as_special_tokens.md)
(`[LANG:xx]` as special tokens), and
[../adr/2026-08-28__choose_qwen25_tokenizer.md](../adr/2026-08-28__choose_qwen25_tokenizer.md)
(the final tokenizer: Qwen2.5-0.5B over mBERT). `MultilingualTokenizer`
(`zipvoice/tokenizer/multilingual_tokenizer.py`) already defaults to
Qwen2.5-0.5B. See the implementation plan at
[docs/plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/)
for the sequenced breakdown (eval sets first, then vocabulary/special-token
prototype, then the bakeoff, then the frozen-front-end training run).
