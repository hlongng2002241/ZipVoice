# Real-corpus token statistics (Sprint 003, resolves the corpus-access blocker)

The user provided access to two real Vietnamese ASR-transcript corpora:
- `/data4/audio/youtube/exp/01/audio.exist.denoise.info.vol.diarize.mos.asr.abs.final.jsonl`
  (52,734 utterances — YouTube talk-show/interview style)
- `/data4/audio/toptop/exp/04/metadata.dur.norm.cut.dur.sv.wer.punc.hard_merge.dur.wer.dur.flac.jsonl`
  (120,727 utterances — short-video/live-selling style)

**173,461 utterances, 25.68M tokens, 69.1M characters total.** This is the
first real-data validation available for this plan — everything before this
was either hand-written seed sentences or static tokenizer-card claims. Tool:
`zipvoice/bin/analyze_token_stats.py`.

## 1. Real fragmentation is worse than the earlier synthetic estimate

| | Synthetic hand-written VI sentence (earlier) | Real ASR transcripts (this analysis) |
|---|---|---|
| mBERT tokens/char | 0.31 | **0.372** |
| Qwen2.5-0.5B tokens/char | 0.29 | **0.368** |

Real, informal, spoken-register Vietnamese fragments noticeably more than the
earlier clean hand-written test sentences suggested — worth remembering as a
general caution about validating tokenizer claims on synthetic text. **On real
data, mBERT and Qwen2.5-0.5B are effectively tied** (0.372 vs. 0.368), much
closer than the synthetic comparison implied.

## 2. A real, narrow, fixable mBERT vocabulary gap — found and fixed

mBERT showed 1,496 true `[UNK]` tokens (0.0058% of all tokens) — small, but
worth tracing rather than dismissing. Root cause, found by scanning every word
that produced an `[UNK]`: mBERT's cased WordPiece vocabulary is missing a
specific, small set of **capitalized Vietnamese interjections and
emphasis-capitalized words** as usable pieces — their lowercase forms exist
(`"ừ"` → `['ừ']`), but the capitalized forms don't (`"Ừ"` → `['[UNK]']`).
27 distinct words account for all 1,496 occurrences: mostly single-character
discourse particles (`Ờ`, `Ừ`, `Ồ`, `Ơ`, `Ạ` — "uh-huh"/"oh"/"yeah"-type
fillers extremely common in casual spoken Vietnamese but rare in mBERT's
Wikipedia training data) plus a few ALL-CAPS emphasis words (`VIỆT`, `KIẾM`,
`TIỀN`, `CẦN`, `GIỜ`).

**Fixed**: `MultilingualTokenizer` now accepts an `extra_tokens` parameter
(added via `tokenizer.add_tokens`, distinct from the `[LANG:xx]` control
tokens added via `add_special_tokens`) and ships the discovered list as
`MultilingualTokenizer.VI_UNK_GAP_TOKENS` — not applied by default (a generic
multilingual tokenizer class shouldn't silently bake in one base model's
Vietnamese-specific fix), but ready to pass explicitly. Re-running the same
173,461-utterance analysis with `extra_tokens=VI_UNK_GAP_TOKENS` confirms
**`[UNK]` count drops from 1,496 to exactly 0**, with no change to total token
count (each word maps 1:1 from "UNK" to "real token", not a re-segmentation).

**Qwen2.5-0.5B has zero `[UNK]` by construction** (byte-level BPE always has a
byte fallback; WordPiece does not) and, checked directly, already handles all
three of `Ừ`/`Ờ`/`Ồ` as single tokens without needing a fix.

### Correction (2026-08-28): the gap is structural, not a narrow one-off

The 27-word patch above was challenged (correctly) as covering only what this
*one* corpus happened to contain, not the actual size of the problem. Testing
systematically confirms it's much bigger: generated every uppercase Vietnamese
vowel+tone-mark combination (68 total) and checked each against mBERT's
vocabulary directly (not just what appeared in this corpus):

**36 of 68 (53%) systematically-generated uppercase Vietnamese diacritic
vowels are missing from mBERT's vocabulary** as usable tokens — e.g. `Ạ`, `Ầ`,
`Ẫ`, `Ậ`, `Ắ`, `Ằ`, `Ẳ`, `Ẵ`, `Ặ`, `Ẻ`, `Ẽ`, `Ế`, `Ề`, `Ể`, `Ễ`, `Ệ`, `Ỉ`, `Ị`,
`Ỏ`, `Ồ`, `Ổ`, `Ỗ`, `Ộ`, `Ụ`, `Ỳ`, `Ỵ`, `Ỷ`, `Ỹ`, plus `Ù`/`Û`/`Ĩ`/`Ĭ`/`Ŏ`/`Ũ`/
`Ŭ`/`Ŷ`. The pattern correlates with specific tone marks (hook-above, tilde,
dot-below are mostly missing in uppercase) rather than being random —
consistent with these being rarer in mBERT's Wikipedia-based training data,
especially capitalized.

**This means the 27-word patch is not a real fix, only a band-aid for this
specific corpus.** A different or larger corpus will keep surfacing new `[UNK]`
tokens from this same 36-character gap indefinitely; `extra_tokens` can be
extended reactively, but there's no way to be confident the list is ever
complete. Qwen's byte-level BPE has no equivalent risk *class* at all — it is
not that Qwen "happens to" cover these characters, it structurally cannot
produce `[UNK]` for any Unicode input, by construction.

### Does the same "raw embedding has real structure" evidence hold for Qwen?

The embedding-structure check in
[embedding_structure_check.md](embedding_structure_check.md) was run only on
mBERT. Re-ran the identical related/unrelated cosine-similarity test on
Qwen2.5-0.5B's raw (non-contextualized) embedding table before drawing a
combined conclusion:

| | mBERT | Qwen2.5-0.5B |
|---|---|---|
| EN related pairs | 0.429 | **0.510** |
| EN unrelated pairs | 0.056 | 0.125 |
| EN structure gap | 0.373 | **0.385** |
| VI related pairs | **0.324** | 0.199 |
| VI unrelated pairs | 0.169 | 0.147 |
| VI structure gap | **0.155** | 0.052 |

**Neither candidate is a clean winner.** Qwen2.5-0.5B's raw embedding space
has slightly *better* English structure than mBERT's, but **mBERT's
Vietnamese-specific structure (0.155 gap) is 3x stronger than Qwen2.5-0.5B's
(0.052 gap)** — plausibly because mBERT's training explicitly balances
per-language exposure across 104 languages, while Qwen's pretraining, though
larger overall, is more EN/ZH-weighted with Vietnamese as a smaller fraction.

### Revised comparison table (real corpus, all three candidates)

| Metric | mBERT (patched) | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---|---|---|
| Embedding params | **91.8M** | 136.1M | 155.6M |
| Tokens/char (real data) | 0.372 | **0.368** | 0.369 |
| `[UNK]` risk | Structural gap (36/68 chars), patch is reactive/incomplete | **None, by construction** | **None, by construction** |
| Vocab utilization | 15.34% | 14.12% | 12.21% |
| Singleton token % | **21.08%** | 26.38% | 26.50% |
| Raw-embedding EN structure | 0.373 gap | **0.385 gap** | not tested |
| Raw-embedding VI structure | **0.155 gap** | 0.052 gap | not tested |

Qwen2.5-0.5B dominates Qwen3-0.6B on every axis that matters here (same
tokenization, same zero-UNK guarantee, 20M fewer embedding params) — the real
decision is mBERT vs. Qwen2.5-0.5B, and it is now a genuine trade-off between
**smaller + stronger Vietnamese-specific pretrained structure, with a
real-but-patchable UNK risk** (mBERT) vs. **structurally UNK-proof + slightly
better English structure, at +44.3M (+48%) embedding params and weaker
Vietnamese-specific structure** (Qwen2.5-0.5B). This is not a call the data
alone resolves cleanly — see the chat response for the question back to the
user on how to weigh it.

## 3. Vocabulary utilization and the "uniform distribution" question

The plan originally asked whether token distributions should be made "as
uniform as possible" for training. **Measured on this real corpus, the answer
is that literal uniformity is neither observed nor a real objective — and
that's expected, not a tokenizer defect:**

- Only **18,338 of mBERT's 119,578 tokens (15.3%)** are ever used by this
  Vietnamese-only corpus (expected: mBERT's vocabulary spans 104+ languages;
  the unused ~85% is mostly non-Vietnamese vocabulary still needed once
  English and Chinese training data are mixed in).
- The observed token-frequency distribution has **9.36 bits of entropy**,
  against a theoretical maximum of 14.16 bits if usage were perfectly uniform
  across the 18,338 used tokens — i.e. **66.1% of the uniform maximum**. The
  top 5 tokens alone (`.`, `,`, `là`, `cái`, `có`) already account for **8.8%**
  of all 25.68M token occurrences.
- This is [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law) — a small
  number of function words/punctuation dominate any natural-language corpus,
  in any language, under any reasonable subword tokenizer. It is not something
  a different tokenizer choice fixes, and literal uniformity was never a
  realistic target: it would require, e.g., punctuation and "là" (is) to
  appear about as often as rare content words, which contradicts how language
  is actually used.

**The real, actionable risk behind the "uniform is better" intuition** is not
distribution shape but **undertraining of rare tokens**: 3,865 of the 18,338
used tokens (21.1%) are *singletons* — seen exactly once across the entire
25.68M-token corpus. A token's embedding row that gets one gradient update in
an epoch won't learn much from this corpus alone. This is a real concern, and
it's exactly what motivates the recommendation already made in
[embedding_structure_check.md](embedding_structure_check.md): **loading
mBERT's pretrained embedding table gives every token — including these rare
ones — a meaningful starting point instead of random noise**, which is the
correct mitigation for this risk, not chasing an artificially uniform
frequency distribution.

## Recommendation status: settled — Qwen2.5-0.5B (was reopened 2026-08-28)

The mBERT recommendation from earlier in Sprint 003 was reopened here after
the systematic check above showed the `[UNK]` issue is a structural gap
(36/68 uppercase Vietnamese diacritic vowels), not the narrow 27-word issue
first reported — and the embedding-structure comparison showed mBERT and
Qwen2.5-0.5B each winning on a different axis (Vietnamese-specific pretrained
structure vs. structural UNK-immunity + smaller English-structure edge). That
trade-off was resolved by the user in favor of Qwen2.5-0.5B; see
[../../../adr/2026-08-28__choose_qwen25_tokenizer.md](../../../adr/2026-08-28__choose_qwen25_tokenizer.md)
for the final decision and rationale.

What's unaffected either way:
- Real fragmentation (0.368-0.372 tokens/char across all three candidates) is
  worse than the earlier synthetic-sentence estimate suggested, but the three
  candidates are close enough to each other that fragmentation alone doesn't
  decide anything.
- **Still open, unaffected by this analysis**: the actual training-based
  WER/pronunciation validation (this is real *text* data only — no
  small-scale training run was possible here, since that requires running
  the full ZipVoice training pipeline, not just tokenizer analysis).
