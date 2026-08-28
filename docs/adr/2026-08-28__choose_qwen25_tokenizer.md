# Choose Qwen2.5-0.5B's tokenizer over mBERT for the multilingual text front-end

- **Date:** 2026-08-28
- **Status:** Accepted
- **Related:** [../proposals/2026-08-28__mbert_multilingual_tokenizer.md](../proposals/2026-08-28__mbert_multilingual_tokenizer.md);
  refines [2026-08-28__adopt_pretrained_multilingual_tokenizer.md](2026-08-28__adopt_pretrained_multilingual_tokenizer.md),
  which decided to reuse an existing pretrained tokenizer but left the
  specific model open, pending a bakeoff

## Context

[2026-08-28__adopt_pretrained_multilingual_tokenizer.md](2026-08-28__adopt_pretrained_multilingual_tokenizer.md)
established that ZipVoice's multilingual front end would reuse an existing
pretrained tokenizer's vocabulary rather than train a custom one or run a full
transformer at inference, leaving the specific model choice to a bakeoff.
mBERT (`bert-base-multilingual-cased`) was the leading candidate throughout
most of that investigation: smallest embedding table (91.8M params) among
tokenizers with good measured EN/VI/ZH tokenization, and real TTS-integration
precedent (BERT-VITS2-style).

Real-data testing progressively undermined that lead:

1. On the user's own 173,461-utterance Vietnamese ASR-transcript corpus,
   mBERT showed a small (0.0058%) `[UNK]` rate, traced to 27 specific
   capitalized Vietnamese interjection words missing from its vocabulary. A
   systematic check of all 68 possible uppercase Vietnamese diacritic-vowel
   characters found **36 (53%) missing entirely** — not a narrow, patchable
   gap but a structural one, likely to keep surfacing new failures on new
   data.
2. Testing at 10x scale (690M characters each) on real web-scale text
   (FineWeb-2/FineWeb-edu, not just clean spoken-register ASR transcripts) for
   **all three target languages** confirmed the `[UNK]` problem is not
   Vietnamese-specific: mBERT's `[UNK]` rate was 0.969% (English), 0.791%
   (Vietnamese), and 0.899% (Chinese) — pervasive across the board.
3. The same 10x-scale test also found mBERT compresses noticeably worse than
   Qwen2.5-0.5B for English (0.272 vs. 0.257 tokens/char) and dramatically
   worse for Chinese (0.938 vs. 0.692 tokens/char — mBERT is close to one
   token per character, essentially no compression). mBERT retained only a
   marginal edge on Vietnamese fragmentation (0.367 vs. 0.374), not enough to
   offset the rest.
4. Qwen2.5-0.5B (byte-level BPE) had **zero `[UNK]` tokens across every
   corpus tested** — not "happened to cover it," but structurally
   incapable of producing `[UNK]` for any Unicode input, by construction.

Full data: [../plans/2026-08-28__multilingual_tts_frontend/eval_sets/real_corpus_token_stats.md](../plans/2026-08-28__multilingual_tts_frontend/eval_sets/real_corpus_token_stats.md).

An earlier, smaller-scale embedding-structure check (cosine similarity of
raw, non-contextualized embeddings between related/unrelated word pairs) had
found mBERT's Vietnamese-specific structure (0.155 gap) stronger than
Qwen2.5-0.5B's (0.052 gap), while Qwen2.5-0.5B's English structure was
slightly better (0.385 vs. 0.373). This was weighed against the `[UNK]` and
compression findings above and did not change the conclusion.

## Decision

We will use **Qwen2.5-0.5B's tokenizer** (byte-level BPE, vocab 151,669,
hidden dim 896) as `MultilingualTokenizer`'s default base model, replacing
mBERT. The embedding table to load/fine-tune is now `151669 × 896` (~136.1M
params) projected via `Linear(896, 192)`, rather than mBERT's
`119547 × 768` (~91.8M params) via `Linear(768, 192)`.

## Alternatives Considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| mBERT (original leading candidate) | Smaller embedding table (91.8M vs. 136.1M, -44.3M); stronger Vietnamese-specific raw-embedding structure in the small-scale check | Structural `[UNK]` gap confirmed pervasive (~0.8-1.0%) across all three target languages at real scale, not just a patchable Vietnamese-specific list; worse compression for English and much worse for Chinese | The `[UNK]` risk turned out to be structural and universal, not the narrow issue it first appeared to be — a real correctness risk that outweighs the size and Vietnamese-structure advantages |
| Qwen3-0.6B | Same tokenizer family as Qwen2.5-0.5B | 20M more embedding params (155.6M vs. 136.1M) for statistically indistinguishable tokenization quality | Already established as dominated by Qwen2.5-0.5B before this decision; not re-litigated here |
| Qwen2.5-0.5B (chosen) | Zero `[UNK]` by construction (byte-level BPE), confirmed across every corpus tested including at 10x real-web-scale; better compression for English and (substantially) Chinese | +44.3M (+48%) embedding params vs. mBERT; weaker Vietnamese-specific raw-embedding structure in the small-scale check | Structural correctness (no `[UNK]` risk class at all) and better real-data compression for 2 of 3 languages outweigh the size cost and the Vietnamese-structure gap, which was measured on a small hand-picked word list rather than at corpus scale |

## Consequences

- Easier: no `[UNK]`-gap patching (`VI_UNK_GAP_TOKENS`) needed for the default
  configuration — kept in `MultilingualTokenizer` only for reference/if
  reverting to mBERT.
- Harder: the embedding table is 48% larger than the original mBERT plan
  (136.1M vs. 91.8M params), a real cost against the project's stated goal of
  keeping the model small — accepted as the correct trade against the
  `[UNK]` correctness risk.
- Neutral: `MultilingualTokenizer`'s implementation was already generic over
  the base model (mBERT support is retained, just no longer the default) — no
  architecture change was needed to make this switch, only a default-parameter
  and documentation change, confirmed by the existing test suite still passing
  unchanged for the mBERT-specific tests.
- The mBERT-specific raw-embedding-structure advantage for Vietnamese (0.155
  vs. 0.052 gap) is a known, accepted loss — not verified whether it
  translates into a measurable end-to-end quality difference, since that
  requires the training run this decision is a prerequisite for, not
  something resolved by this ADR.
