# Adopt an existing pretrained multilingual tokenizer, drop phonemizer

- **Date:** 2026-08-28
- **Status:** Accepted
- **Related:** [../proposals/2026-08-28__mbert_multilingual_tokenizer.md](../proposals/2026-08-28__mbert_multilingual_tokenizer.md)

## Context

ZipVoice's `EmiliaTokenizer` segments text into language spans using raw Unicode
character ranges, then phonemizes each span per language (espeak for English,
pinyin for Chinese). Vietnamese diacritic characters fall outside the ranges the
segmenter checks for, so mixed English+Vietnamese text is mis-segmented and
mis-phonemized. The author wants to move off phonemizer-based G2P entirely for
this front end rather than patch the segmentation bug, since phonemizer cannot
express that a Latin-script span should be pronounced as English vs. Vietnamese
without external language tagging — brittle for genuinely code-switched text.
The author also wants to keep the model's parameter/latency footprint small,
which rules out running a full pretrained transformer at inference.

Two tokenizer sourcing options were on the table: train a custom tokenizer on
project-specific EN/VI/ZH data, or reuse an existing pretrained multilingual
tokenizer (mBERT, XLM-R, a small LLM's tokenizer, etc.). A third option — using a
phoneme-level contextual encoder such as PL-BERT — was investigated and found
unworkable (no Vietnamese/Chinese checkpoint exists, and it still requires
per-language phonemization upstream, so it doesn't remove the underlying problem).

## Decision

We will replace ZipVoice's phonemizer-based segmentation with the vocabulary and
tokenizer of an **existing, already-calibrated pretrained multilingual model**,
not a custom-trained tokenizer, and not a full pretrained transformer running at
inference. Only the tokenizer and (optionally) the input embedding table are
reused — no transformer blocks are loaded, keeping inference cost close to a
lookup. The specific model (mBERT vs. alternatives) is decided by the bakeoff in
the linked proposal, not fixed by this ADR.

**Resolved (2026-08-28)**: the bakeoff concluded with **Qwen2.5-0.5B**, not
mBERT — see [2026-08-28__choose_qwen25_tokenizer.md](2026-08-28__choose_qwen25_tokenizer.md)
for the real-data findings and reasoning. This ADR's core decision (reuse an
existing pretrained tokenizer, not a custom or full-transformer one) is
unaffected by which specific model won.

## Alternatives Considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Train a custom EN/VI/ZH tokenizer from scratch | Tuned exactly to project data and language mix | Requires curating a large multilingual corpus, training infrastructure, and re-validating segmentation quality from zero, duplicating work existing tokenizers already did well | Existing multilingual tokenizers were already carefully calibrated on large corpora; reusing one is lower-risk and faster |
| PL-BERT (phoneme-level contextual encoder) | Reported prosody/MOS gains in its own paper | Not a phonemizer replacement (still needs per-language G2P upstream); no Vietnamese/Chinese checkpoint exists; no splice point in ZipVoice's architecture | Doesn't solve the stated problem and requires pretraining a new model from scratch for VI/ZH |
| Full pretrained multilingual transformer as text encoder (BERT-VITS2-style) | Best-precedented pattern for pretrained-embedding TTS conditioning | Requires running a full transformer (~100M+ params) at inference | Conflicts with the explicit goal of keeping the model small |
| Fix only the Unicode segmentation bug, keep phonemizer | Minimal, low-risk, ~2-line change | Doesn't remove the deeper limitation: phonemizer can't express per-span language identity for code-switched Latin-script text | Author wants to move off phonemizer-based G2P, not patch around it |

## Consequences

- Easier: mixed-language segmentation is handled by the chosen tokenizer's own
  vocabulary instead of hand-maintained Unicode-range logic; no per-language G2P
  pipeline to maintain going forward.
- Harder: the model must learn pronunciation implicitly from paired audio/text
  data instead of relying on explicit phoneme supervision — a real quality risk
  for irregular English spelling if training data is limited, tracked as an open
  risk in the linked proposal.
- Harder: token-to-frame duration allocation can no longer assume roughly uniform
  per-token duration (true for phonemes, false for variable-length subwords).
  Zero-duration handling for the special tokens introduced by
  [2026-08-28__language_hint_as_special_tokens.md](2026-08-28__language_hint_as_special_tokens.md)
  is **implemented**. The script-weighted duration scheme for *real* subword
  tokens described in the linked proposal is **not implemented** — real tokens
  still split the remaining frames evenly, per Sprint 003's framing of
  uniform-vs-weighted as an open training ablation rather than a requirement.
- Neutral: the exact tokenizer and embedding-initialization strategy (pretrained
  vs. from-scratch) remain open, to be settled by the bakeoff in the implementation
  plan, not by this ADR.
