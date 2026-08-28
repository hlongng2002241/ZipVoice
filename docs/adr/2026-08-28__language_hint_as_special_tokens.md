# Represent the primary-language hint as special vocabulary tokens

- **Date:** 2026-08-28
- **Status:** Accepted
- **Related:** [../proposals/2026-08-28__primary_language_conditioning.md](../proposals/2026-08-28__primary_language_conditioning.md),
  depends on [2026-08-28__adopt_pretrained_multilingual_tokenizer.md](2026-08-28__adopt_pretrained_multilingual_tokenizer.md)
  (both add entries to the same vocabulary at the same time)

## Context

OmniVoice lets a caller optionally pass a primary language for an utterance, or
omit it for automatic behavior, implemented as a literal control string
(`<|lang_start|>vi<|lang_end|>`) inside the natural-language prompt fed to its
autoregressive LLM. ZipVoice has no such prompt channel — it is a
non-autoregressive flow-matching model that consumes a fixed token vocabulary,
not free-form text. The author wants an equivalent per-utterance language hint
for ZipVoice to help disambiguate Latin-script spans that could plausibly be
either English or Vietnamese once the tokenizer stops doing per-language
Unicode-range segmentation (see
[2026-08-28__adopt_pretrained_multilingual_tokenizer.md](2026-08-28__adopt_pretrained_multilingual_tokenizer.md)).

ZipVoice already has a working precedent for reserved control tokens:
`DialogTokenizer` treats bracketed strings like `[S1]`/`[S2]` (speaker tags) as
single vocabulary entries mapped through the same `token2id` table as ordinary
text tokens. An external review (Codex, gpt-5.6-sol, high reasoning effort, run
2026-08-28) proposed an alternative: a dedicated global conditioning embedding
injected into the model at every sequence position (or directly into the
decoder), rather than a token in the sequence, arguing this avoids ambiguity
about whether a single prefix token's influence actually propagates through the
whole sequence via attention.

## Decision

We will implement the primary-language hint as reserved special tokens
(`[LANG:en]`, `[LANG:vi]`, `[LANG:zh]`, extensible per language), prepended to
the token sequence, reusing the same mechanism already used for `[S1]`/`[S2]`.
We explicitly considered and rejected Codex's dedicated-embedding alternative.

## Alternatives Considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Special vocabulary token (chosen) | Reuses an existing, working mechanism (`[S1]`/`[S2]`) with no new architecture; implemented alongside the tokenizer/vocabulary work already underway | Effectiveness of a single prefix token's influence over the whole sequence via bidirectional attention is plausible but unverified — must be validated, not assumed | Simpler, consistent with existing precedent, and the author's explicit preference over adding a new conditioning pathway |
| Dedicated global conditioning embedding (Codex's suggestion) | Guarantees the language signal reaches every position directly, independent of attention learning to propagate it; cleaner separation from the acoustic-duration-bearing token stream | New conditioning pathway not used anywhere else in ZipVoice; more implementation surface (where to inject it, how it composes with existing conditioning) for a benefit that is itself unproven without also validating it | Rejected in favor of reusing the proven, simpler `[S1]`/`[S2]`-style mechanism |

## Consequences

- Easier: no new conditioning pathway in the model architecture; the vocabulary
  addition happens in the same step as the tokenizer work in
  [2026-08-28__adopt_pretrained_multilingual_tokenizer.md](2026-08-28__adopt_pretrained_multilingual_tokenizer.md).
- Harder: because it's a token in the sequence rather than a separate
  conditioning signal, it must be explicitly excluded from the per-token
  duration allocation (it carries no acoustic content) — the current duration
  code has no such concept yet and needs it added.
- Risk accepted, to be validated in the implementation plan rather than assumed:
  a single prepended tag token's influence on the full sequence's text condition
  depends on the bidirectional `text_encoder`'s attention actually propagating
  it — this needs an explicit probe (compare outputs across different tags on
  identical text) before being relied on.
- **Decided (2026-08-28, amendment)**: the "no hint given" case always uses an
  explicit `[LANG:auto]` sentinel, never omission — mirroring OmniVoice's
  literal `"None"` tag exactly. The label-dropout rate during training remains
  an open experimental question, not settled by this ADR.
- **Decided (2026-08-28, amendment)**: the true (non-dropout) label is read
  directly from the training dataset's existing per-text-audio-pair ground
  truth — not inferred from text. A text-based majority-word-vote classifier
  was prototyped to handle a hypothesized "corpus metadata may be noisy" case,
  then deleted once the author confirmed their dataset already provides
  reliable per-pair labels, making that inference step unnecessary. See
  [../plans/2026-08-28__multilingual_tts_frontend/eval_sets/primary_language_definition.md](../plans/2026-08-28__multilingual_tts_frontend/eval_sets/primary_language_definition.md).
