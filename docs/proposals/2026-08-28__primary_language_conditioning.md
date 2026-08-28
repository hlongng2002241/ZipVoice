# Optional primary-language conditioning tag (explicit language hint, auto by default)

- **Status**: Proposed, not yet implemented
- **Author**: LongNH (with Claude Code assistance)
- **Date**: 2026-08-28
- **Related**: [2026-08-28__mbert_multilingual_tokenizer.md](2026-08-28__mbert_multilingual_tokenizer.md)
  now owns the actual vocabulary addition — the `[LANG:xx]` special tokens are
  defined as part of that proposal's tokenizer-construction step, since both are
  decided by the same vocabulary. This proposal covers the training-time
  mechanics (dropout, data labeling) and inference API around those tokens.
- **External review**: discussed with Codex (gpt-5.6-sol, high reasoning effort)
  on 2026-08-28. Codex's main suggestion — replacing the special-token approach
  with a dedicated global conditioning embedding injected at every position —
  was considered and **rejected**: the author's explicit decision is to keep
  `[LANG:xx]` as special tokens, consistent with the existing `[S1]`/`[S2]`
  precedent and simpler to implement alongside proposal 1's vocabulary work.
  Codex's other critiques (duration handling, validation rigor, label-noise
  caution) are incorporated below. See
  [docs/plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/)
  for the implementation plan.

## Summary

Add an optional per-utterance "primary language" hint that a user can pass at
inference time (`lang="vi"`, `lang="en"`, `lang="zh"`, or omitted for automatic
behavior), modeled directly on how OmniVoice exposes the same idea. Implemented as
a reserved special token (e.g. `[LANG:vi]`) prepended to the token sequence,
reusing ZipVoice's existing bracket-tag tokenizer infrastructure. Requires a
training-time change (per-utterance ground-truth language label + random tag
dropout) so the model still works when no hint is given.

## Motivation

While investigating the mixed EN/VI/ZH tokenization problem
([2026-08-28__mbert_multilingual_tokenizer.md](2026-08-28__mbert_multilingual_tokenizer.md)),
the user asked whether ZipVoice can offer the same "choose primary language, or
leave it to auto" control that OmniVoice exposes at inference time. Investigation
of `../OmniVoice/omnivoice/models/omnivoice.py` confirmed OmniVoice does exactly
this, and that the same idea transfers to ZipVoice with a small amount of new
machinery — it composes naturally with the shared-vocabulary tokenizer, since an
explicit language hint helps resolve exactly the kind of ambiguity a shared
EN/VI/ZH vocabulary introduces (e.g. a Latin-script word that could plausibly be
read as English or Vietnamese).

## How OmniVoice does it

It is not a separate conditioning mechanism — it's a control tag embedded
directly in the text prompt fed to the LLM (`omnivoice/models/omnivoice.py:1104`):

```python
lang_str = lang if lang else "None"
style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
```

`generate(text, language="vi")` produces a prompt containing
`<|lang_start|>vi<|lang_end|>`; `generate(text)` (no language given) produces
`<|lang_start|>None<|lang_end|>`. This is cheap for OmniVoice specifically because
it's an autoregressive LLM with a full natural-language vocabulary — the tag is
just more text tokens, and the model must have seen both the true-label and the
literal `"None"` string during training (a classifier-free-guidance-style label
dropout) to have learned to fall back to inferring language from content when the
tag is absent/`None`.

## Does ZipVoice have the pieces for this? Yes

ZipVoice is architecturally different (non-autoregressive flow-matching, not an
autoregressive LLM with a natural-language prompt), so it can't reuse OmniVoice's
"just write more text" trick verbatim. But it already has an equivalent
mechanism for reserved control tokens: `DialogTokenizer`
(`zipvoice/tokenizer/tokenizer.py:502-515`) already treats bracketed strings like
`[S1]`/`[S2]` (speaker tags) as single reserved vocabulary entries, recognized by
`is_tag()` and routed through `split_segments()` in `EmiliaTokenizer`
(`tokenizer.py:387-499`), then mapped through `token2id` like any other token.

Adding `[LANG:en]` / `[LANG:vi]` / `[LANG:zh]` / `[LANG:auto]` (the last used
for both the training-time dropout case and the inference-time no-hint
default — see "Decided" below) as new special tokens, prepended to the token
sequence for an utterance, is the same pattern already
used for `[S1]`/`[S2]`, just applied to language instead of speaker. Confirmed
there is currently **no** existing per-utterance mechanism like this — ZipVoice's
only existing `--lang` flag (`zipvoice/bin/infer_zipvoice.py:154`, consumed at
`infer_zipvoice.py:789`) selects the whole-utterance espeak language for
`EspeakTokenizer` in single-language mode; it doesn't apply to `EmiliaTokenizer`'s
per-character auto segmentation and has no "auto" behavior of its own.

## Proposed design

1. **Vocabulary**: add `[LANG:en]`, `[LANG:vi]`, `[LANG:zh]` (extend as more
   languages are added) as reserved tokens — **now specified as part of proposal
   1's tokenizer-construction step**, since both proposals add entries to the
   same vocabulary at the same time. Same mechanism as `[S1]`/`[S2]` today.
2. **Tokenizer**: prepend the corresponding tag token to the token sequence when a
   language is specified; prepend nothing (or a dedicated `[LANG:auto]` token —
   see open question below) when it isn't.
3. **Training data — resolved 2026-08-28**: each training utterance's
   ground-truth primary-language label is read directly from the dataset's
   existing per-text-audio-pair metadata (confirmed available by the author) —
   this is a data-pipeline change (identify and read the correct manifest
   field), not a language-inference problem. Codex's caution that corpus-level
   metadata "may be noisy" motivated an earlier plan to infer the label from
   text with a word-count classifier; that classifier was built and then
   deleted once it became clear the dataset already provides reliable per-pair
   ground truth, making inference unnecessary. See
   [../plans/2026-08-28__multilingual_tts_frontend/eval_sets/primary_language_definition.md](../plans/2026-08-28__multilingual_tts_frontend/eval_sets/primary_language_definition.md).
4. **Label dropout**: during training, randomly replace the true tag with the
   "auto" case for some fraction of utterances (OmniVoice's approach) so the model
   doesn't become dependent on always receiving a hint, and still performs
   acceptably when the user omits `lang`. **Codex correction, accepted**: dropout
   must be applied dynamically per training step/epoch (sampled fresh each time an
   utterance is seen), not baked in once during manifest tokenization — a
   fixed-at-prep-time dropout permanently splits the corpus into "always tagged"
   and "always untagged" examples and defeats the purpose of the regularization.
5. **Model/duration**: no architecture change needed beyond the tokenizer/vocab —
   the tag token becomes one more entry in the token sequence and participates in
   `forward_text_embed`/`forward_text_condition`
   (`zipvoice/models/zipvoice.py:187-251`) like any other token **except for
   duration**: per proposal 1's revision, `[LANG:xx]` tokens must get **zero**
   duration (they carry no acoustic content), not a share of frames like a real
   subword. This part is architecturally straightforward but untested — ZipVoice
   is non-autoregressive, unlike OmniVoice's autoregressive LLM where the tag is
   consumed causally before any audio token is generated. **Codex pushback,
   accepted as a required validation step (not an assumption)**: it is *plausible*
   but not guaranteed that a single prefix token measurably influences the whole
   sequence's text condition through the bidirectional `text_encoder` attention —
   this needs an explicit probe (e.g. compare outputs for the same text with
   different `[LANG:xx]` tags and confirm they differ) before relying on it,
   rather than assuming bidirectional attention makes it work.
6. **Inference API**: expose a `lang: Optional[str] = None` argument on the
   inference entry points (`zipvoice/bin/infer_zipvoice.py` and friends),
   analogous to OmniVoice's `generate(text, language=...)`.

## Decided (2026-08-28, was previously an open question)

- **Explicit `[LANG:auto]` token for the "no hint" case — not omission.**
  Codex had suggested omission matches the desired fallback behavior more
  directly than a third learned condition. The author's explicit decision
  overrides that: use `[LANG:auto]` always, exactly mirroring OmniVoice's
  literal `"None"` tag (`lang_str = lang if lang else "None"`) rather than
  ever dropping the tag boundary. Train and evaluate **three** conditions:
  true tag, `[LANG:auto]`, and a *deliberately incorrect* tag (to check whether
  the model actually uses the tag or ignores it and infers from content
  regardless — if performance is indistinguishable between correct and
  incorrect tags, the tag isn't doing anything).

## Open questions
- **Dropout rate**: OmniVoice's exact dropout probability isn't visible from the
  code paths inspected; needs a reasonable default (e.g. 10-20%, standard for
  classifier-free-guidance-style conditioning) picked empirically on a small run
  rather than assumed.
- **Interaction with genuinely code-switched utterances**: a single primary-language
  tag is a per-utterance hint, not a per-token/per-span label. For utterances that
  are heavily mixed (e.g. roughly half EN, half VI in the same sentence), one tag
  can only bias the whole sequence, not resolve ambiguity locally — the shared
  vocabulary (or the model's own learned disambiguation) still has to do the
  fine-grained work. This feature is a coarse global hint, not a replacement for
  correct per-span handling. **Codex risk, accepted**: a global tag could actively
  hurt balanced code-switched utterances by biasing minority-language spans toward
  the tagged/majority language — worth checking for in evaluation, not just
  assuming the tag is harmless when wrong.
- **Not fully independent of proposal 1 (Codex correction, accepted)**: if the
  chosen tokenizer mis-segments or drops Vietnamese content before the model ever
  sees it, `[LANG:vi]` cannot recover content that was never correctly tokenized
  in the first place. This tag is a pronunciation/disambiguation hint on top of
  correct tokenization, not a substitute for it.
- **API naming**: use `primary_lang` rather than reusing `lang`, since ZipVoice
  already has a `--lang` flag for `EspeakTokenizer`'s language selection
  (`infer_zipvoice.py:154`) — a different, whole-utterance-phonemizer-language
  concept that this proposal shouldn't be confused with.

## Next steps

The decision to implement this as special tokens (not a dedicated conditioning
embedding) is recorded in
[../adr/2026-08-28__language_hint_as_special_tokens.md](../adr/2026-08-28__language_hint_as_special_tokens.md).
See the implementation plan at
[docs/plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/),
which sequences this alongside proposal 1's vocabulary/tokenizer work.
