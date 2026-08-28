# Primary-language definition (Sprint 001 deliverable)

Resolves the open item flagged in
[../../../adr/2026-08-28__language_hint_as_special_tokens.md](../../../adr/2026-08-28__language_hint_as_special_tokens.md):
*""primary language" itself needs a precise, written-down definition... before
labeling."*

## Definition (revised 2026-08-28 — no text-based inference needed)

**An utterance's primary language for `[LANG:xx]` training-label purposes is
whatever the dataset's own per-text-audio-pair metadata says it is.** The
author confirmed their training data already carries reliable, per-pair
ground-truth language labels — there is no need to *infer* the language from
the text content itself.

An earlier draft of this document proposed a word-count-majority heuristic
(`classify_primary_language`/`classify_word` in a since-deleted
`zipvoice/utils/language_id.py`) to infer the label from text, motivated by
Codex's caution that "corpus-level language metadata may be noisy... for
genuinely code-switched utterances." That concern doesn't apply once the
dataset provides reliable per-pair ground truth directly — the heuristic
classifier was solving a problem this dataset doesn't have, and was removed
rather than kept as unused machinery.

## The `[LANG:auto]` case (no language provided)

Per the author's decision, mirroring OmniVoice's `lang_str = lang if lang else
"None"` pattern exactly: when no language is specified, use an explicit
`[LANG:auto]` sentinel token — not omission, and not a text-based guess.
`[LANG:auto]` is produced two ways, neither of which involves classifying
text:

1. **At inference**, when the caller doesn't pass a `lang` argument, the
   inference API prepends `[LANG:auto]` directly.
2. **During training**, per proposal 2's label-dropout mechanism, the true
   ground-truth tag is randomly replaced with `[LANG:auto]` for some fraction
   of utterances (dynamically, per training step — see
   [../2026-08-28__sprint_002__special_tokens_and_duration.md](../2026-08-28__sprint_002__special_tokens_and_duration.md)),
   so the model learns to behave reasonably when the hint is absent, the same
   way it will see `[LANG:auto]` at inference.

In both cases `[LANG:auto]` is a fixed vocabulary token like any other
`[LANG:xx]` tag — the model's own `text_encoder` is what has to learn to
disambiguate content when it sees this token, not any external classifier.

## What's left for sprint 001 to actually do here

With label inference removed, this document no longer needs an ambiguity
resolution/tie-break procedure. What sprint 001 still needs, per the overview
plan:
- Confirm the exact field/column in the dataset's manifest format that carries
  the per-pair language label, so sprint 002's training pipeline reads from it
  directly instead of hard-coding a mapping.
- The eval-set curation work (held-out utterances per language and
  code-switch combination) is unaffected by this revision and remains blocked
  on real corpus access — see
  [README.md](README.md).
