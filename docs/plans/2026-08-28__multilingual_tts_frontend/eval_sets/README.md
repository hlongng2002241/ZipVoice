# Sprint 001 eval-set artifacts

Deliverables of
[../2026-08-28__sprint_001__eval_and_data_contracts.md](../2026-08-28__sprint_001__eval_and_data_contracts.md).

- `primary_language_definition.md` — **revised 2026-08-28**: the primary
  language for training purposes is read directly from the dataset's existing
  per-text-audio-pair ground truth, not inferred from text. A text-based
  classifier (`zipvoice/utils/language_id.py`) was built, then removed once it
  became clear the dataset already provides reliable per-pair labels and the
  classifier was solving a problem that doesn't exist here — see the doc for
  the full explanation, including how `[LANG:auto]` is produced instead
  (OmniVoice-style fixed sentinel, not a text-based guess).
- `ambiguous_words_en_vi.jsonl` — curated EN/VI homograph and cross-language
  ambiguous-word spot-check list, for pronunciation checking once phonemes are
  dropped (per the mBERT proposal's implicit-G2P risk). Hand-curated from
  linguistic knowledge, independent of the (now-removed) classifier.
- `code_switched_seed_sentences.jsonl` — 52 hand-written text-only sentences
  across 7 buckets (`en`, `vi`, `zh`, `en+vi`, `en+zh`, `vi+zh`, `en+vi+zh`),
  for early sprint 002/003 prototyping ahead of real corpus access.

## What this is, and isn't, yet

This is a **text-only seed set**, hand-written rather than mined from a real
training corpus. **Blocker, surfaced rather than worked around**: this sandbox
has no local corpus data — neither the repo-relative `data/` directory
(gitignored) nor the absolute `/data/longnh/data/` path the existing VI
data-prep scripts (`scripts/vi/m00_collect_dataset.py`) reference exists here.
The sprint 001 plan's "curate held-out utterance sets ... from existing
corpora" item is **not yet done** — it needs access to the actual Emilia and
Vietnamese training manifests, which aren't present in this environment.

The originally-planned "audit existing corpus language metadata for noise"
item is **no longer needed as a classifier-based audit**: the dataset already
carries reliable per-text-audio-pair language ground truth (confirmed by the
author), so there's nothing to infer or cross-check against text content. What
remains is simply confirming which manifest field carries that ground truth so
sprint 002's training pipeline reads it directly — see
`primary_language_definition.md`.

What's usable now, independent of the corpus-access blocker:
- The primary-language definition, now resolved to "read from the dataset's
  existing per-pair label" rather than an inferred/heuristic one.
- The ambiguous-word and code-switched seed sentences as an initial,
  text-only smoke-test set for sprint 002's probe and early sprint 003
  prototyping — expand with real corpus-mined utterances once the corpus is
  accessible.

Once the real corpora are accessible, expand these seed sets with real mined
utterances rather than treating this hand-written set as sufficient on its own
for sprint 003/004's bakeoff and regression testing.
