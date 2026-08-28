# Sprint 001 — Evaluation sets & data contracts

- **Date:** 2026-08-28
- **Author:** LongNH (with Claude Code assistance)
- **Status:** In Progress — see "Progress" below; blocked on real corpus access
  for the corpus-dependent items

## Goal

Build the fixed, held-out evaluation sets and define the data contracts (a
precise "primary language" definition, an audit of existing corpus language
metadata) that every later sprint in this plan — and the bakeoff in sprint 003
specifically — will be judged against.

## Background

Per [2026-08-28__overview.md](2026-08-28__overview.md), this is a prerequisite
for all later sprints: without fixed eval sets, sprint 003's bakeoff has no
consistent way to be scored. Codex's review had flagged corpus-level language
metadata as "may be noisy, absent, or ill-defined for genuinely code-switched
utterances," which originally motivated a text-based classifier to infer the
label — **that classifier was built, then removed on 2026-08-28**: the author
confirmed their dataset already carries reliable per-text-audio-pair language
ground truth, so there's nothing to infer. See
[eval_sets/primary_language_definition.md](eval_sets/primary_language_definition.md)
for the full revision, including how the `[LANG:auto]` no-hint case is produced
(an OmniVoice-style fixed sentinel via label dropout on the known ground truth,
not a text-based guess).

## Scope

- **In scope:**
  - Held-out evaluation utterances for EN, VI, ZH individually, plus a
    code-switched EN+VI, EN+ZH, and EN+VI+ZH set, including a curated subset of
    ambiguous Latin-script words/homographs (plausibly EN or VI) for
    pronunciation spot-checking.
  - Confirming which manifest field carries the dataset's existing per-pair
    language ground truth, so sprint 002's training pipeline reads from it
    directly.
- **Out of scope:** any tokenizer, embedding, or model training/selection work
  (sprints 002-004); inferring language from text (not needed — see
  Background).

## Approach

1. Curate held-out utterance sets per language and per code-switch combination,
   sized for repeated use across sprints 002-004 (not single-use).
2. Curate a dedicated ambiguous-word/homograph subset for pronunciation
   spot-checking (not just aggregate WER).
3. Confirm the manifest field carrying the dataset's ground-truth per-pair
   language label, and document it for sprint 002.
4. Check the eval sets themselves with a native/fluent Vietnamese speaker pass
   (or equivalent) before relying on them.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Eval sets too small or unrepresentative | Bakeoff/training decisions in later sprints rest on noisy signal | Size sets deliberately for reuse and cross-check with the ambiguous-word subset specifically, not just aggregate size |
| "Primary language" definition chosen here turns out inadequate for real code-switch patterns later | Rework needed in sprint 002/003 | Document the definition explicitly so it can be revisited with a clear before/after, rather than silently redefined |

## Testing & Validation

Native/fluent-speaker spot check of the Vietnamese portions of the eval sets;
manual review of the ambiguous-word subset for genuine ambiguity (not
accidentally unambiguous items).

## Progress (2026-08-28)

**Done, independent of corpus access:**
- Ambiguous EN/VI word spot-check list (15 entries, including genuine
  cross-language homographs like "an"/"do"/"me" and English-internal
  homographs like "read"/"wind"/"live" worth checking once phonemes are
  dropped), hand-curated from linguistic knowledge:
  [eval_sets/ambiguous_words_en_vi.jsonl](eval_sets/ambiguous_words_en_vi.jsonl).
- 52 hand-written text-only seed sentences across all 7 language/code-switch
  buckets: [eval_sets/code_switched_seed_sentences.jsonl](eval_sets/code_switched_seed_sentences.jsonl).

**Superseded (built, then removed as unnecessary):** a text-based
majority-word-vote classifier (`zipvoice/utils/language_id.py` and its
wordlists/tests) was built to infer an utterance's primary language, based on
Codex's caution that corpus metadata "may be noisy." The author confirmed
their dataset already provides reliable per-pair ground truth directly, making
the classifier redundant for label generation — it was deleted rather than
kept as unused scaffolding. See
[eval_sets/primary_language_definition.md](eval_sets/primary_language_definition.md)
for the resolved definition (read from the dataset directly) and how
`[LANG:auto]` is produced instead (fixed sentinel via label dropout, not
inference).

**Still blocked — genuinely missing, not worked around:** this sandbox has no
local training corpus. Neither the repo-relative `data/` directory
(gitignored) nor the absolute `/data/longnh/data/` path referenced by
`scripts/vi/m00_collect_dataset.py` exists here. As a result, this sprint's
"curate held-out utterance sets ... from existing corpora" item is **not
done**, and "confirm the manifest field carrying the language label" can't be
verified against a real manifest yet. See
[eval_sets/README.md](eval_sets/README.md) for what's usable now vs. still
pending. **Needs input from the user**: where does the actual Emilia/Vietnamese
training data live (a different machine, an unmounted path, a download step
not yet run), and how should it be made accessible here to finish this sprint
and unblock sprint 003's bakeoff?

## Rollout

Eval sets are checked into the repository as reusable fixtures for sprints
002-004 and, per the overview, are not expected to change once sprint 002
begins consuming them — the seed sentences should be extended with real
corpus-mined data once accessible, not replaced.
