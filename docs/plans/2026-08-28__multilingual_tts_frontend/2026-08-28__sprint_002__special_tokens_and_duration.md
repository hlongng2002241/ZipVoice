# Sprint 002 — `[LANG:xx]` special tokens & zero-duration handling

- **Date:** 2026-08-28
- **Author:** LongNH (with Claude Code assistance)
- **Status:** In Progress — code + tests done; the influence probe is blocked
  on a trained checkpoint (see Progress)

## Goal

Add `[LANG:en]`/`[LANG:vi]`/`[LANG:zh]` as reserved vocabulary tokens, make
ZipVoice's duration-allocation code correctly give special/control tokens zero
acoustic duration, implement dynamic per-step label dropout, and prove — not
assume — that a prepended language tag measurably influences the model's output.

## Background

Per [../../adr/2026-08-28__language_hint_as_special_tokens.md](../../adr/2026-08-28__language_hint_as_special_tokens.md),
these tokens reuse the same mechanism as ZipVoice's existing `[S1]`/`[S2]`
speaker tags. Codex's review identified two concrete gaps the original proposals
didn't specify: special tokens need explicit zero-duration handling (the current
`prepare_avg_tokens_durations`/`get_tokens_index` in `zipvoice/utils/common.py`
has no such concept), and a single prefix token's influence over the whole
sequence via bidirectional attention is plausible but unverified.

## Scope

- **In scope:**
  - Vocabulary reservation mechanism that works on top of whichever external
    pretrained tokenizer sprint 003 ultimately selects (not hard-coded to
    ZipVoice's current token file format).
  - Zero-duration exclusion for special/control tokens in the duration
    allocation path.
  - Dynamic per-training-step label dropout (sampled fresh each time an
    utterance is seen, not baked in once at data-prep time — the mistake Codex
    flagged in the original draft).
  - A prefix-token-influence probe: compare model outputs for identical text
    under different `[LANG:xx]` tags (and a deliberately incorrect tag) to
    confirm the tag changes behavior rather than being ignored.
- **Out of scope:** choosing which base tokenizer these tokens ride on top of
  (sprint 003); the full-scale training run (sprint 004).

## Approach

1. Implement `[LANG:en]`/`[LANG:vi]`/`[LANG:zh]`/`[LANG:auto]` as reserved
   vocabulary entries. **Decided 2026-08-28**: the no-hint case always emits
   the explicit `[LANG:auto]` sentinel — never omission — mirroring
   OmniVoice's literal `"None"` tag exactly. The true (non-dropout) label comes
   directly from the dataset's existing per-text-audio-pair ground truth (see
   [eval_sets/primary_language_definition.md](eval_sets/primary_language_definition.md));
   no text-based inference is needed or used.
2. Extend `prepare_avg_tokens_durations`/`get_tokens_index` (or their
   replacement, depending on sprint 003's duration-scheme outcome) to assign
   zero frames to designated special tokens.
3. Implement label dropout inside the training data pipeline/dataloader
   (per-step, not per-prep): for each utterance, use the ground-truth tag most
   of the time, replaced with `[LANG:auto]` for a configurable fraction of
   steps, and occasionally a deliberately wrong tag for the probe in step 4.
4. Build the prefix-token-influence probe and run it on a small pretrained-ish
   or partially-trained checkpoint; treat a null result (no measurable
   difference across tags) as a gate failure requiring investigation before
   sprint 004 proceeds, not a footnote.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Label dropout accidentally implemented at prep-time instead of dynamically | Permanently splits the corpus into always-tagged/always-untagged, defeating the regularization | Implement in the dataloader/training loop, not the manifest-prep script; covered by this sprint's own review |
| Probe shows the tag has no measurable effect | The whole feature is a no-op | Treat as a blocking finding — investigate (e.g. embedding not receiving gradient, duration exclusion swallowing the token's position) before sprint 004, don't ship silently |
| Zero-duration handling interacts badly with the script-weighted duration split from sprint 003 | Broken frame allocation, subtle audio artifacts | Coordinate directly with sprint 003 since both touch the same duration code path |

## Testing & Validation

Unit test confirming special tokens receive exactly zero frames in the duration
allocation; the prefix-token-influence probe's results, both for a correct tag
and a deliberately wrong one; dropout-rate sanity check (empirical
tag/`[LANG:auto]`/wrong distribution in a training epoch matches the configured
rate).

## Progress (2026-08-28)

**Done:**
- `zipvoice/tokenizer/multilingual_tokenizer.py`: `MultilingualTokenizer`
  wraps any HF `AutoTokenizer` and registers `[LANG:en]`/`[LANG:vi]`/`[LANG:zh]`/
  `[LANG:auto]` via `add_special_tokens`, confirmed (test) to tokenize as
  exactly one token each, at IDs beyond the base vocabulary. Exposes
  `zero_duration_mask(token_ids)` for step 2.
- `zipvoice/utils/common.py`'s `prepare_avg_tokens_durations` now accepts an
  optional `zero_duration_mask`; masked tokens get exactly 0 frames and the
  remaining tokens split the utterance's real duration evenly. Confirmed via
  `tests/test_common.py` (including that `get_tokens_index` never assigns any
  frame to a zero-duration token). Backward compatible: omitting the mask
  reproduces the original uniform-split behavior exactly.
- `sample_lang_tag()` implements the true/`[LANG:auto]`/wrong-tag three-way
  sampling from proposal 2's "Decided" section, parameterized by
  `auto_prob`/`wrong_tag_prob` and an explicit `random.Random` (not global
  state, for reproducibility). Verified the sampled distribution matches
  configured probabilities within noise (`tests/test_multilingual_tokenizer.py`).
- **Found and fixed a real design flaw while implementing this**:
  `MultilingualTokenizer` was initially added to the existing
  `zipvoice/tokenizer/tokenizer.py`, which hard-imports `jieba` and
  `piper_phonemize` at module load time (confirmed: neither is installed in
  this sandbox, so the whole module failed to import). Coupling a
  phonemizer-free tokenizer to a module that requires a phonemizer stack just
  to be importable would undermine the entire point of
  [../../adr/2026-08-28__adopt_pretrained_multilingual_tokenizer.md](../../adr/2026-08-28__adopt_pretrained_multilingual_tokenizer.md).
  Fixed by extracting the `Tokenizer` ABC into a new dependency-free
  `zipvoice/tokenizer/base.py`, imported by both `tokenizer.py` (backward
  compatible, no import changes needed elsewhere) and the new
  `multilingual_tokenizer.py`. Confirmed
  `import zipvoice.tokenizer.multilingual_tokenizer` pulls in neither `jieba`
  nor `pypinyin`.
- 11/11 tests passing across `tests/test_multilingual_tokenizer.py` and
  `tests/test_common.py`.

**Not done — genuinely blocked, not skipped:** the prefix-token-influence
probe (comparing model outputs across different `[LANG:xx]` tags on identical
text) requires a model that has actually been trained with these tags — there
is nothing to probe on an untrained/randomly-initialized embedding. This is
inherently gated on sprint 004's training run, which is itself blocked on real
corpus access (see sprint 001's Progress note). Tracked here as a real open
item, not silently dropped.

## Rollout

Land behind a config flag so `[LANG:xx]` conditioning can be toggled per
training run independent of the tokenizer bakeoff in sprint 003.
