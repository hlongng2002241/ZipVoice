# Sprint 000: Build the new wrapping tokenizer

- **Date:** 2026-09-02
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Done -- implemented as `zipvoice/tokenizer/fusion_tokenizer.py`
  (`FusionTokenizer`, `FusionTokenizerArtifact`), tested in
  `tests/test_fusion_tokenizer.py`: real-corpus equivalence tests reading up
  to 1000 samples each from `data/corpus/{en,zh}/*.jsonl` (English/Chinese)
  and `tests/fixtures/vi_normalized_samples.jsonl` (Vietnamese -- see the
  normalization note below for why), comparing against
  `EspeakTokenizer`/`EmiliaTokenizer` (skips with a `pytest` warning rather
  than failing if the corpus/fixture files aren't present), plus a
  `LanguageModelTokenizer` integration test, `__post_init__` cross-field
  validation tests for `FusionTokenizerArtifact`, and small synthetic
  artifact-invariant tests. All 22 tests passing. Four things found/decided
  during implementation, documented inline below and in the ADR: no
  zero-duration masking for punctuation/spaces, and `zero_duration_mask`
  living entirely on the `lm_tokens` side rather than the phone side (see
  step 1's note), a real statefulness quirk in the underlying
  `piper_phonemize`/espeak-ng engine (see step 3's note), and per-language
  opt-in normalization scoped only to `FusionTokenizer` itself (see step
  1's normalization note).
- **Related:** [2026-09-02__qwen_phone_fusion_overview.md](2026-09-02__qwen_phone_fusion_overview.md),
  [../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md](../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md)
  (section 5, "Phone branch: a new, wrapping tokenizer")

## Goal

Build a new tokenizer class that wraps the existing `EmiliaTokenizer`/
`EspeakTokenizer` (never modifies them) and produces phones, `lm_tokens`
(Qwen/BPE output), and a group mapping between them -- with an automated test
proving its phone output is identical to what the existing tokenizers already
produce for the same text.

## Background

See the ADR's Terminology note: `tokens` continues to mean phones (unchanged
from inherited code), `lm_tokens` is the new name for Qwen/BPE output. No
inherited function/variable in `zipvoice/tokenizer/`, `zipvoice/utils/common.py`,
or `zipvoice/models/zipvoice.py` is renamed by this work.

This sprint touches **no model-architecture code**
(`zipvoice/models/zipvoice.py`, `zipvoice/models/modules/zipformer.py`) --
only tokenizer code and its own tests. It can proceed in parallel with sprint
001.

## Scope

- **In scope:** the new tokenizer class; whole-utterance G2P with group
  recovery for English/Vietnamese (via `phonemize_espeak`'s literal-space
  markers) and Chinese (via `jieba.cut()` segment boundaries, preserving
  `EmiliaTokenizer`'s existing Han-vs-Latin-script routing exactly); the new
  versioned per-utterance artifact (not a bare tuple); the equivalence test.
- **Out of scope:** Qwen N-layer feature extraction (sprint 003); any change
  to `train_zipvoice.py`/`SpeechSynthesisDataset`/`prepare_input()` to
  actually consume this artifact in training (sprint 003 -- this sprint
  produces and tests the artifact in isolation); model-architecture code.

## Approach

1. **New tokenizer class**, wrapping the existing tokenizers rather than
   modifying them:
   - For English/Vietnamese text: call `phonemize_espeak()` at
     whole-sentence granularity (as today), but instead of flattening
     immediately, split the returned phone list on literal `' '` elements to
     recover word/group boundaries (confirmed empirically this session: the
     space characters are present as ordinary list elements in
     `phonemize_espeak`'s output).
   - For Chinese text: preserve `EmiliaTokenizer.texts_to_tokens()`'s
     existing top-level segmentation (Han segments to the Chinese
     jieba+pypinyin path, ASCII-alphabetic segments to the English espeak
     path, pinyin/tag segments to their own paths, per `tokenizer.py:227-244,368-399`)
     exactly as-is. Within the Chinese path specifically, call `jieba.cut()`
     once and `lazy_pinyin()` per resulting segment (not on the whole
     segment list at once) to recover group boundaries -- confirmed
     phonologically equivalent to today's single-call behavior, since
     `pypinyin`'s tone-sandhi converter already processes each supplied list
     element independently.
   - Build the group mapping: each group is a set of consecutive phones plus
     the set of consecutive `lm_tokens` (via `LanguageModelTokenizer`'s
     underlying fast HF tokenizer's `return_offsets_mapping=True`) that
     correspond to the same underlying word-equivalent span.
   - **Corrected during implementation**: punctuation and spaces do *not*
     get zero-duration treatment, and there is no phone-side
     `zero_duration_mask` at all. Verified directly against the source
     checkpoint's actual phone vocabulary (`tokens.txt`): the literal space
     character and ordinary punctuation are already normal vocabulary
     entries, and a real sentence's phones map through with zero OOV drops
     -- the converged decoder was trained throughout with these receiving a
     normal share of duration (punctuation is silence, and silence still
     takes time). The only token with no real acoustic content at all is
     the `[LANG:xx]` control tag on the `lm_tokens` side (pure metadata),
     so `zero_duration_mask` lives entirely on the artifact's `lm_tokens`
     side, sourced 100% from `LanguageModelTokenizer.zero_duration_mask()`
     -- see step 2.
   - **Normalization is opt-in, per-instance, and scoped only to
     `FusionTokenizer`** -- `EmiliaTokenizer`/`EspeakTokenizer` are never
     modified. Since one instance is always locked to a single `lang`,
     `FusionTokenizer(lang=..., use_normalizer=...)` takes a plain `bool`
     (not a per-language dict); leaving it `None` resolves the default from
     a private `_DEFAULT_USE_NORMALIZER = {"en": True, "zh": True, "vi":
     False}` table keyed by `lang`: English/Chinese default on (matching
     `EmiliaTokenizer`'s own always-on normalizers, so raw-corpus
     equivalence holds by default for those two languages), Vietnamese
     defaults off and stays off in practice (no Vietnamese normalizer
     exists in this module -- enabling it is a logged no-op), since
     Vietnamese text is normalized by a separate, external pipeline before
     it reaches training. Whether normalization is enabled/disabled is
     logged at construction time (`INFO`/`WARNING`). Both phones and `lm_tokens` are
     derived from the *same* canonical (post-normalization, if enabled)
     text, which is what makes the group correspondence in step 2 possible
     without reconciling `EmiliaTokenizer`'s hardcoded internal
     normalization against `LanguageModelTokenizer`'s no-normalization
     default.
2. **New output artifact** (`FusionTokenizerArtifact`, schema version 3): a
   distinct, versioned per-utterance object (not a bare tuple/`NamedTuple`
   used as the sole safeguard) carrying: phones, phone IDs, the group
   mapping (`phone_groups`), the language used, `lm_tokens`, `lm_token`
   IDs, an `lm_tokens`-aligned `zero_duration_mask` (see below), and a
   schema version. `lm_token_groups` (one entry per `phone_groups` entry,
   or `None`) is only
   populated when a `LanguageModelTokenizer` instance is passed in via
   `FusionTokenizer(lm_tokenizer=...)` -- computed by intersecting each
   group's canonical-text character span (recovered from whitespace/jieba
   segment boundaries) with the `lm_tokenizer`'s
   `return_offsets_mapping=True` output over that same canonical text.
   **`lm_tokens`/`lm_token_ids` always have the deterministic
   `[LANG:{lang}]` control token prepended as index 0** -- since this
   fusion architecture requires `--lang-auto-prob`/`--lang-wrong-prob` both
   be `0` (point 2 of the ADR), the tag is never sampled/variable, so
   `FusionTokenizer` prepends it itself rather than leaving it to a later
   pipeline stage (which would also have to remember to shift every
   `lm_token_groups` index). `lm_token_groups` indices are computed with
   that `+1` offset already applied; index `0` never appears in any group.
   `lm_tokenizer` must have been constructed with a matching `[LANG:{lang}]`
   entry in its `lang_tags` (true by default), or `text_to_artifact()`
   raises. When
   the phone-group count and word/segment-span count don't line up for an
   utterance (the residual risk flagged in the risk table below),
   `lm_token_groups` is left `None` for that utterance rather than
   asserting or silently misaligning. **`__post_init__` validates the
   artifact's cross-field invariants at construction time** (length
   agreement between `phones`/`phone_ids`, and between
   `lm_tokens`/`lm_token_ids`/`zero_duration_mask` -- note
   `zero_duration_mask` pairs with `lm_token_ids`, not `phone_ids`;
   `phone_groups` exactly, monotonically, and non-overlappingly partitions
   `range(len(phone_ids))`; `lm_token_groups`, when present, has one entry
   per `phone_groups` entry with in-range indices; `schema_version` matches
   the module's current `SCHEMA_VERSION`, rejecting a stale cached artifact
   rather than silently accepting it) -- a malformed artifact raises
   immediately rather than failing confusingly downstream in duration
   allocation or dataset collation. **`EmiliaTokenizer`/`EspeakTokenizer`
   themselves are untouched** -- their existing methods (`texts_to_tokens()`,
   `texts_to_token_ids()`, `tokens_to_token_ids()`, etc.) keep their existing
   return contracts unchanged for any caller still using those classes
   directly. `FusionTokenizer` is a new, separate class alongside them, not
   a drop-in replacement implementing that same interface -- it only
   exposes `flat_phones()`, `phone_groups()`, and `text_to_artifact()` (see
   `zipvoice/tokenizer/fusion_tokenizer.py`); a caller needing the artifact
   (sprint 003 onward) must call `text_to_artifact()` explicitly, not treat
   `FusionTokenizer` as a swap-in for the legacy tokenizer interface.
3. **Equivalence test** (the hard requirement from the ADR): implemented
   against real text -- up to 1000 samples each for English/Chinese from
   `data/corpus/{en,zh}/*.jsonl` (one short line per JSON document, to keep
   runtime practical against multi-paragraph web text while still
   exercising diverse real content), asserting that flattening the new
   tokenizer's phone groups reproduces exactly what the corresponding
   existing tokenizer (`EspeakTokenizer` for EN, `EmiliaTokenizer` for ZH)
   already produces for that text -- for English, both sides are compared
   after applying the *same* normalization (the old tokenizer is fed
   pre-normalized text, matching `FusionTokenizer(lang="en")`'s default
   `use_normalizer=True` behavior on raw text), and a second test confirms
   `use_normalizer=False` reproduces raw `EspeakTokenizer` output exactly
   (the opt-out path real, already-normalized training data uses). Skips
   with a `pytest` warning (doesn't fail/error) if the corpus files aren't
   present in the checkout.

   **Vietnamese samples come from a small fixture, not
   `data/corpus/vi/*.jsonl`**: `tests/fixtures/vi_normalized_samples.jsonl`
   holds 1000 real, already-normalized Vietnamese utterances extracted from
   `data/all/manifests/train.jsonl.gz` (the real training manifest, not the
   raw web corpus), via `scripts/all/build_vi_tokenizer_fixture.py`. **Not
   tracked in git** (`.gitignore`'s `tests/fixtures/*.jsonl` entry) --
   real training-utterance text carries the same sensitivity concerns as
   `data/` (also gitignored, "Recommended for sensitive data"), so
   committing an extracted copy into `tests/` would defeat that exclusion.
   Anyone with local access to `data/all/manifests/train.jsonl.gz` can
   regenerate it by running the script above; if it's absent, the VI test
   warns and skips, exactly like the EN/ZH corpus-based tests do when
   `data/corpus/{en,zh}/*.jsonl` are absent. This matters because
   `FusionTokenizer(lang="vi")`'s `use_normalizer` defaults to `False` and
   is expected to stay `False` -- Vietnamese normalization
   happens in a separate, external pipeline before data reaches training,
   so the raw, unnormalized `data/corpus/vi/*.jsonl` web text would not be
   representative of what this tokenizer actually sees for Vietnamese in
   practice. The Vietnamese equivalence check itself is unaffected by
   `use_normalizer` (neither side normalizes), but the fixture makes the
   test exercise realistic input instead of arbitrary raw web text.

   **Real finding, corrected during implementation**: `piper_phonemize`'s
   underlying espeak-ng engine is **stateful across calls within a
   process** -- phonemizing the same text immediately after a *different*
   text can produce different phone output than phonemizing it in
   isolation (confirmed directly: calling `phonemize_espeak` on two fixed
   texts back-to-back is reproducible, but interleaving calls to two
   *different* tokenizer instances on the same texts is not, since both
   instances share the same underlying stateful engine). The equivalence
   test computes each tokenizer's output over *all* samples in one clean,
   uninterrupted pass before comparing -- comparing per-sample with calls
   interleaved between the two tokenizers produced false mismatches purely
   from this state-sharing artifact, not any real difference in
   tokenization logic. **This is a pre-existing property of
   `piper_phonemize`/espeak-ng, not something introduced by this sprint's
   code** -- it plausibly also affects any *existing* code path that
   phonemizes many different texts in sequence through a shared process
   (e.g. `tokenizer.py`'s `add_tokens()` processing a whole manifest via
   `EspeakTokenizer`/`EmiliaTokenizer`), which is worth a dedicated
   follow-up investigation independent of this fusion work.
4. **Artifact invariant checks**, run over the same real corpus sample: every
   phone belongs to exactly one group; groups are monotonic and
   non-overlapping; every relevant `lm_token` maps to a group or an explicit
   ignored/control classification; `len(zero_duration_mask) ==
   len(lm_token_ids)` (not `phone_ids` -- see step 1's note).

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Word-count mismatches between the new tokenizer's phone groups and `lm_token` groups on real corpus text (numbers, abbreviations, contractions not fully normalized upstream) | Broken group correspondence for affected utterances | Run the equivalence and invariant checks over real corpus samples, not just clean synthetic strings; document and handle (or reject) any residual mismatch class found |
| Chinese segment-routing/punctuation decisions were made by Claude Code, not the author (who does not read Chinese) | A linguistic consideration might be missed | Explicitly flagged in the ADR; a native-speaker review pass is recommended before sprint 003, not required to unblock this sprint |
| Vietnamese diacritics not recognized by `EmiliaTokenizer`'s ASCII-alphabetic segmenter, if VI text ever needs to pass through that specific path | Could misroute VI text if this tokenizer's scope ever expands beyond EN/VI (Espeak-direct)/ZH (Emilia-style) | Out of scope for this sprint since VI is always handled via the direct Espeak path (`EspeakTokenizer`-equivalent), not `EmiliaTokenizer`'s segmenter -- noted here in case the new tokenizer's scope changes later |

## Testing & Validation

- Automated equivalence test (phone output matches existing tokenizers
  exactly) as the primary pass/fail gate for this sprint.
- Artifact invariant checks over a real corpus sample.
- No model training or generation involved -- purely a tokenizer-level
  deliverable.

## Rollout

The new tokenizer class is added alongside the existing ones (`EspeakTokenizer`,
`EmiliaTokenizer`, `LanguageModelTokenizer`) without modifying any of them.
Nothing in the existing training/inference pipeline is wired to use it until
sprint 003.
