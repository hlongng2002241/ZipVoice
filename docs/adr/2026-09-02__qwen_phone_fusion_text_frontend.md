# Fuse a context-aware Qwen branch with a phoneme branch, feeding a repositioned `in_proj`

- **Date:** 2026-09-02
- **Status:** Accepted (design finalized after three rounds of Codex review;
  see git history of this file for the pre-finalization version)
- **Related:** [../proposals/2026-09-01__embedding_interface_warmup_curriculum.md](../proposals/2026-09-01__embedding_interface_warmup_curriculum.md)
  (full analysis and design evolution this decision is drawn from),
  [../proposals/2026-08-31__warmstart_from_pretrained_checkpoint.md](../proposals/2026-08-31__warmstart_from_pretrained_checkpoint.md)
  (the warm-start this decision modifies the text frontend of).

## Terminology note (read this first)

This document uses a convention chosen specifically to avoid renaming any
inherited code:

- **`tokens`** means exactly what it already means in this codebase's
  inherited code (`master`'s convention, unchanged by this ADR): the
  acoustic-alignment unit that duration allocation and frame-gathering
  operate over. In `master`, that was always phones (no BPE tokenizer
  existed, so "token" and "phone" were synonyms). In this fusion
  architecture, **`tokens` still means phones** -- nothing inherited
  (`prepare_avg_tokens_durations`, `get_tokens_index`, `token_duration`,
  `max_tokens`, `supervision.tokens`, etc.) needs renaming.
- **`lm_tokens`** is the new name for the one genuinely new concept this
  architecture introduces: `LanguageModelTokenizer`'s Qwen/BPE output. This
  never existed in `master`'s world and has no legacy name to collide with.
- **Groups**: the shared alignment unit between `tokens` (phones) and
  `lm_tokens` (BPE pieces) -- a group is the set of consecutive phones and
  the set of consecutive `lm_tokens` that correspond to the same underlying
  word-equivalent span of text.

## Context

The warm-started multilingual model (887/889 tensors transplanted from a
converged master-architecture checkpoint; `EspeakTokenizer`, phoneme tokens)
produces natural-sounding speech but wrong word content, unchanged from 48k to
76k training batches. Word-order sensitivity is empirically confirmed (rules
out simple conditioning collapse), so the problem is that the
`embed`->`in_proj`->`text_encoder` pathway uses text context but hasn't
converged to *correct* content.

Three structural facts drive this decision:

1. **`embed` only imports Qwen2.5-0.5B's embedding table** (`_make_pretrained_embedding`,
   `zipvoice.py:35-88`) -- a static, context-free per-token lookup. It carries
   none of Qwen's actual pretrained transformer reasoning. Checked
   `../OmniVoice/omnivoice/training/builder.py:95-116` and
   `omnivoice.py:418-424` directly: OmniVoice loads Qwen via
   `AutoModel.from_pretrained` (full pretrained transformer stack, not just
   embeddings) and runs every layer on every forward pass -- genuinely
   context-aware, unlike our current setup.
2. **espeak-style G2P cannot identify per-word language in code-switched
   Latin-script text.** A from-scratch BERT token-classifier for this exact
   problem (trained on synthetic monolingual-corpus mixtures) worked
   normally but reliably failed on (a) surface-form-ambiguous EN/VI words and
   (b) short spans of one language embedded in long spans of the other --
   plausibly because the synthetic training data underrepresented that
   pattern. This rules out repeating an isolated, upstream hard-classification
   step as the fix.
3. **The real training corpus already contains genuine EN/VI code-switched
   utterances** (not just synthetic text mixing), each with a per-utterance
   primary-language label from existing dataset metadata (`supervision.language`).
   This is real, if imperfect (utterance-level, not per-span), ground truth to
   build the phone branch from.

## Decision

We will restructure the text frontend to fuse two branches -- a
context-aware **Qwen branch** and an explicit **phoneme branch** -- combined
by a learned, per-group gate, feeding into `text_encoder` through a
repositioned `in_proj`:

```
text --+--> Qwen embedding + N transformer layers --> last-subtoken pooling per group --> linear(896->192, NEW) --------+
       |                                                                                                                |
       |                                                                                                                +--> per-group fusion gate --> group-to-phone expansion --> in_proj (192->192, SOURCE-initialized) --> [existing, unmodified] text_encoder Zipformer stack --> ... --> fm_decoder
       |                                                                                                                |
       +--> new wrapping tokenizer: whole-utterance G2P (immutable supervision.language) --> phone embedding (192-dim; SOURCE-initialized where covered) --+
```

### 1. Language selection for the phone branch (the central open problem, now resolved as an explicit research bet)

The phone branch phonemizes each utterance under its **immutable
`supervision.language`** ground truth (not the sampled/noisy training tag
used elsewhere for CFG-style regularization -- see point 6). This means
minority-language spans within a genuinely code-switched utterance get
phonemized under the *wrong* language's rules -- e.g. an English word inside
a Vietnamese-primary utterance gets Vietnamese phonological rules applied to
it, producing wrong phones for that word specifically.

**This is deliberately not fixed at the phonemizer level.** Instead, the
Qwen branch is expected to help the model implicitly recognize a span's true
language (no explicit per-span label anywhere) and use that recognition to
**correct** the wrong phonetic content for that span -- not merely flag the
phone branch as "unreliable" there. This is explicitly the harder of two
possible roles for the Qwen branch (the other being "just a reliability
signal, phones still dominate content"), and it is **an unproven, high-risk
experiment, not a guaranteed fix**:

- It only has any chance of working because the gate operates per-group (see
  point 3) -- an utterance-level gate could never favor phones for a reliable
  majority-language span while favoring Qwen for a short embedded
  minority-language span.
- It relocates (not eliminates) the same data-coverage failure mode that
  broke the earlier BERT classifier -- if short minority-language spans are
  underrepresented in training data, the gate can still learn to default to
  trusting the (wrong) phone branch on exactly those spans, just via a
  smoother, implicit training signal (flow-matching loss) instead of an
  explicit classification objective.
- No extra training tricks are used to mitigate this (see point 4) --
  **gate collapse toward phone-only is an accepted risk, not something this
  design mitigates.** If the Qwen branch never activates meaningfully during
  training, that is a failed experiment result to learn from, not evidence
  that more training time is automatically needed.

Sprint 003's fixed-gate experiment (see the plan; see also "Why a separate
pre-architecture oracle-validation sprint was dropped" below) is a **hard
prerequisite** for validating this bet before the learned version is built.

### 2. Config enforcement required for training-time consistency

Training samples a per-utterance language *tag* (`sample_lang_tag`,
`train_zipvoice.py:934-971`) independently of `supervision.language`,
including `[LANG:auto]` (`--lang-auto-prob`) and a deliberately-wrong tag
(`--lang-wrong-prob`) for the text/`lm_tokens` side's own regularization.
Since the phone branch always uses the immutable `supervision.language`
regardless of the sampled tag, training this fusion architecture with either
probability nonzero would create a text/phone inconsistency (the `lm_tokens`
branch sees "no hint" or "wrong hint" while the phone branch stays grounded
in truth). **Both `--lang-auto-prob` and `--lang-wrong-prob` must be enforced
at `0` when training this architecture, validated in code** (train_zipvoice.py
currently accepts and validates arbitrary legal values for both,
`train_zipvoice.py:1181` -- a fusion-specific rejection of nonzero values must
be added, not left to documentation or shell-script convention).

At inference, the current `--lang` default of `None` (which becomes
`[LANG:auto]` on the `lm_tokens`/tag side) provides no language for the
phonemizer to use at all. **The inference-time default reverts to `en-us`**
(master's original default for its own `--lang` flag), requiring an explicit
language-code mapping between namespaces that don't otherwise line up:
phonemizer locale code (`en-us`, `vi`, ...) <-> canonical short code (`en`,
`vi`, ...) <-> `lm_tokens` tag (`[LANG:en]`, `[LANG:vi]`, ...). This mapping
must be built explicitly; today's code has no single normalized namespace
across all three.

### 3. Gate: per-group, not per-utterance

The fusion gate operates at the same **group** granularity as the phone/`lm_token`
correspondence itself (point 5) -- per word-equivalent unit, not a single
scalar per utterance. This is what makes point 1's bet even possible in
principle: the gate can favor phones for a reliable majority-language group
while favoring Qwen for a specific minority-language group in the same
utterance.

Each group's Qwen-branch representation uses **last-subtoken pooling**: the
final `lm_token` belonging to that group, taken from the N-layer Qwen
extraction. This choice is specifically informed by Qwen2's **causal**
attention (it cannot see tokens after the current position) -- the last
subtoken in a group has seen that entire group's own content plus everything
before it, which substantially (though not completely -- it still can't see
subsequent groups) mitigates the "can't identify a foreign span from its
first token alone" concern. This is an accepted, documented, non-blocking
limitation, not something requiring a bidirectional-Qwen workaround.
**Mean pooling across the group's `lm_tokens`** is a documented alternative,
not adopted now -- worth trying later (sprint 004+) if last-subtoken pooling
underperforms; it trades away the "most-context-seen" property for less
sensitivity to any single subtoken's idiosyncrasies.

**Only the Qwen branch is collapsed to group granularity -- the phone branch
is not.** Each phone keeps its own individual embedding at its own sequence
position; the group-level fusion gate combines each phone's own embedding
with the *same* pooled-and-broadcast Qwen group vector, repeated across every
phone position in that group. This is a clarification made explicit during
sprint 003 planning (an earlier reading of "group-level fused representation
... broadcast onto the phone-granular sequence" could be misread as also
collapsing the phone branch to one vector per group before broadcasting,
which would discard per-phone identity and stop approximately reproducing
the source checkpoint's per-phone conditioning even at phone-only gate
values -- not the intent). Concretely, for phone `i` in group `g`:
`fused[i] = gate[g] * phone_embed[i] + (1 - gate[g]) * qwen_group_vec[g]`
(or whatever exact combination the implementation uses -- the point is
`phone_embed[i]` stays indexed by `i`, not by `g`).

### 4. Gate initialization and training

The gate is initialized near-phone-only (`eps=0.01` mixing weight for the
Qwen branch) but is **not architecturally floored** -- it can move to any
value, including exactly favoring one branch, during training. This resolves
a prior internal inconsistency: an earlier draft wanted both "exact source
reproduction at initialization" (requires the Qwen contribution to be exactly
zero) and "a gate floor preventing collapse" (requires it to never be zero) --
these are incompatible. The resolved position is **approximate**
source-like initialization (not exact), with no floor.

`eps=0.01` at initialization corresponds to a strongly saturated sigmoid
(`logit(0.01) ≈ -4.6`), giving roughly 25x less gradient at that point than a
neutral 0.5 gate would. Combined with a source-initialized, already-working
phone path and a fresh, unproven Qwen projection, there is a real risk the
optimizer suppresses the Qwen branch before it ever becomes useful. **No
mitigation is applied for this at this stage** -- no phone-branch dropout, no
gate warm-up schedule, no entropy regularization, no explicit reliability
supervision. The model must learn routing purely from flow-matching loss,
since there is no ground-truth per-group reliability label to supervise
against, and some words legitimately should route ~100% to phones (a
permanent regularization would fight that). This is a deliberate, accepted
risk (see point 1); sprint 004 logs gate-weight distributions during training
specifically to detect collapse early, without requiring the external
evaluation project to discover it.

Branch magnitude is normalized, which this ADR originally left open as an
implementation detail for the build sprint. **Resolved as follows:** the
projected Qwen vector is RMS-normalized and then multiplied by a learnable
`qwen_scale`, initialized by `calibrate_qwen_scale()` to the phone
embedding's own RMS. `eps=0.01` therefore means "1% of a vector whose
magnitude matches the phone branch's", which is the precise functional
meaning it previously lacked. The calibration must be re-run *after* a
checkpoint transplant, since at construction it can only measure the random
initialization's RMS -- `scripts/fusion/m00_build_warmstart_checkpoint.py`
does this, and measured 0.4887 against the k2-fsa source.

**Gate weights are initialized small-random (std 0.01), not zero.** Zero is
the obvious way to make the gate start at exactly `eps` -- the logit would be
the bias alone -- and it is wrong here for an optimizer reason discovered
during implementation review. `ScaledAdam` scales every non-scalar tensor's
update by that tensor's own RMS, floored at `param_min_rms=1e-5`, and the
scale-growth path cannot compensate: `scale_grads = (p * grad).sum(...)` is
identically zero when `p` is zero, and the below-floor mask zeroes the scale
step outright. Measured with a constant unit gradient, a zero-initialized
tensor needs ~10k steps just to reach the floor (RMS 1e-5 at 10k, 5.2e-5 at
25k, 0.198 at 100k) -- a severe update-scale bottleneck across exactly the
early phase of a warm-start fine-tune, during which the gate is effectively
its scalar bias alone: one global mixing constant rather than the per-group
input-dependent gate this point specifies. It is a bottleneck rather than a
permanent freeze, and the real escape time depends on the gradient sequence,
which the constant-gradient probe does not model. Small-random weights avoid
the question entirely while keeping the gate at ~`eps`: measured on the real
config with the transplanted checkpoint, the gate spans 0.0086..0.0109 across
real groups and the fused output stays within 1.3% of plain phone-only.

Confirmed afterwards on real training rather than only on a synthetic
probe: two otherwise-identical GPU runs over the same data and optimizer,
differing only in this initialization, moved the gate weight 3.68e-07 per
step (small-random) versus 4.75e-10 per step (zero) -- a 775x difference.

This is a concrete, measured instance of the optimizer-interaction risk
flagged two paragraphs above. It does not change the "no mitigation applied"
position there -- no dropout, warm-up schedule or entropy regularization was
added -- but it does mean one specific way the optimizer could have
suppressed the Qwen branch has been removed by construction rather than left
to be detected by sprint 004's gate logging.

### 5. Phone branch: a new, wrapping tokenizer (not a modification of existing tokenizers)

A **new tokenizer class** will be built, wrapping (not modifying in place)
the existing `EmiliaTokenizer`/`EspeakTokenizer`. Hard requirement, verified
by an automated equivalence test: **for any given text, the flat phone
sequence the new tokenizer produces must be identical to what the
corresponding existing tokenizer already produces for that text** -- the new
tokenizer adds grouping/mapping metadata on top of unchanged phone output, it
does not reimplement or alter phonemization. Concretely:

- Whole-utterance G2P is used (not per-word calls), preserving cross-word
  phonological rules (Chinese tone sandhi, English/Vietnamese connected
  speech) that a naive per-word decomposition would lose.
- Word/group boundaries are recovered from each phonemizer's own natural
  output structure rather than reimplemented: `phonemize_espeak`'s
  sentence-level output contains literal space characters marking word
  boundaries (confirmed empirically) for English/Vietnamese; for Chinese,
  `jieba.cut()`'s own segment boundaries serve the same role, calling
  `lazy_pinyin()` per segment -- confirmed equivalent to today's
  `EmiliaTokenizer.tokenize_ZH()` behavior, since `pypinyin`'s tone-sandhi
  converter already processes each supplied list element independently
  (verified against `pypinyin` source), so no sandhi quality is lost by
  preserving boundaries this way instead of the current flatten-and-discard
  approach.
- **`EmiliaTokenizer`'s existing Han-vs-Latin-script segment routing is
  preserved exactly, not replaced.** `EmiliaTokenizer.texts_to_tokens()`
  already routes ASCII-alphabetic segments to `tokenize_EN()` (proper English
  espeak phonemization), not through the Chinese pinyin path -- English words
  embedded in Chinese text are *already* correctly phonemized today. The new
  tokenizer must not degrade this by treating non-Han segments generically as
  "wrong-language content" -- that would both violate the equivalence
  requirement and be a regression. (Chinese-specific decisions in this ADR,
  including this one and punctuation handling below, were made by Claude
  Code, not the author, who does not read Chinese -- flagged here explicitly
  rather than presented as the author's own linguistic judgment.)
  This does **not** solve EN-vs-VI ambiguity within Latin-script text --
  `EmiliaTokenizer`'s segmenter has no Vietnamese-specific path and cannot
  distinguish EN from VI within the same ASCII-alphabetic classification.
  That remains exactly the open problem points 1/3/4 exist to address.
- **Punctuation and spaces do *not* get zero-duration treatment, and
  `zero_duration_mask` lives entirely on the `lm_tokens` side, not the
  phone side -- corrected during implementation.** An earlier draft of
  this ADR planned to zero punctuation/space duration via a phone-aligned
  `zero_duration_mask`. Verified directly against the source checkpoint's
  actual phone vocabulary (`tokens.txt`) during sprint 000's
  implementation: the literal space character and ordinary punctuation
  (`,`, `.`, `?`, ...) are already ordinary vocabulary entries with their
  own ids, and a real sentence's phones map through to token ids with
  **zero OOV drops** -- meaning the converged source checkpoint was
  trained throughout with these receiving a normal, non-zero share of
  duration like any other phone. Punctuation is silence, and silence still
  takes time; zeroing its duration now would be an unjustified divergence
  from what the transplanted decoder actually learned to expect. The only
  token that legitimately has *no* corresponding acoustic content is the
  `[LANG:xx]` control tag on the `lm_tokens` side (pure metadata, not a
  sound) -- so `zero_duration_mask` is entirely an `lm_tokens`-aligned
  field on the new artifact, sourced 100% from
  `LanguageModelTokenizer.zero_duration_mask()`; there is no phone-side
  zero-duration concept at all.
- A 192-dim phone embedding table is initialized from the source checkpoint's
  original scratch embedding where phoneme coverage overlaps, with fresh rows
  for anything uncovered (Vietnamese tones, Chinese finals) -- mirroring how
  `_make_pretrained_embedding` already handles vocab-size mismatches for the
  Qwen embedding.
- **Normalization is opt-in, per-instance, and scoped only to this new
  tokenizer** -- `EmiliaTokenizer`/`EspeakTokenizer` are never modified.
  Since one `FusionTokenizer` instance is always locked to a single `lang`
  (see point 5's language-selection discussion), `use_normalizer` is a
  plain `bool` (`FusionTokenizer(lang=..., use_normalizer=...)`), not a
  per-language dict; its default (when left `None`) is looked up per
  language from a private `_DEFAULT_USE_NORMALIZER` table --
  `{"en": True, "zh": True, "vi": False}`. Following
  `LanguageModelTokenizer`'s existing "no normalization by design"
  principle: English/Chinese default on, matching `EmiliaTokenizer`'s own
  always-on normalizers for those languages, so raw-corpus equivalence
  still holds by default; Vietnamese defaults off and is expected to stay
  off, since Vietnamese text is normalized by a separate, external pipeline
  before it reaches training (enabling it is a logged no-op -- no
  Vietnamese normalizer exists here). Whether normalization is
  enabled/disabled is logged at construction time (`INFO` for enabled,
  `WARNING` for disabled, since a disabled language's text must already be
  normalized upstream). This directly resolves the
  character-offset-alignment problem needed for the group correspondence
  in point 8: both phones and `lm_tokens` are derived from the *same*
  canonical (post-normalization, if enabled) text, so `lm_token` character
  offsets (`return_offsets_mapping`) always land on the same string the
  phone groups were split from -- no separate reconciliation between
  `EmiliaTokenizer`'s hardcoded internal normalization and
  `LanguageModelTokenizer`'s no-normalization default is needed.

**Output format**: the new tokenizer's legacy-compatible methods keep their
existing return contracts unchanged (for any caller still using them). A new
method returns a distinct, versioned per-utterance artifact (phones,
`lm_tokens`, group mapping, `lm_tokens`-aligned zero-duration mask,
normalization/language provenance, schema version) -- **not** a bare
tuple/`NamedTuple` alone, since that would not by itself prevent the
following real, already-verified data-integrity risk:
`train_zipvoice.tokenize_text()` currently reuses `supervision.tokens`
whenever that field already exists, without checking which tokenizer
produced it (`train_zipvoice.py:934-957`), and `SpeechSynthesisDataset`
reads it straight into `batch["tokens"]` (`dataset.py:87`). To actually
close this gap: `tokenize_text()` must validate the artifact's schema/kind
rather than blindly accepting a pre-existing `supervision.tokens`, and
`SpeechSynthesisDataset`/`prepare_input()` must be updated to collate the
additional fields (`lm_tokens`, group mapping, `lm_tokens`-aligned mask)
instead of only passing through the legacy `tokens` field. This is explicit
build-sprint scope, not something the tokenizer's return type can solve by
itself. The artifact's own `__post_init__` validates its cross-field
invariants at construction time (see below), so a malformed artifact
raises immediately rather than failing confusingly downstream in duration
allocation or dataset collation.

**Artifact invariants**, validated by `FusionTokenizerArtifact.__post_init__`
at construction time for every corpus/inference item (not just a finite
test fixture): every phone belongs to exactly one group; groups are
monotonic and non-overlapping; `lm_token_groups`, when present, has exactly
one entry per group with in-range indices; `len(zero_duration_mask) ==
len(lm_token_ids)` (**not** `len(phone_ids)` -- see above); the
normalization and language used to build the artifact are recorded; the
artifact carries a schema/version and rejects a stale one (cached Qwen
features and mappings can become stale independently of phone-sequence
equivalence).

**`zero_duration_mask` is never passed into phone-side duration allocation**
(`prepare_avg_tokens_durations()`/`get_tokens_index()` in
`zipvoice/utils/common.py`) -- clarified during sprint 003 planning after a
review round raised it as a possible shape-mismatch bug (the mask is sized
to `lm_token_ids`, while those functions expect a mask sized to
`tokens_lens`, which is phone count under this architecture). The mask only
ever made sense back when `tokens` meant `lm_tokens` (the earlier,
pre-fusion multilingual-tokenizer design, where the `[LANG:xx]` tag lived in
the same sequence phone duration allocation was never involved with). Under
this architecture, the phone sequence never contains a `[LANG:xx]` tag at
all (it's produced independently by phonemization), so
`prepare_avg_tokens_durations()`/`get_tokens_index()` need no masking
whatsoever and can keep operating on raw phone counts exactly like
`master`'s original, unmodified behavior -- no code change needed there, only
this explicit scoping note so the confusion doesn't resurface.

### 6. Qwen extraction contract

Run text through Qwen2.5-0.5B's embedding table plus a genuinely **truncated**
forward through the first N of its 24 pretrained transformer layers (not an
intermediate hidden state sliced out of a full 24-layer forward -- these are
different features). Qwen's own final RMSNorm (calibrated for 24-layer-deep
statistics) is **not** applied to the N-layer output -- the raw layer-N
hidden state is used, since the freshly-trainable `linear(896->192)`
projection immediately downstream can absorb whatever scale/distribution
results, and each decoder layer already contains its own internal
input/post-attention RMS normalization. `use_cache=False` is set explicitly
(Qwen2 defaults to `use_cache=True`, building an unused KV-cache on every
forward pass otherwise -- harmless to correctness for a one-shot extraction,
but wasteful and worth pinning down explicitly as part of a reproducible
extraction contract).

This computation is **deterministic and cacheable in principle** for fixed
training-corpus text (frozen Qwen weights, fixed input) -- but **v1 runs it
live instead**, inline during each training step, rather than
precomputing/caching to disk. Decided during sprint 003 planning: Qwen2.5-0.5B
is small and only N=4 layers are ever run, so a live forward pass is expected
to be cheap enough that the added engineering cost of a cache (serialization
format, invalidation whenever the extraction contract or normalization
changes, collation plumbing) isn't justified yet. Revisit only if profiling
during sprint 003/004 shows the live forward pass is actually a training
bottleneck. Inference-time text always requires a live Qwen forward pass
regardless (arbitrary text can't be precomputed). **`N=4` layers,
empirically validated** (14.9M params/layer) via a linear-probe experiment
(`temp/tokenizer/{build_lang_probe_dataset,train_lang_probe}.py`, not
committed to the repo -- local/throwaway by convention): a frozen-Qwen,
word-level EN/VI language probe, trained and tested on disjoint vocabulary
(no word appears in both splits) restricted to genuinely surface-ambiguous
plain-ASCII words (real Vietnamese words missing diacritics vs. real
English words/names -- diacritic-marked Vietnamese was excluded as a
trivial, uninformative case: Qwen's byte-level BPE already tokenizes
diacritic vs. plain-ASCII content completely differently, so including it
produced a meaningless ~100%). Result: ~96-98% accuracy at every depth
tested (N=2/4/8/12/24), with no meaningful difference between N=2 and N=4
(within noise on a 300-word test set) and N=24 (full model) slightly
*worse* -- so depth barely matters once past the earliest layers. `N=4` was
kept over the equally-valid `N=2` as the lower-risk choice (small added
compute cost, some margin over the bare minimum, consistent with this
ADR's original starting guess) rather than because the data required it.
Caveat: this experiment mainly shows Qwen has a strong per-word *lexical*
prior for which language a given surface form typically belongs to, not
that it resolves a word genuinely balanced across both languages via
sentence context (rare enough in practice that it wasn't tested directly).

### 7. Duration allocation

Duration allocation reverts to **phone-count**, not BPE/`lm_token`-count. The
group-level fused representation (Qwen branch + phone branch, combined by the
per-group gate) must be **deterministically expanded onto the phone
sequence** before `forward_text_condition()`, since `prepare_avg_tokens_durations()`/
`get_tokens_index()` gather from a phone-granular sequence, not a
group-granular one (`zipvoice.py:343`, `common.py:342`). This is a more
principled proxy for true phonetic duration than the current uniform-per-BPE-token
scheme, though still an approximation (equal duration per phone, not real
phoneme-specific durations).

### 8. Rejected: learned cross-attention for alignment

We explicitly reject learned cross-attention between the phone and
`lm_token` sequences as the mechanism for establishing their correspondence.
The word/group-to-phone and word/group-to-`lm_token` correspondences are
deterministic and computable (via each phonemizer's own boundary markers and
the Qwen tokenizer's `return_offsets_mapping` support, confirmed against the
actual saved tokenizer artifact and installed `transformers` fast-tokenizer
backend) -- not something the model should have to re-learn from limited
training data. (What genuinely *is* learned, and is not deterministic, is
the *gate* deciding how much to trust each branch per group -- that is a
different, intentional learning problem from establishing correspondence.)

## Alternatives Considered

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Status quo: keep joint-training the current BPE-only architecture longer | Zero implementation cost | 76k-batch plateau, no improvement from 48k; argues this alone won't converge | Empirically not working |
| Per-word language-ID classifier (repeat author's BERT-classifier approach) inside the pipeline, then language-aware phonemization | Simple, well-understood approach | Already tried, fails on exactly the hard cases (surface-form-ambiguous words, short spans in long context) | Demonstrated insufficient accuracy |
| Multi-candidate (per-language) phoneme gating, blending several G2P hypotheses per span | More robust to genuinely ambiguous spans in principle | Real added complexity (multiple G2P calls per span, more gate parameters); author explicitly does not want this | Simplified away in favor of a single phone branch |
| Fuse-then-project (concatenate raw Qwen/phone representations, then project to 192) | Fewer total modules | Forces one projection to do two jobs (compress Qwen's space *and* merge phoneme info); `in_proj` can't be reused/transplanted | Weakens source-interface preservation |
| Fuse after independent 192-dim projections, but keep feeding `text_encoder`'s *existing* `in_proj` (unmodified, still 896-dim-input) for the Qwen branch specifically | Keeps `in_proj`'s original role literally unchanged | Requires modifying `TTSZipformer.forward()` to skip its own internal `in_proj` call for this path; `in_proj` stays un-transplantable (still 896-dim shape) | More invasive code change, and forgoes the `in_proj` transplant opportunity |
| Cross-attention between phone and `lm_token` sequences (learned alignment, no explicit group mapping) | No word-boundary/offset-mapping engineering needed; flexible | Re-learns a deterministic, computable correspondence from limited training data; adds new trainable parameters competing for scarce gradient signal; undermines the low-data motivation for the whole effort | Wrong tradeoff given this project's actual data constraint |
| Hard gate floor / permanent entropy regularization to prevent gate collapse | Prevents total collapse toward one branch | Directly contradicts approximate-source-reproduction-at-init; also fights legitimate ~100%-phone routing for genuinely monolingual/reliable words | Rejected in favor of accepting the collapse risk and monitoring for it (point 4) |
| Embedding-only training curriculum (freeze all but `embed`+`in_proj`) as the primary fix | Real precedent (Artetxe et al. 2020); simpler than fusion | Doesn't address the root cause (BPE tokenizer + limited data can't learn code-switch disambiguation on its own); real gaps in this codebase's freeze/unfreeze machinery | Deprioritized behind fusion, retained as background |
| Embedding-space alignment/distillation toward the source's phoneme embeddings | Real TTS precedent (Tu et al. 2019, Huang et al. 2022) | Same underlying problem as embedding-only curriculum -- doesn't give the model new pronunciation information, just a better starting point | Deprioritized behind fusion, retained as background/complementary |

## Consequences

**Easier:**
- `in_proj` and (partially) the phone embedding table become transplantable
  from the converged source checkpoint again -- more of the model starts from
  known-good weights than the current warm-start achieves.
- No changes needed inside `TTSZipformer`/`text_encoder`'s own code --
  isolates the new complexity to the text-frontend construction path
  (`forward_text_embed()` and the new fusion modules), lower regression risk
  to the already-working, transplanted decoder/encoder internals.
- Duration allocation moving to phone-count is a more principled proxy for
  true phonetic duration than the current uniform-per-BPE-token scheme.
- Chinese English-insertion handling and tone sandhi require no new
  linguistic logic -- both are already correctly handled by preserving
  `EmiliaTokenizer`'s existing behavior.
- No inherited function/variable needs renaming for terminology clarity (see
  Terminology note) -- `lm_tokens` is purely additive.

**Harder:**
- New preprocessing pipeline required: a new wrapping tokenizer with a
  strict phone-equivalence test, Qwen N-layer feature extraction with a
  precisely pinned-down contract, group construction and validation,
  offline caching with schema/versioning -- real engineering, not a small
  patch.
- Duration allocation logic (`prepare_avg_tokens_durations`,
  `get_tokens_index`) needs a deterministic group-to-phone expansion step
  before it, touching code shared with the training path.
- Manifest/dataset plumbing changes are required beyond the tokenizer itself:
  `train_zipvoice.tokenize_text()`'s blind `supervision.tokens` reuse and
  `SpeechSynthesisDataset`'s tokens-only batching both need updating to
  validate and collate the new artifact's fields.
- New failure surfaces, several accepted rather than mitigated: gate collapse
  toward phone-only (accepted risk, monitored not prevented); the phone
  branch is deliberately wrong on minority-language spans by design (accepted
  bet, not a bug). Two risks listed here originally were subsequently
  retired by measurement, not by mitigation: **phone-inventory gaps**
  (Vietnamese tones, Chinese finals) versus the source checkpoint's phoneme
  set did not materialize -- `FusionTokenizer` produces zero phonetic OOV
  against the source's 360-symbol vocabulary (VI 0/276,428; EN 0/87,011; ZH
  300/56,860, all punctuation), and both candidate source checkpoints ship
  that identical `tokens.txt`, making the embedding transplant 1:1 with no
  uninformed rows; and **branch-magnitude normalization** is now defined
  (see point 4).
- Flow-matching loss alone (already shown to plateau without indicating
  content-correctness) is not sufficient to judge this work's success.
  Quality evaluation is handled in a separate project outside this repo.

**Neutral / deferred:**
- This ADR does not resolve whether the *current* checkpoint's plateau is
  actually caused by the phone/`lm_token`-granularity problem this fusion
  targets, versus a data/tokenizer correctness bug, versus the missing
  `--finetune` flag on the warm-start training script. The implementation
  plan sequences cheap diagnostics for these *before* building any of the
  architecture described here, specifically to avoid building real
  engineering on top of an unconfirmed diagnosis.
- Whether the "Qwen branch corrects wrong phones" bet succeeds at all is
  explicitly unresolved by this ADR -- see point 1 and the plan's sprint 003
  (whose step 1 now carries the fixed-gate validation formerly proposed as a
  separate sprint 002 -- see the note below).

## Implementation review (2026-09-06)

The built implementation was reviewed adversarially over six rounds against
an external reviewer (Codex, `gpt-6-astra`, high reasoning effort), which
proposed checks and read diffs but ran no commands itself -- every
measurement below came from this project's own runs, so this is a second
reading plus command design, **not** independent verification.

**Verified with evidence, not argument:**

- *Truncation is exact.* The physically-shortened 4-layer model's output is
  bit-identical (`max_abs = 0.0`) to the full 24-layer model's
  `hidden_states[4]`, in fp32 and under fp16 autocast. Replacing
  `model.norm` with `Identity()` and running only the first N layers
  reproduces the intended intermediate state exactly.
- *No padding leakage.* Replacing every pad position's token id with a
  random real token leaves outputs bit-identical in both precisions. Repeats
  at fixed shape are also exact, so the extractor is deterministic per shape;
  only batch *shape* perturbs results (~1e-6 relative in fp32, ~5e-4 in fp16,
  i.e. fp16 epsilon), which is accepted as frozen-extractor noise.
- *Transplant is byte-exact.* Every non-fresh tensor equals its source under
  `torch.equal`, no source tensor goes unused, and the source and target
  vocabularies (360 symbols), model config and feature config are identical.
- *No other initialization traps.* All 894 loaded trainable tensors are
  finite and none is near `param_min_rms`. The 23 tensors that are zero at
  construction are all overwritten by the transplant.

**Defects found and fixed** (each has a regression test): the zero-init gate
weight described in point 4; a crash when a batch mixed an alignment-failed
utterance with aligned ones, which killed the batch instead of degrading to
phone-only; a group reduction that was dtype-correct only because RMS
normalization happened to promote fp16 back to fp32; supervision metadata
leaking across tokenizer switches; integer `tokens` from a fused cut being
silently dropped by a string-keyed vocabulary into an empty phone sequence;
`include_lang_tag=False` being accepted and then ignored, which would have
run the identical computation while claiming the tag was removed; and a
resume that silently changed `qwen_layers`, `tokenizer`, `text_frontend` or
`pretrained_tokenizer_name` without complaint.

**Subsequently closed by execution** (2026-09-06, after the review): a
bounded GPU run on a 60-cut subset completed two epochs under fp16 across
both languages, then resumed from `epoch-1.pt` and continued -- so the
save/resume round trip, which the review had flagged as never exercised,
now has been. The resume guard was verified under real conditions too:
resuming with a changed `--qwen-layers` exited non-zero with the intended
error and never reached training. The same run gave an end-to-end A/B of
the gate fix on real data and the real optimizer -- identical runs differing
only in the gate weight's initialization moved it 3.68e-07 per step
(small-random) versus 4.75e-10 (zero), a **775x** difference, confirming on
real gradients what the synthetic probe predicted.

**Still not covered**, and therefore still unknown: sustained numerical
stability over a full-length run (one is now in progress, see sprint 004's
run log); multi-GPU execution (the DDP path was read, never exercised);
corpus-wide alignment accuracy, convergence or speech quality; and the
deferred inference path, which still raises `NotImplementedError`.

## Why a separate pre-architecture oracle-validation sprint was dropped

An earlier version of this plan (sprint 002, since removed) proposed
validating -- before building any of the model-architecture changes above --
that a source-compatible phone path could recover correct EN/VI
code-switched pronunciation on the *existing* public source checkpoint,
using hand-verified "oracle" phones to bypass the language-ID problem
entirely. Attempting to execute it surfaced two reasons to drop it as a
separate, dedicated pre-architecture sprint:

1. **EN/VI code-switching has no rule-based per-word signal, unlike EN/ZH.**
   Han-vs-Latin script separates EN/ZH code-switching almost for free
   (`EmiliaTokenizer`'s existing routing already does this) -- an
   oracle-phone validation run on EN/ZH content would prove little about the
   actually-hard EN/VI case this project needs, since a from-scratch
   classifier already failed specifically on EN/VI's hard cases (see
   Context, point 2). Any oracle validation has to be run on genuine EN/VI
   content, not EN/ZH, to say anything relevant here.
2. **The only currently-cached checkpoint is not a stable target to validate
   against.** `hynt/ZipVoice-Vietnamese-2500h` -- the actual checkpoint this
   project's own warm-start (`exp/all_warmstart/`) was transplanted from --
   was fine-tuned exclusively on Vietnamese data and has lost the ability to
   pronounce English. An oracle-phone test against it would confound "can
   the backbone render EN/VI code-switched content correctly" with "did this
   specific checkpoint's fine-tuning regime retain English pronunciation at
   all" -- a poor result would say nothing about backbone capacity in
   general, only about this one checkpoint's own training history.
   Compounding this: the plan (per author discussion) is to switch the
   warm-start source to the *original* upstream checkpoint once the fusion
   architecture is built, since it was never Vietnamese-only fine-tuned, and
   this project's own training corpus (which does retain some English
   content, even if currently mislabeled -- see sprint 001's findings)
   should avoid re-inducing the same forgetting going forward. There is
   therefore no *stable* checkpoint available right now to validate against
   before the architecture exists -- validating now against a checkpoint
   about to be discarded would not inform the actual go/no-go decision.

The oracle-validation idea is not abandoned, just relocated: it survives as
sprint 003's fixed-gate (oracle/uniform/wrong) two-branch comparison (see
that sprint's Approach, step 1), which runs against whichever checkpoint is
actually the current warm-start source at that point in the project, and
which already needs the new architecture's code to exist anyway. A separate,
dedicated pre-architecture sprint for it would have added a validation step
that could only be run meaningfully once, against a checkpoint immediately
superseded by the plan's own next step -- not a useful gate.
