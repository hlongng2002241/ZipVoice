# Sprint 003: Build the Qwen+phoneme fusion text frontend

- **Date:** 2026-09-02
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Draft
- **Related:** [2026-09-02__qwen_phone_fusion_overview.md](2026-09-02__qwen_phone_fusion_overview.md),
  [2026-09-02__sprint_000__new_tokenizer.md](2026-09-02__sprint_000__new_tokenizer.md)
  (provides the phone/`lm_token`/group artifact this sprint consumes),
  [2026-09-02__sprint_001__diagnostics.md](2026-09-02__sprint_001__diagnostics.md)
  (must support proceeding first),
  [../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md](../../adr/2026-09-02__qwen_phone_fusion_text_frontend.md)
  (see also "Why a separate pre-architecture oracle-validation sprint was
  dropped" -- what used to be sprint 002 is now this sprint's step 9)

## Goal

Implement the accepted fusion architecture: a context-aware Qwen branch and a
phoneme branch, fused by a learned per-group gate, feeding a repositioned,
partially source-transplanted `text_encoder`.

**This is the first sprint in this plan that touches model-architecture code**
(`zipvoice/models/zipvoice.py`, `zipvoice/models/modules/zipformer.py`) --
sprints 000-001 deliberately do not, per the overview's explicit constraint.

## Background

Only start this sprint once sprint 001's diagnostics have been reviewed (no
structural bug found that would explain the plateau on its own). See the ADR
for the full architecture diagram and rationale, and its Terminology note for
the `tokens` (= phones) / `lm_tokens` (= Qwen/BPE) convention used throughout.

A dedicated pre-architecture oracle-validation sprint (formerly sprint 002)
was dropped -- see the ADR's note on why. In short: the only checkpoint
available to validate against is not the one this project will actually
warm-start the fused architecture from (the currently-cached
`hynt/ZipVoice-Vietnamese-2500h` was fine-tuned Vietnamese-only and has lost
English pronunciation; the plan is to switch to the original upstream
checkpoint once this sprint's architecture exists), so that validation now
happens near the *end* of this sprint instead (step 9), against whichever
checkpoint gets pinned as authoritative in step 1 -- not assumed fixed when
this plan was written.

## Scope

- **In scope:** locking down the fusion equation, Qwen extraction, and
  two-vocabulary config contracts before writing model code; building a
  small (10-20 utterance) manually-annotated EN/VI code-switch evaluation
  set; new model code (phone embedding table, Qwen-branch projection,
  per-group fusion gate, group-to-phone expansion with its inherited
  sentinel-padding requirement, `text_encoder.in_dim=192`); source-transplant
  of `in_proj` and the phone embedding table; duration allocation moved to
  phone-count; the manifest/dataset plumbing changes needed to safely
  consume sprint 000's tokenizer artifact; config enforcement for
  fusion-mode training; a new warm-start checkpoint builder with explicit
  tensor-name mappings; a short calibration pass for the new components;
  and, last, the fixed-gate (oracle/uniform/wrong) two-branch comparison
  against the evaluation set (this absorbs what used to be sprint 002's
  job, since it needs this sprint's model code -- and a calibrated one --
  to exist first).
- **Out of scope:** offline caching of Qwen features (v1 runs Qwen live
  during training -- see step 1); training the model to convergence
  (sprint 004); multi-candidate per-language gating (rejected, see ADR);
  cross-attention alignment (rejected, see ADR); finalizing the
  inference-time composition/chunking contract (deferred, see Approach).

## Approach

Reordered after a Codex-assisted implementability review found the original
9-step list wasn't actually sequential (step 1 secretly depended on most of
the others, among other gaps -- see git history of this file for the
pre-review version). New order locks contracts first, builds what's
independent early, and puts the fixed-gate comparison last, once there's
something real to compare.

1. **Lock down contracts before writing model code** -- each of these
   resolves a specific gap the review round found:
   - **Fusion equation**: only the Qwen branch is collapsed to group
     granularity -- the phone branch is not. Each phone keeps its own
     individual embedding at its own position; the per-group gate combines
     each phone's own embedding with the *same* pooled-and-broadcast Qwen
     group vector, repeated across every phone in that group:
     `fused[i] = gate[g] * phone_embed[i] + (1 - gate[g]) * qwen_group_vec[g]`
     for phone `i` in group `g` (or whatever the exact combination turns out
     to be -- the point is `phone_embed[i]` stays indexed by `i`, never
     collapsed to `g`). See the ADR's point 3 for the full note on why this
     needed to be made explicit.
   - **Qwen extraction**: N=4 layers (already validated, see step 2 note
     below), **last-subtoken pooling** per group (mean pooling documented
     as a later alternative if this underperforms, see ADR point 3), run
     **live during training** (inline forward pass each step, not
     precomputed/cached -- Qwen2.5-0.5B is small and only 4 layers run, so
     the cache's added complexity isn't justified for v1; see ADR point 6).
   - **`zero_duration_mask` scoping**: it is `lm_token`-aligned and is
     **never** passed into `prepare_avg_tokens_durations()`/
     `get_tokens_index()` (which operate on phone counts and need no
     masking at all under this architecture -- see ADR's "Artifact
     invariants" section). This was a real point of confusion during
     review (the mask concept only ever made sense when `tokens` meant
     `lm_tokens`, in the pre-fusion design) -- stating it here so it
     doesn't resurface as a "shape mismatch bug" during implementation.
   - **Two-vocabulary model config**: `train_zipvoice.run()` currently
     derives one `vocab_size`/`pad_id` from one tokenizer and passes it
     into `ZipVoice`. Fusion needs a phone vocabulary/pad-id pair *and* a
     Qwen tokenizer/model identity, both threaded through model
     construction -- define this as an explicit fusion-mode config branch,
     not a single changed literal (`in_dim=192` alone is not sufficient --
     see step 3).
   - **Pin the authoritative source checkpoint**: `k2-fsa/ZipVoice`,
     file `zipvoice/model.pt` (the upstream author's own release, the
     `HUGGINGFACE_REPO` `infer_zipvoice.py` already points at), per the
     author's decision to move off `hynt/ZipVoice-Vietnamese-2500h` once
     this architecture exists (hynt's is Vietnamese-only fine-tuned and has
     lost English). Recorded explicitly because "whichever checkpoint is
     currently the warm-start source" is not reproducible enough for a
     result anyone should treat as a gate.

     **Checked, not assumed** -- two facts that materially simplify steps 3
     and 7:
     - The two candidate checkpoints ship the **identical 360-entry
       `tokens.txt`** (compared directly; zero symbols differ in either
       direction). hynt's Vietnamese fine-tune never extended the phone
       inventory, so the checkpoint choice has *no* effect on phone-vocabulary
       coverage or on the transplant's shape.
     - `FusionTokenizer`'s real output stays inside that inventory:
       measured over 400 utterances per language, **VI 0 OOV / 276,428
       phones, EN 0 OOV / 87,011 phones, ZH 300 OOV / 56,860 (0.53%)** --
       and every ZH miss is exotic punctuation (（）「」《》—…·), not a
       phone. So the phone embedding transplants **1:1 with no fresh rows**,
       and the "Vietnamese tones / Chinese finals missing from the source
       inventory" worry does not materialize.
   - **`lm_token_groups` fallback**: when `FusionTokenizerArtifact.lm_token_groups`
     is `None` for an utterance (alignment failed -- a known residual risk
     from sprint 000), treat that utterance as **phone-only** (zero Qwen
     contribution for every group in it), not a crash or a silent
     misalignment. Simple, safe, consistent with never crashing on this
     documented edge case.
2. **Build a manually-annotated EN/VI code-switch evaluation set** (10-20
   utterances, per-span hand-verified phones -- correcting for wherever
   primary-language-only phonemization would get a minority-language span
   wrong). Independent of everything else here -- can be built any time,
   does not need model code. This is the artifact step 9's fixed-gate
   comparison consumes.
3. **Model code changes** (`zipvoice/models/zipvoice.py` and construction
   path). **Implemented** as
   `zipvoice/models/modules/qwen_extractor.py` (`TruncatedQwenExtractor`),
   `zipvoice/models/modules/fusion.py` (`PhoneQwenFusion`,
   `build_phone_group_index`), and a `text_frontend="fusion"` branch in
   `ZipVoice`; tested by `tests/test_qwen_extractor.py`,
   `tests/test_fusion_module.py`, `tests/test_zipvoice_fusion.py` (30 tests,
   all passing; the whole suite stays green, so the pre-fusion "embedding"
   frontend is provably untouched). Three things the ADR left open and this
   implementation pinned down, each recorded in the relevant module's
   docstring:
   - **Gate input** = `[projected Qwen group vector ; mean phone embedding
     of that group]`. The gate's real question is a *mismatch* one -- the
     phone branch is wrong exactly where a group's true language differs
     from the language the utterance was phonemized under -- so it needs to
     see both sides, not just Qwen. The phone mean feeds the gate scalar
     only; the output path still uses each phone's own embedding.
   - **Gate shape** = one scalar per group (not per-channel), matching the
     ADR's "eps mixing weight" language; widening it to per-channel is a
     one-line change if wanted later.
   - **Branch-magnitude normalization** (the ADR's must-resolve item): the
     Qwen branch is RMS-normalized then rescaled by a learnable scalar
     initialized to the phone embedding's own RMS, so `eps=0.01` means "1%
     of a phone-embedding-sized vector" rather than 1% of an arbitrary
     fresh projection's scale. `PhoneQwenFusion.calibrate_qwen_scale()`
     re-derives it after the transplant replaces the random init -- step 7
     must call it.

   Also decided while implementing: **Qwen is not an `nn.Module` member of
   `ZipVoice`.** Its weights are frozen and reproducible from the
   HuggingFace id, so registering them would add ~196M frozen parameters to
   every checkpoint (and to `model_avg`'s float64 copy) and force step 7's
   builder to reason about tensors that never change. The extractor runs
   beside the model and its pooled output is passed in, exactly like
   `tokens` already are -- which also keeps "run live vs. cache later" a
   caller-side decision, swappable without touching model code.
   - New `linear(896->192)` for the Qwen branch (fresh init -- nothing in
     the source checkpoint informs this layer).
   - New phone embedding table (192-dim; rows initialized from the source
     checkpoint's original scratch embedding where phoneme coverage
     overlaps, fresh rows for anything uncovered -- mirroring
     `_make_pretrained_embedding`'s existing vocab-mismatch handling).
   - New per-group fusion gate module, implementing step 1's fusion
     equation (per-phone phone embedding, group-broadcast Qwen vector).
     Initialized near-phone-only (`eps=0.01` Qwen mixing weight) but with
     **no hard floor and no entropy regularization** -- per the ADR, gate
     collapse toward phone-only is an accepted risk, monitored (sprint 004)
     rather than architecturally prevented. Branch-magnitude normalization
     between the Qwen projection's output and the phone embedding's output
     must be defined here (not left implicit), since `eps=0.01` has no
     precise functional meaning without it.
   - **Group-to-phone expansion, including the inherited sentinel-padding
     position**: `pad_labels()` (`zipvoice/utils/common.py`) appends one
     extra pad position beyond the real phone count, which
     `get_tokens_index()` relies on being able to gather from (a residual
     index equal to `n_tokens` itself) -- this works today only because the
     plain embedding-table lookup naturally produces a valid vector there.
     The new fused, group-expanded sequence must **explicitly append the
     same pad-embedding vector** (`self.embed(pad_id)`) as its own final
     position, or `get_tokens_index()` will index out of range on the
     longest sequence in a batch -- easy to miss since shorter sequences in
     the same batch would already have slack from ordinary padding and
     wouldn't trigger it. No change needed to `get_tokens_index()` itself.
   - `text_encoder` constructed with `in_dim=192` instead of 896 via the
     fusion-mode config branch (step 1); `in_proj` initialized from the
     source checkpoint's original `in_proj` (now shape-compatible). No
     changes needed inside `TTSZipformer`/`text_encoder` itself.
   - `forward_text_embed()` **and every other call site currently reaching
     the tokens-only interface** updated to compute the fused,
     group-expanded representation instead of `self.embed(tokens)`
     directly: `forward_text_inference_gt_duration()`,
     `forward_text_inference_ratio_duration()`, `sample()`,
     `sample_intermediate()`, and the training-time loss/OOM-scan call
     sites in `train_zipvoice.py` -- enumerate and check off each one
     rather than assuming the two most obvious paths are the only ones.
4. **Duration allocation to phone-count. Training side: nothing to change**
   -- keeping the artifact's phone ids on the existing `supervision.tokens`
   field (the ADR's terminology note: `tokens` *means* phones) means
   `tokens_lens` already *is* the phone count, so
   `prepare_avg_tokens_durations()`/`get_tokens_index()` operate on phone
   units with no edit at all, and no mask is threaded into them (step 1).
   Verified end-to-end by `tests/test_zipvoice_fusion.py`'s
   sentinel/duration test. The remaining pieces --
   `forward_text_inference_ratio_duration()` and `infer_zipvoice.py`'s
   chunking (`chunk_tokens_punctuation()`/`batchify_tokens()`) -- are
   inference-side and blocked on the deferred composition decision below.
5. **Manifest/dataset plumbing. Implemented.** The manifest itself is
   unchanged -- what was missing was the in-memory path: `tokens` (plus the
   optional mask) was all that travelled from tokenizer to model, and the
   fusion model needs three more per-utterance fields. Now:
   `tokenize_text_fusion()` (new) picks the right per-language
   `FusionTokenizer` from the cut's own `supervision.language` and attaches
   `tokens`/`phone_groups`/`lm_token_ids`/`lm_token_groups`;
   `SpeechSynthesisDataset` collates the three extras (only when present, so
   every other tokenizer keeps its exact batch schema); `prepare_input()`
   grows a `return_fusion_fields` flag on the same "don't change existing
   return shapes" principle as the mask; `compute_fbank_loss()` runs the
   Qwen extractor live and passes the pooled features to the model. Tested
   end-to-end by `tests/test_fusion_pipeline.py` (real text -> real loss
   with gradients reaching the Qwen projection).

   Note the artifact-schema-validation part of this step turned out to be
   defensive only, not a live bug: sprint 001 checked all four manifests and
   **none carries a precomputed `tokens` field** (0 of 171,961
   supervisions), so `tokenize_text()`'s blind-reuse branch never fires on
   this data.
6. **Config enforcement. Implemented** for the training-side half: fusion
   mode now raises if `--lang-auto-prob`/`--lang-wrong-prob` are nonzero
   (ADR point 2 -- the phone branch always uses the true
   `supervision.language`, so a sampled or wrong tag would desynchronize the
   branches), with the reason in the error message. The language-code
   mapping table and the inference-time `--lang` default are inference-side
   and wait on the deferred decision below.
7. **Build a new warm-start checkpoint. Implemented** as
   `scripts/fusion/m00_build_warmstart_checkpoint.py` (new `scripts/fusion/`
   folder, alongside `model.json` and `m01_train.sh`; the manifests from
   `scripts/all/m00_prepare_manifest.py` are reused unchanged). Kept
   separate from `scripts/all/m04_build_warmstart_checkpoint.py` because
   that one asserts identical key sets and copies same-named tensors --
   which here would either abort or, worse, silently leave the phone
   embedding random. The new builder states the mapping explicitly
   (`embed.weight -> fusion.phone_embed.weight`, everything else by name,
   five genuinely-new fusion tensors fresh), refuses to run if any
   *unexpected* tensor finds no source counterpart, and refuses if any
   source tensor would be silently discarded.

   **Result, run against `k2-fsa/ZipVoice`: 889 of 894 tensors
   transplanted** -- better than the pre-fusion warm-start's 887/889,
   because `text_encoder.in_proj` is 192->192 again and therefore
   transplantable (verified byte-identical to the source, as is the full
   phone embedding and every `fm_decoder` tensor). Only the Qwen
   projection, the gate, and the scale start fresh. `calibrate_qwen_scale()`
   runs after the transplant (giving 0.4887, the transplanted embedding's
   RMS) so `--gate-init-eps` has its intended meaning; verified to survive
   the save/load round-trip and to load `strict=True` into a model built
   exactly as `run()` builds it (122,837,414 params).
8. **Short calibration pass before the fixed-gate comparison is meaningful.**
   The Qwen projection and gate are freshly initialized -- an "oracle" gate
   that routes to an untrained, random projection doesn't test anything
   about attainable pronunciation benefit, it tests a random vector against
   a working decoder. Train just the new components (Qwen projection, gate;
   optionally a light fine-tune of the rest) with **`--finetune True`
   explicitly set** (see sprint 001's finding: this flag is easy to forget
   and materially changes training dynamics for a warm-started model) for
   a short, defined number of steps before step 9 runs. This step's
   training protocol (which parameters move, how many steps, what loss) is
   itself an open item to define during implementation, not deferred
   further than that.
9. **Fixed-gate two-branch comparison**, now that there's a calibrated
   model to compare: using step 2's evaluation set and the checkpoint
   pinned in step 1, compare an oracle gate (favor whichever branch is
   known-correct per group), uniform averaging of both branches, and a
   deliberately-wrong gate. This establishes the attainable benefit and
   whether the two branches' outputs remain acoustically separable at all
   -- confirm this before investing in training a *learned* gate.

**Deferred, not decided here**: exact inference-time composition (how a
prompt artifact and a target artifact combine without producing two
`[LANG:xx]` tags in one causal Qwen context, and how long-text chunking
preserves phone/`lm_token` group boundaries). Per the author: this will be
decided empirically, by trying it, once the model exists and can actually
be run -- not resolved on paper in advance. Flagged here so it isn't
mistaken for an oversight; `apply_lang_tag()`'s existing prompt-only-tagging
convention in `infer_zipvoice.py` is the natural starting point to try
first. **The code enforces the deferral rather than guessing**: under
`text_frontend="fusion"`, `forward_text_inference_gt_duration()` and
`forward_text_inference_ratio_duration()` (and therefore `sample()`) raise
`NotImplementedError` pointing at this note, so generation cannot silently
run on an invented composition contract. Training is unaffected.

**Known accepted data-quality issue, not addressed by this sprint**: sprint
001 found ~95% of "English"-labeled training utterances are actually
Vietnamese text with embedded English words (a mislabeled upstream field,
left unfixed per the author's explicit choice). Under this architecture,
`supervision.language` is treated as immutable ground truth for *both* the
phone branch's phonemization and the deterministic `[LANG:xx]` tag -- so
those mislabeled rows get both branches corrupted together, not just one.
This sprint does not filter, relabel, or downweight them; they remain in
training as accepted noise. Worth stratifying gate-weight/quality
monitoring by this known-noisy subset during sprint 004, rather than
averaging over it silently.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Fixed-gate mixtures (step 9) are not acoustically separable (oracle vs. wrong gate sound similar) | A learned gate would have nothing meaningful to learn to select between | Re-examine whether the phone/Qwen branches actually carry distinguishable information at the fusion point before proceeding further in this sprint |
| An oracle gate at step 9 still looks bad because the calibration pass (step 8) under-trained the new components, not because the branches lack information | Could be misread as "the Qwen branch has no useful signal" when it's actually "the new components never got a fair chance to learn to use it" | Define step 8's protocol (steps/params/loss) deliberately enough to rule this out before treating step 9's result as conclusive; if in doubt, extend calibration before concluding |
| Word-count/group mismatches on real corpus text beyond what sprint 000 already validated | Broken correspondence for some fraction of utterances | `lm_token_groups=None` -> phone-only fallback for that utterance (step 1), not a crash or silent misalignment |
| Gate collapse toward phone-only (accepted risk, not mitigated by construction) | The Qwen-correction bet may never activate | Monitored via gate-weight logging in sprint 004, not prevented here; a collapsed gate is a valid (negative) experiment outcome |
| Branch-magnitude mismatch between the fresh Qwen projection and the source-initialized phone embedding | `eps=0.01` init has no defined functional meaning without this | Must be resolved in step 3 (explicit normalization or measured norm ratio) before training, not deferred |
| ~~Phone inventory gaps (Vietnamese tones, Chinese finals not in source's original phoneme set)~~ | ~~Those phones start from fresh (uninformed) embeddings~~ | **Resolved, measured** (step 1): zero phonetic OOV across VI/EN/ZH against the source's 360-symbol vocabulary (only exotic ZH punctuation misses, 0.53% of ZH phones), and both candidate checkpoints share that identical vocabulary -- the phone embedding transplants 1:1 with no fresh rows |
| The Qwen branch's contribution is necessarily uniform across every phone in a group (only the phone branch stays per-phone -- see step 1/ADR point 3) | Possible quality ceiling for groups where per-phone Qwen nuance would have helped | Accepted as the simplest correct starting point given Qwen operates on `lm_tokens`, not phones; revisit only if sprint 004 results indicate it's a binding constraint |
| Forgetting the sentinel-padding position in group-to-phone expansion (step 3) | `get_tokens_index()` indexes out of range, but only for the longest sequence in a batch -- easy to miss in small-scale testing | Explicit requirement in step 3; test with a batch whose longest sequence is the one under test, not just short smoke-test sentences |
| Running Qwen live (not cached) turns out to meaningfully slow training | Wasted training wall-clock, possible pressure to revisit the caching decision mid-sprint | Profile early (step 3-4) rather than discovering it deep into sprint 004's training; the ADR's point 6 already names this as the condition for revisiting |
| Mislabeled-language rows (sprint 001 finding, ~95% of "English"-labeled utterances) corrupt both branches together for those utterances | Training signal noise of unknown magnitude; could suppress or distort gate learning on affected utterances | Left unfixed per the author's choice (see Approach); stratify sprint 004's monitoring by this known-noisy subset rather than averaging over it silently |

## Testing & Validation

- Fixed-gate comparison (step 9) judged for acoustic separability before
  proceeding to the learned gate -- interpreted alongside step 8's
  calibration protocol (see the risk above about conflating "no signal"
  with "undertrained new components").
- Unit-level: group-to-phone expansion (including the sentinel-padding
  position) and offset-mapping correspondence tested against a real sample
  of the training corpus (exact match / documented mismatch rate), building
  on sprint 000's equivalence test. Include at least one test batch where
  the longest sequence is the one exercising the sentinel-padding path.
- Strict-load sanity check on the new warm-start checkpoint (all
  source-transplanted tensors load without shape errors) -- built against
  the checkpoint pinned in step 1, with its new explicit tensor-name
  mapping (step 7), not a shape-only copy loop.
- Smoke test: a handful of real training steps complete without error (same
  pattern as the original warm-start's smoke test), covering every call
  site enumerated in step 3, not just `forward_text_embed()`.

## Status (2026-09-06)

**Built and committed** (`1ac4b28`): the fusion module, the truncated Qwen
extractor, the `fusion` tokenizer path through the trainer, and
`scripts/fusion/` with the warm-start builder and launch script. 108 tests
pass. The warm-start checkpoint transplants 889/894 tensors byte-exactly.

Reviewed adversarially over six rounds; seven defects found and fixed, four
properties verified by measurement rather than argument. Both are recorded in
the ADR's "Implementation review" section rather than duplicated here,
together with the boundary of what that review did *not* cover.

**Smoke test (the sprint's "handful of real training steps" validation, and
then some), 2026-09-06.** A 60-cut / 20-dev subset covering both languages
ran two full epochs on GPU under fp16: 9 batches per epoch, validation
0.0406 -> 0.0472 -> 0.0423, peak 7152MB, checkpoints written at both the
`--save-every-n` cadence and epoch boundaries. It then **resumed** from
`epoch-1.pt` and ran on to epoch 3, and separately **refused** to resume
with a changed `--qwen-layers` -- exercising the resume guard end to end
rather than only in unit tests. `exp/fusion/warmstart_fusion.pt` was
verified unchanged afterwards.

That run also A/B'd the gate-initialization fix on the real model and data,
which the ADR's point 4 records: 3.68e-07 per step with the small-random
init versus 4.75e-10 with the old zero init, a 775x difference on real
gradients.

**Still outstanding in this sprint:**

- **Step 2** -- the 10-20 utterance manually-annotated EN/VI code-switch
  evaluation set. Nothing downstream can be *measured* without it; sprint
  004's comparison depends on it existing.
- **Step 8** -- the short calibration pass.
- **Step 9** -- the fixed-gate (oracle / uniform / wrong) comparison, which
  is where the ADR's relocated oracle validation actually lands.

Steps 8 and 9 are the sprint's own go/no-go on whether the Qwen branch
carries usable signal, so training the learned gate before they run means
accepting that risk knowingly rather than having retired it.

## Rollout

New architecture trains into a separate experiment directory, distinct from
both the from-scratch (`exp/all/`) and current warm-start (`exp/all_warmstart/`)
runs, so both remain available as baselines for sprint 004's comparison.
