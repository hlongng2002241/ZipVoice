# Restore backward compatibility with upstream `master`

- **Date:** 2026-08-29
- **Author:** LongNH (with Claude Code assistance)
- **Status:** Done. Steps 1-6 implemented and verified (steps 3-5 are one-line
  fixes/comments, folded into the same pass); step 7 (optional
  `Tokenizer.__module__`) skipped as genuinely not worth it. Additionally
  validated against the real, published `k2-fsa/ZipVoice` HuggingFace
  checkpoint on GPU: real inference (produced real, non-silent audio) and 5
  real fine-tuning steps continuing from that checkpoint's weights both
  succeeded. That real-world test caught one bug this plan's own synthetic
  tests missed (see "Testing & Validation").
- **Related:** implements
  [../proposals/2026-08-29__master_backward_compatibility.md](../proposals/2026-08-29__master_backward_compatibility.md)

## Goal

Make every default (non-multilingual) codepath in `mine` behave identically to
`master`'s — including model/optimizer-checkpoint resume and the
`prepare_input`/dataset-batch public contracts — **except** for two behavior
changes the user explicitly requested this session (skip-validation-at-batch-0,
unconditional `tqdm`), which are kept as named, documented exceptions rather
than reverted. Do this without touching any of this session's actual
multilingual functionality, and without checking out or rewriting history
against `master` (decided in the proposal — this is a fix-forward plan on
`mine`).

## Background

See the proposal for the full audit and the rewrite-vs-fix-forward decision.
Two independent LLM passes (a general-purpose diff audit, then two rounds of
Codex gpt-5.6-sol critique — one against the raw audit, one against the
written proposal/plan themselves) found real, previously-unknown compatibility
gaps and two incomplete fix designs in this plan's first draft, both corrected
below. Most importantly: `ZipVoice`'s constructor registers `self.embed` in a
different position *and* (for the scratch-embedding case) consumes the RNG
stream in a different order than `master`, which breaks optimizer-checkpoint
resume from a `master`-trained run even though model-weight loading still
works fine.

## Scope

- **In scope:** proposal items A1-A3 (with A1 and A2's complete, corrected fix
  designs below), B4 (documentation only), E9 — concrete, mechanical fixes and
  cleanup to files this session touched, verifiable without a full training
  run.
- **Out of scope:** items C5-C6 (intentional, already correct — no code
  changes, just confirmed-and-commented here as accepted exceptions), D7-D8
  (optional, listed as a nice-to-have at the end, not required for the
  compatibility goal), G (the double-tag inference bug — separate feature bug,
  not a `master`-compat issue), the pre-existing unrelated
  `app/`/`tools/mos/`/`.gitignore` divergence, and building any parallel
  from-`master` branch (explicitly decided against for now — see the
  proposal's "Alternatives Considered").

## Approach

1. **Fix `ZipVoice.__init__`'s parameter registration order *and* RNG
   consumption order** (`zipvoice/models/zipvoice.py`).

   The complete fix, correcting the first draft's incomplete version (which
   only reordered *assignment*, not *construction*, missing the scratch-case
   RNG-order problem Codex's second review caught):

   - **Scratch path** (`embed_source="scratch"`): don't construct
     `nn.Embedding(...)` early at all. `text_embed_dim` is already fully known
     from the constructor argument in this case — nothing needs to be learned
     from constructing anything. Build `self.text_encoder` first using the
     given `text_embed_dim`, exactly as `master` does, then construct and
     assign `self.embed = nn.Embedding(vocab_size, text_embed_dim)`
     **afterward** — this reproduces `master`'s original code path exactly:
     same construction order, same RNG consumption order, same parameter
     registration order. A from-scratch model built with the same seed as
     `master` should now produce bit-identical initial weights.
   - **Pretrained path** (`embed_source="pretrained"`): `text_embed_dim` here
     genuinely cannot be known without loading the source model (to read its
     hidden size), so `_make_pretrained_embedding(...)` still needs to run
     before `text_encoder` is sized. This is fine — `master` has no pretrained
     path at all, so there's no RNG-order compatibility to preserve here, only
     registration order. Store its result in a local variable (not yet
     assigned to `self`), build `self.text_encoder` next, then assign
     `self.embed = <that local variable>` afterward — so parameter
     *registration* order still matches `master`'s (`text_encoder` before
     `embed`) even though construction had to happen earlier out of necessity.
   - Net effect for both paths: `self.text_encoder = TTSZipformer(...)` is
     assigned before `self.embed = ...` in both branches, matching `master`'s
     registration order; the *scratch* path additionally matches `master`'s
     RNG consumption order exactly.

   - **Verify:**
     - `list(model.named_parameters())` order for a `mine`-built model (both
       `embed_source` values) must exactly match a `master`-built model's
       order (not just the same set of names) — this was already checked once
       for the incomplete fix; re-check after the corrected version.
     - For the scratch path specifically: build a model with `master`'s code
       and a model with `mine`'s (corrected) code, same seed, same
       `TINY_KWARGS`-style config; assert every parameter tensor is
       bit-identical, not just same-shaped. This directly tests the RNG-order
       part Codex's second review caught, which the first draft's plan didn't
       test at all.
     - **Corrected regression test for the optimizer-resume bug** (the first
       draft's proposed test didn't actually exercise the bug — it saved and
       reloaded within `mine` only, which would have worked even before the
       fix). The real test: (a) build a model + `ScaledAdam` optimizer using
       **`master`'s actual code** (e.g. via `git show master:... ` into a
       temp module, as already done once during this audit), take one
       training step on a fixed synthetic batch, save the optimizer's
       `state_dict()`; (b) build a same-config model + optimizer using
       `mine`'s (corrected) code, `load_state_dict()` the saved `master`
       optimizer state into it; (c) take one more step in *both* — continue
       the original `master` model+optimizer one more step, and continue the
       `mine` model+optimizer (loaded from `master`'s state) one more step, on
       the same synthetic batch — and assert the resulting parameters match
       (small floating tolerance). Checking only "does loading throw an
       exception" is insufficient, since same-shaped parameters at swapped
       positions can silently bind to the wrong per-parameter optimizer state
       without ever raising. Land this as a permanent test (e.g.
       `tests/test_checkpoint_compat.py`), not just a one-off manual check —
       this exact class of regression is easy to silently reintroduce later.

2. **Restore `prepare_input()`'s default return contract, and the dataset
   batch schema underneath it** (`zipvoice/utils/common.py`,
   `zipvoice/dataset/dataset.py`). Two layers, both need fixing — the first
   draft only fixed the first:

   - **`prepare_input()`**: add a new parameter,
     `return_zero_duration_mask: bool = False`. When `False` (the default)
     and `return_tokens=True`, return exactly `[tokens, ...]` as `master`
     does — identical arity to `master`'s original contract. When `True`,
     return `[tokens, zero_duration_mask, ...]` as today. Explicitly define
     the combination Codex flagged as undefined in the first draft: raise
     (e.g. `assert not (return_zero_duration_mask and not return_tokens)`) if
     `return_zero_duration_mask=True` is passed together with
     `return_tokens=False`, since a mask with no tokens to apply it to is
     meaningless — don't silently ignore the flag.
   - **`SpeechSynthesisDataset.__getitem__`**: only add the
     `"zero_duration_mask"` key to the batch dict when at least one cut in the
     batch actually has that attribute set (i.e., when using a tokenizer that
     produces it). For every batch built from the four original tokenizer
     types, the key is **absent entirely** — not present-but-`None` — making
     the batch dict schema identical to `master`'s, not just
     "additive-and-presumed-harmless." For multilingual-tokenizer batches, the
     key is present as today.
   - **Call sites**: `zipvoice/bin/train_zipvoice.py`'s three call sites pass
     `return_zero_duration_mask=(params.tokenizer == "multilingual")`
     explicitly (gated, not unconditional — cheap clarity improvement Codex
     asked for, even though passing it unconditionally was already numerically
     harmless once the dataset-layer fix lands). The nine call sites across
     `train_zipvoice_dialog.py`, `train_zipvoice_dialog_stereo.py`, and
     `train_zipvoice_distill.py` go back to their **original 3-value unpack**
     — they never need to know this parameter exists.

   - **Verify:** all 18 existing tests still pass, plus new direct tests for:
     `return_zero_duration_mask=False` (default) returns a 3-tuple identical
     in shape to `master`'s; `True` returns the 4-tuple with the correct mask;
     an all-`None` mask list normalizes to a bare `None`; a mixed `None`/mask
     batch normalizes untagged entries to all-`False`; the
     `return_zero_duration_mask=True` + `return_tokens=False` combination
     raises; a legacy-tokenizer dataset batch has **no** `"zero_duration_mask"`
     key at all (not `None` — absent); a multilingual-tokenizer batch does
     have the key. Also confirm via `git diff master -- <file>` that the
     three sibling-script files now show zero diff from `master` for this
     specific code path.

3. **Restore `log_interval` default to 50**
   (`zipvoice/bin/train_zipvoice.py`, `get_params()`). No one has asked for
   10; treat it as accidental drift and restore `master`'s value.
   - **Verify:** `git diff master` shows no change to this line.

4. **Document the two accepted exceptions explicitly in code**, so they read
   as decisions, not bugs, to the next reader (including future-Claude):
   - A one-line comment at the `batch_idx_train > 0` check in `train_one_epoch`
     noting this is a deliberate, permanent change from `master`'s original
     validate-at-batch-0 behavior, made at the user's explicit request
     (already has a version of this comment from the earlier session fix —
     confirm it's still accurate and present).
   - A one-line comment at the `tqdm(...)` wrapper noting it's an intentional
     UX addition applying to all tokenizer types, not multilingual-specific,
     also at the user's explicit request.
   - No behavior change in this step — purely comment/documentation, and
     purely to keep these two items from being mistaken for unfixed drift
     later.

5. **Document the checkpoint dtype-cast fix as an accepted exception**
   (`zipvoice/utils/checkpoint.py`). Add a short note (in the existing
   docstring/comment already added for this fix) stating precisely what was
   verified: loaded floating-point values and dtypes match `master`'s output
   elementwise; the *serialized file size* differs, most likely due to
   building a plain dict via comprehension rather than calling `.state_dict()`
   on the module directly (plausible cause, not fully isolated) — so nobody
   re-discovers the size difference later and mistakes it for a fresh
   regression, and nobody overclaims byte-identity it doesn't have.
   - **Verify:** no code change; just confirm the existing comment is accurate
     and doesn't overclaim byte-identity.

6. **Reformatting cleanup for `zipvoice/bin/train_zipvoice.py` and
   `zipvoice/utils/checkpoint.py`.**
   Per Codex's recommendation: rebuild the functional diff cleanly rather than
   hand-reverting individual reformatted lines. Concretely: take a fresh
   checkout of `master`'s version of each file, re-apply *only* the genuine
   functional changes from this session (multilingual tokenizer wiring, the
   corrected `zero_duration_mask` threading from step 2, `tqdm`, the
   `--resume-from-checkpoint`/`--warmup-batches` additions, the checkpoint
   dtype fix, etc.) by hand, preserving `master`'s original formatting/line
   wrapping everywhere the logic is actually unchanged. Do this last, after
   items 1-5 are settled (so the "genuine functional diff" being re-applied is
   final, not a moving target).
   - **Verify:** `git diff master -- zipvoice/bin/train_zipvoice.py`
     after cleanup should be visibly smaller and every remaining hunk should
     correspond to an actual behavior change described in this plan or the
     multilingual-frontend plan — no line should change for formatting
     reasons alone. Full test suite must still pass.

7. **(Optional, low-priority) `Tokenizer.__module__` fix**
   (`zipvoice/tokenizer/tokenizer.py`): if the user wants full fidelity, add
   `Tokenizer.__module__ = "zipvoice.tokenizer.tokenizer"` right after the
   re-export from `zipvoice/tokenizer/base.py`. Skip entirely if not judged
   worth the (very small) added surface.

## Risks & Open Questions

| Risk / Question | Impact | Mitigation / Owner |
| --- | --- | --- |
| Step 1's scratch-path fix (construct `nn.Embedding` only after `text_encoder`) must not accidentally change *which* value of `text_embed_dim` is used to size `text_encoder` | High if botched — would silently change model architecture | `text_embed_dim` for the scratch path is the constructor argument itself, never derived from constructing anything; only the *embedding module's construction* moves, not the dimension value used earlier |
| Step 1's pretrained-path fix still constructs (and downloads) the source model before `text_encoder` exists | Low — this is unavoidable and doesn't regress anything, since `master` has no pretrained path to match | Keep the download/dimension-computation call exactly where it is today; only move the `self.embed = ...` assignment line itself |
| Step 2's dataset-layer fix (conditionally adding the batch key) must not break the multilingual path, which relies on the key being present | Medium | Explicit test asserting the key **is** present for multilingual batches, not just absent for legacy ones |
| Step 6 (reformatting cleanup) is manual and error-prone at this file size | Medium — risk of accidentally reverting a real functional line while "restoring formatting" | Do it last, after 1-5 land and tests are green; diff the result against the pre-cleanup version function-by-function, not just eyeball it once |
| The double-tag inference bug (proposal item G) isn't fixed by this plan | Low for `master` compatibility (unrelated), but a real quality gap in the new inference feature | Left as an explicit open question in the proposal for the user to decide whether to fold in or track separately |

## Testing & Validation

- Full existing test suite (`pytest tests/ --ignore=tests/test_api.py`) must
  pass after every step, not just at the end.
- New tests, all permanent (not one-off manual checks):
  - Step 1: `named_parameters()` order match (both `embed_source` values),
    bit-identical scratch-path initial weights vs. `master` at the same seed,
    and the corrected master-state-loaded-into-mine optimizer-continuation
    test described above.
  - Step 2: the seven cases listed in step 2's Verify section (default arity,
    explicit-mask arity, all-`None` normalization, mixed-mask normalization,
    the newly-defined invalid-combination error, legacy batch has no mask key,
    multilingual batch has the mask key).
- **Real-world GPU validation** (beyond synthetic unit tests): downloaded the
  actual published `k2-fsa/ZipVoice` checkpoint from HuggingFace and, using
  this plan's fixed code:
  1. Ran real inference (`infer_zipvoice.py`, `--tokenizer emilia`, the
     checkpoint's own `model.json`/`tokens.txt`) — the checkpoint's 889-key
     `state_dict` loaded with `strict=True` (unaffected by the registration
     reorder, since it's name-keyed), and produced real, non-silent audio
     (8.6s, RMS 0.13) from a real prompt.
  2. Ran 5 real fine-tuning steps (`train_zipvoice.py --checkpoint <path>`)
     continuing from that checkpoint's weights on real English utterances
     from this session's own corpus, with `--use-fp16 True` and on-the-fly
     feature extraction — completed cleanly.
  This surfaced a real bug step 2's own synthetic tests missed: making
  `return_zero_duration_mask` conditional on `params.tokenizer ==
  "multilingual"` in `train_zipvoice.py`'s three call sites, without also
  conditioning the (always-4-value) LHS unpack, broke every non-multilingual
  tokenizer immediately (`ValueError: not enough values to unpack`) — caught
  the moment a real `--tokenizer emilia` run actually executed that code
  path, which no unit test happened to exercise. Fixed by requesting the mask
  unconditionally in `train_zipvoice.py`'s own three call sites instead
  (`compute_fbank_loss`/`model.forward()` already handle
  `zero_duration_mask=None` correctly for non-multilingual tokenizers) — this
  keeps the sibling scripts' arity restoration (which was the actual
  master-compatibility goal) while fixing train_zipvoice.py's own internal
  consistency.
- **A third Codex revalidation pass** (after implementation) found a real
  flaw in `test_optimizer_state_resume_from_master` itself: it called
  `saved_state = opt_master.state_dict()` without deep-copying, then advanced
  `opt_master` one more step *before* loading `saved_state` into `opt_mine` --
  since `ScaledAdam` mutates its state tensors in place on every `step()`,
  `saved_state` was silently aliased and mine ended up loading step-2 state,
  which is what actually forced the originally-reported loose tolerance
  (`atol=1e-2, rtol=5e-2`), not genuine independent-load floating-point drift.
  Fixed by `copy.deepcopy`-ing both the model and optimizer state immediately
  after step 1, loading those frozen states into fresh objects instead of
  hand-reproducing step 1, and syncing the global Torch RNG state before each
  side's second forward (`condition_time_mask` samples `torch.rand`). Also
  seeded `_synthetic_batch()` explicitly (it previously relied on ambient
  global RNG state left over from whatever ran before it in the same pytest
  session, making the test's actual numerical behavior depend on test
  execution order). With both fixed, the test now passes at `atol=5e-3,
  rtol=0` -- measured empirically (max observed residual 0.0028, attributable
  to ordinary floating-point non-associativity between two independently
  `importlib`-loaded copies of structurally identical code), a ~2-10x tighter
  bound than the original, and confirmed stable across repeated runs and
  execution order.
- Manual smoke check: after all steps, run
  `git diff master --stat -- zipvoice/ scripts/` (excluding the
  genuinely new multilingual-only files) and confirm the remaining diff is
  small, explainable line-by-line, and free of arity/order/default surprises.

## Rollout

No deployment/migration concerns — this is a single-repository code-quality
and compatibility fix, not a service change. Land as one or a small number of
focused commits (one per numbered step above is reasonable, to keep review
tractable), only after the user has reviewed and approved this plan and the
proposal it implements. No branch checkout or history rewrite against
`master` — all work happens as new commits on `mine`.
