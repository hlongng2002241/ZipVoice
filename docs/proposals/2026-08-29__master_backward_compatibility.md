# Restore backward compatibility between `mine` and upstream `master`

- **Status**: Accepted
- **Author**: LongNH (with Claude Code assistance)
- **Date**: 2026-08-29
- **Related**: covers the divergence introduced by this session's multilingual
  work — [2026-08-28__mbert_multilingual_tokenizer.md](2026-08-28__mbert_multilingual_tokenizer.md),
  [2026-08-28__primary_language_conditioning.md](2026-08-28__primary_language_conditioning.md),
  and the plan at
  [../plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/).
- **External review**: audited independently twice — a general-purpose research
  pass, then two rounds of critique/re-verification by Codex (gpt-5.6-sol,
  high reasoning effort): once against the raw audit, once against this
  document itself (the second pass found real gaps in the first draft's fix
  designs and some overclaiming in the wording, both corrected below). Every
  claim that materially affects the recommendation was also re-verified by
  hand in this session — not just trusted from either pass — before being
  written down.
- **Decided (2026-08-29)**: fix forward on `mine`. No checkout back to
  `master`, no fresh-branch reimplementation. See "Alternatives Considered."

## Summary

`mine` (this fork's working branch, HEAD `87aa83b`) is 18 commits ahead of
upstream `master` (HEAD `2f7326f`) and 0 commits behind — `master` has not
moved independently, so this is pure addition, not a true fork split. The
question this proposal answers: **can a user who never opts into the new
multilingual features get behavior identical to running `master`'s original
code, and can future `master` commits still be merged into `mine` without
excessive pain?**

Short answer: **not fully today, but achievable with small, well-understood,
independent fixes** — one real correctness gap (optimizer-checkpoint resume
from a `master`-trained checkpoint is broken), one public API contract that
is *still* broken today (in-tree callers were patched around it, not fixed),
a couple of small unintentional default-value drifts, and a formatting-noise
problem that will make every future `master` merge into one file painful.
None of these require architectural changes. Two behavior changes are staying
as deliberate, permanent exceptions to the "identical to master" goal — see
"Accepted exceptions" below — because the user explicitly requested them
earlier this session; the goal is not literal byte-for-byte identity in every
respect, it's "identical except where we chose otherwise, on purpose, and
said so."

## Motivation

This session added substantial new functionality (multilingual tokenizer,
pretrained-embedding support, per-utterance language tags, inference-time
language hints) to files `master` also uses for its original, non-multilingual
codepath (`zipvoice/models/zipvoice.py`, `zipvoice/bin/train_zipvoice.py`,
`zipvoice/utils/common.py`, `zipvoice/utils/checkpoint.py`,
`zipvoice/bin/infer_zipvoice.py`, and the sibling training scripts
`train_zipvoice_dialog(_stereo).py` / `train_zipvoice_distill.py`). The user
asked directly: is this actually backward compatible, and if not, what would
it take? This proposal is that audit, written up for review before any further
code changes happen.

**Scope note**: `mine`'s full divergence from `master` is much larger than
this session's work — 106 files, ~30,800 insertions, spanning pre-existing,
unrelated work (an `app/` F5-TTS server, a Vietnamese text normalizer, MOS
tooling, etc. — the earlier commits of `mine`'s history over `master`, before
this session's work). This proposal is scoped **only** to the
multilingual-frontend session's changes (the most recent 4 commits, after the
last `master` merge). The pre-existing divergence is out of scope here; it
isn't part of what this session touched and would need its own separate audit
if it ever matters.

## Proposed Design

Audit findings, organized by what should actually happen to each one. "Verify"
means re-checked by hand in this session with a runnable script or direct
diff read, not just taken from either automated pass.

### A. Real problems to fix

1. **`ZipVoice.__init__` parameter registration order breaks optimizer-state
   resume** (`zipvoice/models/zipvoice.py`). Verified directly: a tiny model
   built from `master`'s code has `embed.weight` at parameter index 286 (last
   of 287 — `master` constructs `self.embed` *after* `self.text_encoder`);
   the same model built from `mine`'s code has `embed.weight` at index 239,
   because `mine` constructs `self.embed` *before* `self.text_encoder` (to
   read the pretrained embedding's hidden size before sizing
   `text_encoder`'s `in_dim`). Total parameter count and every parameter's
   *name* and *shape* are unchanged — plain `model.load_state_dict()` (keyed
   by name) still works with `strict=True` either direction. But
   `ScaledAdam`'s (and any standard PyTorch optimizer's) `state_dict()` is
   keyed by **position**, not name. Codex confirmed loading a `master`-saved
   `ScaledAdam` state into `mine`'s model and taking one more training step
   raises a tensor-shape mismatch. **Anyone resuming or fine-tuning a
   `master`-trained checkpoint's optimizer state under `mine`'s code today
   cannot** (it errors, which is at least loud, not silent corruption — but
   it's still broken). A second, subtler part of this same problem, found on
   Codex's second review pass: for the **scratch**-embedding path
   specifically, simply reordering *when the module gets assigned* to
   `self.embed` isn't enough, because `nn.Embedding(...)` draws its random
   initial weights at *construction* time, not assignment time — constructing
   it before `text_encoder` still consumes the RNG stream in a different order
   than `master`, so even a fixed-registration-order model wouldn't get
   bit-identical initial weights to `master` from the same seed. See the plan
   for the complete fix (both the ordering and the RNG-order parts).
2. **`prepare_input()`'s public contract is still broken today**
   (`zipvoice/utils/common.py`). When `return_tokens=True`, it used to return
   `[tokens, ...]`; it now unconditionally returns
   `[tokens, zero_duration_mask, ...]`. This already broke 9 call sites in
   `train_zipvoice_dialog.py`, `train_zipvoice_dialog_stereo.py`, and
   `train_zipvoice_distill.py` earlier this session — those 9 call sites were
   patched to unpack 4 values instead of 3, which keeps them working today.
   But that only papered over the symptom at each known call site; the
   function's own public contract (its arity when called the same way
   `master` calls it) is **still changed**, not restored. Any future `master`
   commit that adds a new `prepare_input` caller, or any external code
   importing this helper directly, still breaks on merge with no import
   error, just a wrong-arity unpack. The dataset layer that feeds this
   function has a related, separate gap: `SpeechSynthesisDataset.__getitem__`
   (`zipvoice/dataset/dataset.py`) unconditionally adds a `zero_duration_mask`
   key to every token-bearing batch dict, even for the four original
   tokenizer types (as a list of `None`) — so even a caller that ignores the
   mask still sees a different batch *schema* than `master` produces. See the
   plan for the fix covering both layers.
3. **`log_interval` default silently changed 50 → 10**
   (`zipvoice/bin/train_zipvoice.py`, `get_params()`). Verified via
   `git diff master`. A 5x increase in log/TensorBoard-write frequency
   for *every* user, including non-multilingual ones. No comment or doc
   explains why; it appears to be incidental drift, not a deliberate choice.
   Restoring it to 50 is the plan's default action; nobody has asked for 10
   specifically.

### B. Fixed, but not byte-identical to `master` — accepted as a documented exception

4. **`save_checkpoint`'s `model_avg`/`model_ema` dtype-casting fix**
   (`zipvoice/utils/checkpoint.py`) — this session's fix for a real bug (the
   old `.to(torch.float32)` mutated the live float64 running average in
   place; see prior session work). Verified: the *floating* tensor values and
   dtypes in the saved checkpoint are elementwise-equal to `master`'s output
   for ZipVoice's actual parameters, and loading the checkpoint back works
   correctly. However, the **serialized file size differs** from `master`'s.
   The most likely cause (plausible, not proven) is that the new code builds
   a plain `dict` via comprehension rather than calling `.state_dict()` on
   the (unmutated) module directly, which may carry different non-tensor
   metadata through `torch.save` — this hasn't been isolated further since it
   doesn't affect correctness. This is a **known, intentional, documented
   compatibility exception** (fixes a real bug; produces different serialized
   bytes but identical loaded values), not something to revert.

### C. Accepted exceptions to the "identical to master" goal — user-requested, not drift

These two are **not** claimed to be backward compatible with `master`'s
default behavior. They're deliberate, permanent product decisions the user
made explicitly earlier this session, kept as named exceptions rather than
silently breaking the "identical" claim elsewhere in this document.

5. **Skipping validation at `batch_idx_train == 0`**
   (`zipvoice/bin/train_zipvoice.py`, `train_one_epoch`). `master` validates
   at batch 0 before any training step; `mine` was explicitly changed this
   session at the user's direct request ("if it is, I don't want that") to
   skip it.
6. **`tqdm` progress bar enabled unconditionally**
   (`zipvoice/bin/train_zipvoice.py`, `train_one_epoch`) — also added this
   session at the user's explicit request. Changes console output and adds
   minor overhead for every invocation, including non-multilingual ones.

### D. Small, low-risk items — optional cleanup, not required

7. **`Tokenizer` ABC relocated** from `zipvoice/tokenizer/tokenizer.py` to a
   new `zipvoice/tokenizer/base.py`, re-exported so
   `from zipvoice.tokenizer.tokenizer import Tokenizer` still works
   identically for all practical purposes. The one observable difference:
   `Tokenizer.__module__` now reports `zipvoice.tokenizer.base` instead of
   `zipvoice.tokenizer.tokenizer` — affects only code that introspects or
   pickles the abstract base class itself (not its concrete subclasses,
   whose `__module__` is unaffected). Very low real-world impact.
8. **An invalid (empty-token) input to `prepare_avg_tokens_durations` now
   raises a different exception than `master`** (an explicit assertion vs.
   `master`'s division-by-zero). Both are errors either way; only the
   exception type/message differs. Low importance, noted for completeness
   since the goal is literally "identical behavior."

### E. Formatting/mergeability hygiene (separate from correctness)

9. **`zipvoice/bin/train_zipvoice.py` (and, more mildly,
   `zipvoice/utils/checkpoint.py`) contain reformatting mixed into the
   functional diff that is not purely whitespace** — multi-line statements
   collapsed to single lines. Most of this line-reflow survives
   `git diff -w` (which ignores whitespace-only line changes but not line
   *joins*, since those still remove/add lines), so the whitespace-diff line
   count doesn't by itself prove semantic change — but reading the actual
   hunks directly does confirm real reflow (e.g. a 3-line
   `argparse.ArgumentParser(...)` call collapsed to 1 line; a wrapped
   help-string joined onto one line). Codex's assessment, adopted here: don't
   hand-revert individual reformatted lines line-by-line, and don't declare
   the reformatting "intentional after the fact" (that wouldn't reduce future
   conflicts) — instead, rebuild the *functional* diff cleanly against
   `master`'s original formatting for these two files specifically, so future
   `master` commits touching them don't spuriously conflict on lines that
   carry no logic change.

### F. Verified safe, no action needed (for completeness — not previously listed)

- `zipvoice/dataset/datamodule.py`: zero diff from `master`.
- `zipvoice/bin/infer_zipvoice.py`: all pre-existing parser defaults are
  unchanged; every new option is gated behind `--tokenizer multilingual`; the
  four original tokenizer types take a verified no-op path through the new
  `apply_lang_tag()` helper.
- `--warmup-batches 500` (new CLI flag): matches `Eden`'s own pre-existing
  implicit default — not a behavior change, just makes an existing default
  explicit and overridable.
- `prepare_avg_tokens_durations(..., zero_duration_mask=None)`: reproduces
  `master`'s original `utt_duration // n_tokens` floor-division exactly for
  valid, non-empty input. The extra pad token `pad_labels()` appends is
  correctly excluded from both the token-length count and the mask, so
  there's no off-by-one.
- A batch with a mix of `None` (untagged) and real masks is normalized
  correctly (untagged entries become all-`False`, not a crash) — fixed
  earlier this session.

### G. Found along the way, out of scope for this proposal

- **Double `[LANG:xx]` tag at inference with no zero-duration mask** (fixed
  2026-08-31; see [2026-08-28__primary_language_conditioning.md](2026-08-28__primary_language_conditioning.md)).
  This session's `--lang` inference support (for `--tokenizer=multilingual`)
  tagged both `--text` and `--prompt-text`; `forward_text_inference_gt_duration`
  / `forward_text_inference_ratio_duration` then concatenate prompt+text
  tokens, producing two `[LANG:xx]` tokens in the sequence — and
  `zero_duration_mask` was only threaded through the *training* path
  (`forward()`/`forward_text_train()`), not these inference methods, so both
  tags get real acoustic duration at inference despite always getting zero
  duration in training. This is a **correctness bug in this session's own new
  feature**, not a `master`-compatibility issue — it doesn't affect any
  `master` behavior since `master` has no such tags at all. Recorded here
  because it surfaced during this audit; left as an open question below for
  whether it belongs in this plan or a separate one.
- **Pre-existing unrelated divergence** (`app/`, `tools/mos/`, `.gitignore`
  changes, `tests/test_api.py`'s HTTP-request-during-import side effect) —
  none of this is from the multilingual session; out of scope here.

## Alternatives Considered

- **Declare the current state "good enough" and move on.** Rejected: item A1
  (optimizer-resume breakage) is a real, previously-unknown correctness gap
  that directly contradicts "backward compatible," not a style nitpick.
- **Check out `master` fresh and re-implement the multilingual feature from
  scratch there**, so the new code is backward-compatible by construction
  instead of retrofitted. Raised by the user after reading the first draft of
  this audit; discussed with Codex specifically. **Rejected for now**: only
  two files (`train_zipvoice.py`, `checkpoint.py`) actually have the messy
  reformatting-mixed-with-logic problem that motivates "start clean" —
  everything else in the multilingual feature (the tokenizer module, the
  embedding wiring, the dataset/inference changes) is already correct and
  only needs the small, mechanical fixes in items A1-A3, not a rewrite. A
  full reimplementation would re-risk already-verified, subtle work (pretrained
  vocab sizing, zero-duration alignment, mixed-mask handling, language
  dropout, checkpoint dtype preservation) for no correctness benefit — the
  real appeal of a from-`master` branch is a smaller, cleaner *review surface*
  for an eventual upstream contribution, not correctness. That's a packaging
  concern, not an implementation one: if an actual upstream contribution
  becomes a concrete goal later, the right sequence is to fully fix and verify
  `mine` first (this plan), reconstruct just the two messy files against
  `master`'s formatting (item E9, which already achieves the same cleanliness
  a rewrite would for those two files), and only then transplant the
  already-reviewed, minimal functional patches onto a fresh branch as a
  packaging step — not a second implementation. **Decision: stay on `mine`,
  fix forward, no branch checkout to `master`.** This alternative is not
  planned; it can be revisited later if upstreaming becomes a real goal, but
  is not part of this proposal.
- **Rewrite `mine`'s git history to look like a clean patch on top of
  `master`.** Rejected, per Codex's explicit recommendation: history rewriting
  isn't necessary and adds its own risk; a forward-looking cleanup commit (or
  small set of commits) that fixes the concrete items above achieves the same
  mergeability goal without touching history other collaborators may have
  already based work on.

## Open Questions

- Should the double-`[LANG:xx]`-tag inference bug (item G) be fixed as part of
  this same effort, or tracked as its own separate, smaller follow-up? It's
  unrelated to `master` compatibility but was found during this audit.
- For item E (reformatting cleanup): is a single dedicated commit acceptable,
  or does the user want it split further (e.g., `train_zipvoice.py` formatting
  separate from `checkpoint.py`)?
- Item D7 (`Tokenizer.__module__`) — worth the one-line fix
  (`Tokenizer.__module__ = "zipvoice.tokenizer.tokenizer"` after the
  re-export) for completeness, or genuinely not worth touching?

## Next steps

See the implementation plan at
[../plans/2026-08-29__master_backward_compatibility.md](../plans/2026-08-29__master_backward_compatibility.md)
for the concrete, ordered fix list and verification steps. Per the user's
request, no code changes happen until both documents are reviewed and
approved.
