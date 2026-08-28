# RL post-training for ZipVoice via FlowTTS-GRPO

- **Status**: Proposed, not yet implemented
- **Author**: LongNH (with Claude Code assistance)
- **Date**: 2026-08-28
- **Reference paper**: [FlowTTS-GRPO: Online Reinforcement Learning with
  Multi-Objective Reward Optimization for Flow-Matching Based Text-to-Speech](../references/2606.23190v2.pdf)
  (arXiv:2606.23190v2), with supporting background from
  [Flow-GRPO: Training Flow Matching Models via Online RL](../references/2505.05470_flow_grpo.pdf)
  (arXiv:2505.05470, NeurIPS 2025)
- **External review**: discussed with Codex (gpt-5.6-sol, high reasoning effort)
  on 2026-08-28. Verdict: *"highest-risk/highest-reward proposal... nowhere near
  ready for an expensive training run."* All of Codex's recommendations are
  accepted and incorporated below — see the sections marked "(revised per Codex)".
  See [docs/plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/)
  and its sibling RL plan for sequencing against the other two proposals.

## Summary

The user found FlowTTS-GRPO, an online RL (GRPO) post-training method demonstrated
on F5-TTS and CosyVoice 3.0's flow-matching (FM) component, and asked whether it
transfers to ZipVoice given how architecturally similar ZipVoice is to F5-TTS.
Investigation (reading both codebases directly, plus the FlowTTS-GRPO and
Flow-GRPO papers) found that the time-direction convention used in the paper's SDE
equations matches ZipVoice's own convention exactly — resolved by reading source
code rather than assuming. **Revised claim, per Codex**: matching time direction
is *necessary but not sufficient* for the formulation to transfer — the stochastic
transition kernel, time discretization, and endpoint handling must also match, and
several of those are not yet verified (see "Correctness gates" below). This is a
promising but unproven direction requiring a bounded feasibility prototype with
explicit correctness checks, not a "yes, transfers directly" conclusion.

## What FlowTTS-GRPO does

An online RL framework that fine-tunes an already-trained flow-matching TTS model
using Group Relative Policy Optimization (GRPO):

1. Converts the model's deterministic ODE sampling into an equivalent
   Stochastic Differential Equation (SDE) that preserves the same marginal
   distribution but introduces the stochasticity RL needs for exploration
   (adapted from Flow-GRPO, arXiv:2505.05470, originally developed for
   text-to-image diffusion/flow models).
2. Samples a group of G candidate outputs per prompt, computes a reward for each,
   and uses the group-relative advantage (reward normalized by the group's mean
   and std) as the GRPO training signal — no value network, no preference pairs,
   no separately trained reward-to-token model.
3. Combines three rewards with per-batch standard-deviation normalization (so a
   fixed weighting actually behaves as intended despite each reward having a
   different natural scale): speaker similarity (cosine similarity of
   ERes2Net/WavLM speaker embeddings), an ASR-based intelligibility reward
   (1 − WER/CER from an off-the-shelf ASR model), and a DNSMOS-based perceptual
   quality reward.
4. Adds three practical optimizations: omitting classifier-free guidance (CFG)
   during RL sampling (not during the original pretraining) accelerates
   convergence by increasing exploration; synthesizing "hard case" text
   (word/sentence repetition, tongue-twister-like patterns) improves robustness;
   and — specifically relevant here — for FM-only models like F5-TTS (as opposed
   to LLM+FM hybrids like CosyVoice 3.0, where intelligibility is bottlenecked by
   the LLM), RL on the FM component alone improves *both* audio-detail metrics
   *and* intelligibility (reduced WER), since F5-TTS synthesizes directly from
   text with no separate LM stage to bottleneck it.
5. Reported results: FM-RL fine-tuning improves speaker similarity, DNSMOS, and
   (for F5-TTS specifically) WER on Seed-TTS-Eval, confirmed by subjective A/B
   preference tests favoring the RL-tuned model over the baseline in both English
   and Chinese.

## Why it's relevant to ZipVoice specifically

ZipVoice, like F5-TTS, is an FM-only architecture (no LLM stage) — of the two
models the paper evaluates, F5-TTS is architecturally the closer match, and the
paper's own finding that FM-only models get *both* audio-detail and
intelligibility gains from FM-RL (unlike the LLM+FM hybrid, where RL on the FM
component alone doesn't move intelligibility) is the more encouraging case for a
ZipVoice application, not the more equivocal CosyVoice 3.0 case.

## Formulation match (verified against source code, not assumed)

The MDP in the paper treats a flow-matching model's Euler-integrated velocity
field as a stochastic policy: state `(condition, t, x_t)`, action = predicted
velocity, terminal reward evaluated at `t=1`. This requires the model's own
`x_t`/`v_t` formulation to determine the correct SDE conversion equation. Rather
than assume ZipVoice "is basically F5-TTS" and copy the paper's equations
verbatim, the actual training code of both was read directly:

- **ZipVoice** (`zipvoice/models/zipvoice.py:372-373`):
  ```python
  xt = features * t + noise * (1 - t)   # t=0 -> noise, t=1 -> data
  ut = features - noise                  # velocity = data - noise
  ```
- **F5-TTS** (`temp/F5-TTS/src/f5_tts/model/cfm.py:267-280`):
  ```python
  x1 = inp                       # data (mel)
  x0 = torch.randn_like(x1)      # noise
  φ = (1 - t) * x0 + t * x1      # t=0 -> noise, t=1 -> data
  flow = x1 - x0                 # velocity = data - noise
  ```
  Identical convention to ZipVoice, confirmed directly from source.

- **The original Flow-GRPO paper** (arXiv:2505.05470, "Preliminaries" section)
  defines it the *opposite* way: *"Let x0 ∼ X0 be a data sample... and x1 ∼ X1
  denote a noise sample,"* with `x_t = (1-t)x0 + t*x1` — i.e. in Flow-GRPO's own
  notation `t=0` is data and `t=1` is noise, the reverse of F5-TTS/ZipVoice, and
  their MDP's generation direction runs `t: 1 -> 0`.

This is exactly what the FlowTTS-GRPO paper means by *"our FM uses the opposite
time ordering to the original Flow-GRPO... we change Equation 6 to Equation 7"* —
"our FM" (F5-TTS and CosyVoice 3.0-FM, described uniformly) both use the forward
`t=0→noise, t=1→data` direction, opposite to Flow-GRPO's own paper convention, and
their **Equation 7** is the SDE update rule specifically re-derived for that
forward direction:

```
x_{t+Δt,mean} = x_t + [v_θ(x_t,t) + σ_t²/(2(1-t))·(-x_t + t·v_θ(x_t,t))]·Δt,
σ_t = a·√((1-t)/t)
```

Since ZipVoice's convention is verified identical to F5-TTS's, **this equation is
the correct starting point for ZipVoice** — it was not a CosyVoice-specific detail
and not a case of the paper leaving the "correct" equation ambiguous for a third
model. **Revised per Codex**: matching the time-direction sign is necessary but
not sufficient — the equation still needs to be validated against ZipVoice's own
implementation (exact `σ_t` schedule, discretization, endpoint behavior) via the
correctness gates below before being trusted in a training loop, rather than
treated as a drop-in transplant.

### Correctness gates (added per Codex review)

Before wiring the SDE rollout into any GRPO loss, verify each of these on
synthetic/small-scale tensors:

1. **Noise-scale limit**: as `σ_t → 0`, the SDE update must reduce exactly to the
   existing deterministic Euler step already implemented in `EulerSolver`.
2. **Marginal-distribution check**: empirically sampled rollout marginals at a
   fixed `t` should match the expected distribution the SDE is supposed to
   preserve (this is the entire justification for calling it "equivalent" to the
   ODE).
3. **Log-probability agreement**: the closed-form Gaussian log-probability (paper
   Eq. 11) and the actual code computing it must agree on hand-constructed
   synthetic tensors, not just "look right."
4. **Ratio sanity**: the GRPO importance ratio `r_t(θ) = p_θ / p_θ_old` must be
   ≈1 immediately before the first optimizer update (old and current policy are
   the same weights at that point).
5. **KL sanity**: the KL penalty must be exactly 0 when the policy and reference
   model have identical weights.
6. **CFG-off still conditional**: with `guidance_scale=0.0`, generation must
   remain *conditional* on the text/speech condition (unconditional-only or a
   miscombination would silently produce garbage) — the exact meaning of
   `guidance_scale=0.0` needs to be verified against `DiffusionModel.forward`'s
   precise branching (`solver.py`), not assumed from the paper's description.
7. **No supervised regression**: after the SDE/GRPO machinery is wired up but
   before any real RL reward optimization, running it with the reward replaced by
   the original supervised flow-matching loss should not change ordinary-text
   quality or validation loss.

## Existing ZipVoice infrastructure that already covers part of this

- **Classifier-free guidance and Euler ODE solving already exist**:
  `zipvoice/models/modules/solver.py`'s `DiffusionModel`/`EulerSolver` implement
  exactly this (CFG combination, deterministic Euler stepping). The paper's
  "omit CFG during RL sampling" optimization needs no new code — calling
  `EulerSolver.sample(..., guidance_scale=0.0)` already does unguided sampling.
  Training-time CFG condition dropout also already exists
  (`condition_drop_ratio` in `zipvoice.py:339,365-370`).
- **Reward models exist in the codebase in offline-evaluation form**, so they
  don't need to be sourced or built from scratch, but **revised per Codex**:
  don't assume they're usable as RL reward functions as-is — they may be
  file-oriented, unbatched, slow, or in conflict with GPU memory needed for
  rollout generation, and need to be adapted for mid-rollout, batched, GPU-resident
  use before they can serve as reward functions in a training loop:
  - `zipvoice/eval/speaker_similarity/sim.py` + `cpsim.py` (WavLM/ECAPA speaker
    similarity) ≈ the paper's SS1 reward.
  - `zipvoice/eval/mos/utmos.py` (UTMOS perceptual quality score) plays a similar
    *role* to the paper's DNSMOS P.835 proxy reward, but **revised per Codex**:
    UTMOS and DNSMOS are not interchangeable — their behavior across languages and
    audio domains must be validated independently before trusting either as a
    training signal, especially for Vietnamese.
  - `zipvoice/eval/wer/seedtts.py` + `hubert.py` (WER/CER) ≈ the paper's
    ASR-based intelligibility reward.
  - Vietnamese-specific reward models (ASR/speaker/quality) are already available
    separately (per the user) and don't need to be sourced for this proposal, but
    are subject to the same batched/online-usability caveat above.
  - Rewards are **terminal** (evaluated once on the fully generated, vocoded
    audio), not computed mid-denoising-step — "reward orchestration" below means
    wrapping these as callables invoked once per completed rollout, not a
    stepwise/partial-decoding reward design.
- **LoRA/adapter tuning first, not skipped (reversed per Codex review).** The
  original draft argued ZipVoice's FM component is small enough (118.5M
  `fm_decoder` + 4.1M `text_encoder`, 122.8M total excluding the embedding) to
  skip LoRA and fine-tune all parameters directly, since the paper's LoRA budget
  (~10M params, rank 32) was a much smaller fraction of F5-TTS/CosyVoice 3.0's
  larger base models. Codex correctly pushed back: **rollout/training cost in
  RL is dominated by group sampling, vocoding, reward-model inference, and
  retained activations across the denoising window — not raw parameter count**,
  so "the model is small" doesn't actually justify skipping LoRA. More
  importantly, full-parameter updates driven by RL reward risk damaging
  pronunciation and multilingual generalization that supervised training
  established — a real concern if this runs after the front-end proposals land.
  **Revised plan**: start with LoRA or decoder-only (freeze `text_encoder` and the
  embedding, tune only `fm_decoder`) adaptation, establish training stability
  there first, and only compare against full-parameter fine-tuning as a follow-up
  ablation once the constrained version works.

## What's genuinely new engineering (not reuse)

1. **Stochastic (SDE) rollout mode for the solver**, applied only to a step
   window (`S_min`, window size — the paper's "window training" trick, reducing
   compute by only backpropagating through a small subset of denoising steps),
   tracking the per-step Gaussian log-probability (paper's Eq. 11) needed for the
   GRPO importance ratio `r_t(θ)`. This extends `EulerSolver`/`DiffusionModel`
   rather than replacing them.
2. **The GRPO loss itself**: group sampling (G rollouts per prompt), group-relative
   advantage normalization (subtract group mean, divide by group std), the
   clipped surrogate objective, and a KL penalty against a frozen reference copy
   of the model.
3. **Reward orchestration**: wrapping the existing eval models as callable reward
   functions mid-rollout, the std-normalized weighted combination (paper's
   Eq. 12), and discarding zero-std groups (all G samples got the same reward —
   no learning signal).
4. **Hard-case text augmentation** (local/sparse word repetition, global sentence
   repetition) — small and mechanical to port, meant to target robustness on
   pathological inputs (repeated words, tongue-twisters) specifically. **Revised
   per Codex**: needs its own separately-weighted curriculum (not mixed in at the
   same rate as normal text) plus a dedicated normal-text regression suite, since
   over-training on repetition patterns risks the model learning unnatural
   repetition-heavy behavior as a side effect of chasing robustness on the hard
   subset.
5. A held-out validation protocol analogous to the paper's `dev-easy`/`dev-hard`
   split, to track reward trends and decide which RL checkpoint to keep — the
   paper explicitly selected checkpoints by validation reward rather than a fixed
   step count. **Addition per Codex**: also requires periodic **human evaluation**,
   not just automated reward/proxy-metric trends — selecting checkpoints purely by
   the same reward-model family used for training risks circular evaluation (the
   checkpoint that scores best is the one that has learned to satisfy the
   scoring model, not necessarily the one that sounds best).
6. **Reward-hacking mitigation, added per Codex review**: each of the three
   reward channels has a known failure mode to watch for, not just optimize:
   - ASR-based reward can be gamed by producing audio that transcribes well
     without necessarily sounding natural to a human listener (ASR-favorable
     rather than human-favorable acoustics).
   - Speaker-similarity reward can be gamed by copying prompt acoustics too
     literally, reducing expressiveness/naturalness in exchange for a higher
     similarity score.
   - MOS-predictor rewards (UTMOS/DNSMOS) can be exploited outside their training
     distribution, producing audio that scores well on the predictor without
     actually sounding better.
   - Per-batch std-normalization (paper Eq. 12) can amplify noise into a
     misleadingly large training signal when a reward's true variance in a batch
     is small — worth monitoring, not just applying blindly.
   - Mitigation: add a supervised flow-matching replay loss or another anchoring
     term alongside the KL penalty, as a second, independent guard against
     reward-driven regression (KL alone constrains distance from the reference
     policy in output space, not correctness against ground truth).

## Open questions / risks

- **Reward weighting (`λ1, λ2, λ3`) is dataset/model-specific.** The paper's
  chosen weights (`λ1=λ2=1.0, λ3=0.4` for DNSMOS) were tuned empirically for their
  models and data; ZipVoice will need its own small sweep rather than reusing
  these values directly, following the same std-normalization-first methodology.
- **Noise level `a` and window placement (`S_min`, window size) are also
  empirical hyperparameters** in the paper (e.g. noise level 0.5 was chosen after
  an explicit sweep) — expect a similar tuning pass for ZipVoice rather than
  reusing the paper's exact values, since the two models differ in scale and step
  count.
- **Cross-lingual generalization is untested for ZipVoice's case.** The paper
  observed that FM-RL trained only on Chinese+English generalized to other
  languages in CV3-Eval despite no multilingual RL training data, which is
  encouraging for eventually extending to Vietnamese, but this hasn't been
  verified for ZipVoice and shouldn't be assumed without a dedicated eval pass
  using the Vietnamese reward models the user already has.
- **Revised per Codex: not actually independent of the other two proposals,
  operationally.** The original draft called this "orthogonal" because RL
  post-training can technically run on any trained checkpoint regardless of text
  front-end. Codex's correction: that's true mathematically but misleading in
  practice — RL fine-tunes a *specific* checkpoint, and swapping the tokenizer or
  adding language-conditioning tokens afterward invalidates that investment
  (different vocabulary, different embedding space, likely different duration
  behavior). **Sequencing conclusion: RL training should run only after the
  front-end (proposals 1 and 2) is frozen**, not in parallel with it. What *can*
  run in parallel, reusably, is this proposal's math/infrastructure validation
  (the correctness gates above, batched reward-function wrappers, compute
  profiling) against the *current* stable checkpoint — that work doesn't need to
  be redone once the front-end changes, only the final training run does.

## Next steps

The sequencing/LoRA-first decision is recorded in
[../adr/2026-08-28__sequence_rl_after_frontend_freeze.md](../adr/2026-08-28__sequence_rl_after_frontend_freeze.md).
See the implementation plan at
[docs/plans/2026-08-28__flowtts_grpo_rl/](../plans/2026-08-28__flowtts_grpo_rl/),
which sequences the correctness-gate prototype and reward-wrapper work to run in
parallel with the front-end proposals, and the actual GRPO training run only
after proposals 1 and 2 (see
[docs/plans/2026-08-28__multilingual_tts_frontend/](../plans/2026-08-28__multilingual_tts_frontend/))
are frozen.
