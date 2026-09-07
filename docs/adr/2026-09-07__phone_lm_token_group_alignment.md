# Align phone groups to lm_token groups with variable-size blocks, not one-word groups

- **Date:** 2026-09-07
- **Status:** Accepted
- **Author:** LongNH (with Claude Code assistance)
- **Supersedes:** the group-construction rules in
  [2026-09-02__qwen_phone_fusion_text_frontend.md](2026-09-02__qwen_phone_fusion_text_frontend.md)
  (that ADR's architecture, gate, pooling and transplant decisions all stand;
  only *how a group is formed* changes here)

## Summary

The fusion frontend needs each phone group paired with an `lm_token` group.
Until now a group was required to be **exactly one whitespace word**, and an
utterance whose phones would not divide that way was dropped to phone-only.
Four successive rules were written to make espeak's output divide that way.
The last of them still failed on 0.1% of utterances, and the failure was
declared unavoidable.

It was not unavoidable. The one-word requirement was never a requirement --
it was an inherited assumption. Nothing downstream needs a group to be one
word; the model needs only that groups **partition the phones** and
**correspond 1:1 across the two sides**. Allowing a group to span several
words when the phonemizer forces it makes alignment total: worst case the
whole utterance is one coarse group, which is a valid alignment rather than
a failure.

This ADR records the new construction, and -- at greater length, because it
is the more valuable part -- how four rounds of work and an external review
all missed something this simple.

## Context

`FusionTokenizer.text_to_artifact()` must produce, per utterance:

- `phone_groups`: a partition of the phone sequence
- `lm_token_groups`: one entry per `phone_groups` entry, indexing `lm_tokens`

The Qwen branch pools one vector per group (last-subtoken) and broadcasts it
across that group's phones. So the only real contract is: **the two group
lists must have the same length, and phone groups must partition the
phones.** Group *size* is free.

The original construction split the flat phone sequence on the literal `' '`
phone and required the resulting count to equal the whitespace-word count,
because whitespace words were also the anchor for mapping `lm_token`
character offsets. Any mismatch returned `lm_token_groups=None`, a legal
fallback meaning "train this utterance phone-only".

That fallback fired constantly, because espeak renders **speech**, not a
transcription of text tokens, and its internal boundaries do not correspond
to whitespace:

| espeak behaviour | example | effect on the count |
|---|---|---|
| sentence boundary lost when flattening\* | `"người. Rất"` -> `... j . z ...` | one group short per sentence |
| expands one token into several words | `"1997"` -> five spoken words | several groups long |
| fuses adjacent words | `"book of the"` -> `ɒvðə` | one group short |
| drops punctuation-only tokens | `"a - b"` | one group short |
| context-dependent pronunciation | `the` = `tˈɛ` alone, `ðə` in an English run | content mismatch |

\* **Correction (2026-09-07).** Earlier revisions of this ADR said espeak
"emits no space after a sentence-final period", with a voice-dependent story
about espeak-vi versus espeak-en-us. That is not the mechanism.
`phonemize_espeak` returns **one list per sentence**; the boundary is
destroyed by `_flatten_espeak_output`, which joins them with `reduce(x + y)`
and no separator. That flattening is required -- upstream
`EspeakTokenizer.g2p` (tokenizer.py:142) does the same, and the source
checkpoint was trained on the concatenated sequence -- but the boundary was
available and got discarded, then guessed back from punctuation. The
`.!?;:` set is a workaround for information this code threw away, not a
description of espeak.

`ŋˈyə2j.zˈəɜt̪` was also used as an example of cross-word phonology that
per-word phonemization cannot reproduce. It is not: those are two
independent sentences concatenated, with no phonological interaction. The
genuine example is `of the` -> `ɒvðə` **within** one sentence, which is what
actually motivates whole-utterance phonemization and block widening.

**Fixed, then narrowed.** `_flatten_espeak_output` returns the sentence-end
offsets alongside the concatenated phones. The first attempt then treated
them as **hard cuts** -- a block may not span a sentence boundary -- on the
reasoning that two independently phonemized sentences must never share a
group.

That reasoning does not hold, and review rejected it. Independent
phonemization means widening across a boundary is *unnecessary* to recover
pronunciation context; it does not make a shared conditioning group invalid.
A conditioning group is not required to be a phonological unit -- requiring
that is exactly the restriction this ADR removed, reintroduced under a new
name. It was also strictly harmful in practice: a rejected match fell
through to the whole-remainder fallback, producing a *coarser* block that
crossed the boundary anyway, and two other emission paths ignored the cuts
entirely.

The surviving, justified use is narrow: a separator emitted after a sentence
end belongs to the **next** sentence, so trailing-separator attachment stops
there. That is a matching-correctness point, not a grouping policy. The
punctuation set survives under its real name, `_IGNORABLE_PUNCTUATION`,
whose only job is to let a content match skip punctuation; it no longer
pretends to detect sentences, and `_split_into_groups` is back to splitting
on ' ' alone.


## The mistake

Four rounds of work, each an improvement, all inside the wrong frame:

| round | rule | VI alignment |
|---|---|---|
| 0 | split on `' '`, require count == word count | 7.3% |
| 1 | also break after `.!?;:` | 94.0% |
| 2 | per-word group **count** matching | 99.8% |
| 3 | per-word **content** matching, count fallback | 99.9% |

Every round asked the same question -- *how do I make espeak's boundaries
agree with word boundaries?* -- and answered it slightly better. The
question that ends the problem, *does a group have to be one word?*, was
never asked.

Four specific failures of reasoning produced that, and each has a
generalizable form:

**1. An inherited invariant was treated as a requirement.**
The artifact docstring calls a group a "word-equivalent span" and
`__post_init__` asserts the two group lists have equal length. That was read
as "one group == one word". The assert says nothing of the kind. Nobody
checked what the *consumer* actually needs, which is only the partition and
the 1:1 correspondence.
*Generalizable:* before optimizing inside a constraint, find the code that
would break if it were violated. If there is none, it is an assumption, not
a requirement.

**2. The first bug fixed anchored the framing of every later one.**
The sentence-boundary merge was real and its fix was correct. But it
established "espeak's boundaries are wrong, fix the boundaries" as the shape
of the problem, and rounds 2 and 3 were both more elaborate boundary rules.
*Generalizable:* a correct fix to the first instance of a problem is not
evidence that the problem has been correctly framed.

**3. Diminishing returns were read as convergence, not as a warning.**
7.3 -> 94 -> 99.8 -> 99.9 each felt like progress. Four patches to the same
mechanism, with the increments shrinking, is the signature of a frame that
is nearly exhausted -- not of a solution nearly complete.
*Generalizable:* when the Nth patch to one mechanism buys an order of
magnitude less than the (N-1)th, stop patching and re-derive the
requirement.

**4. The residual was rationalized instead of investigated.**
The last 0.1% was written off as "an espeak property, not our bug", on the
grounds that a word's isolated pronunciation genuinely cannot reproduce its
in-context one. That premise is true. The conclusion did not follow: it
argues against *isolating the word*, not against aligning the utterance.
The evidence needed was already in hand -- `of` and `the` had been *measured*
fusing, and the cause correctly diagnosed -- and the obvious response, put
them in one group, was still not taken.
*Generalizable:* "this is inherent to the tool" is a claim about the
solution space, and needs the same evidence as any other claim. Correctly
diagnosing a cause is not the same as correctly bounding the remedies.

## Why review did not catch it either

The implementation was reviewed adversarially over six rounds by an external
reviewer (Codex, `gpt-6-astra`). That review was valuable and found real
defects: an fp16 dtype hazard, a batch-mixing crash, stale supervision
metadata, a silently-ignored option, a resume that changed the frozen
extractor, and -- pointedly -- that a correctness test was circular and that
a rate test can pass while alignment is wrong.

It did not question the one-word frame, for a reason worth recording: **it
was never asked to.** Each round presented a *fix* and asked whether the fix
was correct. A reviewer given a diff reviews the diff. The frame was
implicit in every prompt and therefore invisible in every response --
including in the prompt that described the residual failures as an espeak
property, which the reviewer accepted.

*Generalizable:* to get an assumption reviewed you must **state it as an
assumption** and ask about it directly. "Here is my fix, is it correct?"
cannot surface "your problem statement is wrong." Reviews should include the
question *what am I taking for granted that I have not justified?*, and
supply the constraint list explicitly so a reviewer can challenge it.

## Decision

A group may span **one or more** consecutive whitespace words. It is built
in four steps:

1. **Phonemize the whole utterance once.** This remains the emitted phone
   sequence -- cross-word phonology is preserved and the output stays
   byte-identical to `EspeakTokenizer`'s, which sprint 000 requires.
2. **Segment the phones into blocks.** For each position, phonemize the
   candidate word span *on its own* and match its phone content against the
   flat sequence, skipping separators and punctuation on both sides. If it
   does not match, widen the block. Widening goes **backward as well as
   forward**: the context that changes a word's pronunciation usually sits
   to its left (`the` becomes `ðə` only once `book of` is in the block).
3. **Tokenize per word** and concatenate. Valid because Qwen's BPE is
   concatenative at whitespace boundaries -- verified, not assumed: 300/300
   real utterances and 0/9 adversarial cases, and the vocabulary's only
   multi-word tokens are runs of whitespace, so no merge crosses a space
   between real words. Re-checked by an assert per call.
4. **Assign tokens to blocks by character offset, then coarsen.** Each
   token belongs to the first block whose end offset it has not passed, so
   whitespace lands on the following block and every token is used exactly
   once. Any boundary an lm_token *straddles* is then dropped and those
   blocks merged.

That last step matters more than it sounds. A token covering substantive
characters from two blocks cannot be split between them, and assigning it to
one silently gives that group a token containing the other group's text --
with every group non-empty and every token used once, so neither the
coverage nor the partition check notices. It is not hypothetical: jieba cuts
`今天|天气|很好` while BPE emits `很好` across that boundary, and `我喜欢`
across two boundaries in `我喜欢学习中文`.

The resolution is that **routing and word boundaries decide which
phonemizer runs; they need not survive as conditioning boundaries.** The
result is the finest segmentation coarser than both -- a *common
coarsening*, not a refinement, since cuts are only ever removed.

(An earlier draft of this ADR described a symmetric two-pointer merge of two
block lists. That is what the exploratory script does; the shipped tokenizer
derives lm groups directly from the phone blocks' character offsets, which
is why the coarsening step is needed and where it lives.)

**It cannot fail** for want of a matching block: the block widens until, in
the limit, it is the whole utterance -- one coarse group, not a dropped
utterance. Alignment is still reported unavailable in two cases, both
deliberate: when no phone group survives OOV filtering (there is nothing to
condition), and when a group would end up with no lm_tokens even after
coarsening.

Note the limits of what "cannot fail" claims. It is a statement about
producing *a* correspondence, not about that correspondence being useful:
a whole-utterance group is one Qwen vector broadcast over every phone, which
is weak conditioning whose value against the phone-only fallback has not
been measured. Coverage is a feasibility measure. The `of the` case shows
this from the other side -- 100% alignment of phones that were themselves
nonsense, before script routing existed.

## Script routing (added after the block construction)

Group construction alone does not decide *which phonemizer* a span goes to,
and that turned out to be a separate hole: with `lang="vi"`, espeak has no
reading for Han characters and announced their class instead -- `你好世界`
became `tʃˈaɪniːzlˈetə` four times, "chinese letter" per character. The
utterance aligned perfectly and was nonsense, which is worth noting on its
own: **alignment coverage says nothing about whether the phones are right.**

The rule, per the author:

- a **Han** segment is phonemized as Chinese whatever `lang` says;
- a **Latin** segment follows `lang`, and `lang="zh"` falls back to
  Vietnamese.

"Latin" is a script class, not a language: `EmiliaTokenizer.get_segment`
only distinguishes Han from not-Han, so `"Tôi thích ăn phở"`,
`"machine learning"` and `"Amazon.co.uk"` are all one bucket and share a
voice. That is workable because espeak-vi reads embedded English acceptably
(`məʃˈiːn lˈɜːnɪŋ`), so a Vietnamese-primary instance handles all three
languages.

Text containing no Han characters skips routing entirely, so the 100% of
this corpus that is VI/EN keeps byte-identical behaviour.

**Accepted cost, raised and confirmed deliberately:** under `lang="zh"`,
embedded English now reads with Vietnamese phonology -- `Amazon.co.uk`
becomes `ˈaməzən tʃˈəɜm kˈɔ` where `EmiliaTokenizer` gave
`ˈæmɐzˌɑːn dˈɑːt`. The author chose consistency with a Vietnamese-primary
project over English-inside-Chinese. Consequently the sprint 000 byte-
identity requirement now holds for **non-Han text only**, and for
`lang="zh"` only on Han-only samples; the equivalence tests are scoped
accordingly rather than weakened.

## Consequences

**Better:**
- Alignment is total. Measured on 200 real utterances: 22,781 groups, zero
  failures, and **22,780 of them are single words** -- coverage costs almost
  no granularity. 5 ms/utterance.
- Mislabelled-English utterances, which previously produced no conditioning
  at all, now produce one coarse group.
- The `offset_mapping` path disappears, and with it the measured 0.30% of
  `lm_tokens` (all standalone `Ġ`) that belonged to no group.
- Three invariants are now checkable per call and are asserted: concatenated
  phones reproduce the whole-utterance phonemization; concatenated
  `lm_tokens` reproduce the whole-text tokenization; words are preserved in
  order.

**Worse / accepted:**
- Per-word espeak calls are needed to find block boundaries. They are cached
  and never emitted; interleaving them was verified not to perturb
  whole-utterance output (0/40 utterances changed).
- A merged group is coarser: `of the` gets one Qwen vector rather than two.
  This is correct -- it is one pronunciation unit -- but it does mean the
  two branches share a coarser unit in those rare cases.
- A degenerate utterance can collapse to a single whole-utterance group.
  That is honest (maximally coarse conditioning) rather than a failure, but
  it is not useful conditioning, and it should be **counted**, not hidden.

**Unchanged:** the emitted phone sequence, the architecture, the gate, the
pooling choice, and the transplant. Only group construction changes.

## Testing

Not "does it align" but "does it align **correctly**" -- the distinction the
external review had to force once already:

- concatenated phones == whole-utterance phonemization (per call assert)
- concatenated lm_tokens == whole-text tokenization (per call assert)
- words preserved in order (per call assert)
- corpus-level: alignment rate **and** an independent phone-to-word check
  that never consults group indices
- block-size distribution reported, so a silent slide toward coarse groups
  is visible rather than hidden behind a 100% alignment rate
