# mBERT raw-embedding structure check (Sprint 003)

Tests Codex's specific objection to loading mBERT's pretrained embedding table
without its transformer layers: *"mBERT's input embeddings were trained
jointly with contextual layers; they are not obviously pretrained pronunciation
features."* This can be tested directly without any training data — it's a
property of the released checkpoint, not of ZipVoice's training corpus.

## Method

Loaded `google-bert/bert-base-multilingual-cased`'s raw input embedding table
(`model.get_input_embeddings().weight`, `[119547, 768]`, no forward pass
through the transformer). For each word, mean-pooled the embedding vectors of
its subword pieces (matching exactly what `MultilingualTokenizer` +
ZipVoice's `nn.Embedding` lookup would see). Computed cosine similarity
between hand-picked **related** pairs (morphological variants, synonyms,
same-domain words, cross-lingual name/adjective pairs) and **unrelated** pairs
(same word count, deliberately unconnected), plus a same-shape
randomly-initialized embedding table as a no-structure baseline.

## Results

**English + cross-lingual pairs** (`play`/`played`, `king`/`queen`,
`Vietnam`/`Vietnamese`, etc. — 10 related, 10 unrelated pairs):

| | mean cosine similarity |
|---|---|
| Related pairs (mBERT) | **0.429** |
| Unrelated pairs (mBERT) | 0.056 |
| Related pairs (random init) | -0.017 |
| Unrelated pairs (random init) | -0.008 |

**Vietnamese-specific pairs** (`cà phê`/`trà`, `học sinh`/`giáo viên`,
`Hà Nội`/`Sài Gòn`, etc. — 6 related, 6 unrelated pairs):

| | mean cosine similarity |
|---|---|
| Related pairs (mBERT) | **0.324** |
| Unrelated pairs (mBERT) | 0.169 |

## Interpretation

- **The random-init baseline confirms the test is well-calibrated**: with no
  training at all, related and unrelated pairs are statistically
  indistinguishable (~0 cosine similarity either way), as expected for random
  high-dimensional vectors.
- **mBERT's raw embeddings clearly do carry real, usable structure for
  English**: a 0.37-point gap between related and unrelated pairs, with zero
  contextualization. Codex's objection doesn't hold up empirically for this
  candidate/language — the pretrained embedding table is not "just noise
  wearing a pretrained label."
- **The Vietnamese signal is real but weaker, and partly confounded**: a
  0.155-point gap exists (structure survives), but the *unrelated*-pair
  baseline itself sits at 0.169 rather than ~0 — noticeably higher than
  English's unrelated-pair baseline (0.056). This suggests Vietnamese
  subwords cluster somewhat in mBERT's embedding space simply by sharing a
  "Vietnamese region" of the vocabulary (diacritic-heavy WordPiece pieces),
  not purely by semantic relatedness. The directional result (related >
  unrelated) still holds and still favors pretrained loading over random
  init, but the effect size for Vietnamese specifically is smaller and less
  clean than for English — worth keeping in mind when interpreting any
  downstream Vietnamese-specific quality gains.

## Conclusion

This is real evidence, not a hunch, but it is **not a substitute for the
actual WER/pronunciation comparison** the original bakeoff scope called for —
that still requires real training data and remains blocked (see
[README.md](README.md)). What this check does establish: there is no basis for
assuming pretrained loading is *useless* without the transformer (Codex's
hypothesis); the embedding table demonstrably retains real relational
structure, more so for English than Vietnamese. Combined with the tokenization
efficiency numbers already gathered, this shifts the embedding-form
recommendation from "an open ablation" to "load pretrained, with a
documented caveat about the weaker Vietnamese-specific signal" — see
[../../../proposals/2026-08-28__mbert_multilingual_tokenizer.md](../../../proposals/2026-08-28__mbert_multilingual_tokenizer.md).
