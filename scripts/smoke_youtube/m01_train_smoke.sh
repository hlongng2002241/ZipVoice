#!/usr/bin/env bash
# 10-step smoke test: confirms the multilingual-tokenizer + pretrained-embedding
# integration (LanguageModelTokenizer default Qwen2.5-0.5B, embed_source=pretrained
# in model.json) runs end-to-end on real data shaped like the user's YouTube
# corpus, using lhotse on-the-fly feature extraction (no compute_fbank step,
# per docs/plans/2026-08-28__multilingual_tts_frontend/). The [LANG:xx] tag
# (with --lang-auto-prob dropout) is derived per-utterance from each cut's own
# supervision.language field (set from the corpus's "language" column in
# m00_prepare_manifest.py) -- not a single flag for the whole dataset.
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python -m zipvoice.bin.train_zipvoice \
    --tokenizer multilingual \
    --lang-auto-prob 0.05 \
    \
    --model-config scripts/smoke_youtube/model.json \
    --seed 42 \
    --exp-dir exp/smoke_youtube \
    --num-iters 10 \
    --save-every-n 1000000 \
    --world-size 1 \
    \
    --use-fp16 True \
    --min-len 1.0 \
    --max-len 20.0 \
    --num-buckets 2 \
    --num-workers 2 \
    --on-the-fly-feats True \
    \
    --dataset custom \
    --train-manifest data/smoke_youtube/manifests/smoke_youtube_cuts_train.jsonl.gz \
    --dev-manifest data/smoke_youtube/manifests/smoke_youtube_cuts_dev.jsonl.gz \
    --max-duration 60
