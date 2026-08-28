#!/usr/bin/env bash
# Real training run on the combined YouTube+TopTop corpus (171,961 train
# utterances; 1000 VI + 500 EN held out as test, see m00_prepare_manifest.py
# and data/all/split_summary.json). Same validated configuration as the
# smoke test (scripts/smoke_youtube/): MultilingualTokenizer (Qwen2.5-0.5B),
# embed_source=pretrained, on-the-fly feature extraction (no compute_fbank
# step), [LANG:xx] tags derived per-utterance from supervision.language with
# 5% auto-dropout. --num-epochs/--max-duration follow egs/zipvoice/run_emilia.sh's
# upstream recipe (--num-epochs 11), scaled down from --world-size 8 to 1 GPU.
set -euo pipefail

# Bounds cuDNN v8's shape-keyed execution-plan cache. Without this, the
# highly variable batch/time shapes from DynamicBucketingSampler bucketing
# make this native (non-Python) cache grow roughly linearly in the main
# process's RSS, which OOM-killed two prior runs after ~1hr. See Codex's
# analysis in this session -- confirm the RSS plateaus with a short run
# before removing this comment.
export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=128

CUDA_VISIBLE_DEVICES=0 python -m zipvoice.bin.train_zipvoice \
    --tokenizer multilingual \
    --lang-auto-prob 0.05 \
    \
    --model-config scripts/all/model.json \
    --seed 42 \
    --exp-dir exp/all \
    --num-epochs 15 \
    --world-size 1 \
    \
    --warmup-batches 5000 \
    --base-lr 0.0001 \
    --lr-epochs 3 \
    --lr-batches 7500 \
    \
    --use-fp16 True \
    --min-len 1.0 \
    --max-len 120.0 \
    --num-buckets 30 \
    --num-workers 8 \
    --on-the-fly-feats True \
    --save-every-n 4000 \
    \
    --dataset custom \
    --train-manifest data/all/manifests/train.jsonl.gz \
    --dev-manifest data/all/manifests/test.jsonl.gz \
    --max-duration 210
