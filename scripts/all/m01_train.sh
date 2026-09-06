#!/usr/bin/env bash
# Real training run on the combined YouTube+TopTop corpus (171,961 train
# utterances; 1000 VI + 500 EN held out as test, see m00_prepare_manifest.py
# and data/all/split_summary.json). Same validated configuration as the
# smoke test (scripts/smoke_youtube/): LanguageModelTokenizer (Qwen2.5-0.5B),
# embed_source=pretrained, on-the-fly feature extraction (no compute_fbank
# step), [LANG:xx] tags derived per-utterance from supervision.language with
# 5% auto-dropout. --num-epochs/--max-duration follow egs/zipvoice/run_emilia.sh's
# upstream recipe (--num-epochs 11), scaled down from --world-size 8 to 1 GPU.
set -euo pipefail

TRAIN_ARGS=(
    --tokenizer multilingual
    --lang-auto-prob 0.05

    --model-config scripts/all/model.json
    --seed 42
    --exp-dir exp/all
    --num-epochs 15
    --world-size 1

    --start-epoch 2
    --resume-from-checkpoint exp/all/checkpoint-5000.pt
    --warmup-batches 5000
    --base-lr 0.0001
    --lr-epochs 3
    --lr-batches 7500

    --use-fp16 True
    --min-len 1.0
    --max-len 120.0
    --num-buckets 100
    --num-workers 8
    --on-the-fly-feats True
    --save-every-n 4000

    --dataset custom
    --train-manifest data/all/manifests/train.jsonl.gz
    --dev-manifest data/all/manifests/test.jsonl.gz
    --max-duration 180
)

# Checked against the actual argument array (not this script's source text
# via grep) -- a literal '--finetune' substring in a comment/warning message
# would otherwise make a naive self-referential grep check always pass.
if [[ ! " ${TRAIN_ARGS[*]} " == *" --finetune "* ]]; then
    echo "WARNING: --finetune is not set in this run. If this run is" \
         "resuming/continuing a warm-started (transplanted-checkpoint)" \
         "model rather than genuinely training from scratch, forgetting" \
         "--finetune True means the full from-scratch dropout-annealing" \
         "schedule and the Eden warmup+decay LR schedule get used instead" \
         "of fine-tuning-appropriate settings (see" \
         "train_zipvoice.py:620,1102). Not fixed here on purpose (per the" \
         "user's explicit request), just flagged at run time."
    read -r -p "Press Enter to continue anyway, or Ctrl+C to abort: "
fi

# Bounds cuDNN v8's shape-keyed execution-plan cache. Without this, the
# highly variable batch/time shapes from DynamicBucketingSampler bucketing
# make this native (non-Python) cache grow roughly linearly in the main
# process's RSS, which OOM-killed two prior runs after ~1hr. See Codex's
# analysis in this session -- confirm the RSS plateaus with a short run
# before removing this comment.
export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=128
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m zipvoice.bin.train_zipvoice "${TRAIN_ARGS[@]}"
