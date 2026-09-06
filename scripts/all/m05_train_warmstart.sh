#!/usr/bin/env bash
# Fine-tune from scripts/all/m04_build_warmstart_checkpoint.py's warm-start
# checkpoint (a converged, master-architecture checkpoint's fm_decoder/
# text_encoder weights, transplanted into our Qwen2.5-0.5B-embedding +
# multilingual-tokenizer architecture -- 887/889 tensors copied unchanged,
# only embed.weight and text_encoder.in_proj.weight are freshly initialized),
# instead of training the whole network from scratch (which the ZipVoice
# paper does for 1M updates on 100k hours -- our from-scratch attempt in
# exp/all was far fewer updates on a much smaller corpus and was clearly
# undertrained). Separate exp-dir (exp/all_warmstart) so this doesn't touch
# the from-scratch run in exp/all.
#
# --checkpoint (not --resume-from-checkpoint) is used deliberately: this is
# a fresh optimizer starting from pretrained weights, not resuming a training
# run's full state -- the source checkpoint's optimizer state wouldn't even
# be shape-compatible with our embed.weight/in_proj anyway.
#
# Run scripts/all/m04_build_warmstart_checkpoint.py first if
# exp/all_warmstart/warmstart_from_pretrained.pt doesn't exist yet.
set -euo pipefail

TRAIN_ARGS=(
    --tokenizer multilingual
    --lang-auto-prob 0.05

    --model-config scripts/all/model.json
    --seed 42
    --exp-dir exp/all_warmstart
    --num-epochs 50
    --world-size 1

    --checkpoint exp/all_warmstart/warmstart_from_pretrained.pt
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
    --max-duration 190
)

# Checked against the actual argument array (not this script's source text
# via grep) -- a literal '--finetune' substring in a comment/warning message
# would otherwise make a naive self-referential grep check always pass.
if [[ ! " ${TRAIN_ARGS[*]} " == *" --finetune "* ]]; then
    echo "WARNING: --finetune is not set in this run. This is a" \
         "warm-start/fine-tuning run from a converged, transplanted" \
         "checkpoint -- without --finetune True, training uses the" \
         "full from-scratch dropout-annealing schedule (from batch 0)" \
         "and the Eden warmup+decay LR schedule instead of a fixed" \
         "fine-tuning LR (see train_zipvoice.py:620,1102). Confirmed" \
         "during sprint 001's diagnostics as a real, live discrepancy" \
         "for this script -- not fixed here on purpose (per the" \
         "user's explicit request), just flagged at run time."
    read -r -p "Press Enter to continue anyway, or Ctrl+C to abort: "
fi

export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=128
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m zipvoice.bin.train_zipvoice "${TRAIN_ARGS[@]}"
