#!/usr/bin/env bash
# Train the Qwen+phoneme fusion architecture (sprint 003 -> sprint 004), from
# the warm-start checkpoint that scripts/fusion/m00_build_warmstart_checkpoint.py
# builds. Run that first if exp/fusion/warmstart_fusion.pt doesn't exist.
#
# Reuses the manifests scripts/all/m00_prepare_manifest.py already produced --
# the data is unchanged by fusion; only the text frontend is.
#
# How this differs from scripts/all/m05_train_warmstart.sh (the pre-fusion
# warm-start run), and why:
#
#   --tokenizer fusion    phones via FusionTokenizer + a truncated Qwen's
#                         per-group features, instead of feeding Qwen BPE
#                         tokens straight in as `tokens`.
#   --token-file          the source checkpoint's own 360-symbol phone
#                         vocabulary. Must be the source's, so phone ids line
#                         up with the transplanted embedding rows.
#   --lang-auto-prob 0    required, and enforced in code: the phone branch
#   --lang-wrong-prob 0   always phonemizes under the true
#                         supervision.language, so a sampled or deliberately
#                         wrong [LANG:xx] tag would put the two branches on
#                         different premises for the same utterance (ADR
#                         point 2).
#   --finetune True       set deliberately here, unlike the earlier scripts.
#                         889 of 894 tensors arrive from a converged
#                         checkpoint, so the from-scratch dropout-annealing
#                         schedule and Eden warmup+decay LR are the wrong
#                         regime -- see sprint 001's finding.
#   separate exp-dir      exp/fusion/, so exp/all/ (from-scratch) and
#                         exp/all_warmstart/ (pre-fusion warm-start) both
#                         remain intact as baselines.
set -euo pipefail

WARMSTART=exp/fusion/warmstart_fusion.pt
TOKEN_FILE=scripts/all/pretrained_model_cache/hynt__ZipVoice-Vietnamese-2500h/tokens.txt

if [ ! -f "$WARMSTART" ]; then
    echo "ERROR: $WARMSTART not found. Build it first:"
    echo "  python3 scripts/fusion/m00_build_warmstart_checkpoint.py \\"
    echo "      --token-file $TOKEN_FILE"
    exit 1
fi

TRAIN_ARGS=(
    --tokenizer fusion
    --token-file "$TOKEN_FILE"
    --lm-layers 4
    --gate-init-eps 0.01
    --lang-auto-prob 0
    --lang-wrong-prob 0

    --model-config scripts/fusion/model.json
    --seed 42
    --exp-dir exp/fusion
    --num-epochs 50
    --world-size 1

    --checkpoint "$WARMSTART"
    --finetune True
    --base-lr 0.0001

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

# Same self-check as the other training scripts, kept for consistency -- but
# checked against the actual argument array rather than this file's text, so
# the '--finetune' mentioned in the comments above can't satisfy it.
if [[ ! " ${TRAIN_ARGS[*]} " == *" --finetune "* ]]; then
    echo "WARNING: --finetune is not set for a warm-started run (see"
    echo "train_zipvoice.py:620,1102 for what it changes)."
    read -r -p "Press Enter to continue anyway, or Ctrl+C to abort: "
fi

export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=128
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python3 -m zipvoice.bin.train_zipvoice "${TRAIN_ARGS[@]}"
