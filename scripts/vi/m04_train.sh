# pip install k2==1.24.4.dev20250807+cuda12.8.torch2.8.0 -f https://k2-fsa.github.io/k2/cuda.html

CUDA_VISIBLE_DEVICES=1 python -m zipvoice.bin.train_zipvoice \
    \
    --finetune False \
    \
    --checkpoint checkpoints/zipvoice_vi/model.pt \
    --model-config checkpoints/zipvoice_vi/model.json \
    --token-file checkpoints/zipvoice_vi/tokens.txt \
    --tokenizer espeak \
    --lang vi \
    \
    --seed 8686 \
    --exp-dir exp/zipvoice_tls_train_v3 \
    --save-every-n 50000000000 \
    --valid-by-epoch True \
    --keep-last-k 0 \
    --num-epochs 100 \
    \
    --warmup-batches 2000 \
    --base-lr 0.0001 \
    --lr-epochs 100 \
    --lr-batches 7500 \
    --use-fp16 True \
    --feat-scale 0.1 \
    --condition-drop-ratio 0.2 \
    --min-len 1.0 \
    --max-len 60.0 \
    --num-buckets 60 \
    \
    --dataset custom \
    --train-manifest data/vi/manifest/custom_cuts_train_with_tokens.jsonl.gz \
    --dev-manifest data/vi/manifest/custom_cuts_test_with_tokens.jsonl.gz \
    --max-duration 160