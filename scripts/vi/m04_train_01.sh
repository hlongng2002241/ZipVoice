# pip install k2==1.24.4.dev20250807+cuda12.8.torch2.8.0 -f https://k2-fsa.github.io/k2/cuda.html

    # --checkpoint checkpoints/zipvoice_vi/model.pt \

CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 python -m zipvoice.bin.train_zipvoice \
    \
    --finetune False \
    \
    --checkpoint exp/version_0/bad-model-0.pt \
    --model-config checkpoints/zipvoice_vi/model.json \
    --token-file checkpoints/zipvoice_vi/tokens.txt \
    --tokenizer espeak \
    --lang vi \
    \
    --start-epoch 22 \
    --resume-from-checkpoint exp/version_0/bad-model-0.pt \
    --seed 8686 \
    --exp-dir exp/version_0 \
    --save-every-n 10000 \
    --valid-by-epoch True \
    --keep-last-k 5 \
    --num-epochs 100 \
    \
    --warmup-batches 5000 \
    --base-lr 0.001 \
    --lr-epochs 100 \
    --lr-batches 7500 \
    --use-fp16 True \
    --feat-scale 0.1 \
    --condition-drop-ratio 0.2 \
    --min-len 1.0 \
    --max-len 30.0 \
    --num-buckets 30 \
    --num-workers 16 \
    \
    --dataset custom \
    --train-manifest data/vi/manifest/custom_cuts_train_with_tokens.jsonl.gz \
    --dev-manifest data/vi/manifest/custom_cuts_test_with_tokens.jsonl.gz \
    --max-duration 210