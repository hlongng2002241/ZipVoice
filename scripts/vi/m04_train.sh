# pip install k2==1.24.4.dev20250807+cuda12.8.torch2.8.0 -f https://k2-fsa.github.io/k2/cuda.html

CUDA_VISIBLE_DEVICES=1 python -m zipvoice.bin.train_zipvoice \
    \
    --finetune True \
    \
    --start-epoch 26 \
    \
    --num-epochs 50 \
    --exp-dir exp/zipvoice_tls \
    --seed 8686 \
    --save-every-n 5000 \
    --valid-by-epoch True \
    \
    --checkpoint checkpoints/zipvoice_vi/model.pt \
    --model-config checkpoints/zipvoice_vi/model.json \
    --token-file checkpoints/zipvoice_vi/tokens.txt \
    --tokenizer espeak \
    --lang vi \
    \
    --base-lr 1e-4 \
    --use-fp16 True \
    --feat-scale 0.1 \
    --condition-drop-ratio 0.2 \
    \
    --dataset custom \
    --train-manifest data/vi/manifest/custom_cuts_train_with_tokens.jsonl.gz \
    --dev-manifest data/vi/manifest/custom_cuts_test_with_tokens.jsonl.gz \
    --max-duration 310
    