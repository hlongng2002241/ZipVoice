python -m zipvoice.bin.train_zipvoice \
    \
    --finetune \
    \
    --num-epochs 100 \
    --exp_dir exp/zipvoice_tls \
    --seed 8686 \
    --save-every-n 5000 \
    --valid-by-epoch True \
    \
    --checkpoint checkpoints/zipvoice_vi/model.pt \
    --model-config checkpoints/zipvoice_vi/model.json \
    --tokenizer espeak \
    --lang vi \
    --token-file checkpoints/zipvoice_vi/tokens.txt \
    \
    --use-fp16 \
    --feat-scale 0.1 \
    --condition-drop-ratio 0.2 \
    \
    --dataset custom \
    --train-manifest data/vi/custom_cuts_train_with_tokens.jsonl.gz \
    --dev-manifest data/vi/custom_cuts_test_with_tokens.jsonl.gz \
    --max-duration 200
    

