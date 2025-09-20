python -m zipvoice.bin.compute_fbank \
    --sampling-rate 24000 \
    --source-dir data/vi/manifest \
    --dest-dir data/vi/fbank \
    --type vocos \
    --dataset custom \
    --subset train


python -m zipvoice.bin.compute_fbank \
    --sampling-rate 24000 \
    --source-dir data/vi/manifest \
    --dest-dir data/vi/fbank \
    --type vocos \
    --dataset custom \
    --subset test