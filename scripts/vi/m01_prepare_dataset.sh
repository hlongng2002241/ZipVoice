python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/vi/train.tsv \
    --subset train \
    --output-dir data/vi/ \
    --sampling-rate 24000


python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/vi/test.tsv \
    --subset test \
    --output-dir data/vi/ \
    --sampling-rate 24000