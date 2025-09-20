python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/vi/manifest/train.tsv \
    --subset train \
    --output-dir data/vi/manifest/ \
    --sampling-rate 24000


python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/vi/manifest/test.tsv \
    --subset test \
    --output-dir data/vi/manifest/ \
    --sampling-rate 24000