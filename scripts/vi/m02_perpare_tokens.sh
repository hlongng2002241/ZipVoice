python -m zipvoice.bin.prepare_tokens \
    --input-file data/vi/custom_cuts_train.jsonl.gz \
    --output-file data/vi/custom_cuts_train_with_tokens.jsonl.gz \
    --tokenizer espeak \
    --lang vi


python -m zipvoice.bin.prepare_tokens \
    --input-file data/vi/custom_cuts_test.jsonl.gz \
    --output-file data/vi/custom_cuts_test_with_tokens.jsonl.gz \
    --tokenizer espeak \
    --lang vi
