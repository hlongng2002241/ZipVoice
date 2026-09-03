#!/usr/bin/env python3
"""Build a warm-start checkpoint for our multilingual ZipVoice architecture
by transplanting a converged, master-branch-architecture checkpoint's weights
(scratch-embedding, e.g. a public zero-shot-TTS fine-tune with many more
training updates than our own from-scratch run) into everything that's
shape-compatible, leaving only the embedding table and the text encoder's
input projection to be learned fresh.

Rationale: comparing our own from-scratch checkpoint's state_dict against a
converged master-architecture checkpoint's, 887 of 889 tensors have identical
shapes -- only `embed.weight` (ours: [vocab_size, 896] Qwen2.5-0.5B-dim vs.
the source's: [smaller_vocab, 192] scratch-dim) and `text_encoder.in_proj.weight`
(896->192 vs 192->192, same reason) differ. Everything else -- the entire
fm_decoder (the bulk of the model's capacity) and the rest of text_encoder --
is architecturally identical and can be copied directly. This should let the
model adapt much faster than training the whole network from scratch (the
ZipVoice paper trains its base model for 1M updates on 100k hours; our own
corpus is far smaller and our from-scratch run was far fewer updates in,
clearly undertrained by comparison -- see this session's discussion).

Output: exp/all_warmstart/warmstart_from_pretrained.pt by default (override
with --output), in the {"model": state_dict} format `train_zipvoice.py
--checkpoint` expects (a fresh-optimizer pretrained-weights load, not a full
--resume-from-checkpoint state).

Usage:
    python3 scripts/all/m04_build_warmstart_checkpoint.py \
        --source-hf-repo <org>/<repo> --source-checkpoint-file <name.pt> \
        [--output exp/all_warmstart/warmstart_from_pretrained.pt]
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json

import torch
from huggingface_hub import hf_hub_download

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.lm_tokenizer import LanguageModelTokenizer

OUR_MODEL_CONFIG = REPO_ROOT / "scripts/all/model.json"
OUR_TOKENIZER_DIR = REPO_ROOT / "exp/all/tokenizer"

DEFAULT_OUT_PATH = REPO_ROOT / "exp/all_warmstart/warmstart_from_pretrained.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-hf-repo",
        required=True,
        help="HuggingFace repo id of the converged, master-architecture "
        "checkpoint to warm-start from, e.g. '<org>/<model-name>'.",
    )
    parser.add_argument(
        "--source-checkpoint-file",
        required=True,
        help="Checkpoint filename within that repo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Path to save the warm-start checkpoint to (default: {DEFAULT_OUT_PATH}).",
    )
    args = parser.parse_args()

    print(f"Downloading source checkpoint ({args.source_checkpoint_file})...")
    source_ckpt_path = hf_hub_download(args.source_hf_repo, filename=args.source_checkpoint_file)
    source_state = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)["model"]

    print("Building our model architecture (Qwen2.5-0.5B embedding + multilingual tokenizer)...")
    with open(OUR_MODEL_CONFIG) as f:
        model_config = json.load(f)["model"]
    tokenizer = LanguageModelTokenizer(pretrained_model_name=str(OUR_TOKENIZER_DIR))
    model = ZipVoice(
        **model_config,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )
    our_state = model.state_dict()

    assert set(our_state.keys()) == set(source_state.keys()), (
        "Key sets differ between our architecture and the source checkpoint -- "
        "this script assumes they match except for embed/in_proj shapes."
    )

    warmstart_state = {}
    copied, kept_ours = [], []
    for key, our_tensor in our_state.items():
        source_tensor = source_state[key]
        if our_tensor.shape == source_tensor.shape:
            warmstart_state[key] = source_tensor.clone()
            copied.append(key)
        else:
            warmstart_state[key] = our_tensor.clone()
            kept_ours.append((key, tuple(our_tensor.shape), tuple(source_tensor.shape)))

    print(f"Copied {len(copied)} tensors from the source checkpoint unchanged.")
    print(f"Kept our own (freshly-initialized) values for {len(kept_ours)} tensors:")
    for key, our_shape, source_shape in kept_ours:
        print(f"  {key}: ours {our_shape} (source was {source_shape}, incompatible)")

    # Sanity check: the resulting state dict must load cleanly into a fresh
    # instance of our own architecture before we trust it for training.
    check_model = ZipVoice(
        **model_config,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )
    check_model.load_state_dict(warmstart_state, strict=True)
    print("Sanity check passed: warmstart_state loads with strict=True into our architecture.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": warmstart_state}, args.output)
    print(f"Saved warm-start checkpoint to {args.output}")


if __name__ == "__main__":
    main()
