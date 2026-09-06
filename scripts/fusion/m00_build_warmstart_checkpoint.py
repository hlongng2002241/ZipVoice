#!/usr/bin/env python3
# Copyright      2026  LongNH (with Claude Code assistance)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the warm-start checkpoint for the Qwen+phoneme fusion architecture
(sprint 003, step 7).

Why this is a separate script from
`scripts/all/m04_build_warmstart_checkpoint.py` rather than a flag on it:
that one assumes the source and target state dicts have **identical key
sets** and copies every same-named tensor whose shape matches. That
assumption does not hold here -- fusion renames one tensor and adds five
that have no source counterpart -- so a shape-only copy loop would either
assert out or, worse, silently leave the phone embedding at its random
initialization. This script therefore states the mapping explicitly:

    source (k2-fsa/ZipVoice, master architecture)   ->  fusion model
    ----------------------------------------------     ------------
    embed.weight                       (360, 192)   ->  fusion.phone_embed.weight
    text_encoder.in_proj.{weight,bias}              ->  same name, same shape
    text_encoder.*  (176 tensors)                   ->  same name
    fm_decoder.*    (712 tensors)                   ->  same name
    (nothing)                                       ->  fusion.qwen_proj.*   [fresh]
    (nothing)                                       ->  fusion.gate_proj.*   [fresh]
    (nothing)                                       ->  fusion.qwen_scale    [calibrated]

Two things this buys that the old builder could not:

  - `text_encoder.in_proj` is transplantable again. Under the pre-fusion
    Qwen-embedding architecture it was 896->192 and had to be discarded;
    repositioning it (ADR point 5) makes it 192->192, matching the source
    exactly, so the text encoder arrives fully intact rather than with a
    freshly-initialized first layer.
  - The phone embedding transplants **1:1 with no fresh rows**: measured
    across VI/EN/ZH, `FusionTokenizer` produces zero phones outside the
    source's 360-symbol vocabulary, and both candidate source checkpoints
    ship that identical `tokens.txt` (see sprint 003, step 1).

After transplanting, `PhoneQwenFusion.calibrate_qwen_scale()` is re-run so
the Qwen branch's output scale matches the *transplanted* phone embedding's
RMS rather than the random init's -- without that, `--gate-init-eps` mixes
in 1% of a vector whose magnitude is arbitrary relative to the phone branch
and "near phone-only initialization" stops meaning anything (ADR point 4).

Usage:
    python3 scripts/fusion/m00_build_warmstart_checkpoint.py \
        --token-file scripts/all/pretrained_model_cache/hynt__ZipVoice-Vietnamese-2500h/tokens.txt \
        [--output exp/fusion/warmstart_fusion.pt]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from huggingface_hub import hf_hub_download

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.fusion_tokenizer import FusionTokenizer

DEFAULT_OUT_PATH = REPO_ROOT / "exp/fusion/warmstart_fusion.pt"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "scripts/fusion/model.json"

# The one tensor that changes name between the two architectures. Written
# out explicitly (rather than inferred from shapes) so a future architecture
# change makes this script fail loudly instead of quietly mis-transplanting.
RENAMES = {"embed.weight": "fusion.phone_embed.weight"}

# Tensors the source checkpoint cannot possibly provide.
EXPECTED_FRESH = {
    "fusion.qwen_proj.weight",
    "fusion.qwen_proj.bias",
    "fusion.gate_proj.weight",
    "fusion.gate_proj.bias",
    "fusion.qwen_scale",
}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-hf-repo",
        default="k2-fsa/ZipVoice",
        help="Source checkpoint's HuggingFace repo -- the upstream author's "
        "own release by default (see sprint 003 step 1 for why this rather "
        "than the Vietnamese-only fine-tune).",
    )
    parser.add_argument(
        "--source-checkpoint-file",
        default="zipvoice/model.pt",
        help="Checkpoint filename within the source repo.",
    )
    parser.add_argument(
        "--token-file",
        required=True,
        help="Phone vocabulary ('{token}\\t{id}' per line) -- must be the "
        "source checkpoint's own tokens.txt, so phone ids line up with the "
        "embedding rows being transplanted.",
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--qwen-hidden-size", type=int, default=896)
    parser.add_argument("--gate-init-eps", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Downloading source checkpoint ({args.source_checkpoint_file})...")
    source_path = hf_hub_download(
        args.source_hf_repo, filename=args.source_checkpoint_file
    )
    source_state = torch.load(source_path, map_location="cpu", weights_only=False)
    if isinstance(source_state, dict) and "model" in source_state:
        source_state = source_state["model"]

    tokenizer = FusionTokenizer(token_file=args.token_file, lang="vi")
    print(
        f"Phone vocabulary: {tokenizer.vocab_size} symbols "
        f"(pad_id={tokenizer.pad_id}) from {args.token_file}"
    )

    with open(args.model_config) as f:
        model_config = json.load(f)["model"]
    model = ZipVoice(
        **model_config,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        text_frontend="fusion",
        qwen_hidden_size=args.qwen_hidden_size,
        gate_init_eps=args.gate_init_eps,
    )
    our_state = model.state_dict()

    warmstart = {}
    copied, renamed, fresh, partial = [], [], [], []
    for key, our_tensor in our_state.items():
        source_key = None
        if key in source_state:
            source_key = key
        else:
            for src, dst in RENAMES.items():
                if dst == key and src in source_state:
                    source_key = src
                    break

        if source_key is None:
            warmstart[key] = our_tensor.clone()
            fresh.append(key)
            continue

        source_tensor = source_state[source_key]
        if source_tensor.shape == our_tensor.shape:
            warmstart[key] = source_tensor.clone()
            (renamed if source_key != key else copied).append(key)
        elif (
            key == "fusion.phone_embed.weight"
            and source_tensor.shape[1:] == our_tensor.shape[1:]
        ):
            # Vocabulary sizes differ: copy the rows that exist and leave the
            # rest at their initialization, mirroring how
            # `_make_pretrained_embedding` already handles this. Reported
            # loudly because it means some phones start uninformed.
            tensor = our_tensor.clone()
            n = min(source_tensor.shape[0], our_tensor.shape[0])
            tensor[:n] = source_tensor[:n]
            warmstart[key] = tensor
            partial.append((key, tuple(our_tensor.shape), tuple(source_tensor.shape)))
        else:
            raise ValueError(
                f"Shape mismatch transplanting {source_key!r} -> {key!r}: "
                f"source {tuple(source_tensor.shape)} vs ours "
                f"{tuple(our_tensor.shape)}. Refusing to guess."
            )

    unexpected_fresh = set(fresh) - EXPECTED_FRESH
    if unexpected_fresh:
        raise ValueError(
            f"These tensors found no source counterpart but were expected to: "
            f"{sorted(unexpected_fresh)}. Either the source checkpoint's "
            f"architecture differs from what this script assumes, or RENAMES "
            f"needs updating -- refusing to silently ship a partly-random model."
        )
    unused_source = set(source_state) - set(our_state) - set(RENAMES)
    if unused_source:
        raise ValueError(
            f"Source tensors that were never transplanted: "
            f"{sorted(unused_source)[:10]}{'...' if len(unused_source) > 10 else ''}. "
            f"That means real pretrained weights are being thrown away."
        )

    print(f"Copied {len(copied)} tensors by name.")
    print(f"Renamed and copied {len(renamed)}: {renamed}")
    for key, ours, theirs in partial:
        print(f"Partially copied {key}: ours {ours}, source {theirs}")
    print(f"Freshly initialized {len(fresh)} (no source counterpart): {sorted(fresh)}")

    # Load before calibrating, so the scale is derived from the transplanted
    # embedding rather than the random one it replaced.
    check_model = ZipVoice(
        **model_config,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        text_frontend="fusion",
        qwen_hidden_size=args.qwen_hidden_size,
        gate_init_eps=args.gate_init_eps,
    )
    check_model.load_state_dict(warmstart, strict=True)
    rms = check_model.fusion.calibrate_qwen_scale()
    warmstart["fusion.qwen_scale"] = check_model.state_dict()["fusion.qwen_scale"].clone()
    print(
        f"Calibrated fusion.qwen_scale to the transplanted phone embedding's "
        f"RMS: {rms:.6f}"
    )

    check_model.load_state_dict(warmstart, strict=True)
    print("Sanity check passed: warmstart state loads with strict=True.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": warmstart}, args.output)
    print(f"Saved warm-start checkpoint to {args.output}")


if __name__ == "__main__":
    main()
