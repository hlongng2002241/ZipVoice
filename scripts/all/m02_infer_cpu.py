#!/usr/bin/env python3
"""CPU-only inference using our own trained checkpoint
(exp/all/checkpoint-64000.pt, multilingual tokenizer + Qwen2.5-0.5B
pretrained embedding, per exp/all/model.json) and a real reference voice
(temp/audio/ref/yen_nhi.{wav,txt}, Vietnamese).

Usage:
    python3 scripts/all/m02_infer_cpu.py "<text to synthesize>" \
        [--out OUT_WAV] [--checkpoint-name NAME] [--primary-lang LANG] \
        [--guidance-scale SCALE]

Forces CPU even if a GPU is visible (CUDA_VISIBLE_DEVICES unset before torch
import), per the user's request to run inference without GPU.

Workaround note: this environment's torchaudio (2.10.0) only has a
torchcodec-backed I/O path, and the installed torchcodec build fails to load
(`OSError: libnvrtc.so.13` -- a CUDA-toolkit-version mismatch, unrelated to
CPU vs GPU and unrelated to this repo's own code). Rather than leave
inference broken, this script monkeypatches torchaudio.load/save to go
through `soundfile` instead, scoped to this script only -- the actual
zipvoice codebase is untouched. Remove this workaround once the environment's
torchcodec/CUDA-toolkit versions are reconciled.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
import torchaudio


def _load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # (channels, samples)
    return wav, sr


def _save_via_soundfile(path, wav, sample_rate, *args, **kwargs):
    sf.write(path, wav.numpy().T, sample_rate)  # (samples, channels)


torchaudio.load = _load_via_soundfile
torchaudio.save = _save_via_soundfile

REF_WAV = REPO_ROOT / "temp/audio/ref/yen_nhi.wav"
REF_TEXT_FILE = REPO_ROOT / "temp/audio/ref/yen_nhi.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "scripts/all/cpu_infer_output.wav"),
        help="Output wav path",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint-64000.pt",
        help="Checkpoint filename inside exp/all/",
    )
    parser.add_argument(
        "--primary-lang",
        default="vi",
        help="[LANG:xx] hint for both prompt and target text "
        "(the reference voice is Vietnamese) -- see --primary-lang in "
        "infer_zipvoice.py for details.",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=8,
        help="CPU threads for PyTorch (kept below full core count to leave "
        "headroom for other processes on this shared machine).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free-guidance scale (see infer_zipvoice.py). Default "
        "(unset) uses infer_zipvoice's own default (1.0 for --model-name "
        "zipvoice), which is NOT conditional-only -- the solver computes "
        "(1+scale)*conditional - scale*unconditional, doubled again for "
        "t<=0.5, so 1.0 actually means ~3*conditional-2*unconditional for "
        "most sampling steps. Pass 0.0 for true conditional-only generation.",
    )
    args = parser.parse_args()

    sys.argv = [
        "infer_zipvoice",
        "--model-name", "zipvoice",
        "--model-dir", "exp/all",
        "--checkpoint-name", args.checkpoint_name,
        "--tokenizer", "multilingual",
        "--primary-lang", args.primary_lang,
        "--num-thread", str(args.num_thread),
        "--prompt-wav", str(REF_WAV),
        "--prompt-text", REF_TEXT_FILE.read_text().strip(),
        "--text", args.text,
        "--res-wav-path", args.out,
    ]
    if args.guidance_scale is not None:
        sys.argv += ["--guidance-scale", str(args.guidance_scale)]

    from zipvoice.bin.infer_zipvoice import main as infer_main

    infer_main()
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
