#!/usr/bin/env python3
"""CPU-only inference against either our own trained checkpoint or any
public HuggingFace ZipVoice checkpoint, using the same reference voice
(temp/audio/ref/yen_nhi.{wav,txt}, Vietnamese) either way.

Two modes, chosen by whether --hf-repo is given:

- Local (default): loads a checkpoint from --model-dir (default: exp/all),
  e.g. our own multilingual-tokenizer + Qwen2.5-0.5B-embedding run, or
  exp/all_warmstart for the warm-start run. --tokenizer defaults to
  "multilingual" in this mode.
- HuggingFace (--hf-repo given): downloads --checkpoint-name plus
  --config-file/--tokens-file from that repo and builds a small local
  directory (scripts/all/pretrained_model_cache/<sanitized-repo-name>/)
  with symlinks under the filenames infer_zipvoice.py expects -- many
  community checkpoints ship a `config.json` (not `model.json`) and
  `tokens.txt` alongside the checkpoint, which the built-in --model-dir
  path doesn't directly support. Nothing is copied/duplicated; the HF
  cache itself is left untouched. This is for checkpoints using the
  original master-branch architecture (scratch embedding, no multilingual
  tokenizer -- e.g. --tokenizer espeak or --tokenizer emilia).
  --tokenizer defaults to "espeak" in this mode.

Usage:
    # Our own checkpoint (local)
    python3 scripts/all/m02_infer_cpu.py "<text to synthesize>" \
        [--out OUT_WAV] [--model-dir DIR] [--checkpoint-name NAME] \
        [--lang LANG] [--guidance-scale SCALE]

    # Public HuggingFace checkpoint (master-branch architecture)
    python3 scripts/all/m02_infer_cpu.py "<text to synthesize>" \
        --hf-repo <org>/<repo> --checkpoint-name <name.pt> \
        [--config-file config.json] [--tokens-file tokens.txt] \
        [--tokenizer espeak] [--lang vi] [--out OUT_WAV]

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
from huggingface_hub import hf_hub_download


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


def _prepare_hf_model_dir(hf_repo, config_file, tokens_file, checkpoint_file):
    local_dir = REPO_ROOT / "scripts/all/pretrained_model_cache" / hf_repo.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    config_path = hf_hub_download(hf_repo, filename=config_file)
    tokens_path = hf_hub_download(hf_repo, filename=tokens_file)
    checkpoint_path = hf_hub_download(hf_repo, filename=checkpoint_file)

    links = {
        "model.json": config_path,
        "tokens.txt": tokens_path,
        checkpoint_file: checkpoint_path,
    }
    for link_name, target in links.items():
        link_path = local_dir / link_name
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target)
    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "scripts/all/cpu_infer_output.wav"),
        help="Output wav path",
    )
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "exp/all"),
        help="[local mode] Directory holding model.json, the checkpoint, and "
        "(optionally) a saved tokenizer/ subdir -- e.g. exp/all_warmstart "
        "for the warm-start run. Ignored when --hf-repo is given.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint-64000.pt",
        help="Checkpoint filename inside --model-dir (local mode), or "
        "within the --hf-repo repo (HuggingFace mode).",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HuggingFace repo id of a pretrained checkpoint, e.g. "
        "'<org>/<model-name>'. Switches to HuggingFace mode.",
    )
    parser.add_argument(
        "--config-file",
        default="config.json",
        help="[HuggingFace mode] Model config filename within the repo.",
    )
    parser.add_argument(
        "--tokens-file",
        default="tokens.txt",
        help="[HuggingFace mode] Token vocabulary filename within the repo.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        choices=["emilia", "libritts", "espeak", "simple", "multilingual"],
        help="Tokenizer type the checkpoint was trained with. Defaults to "
        "'multilingual' in local mode, 'espeak' in HuggingFace mode.",
    )
    parser.add_argument(
        "--lang",
        default="vi",
        help="Language identifier, passed straight through to "
        "infer_zipvoice.py's --lang: an espeak-ng language code when "
        "--tokenizer=espeak, or a '[LANG:xx]' hint for the prompt text "
        "when --tokenizer=multilingual (the reference voice is "
        "Vietnamese) -- see --lang in infer_zipvoice.py for details.",
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
    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="Number of ODE solver steps (see infer_zipvoice.py). Default "
        "(unset) uses infer_zipvoice's own default (16 for --model-name "
        "zipvoice). More steps generally trade inference time for quality.",
    )
    args = parser.parse_args()

    if args.hf_repo is not None:
        model_dir = _prepare_hf_model_dir(
            args.hf_repo, args.config_file, args.tokens_file, args.checkpoint_name
        )
        tokenizer = args.tokenizer or "espeak"
    else:
        model_dir = args.model_dir
        tokenizer = args.tokenizer or "multilingual"

    sys.argv = [
        "infer_zipvoice",
        "--model-name", "zipvoice",
        "--model-dir", str(model_dir),
        "--checkpoint-name", args.checkpoint_name,
        "--tokenizer", tokenizer,
        "--lang", args.lang,
        "--num-thread", str(args.num_thread),
        "--prompt-wav", str(REF_WAV),
        "--prompt-text", REF_TEXT_FILE.read_text().strip(),
        "--text", args.text,
        "--res-wav-path", args.out,
    ]
    if args.guidance_scale is not None:
        sys.argv += ["--guidance-scale", str(args.guidance_scale)]
    if args.num_step is not None:
        sys.argv += ["--num-step", str(args.num_step)]

    from zipvoice.bin.infer_zipvoice import main as infer_main

    infer_main()
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
