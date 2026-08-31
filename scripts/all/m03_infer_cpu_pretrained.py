#!/usr/bin/env python3
"""CPU-only inference using any public HuggingFace ZipVoice checkpoint that
uses the original master-branch architecture (scratch embedding, no
multilingual tokenizer -- e.g. `--tokenizer espeak` or `--tokenizer emilia`),
with the same reference voice as m02_infer_cpu.py
(temp/audio/ref/yen_nhi.{wav,txt}, Vietnamese).

Usage:
    python3 scripts/all/m03_infer_cpu_pretrained.py "<text to synthesize>" \
        --hf-repo <org>/<repo> --checkpoint-file <name.pt> \
        [--config-file config.json] [--tokens-file tokens.txt] \
        [--tokenizer espeak] [--lang vi] [--out OUT_WAV]

Forces CPU even if a GPU is visible (CUDA_VISIBLE_DEVICES unset before torch
import).

Model-dir note: many community checkpoints ship a `config.json` (not
`model.json`) alongside their checkpoint and `tokens.txt`, and aren't the
k2-fsa/ZipVoice repo infer_zipvoice.py's built-in --model-dir-less download
path supports. This script downloads the 3 files via huggingface_hub (cached
normally under ~/.cache or $HF_HOME) and builds a small local directory
(scripts/all/pretrained_model_cache/<sanitized-repo-name>/) with symlinks
under the filenames infer_zipvoice.py expects, so nothing is copied/
duplicated and the HF cache itself is left untouched.

Same torchaudio/torchcodec workaround as m02_infer_cpu.py -- see that
script's docstring for why.
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


def _prepare_local_model_dir(hf_repo, config_file, tokens_file, checkpoint_file):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--hf-repo",
        required=True,
        help="HuggingFace repo id of the pretrained checkpoint, e.g. "
        "'<org>/<model-name>'.",
    )
    parser.add_argument(
        "--checkpoint-file",
        required=True,
        help="Checkpoint filename within the repo (varies per repo, e.g. "
        "'model.pt' or an 'iter-N-avg-K.pt'-style averaged checkpoint).",
    )
    parser.add_argument(
        "--config-file",
        default="config.json",
        help="Model config filename within the repo.",
    )
    parser.add_argument(
        "--tokens-file",
        default="tokens.txt",
        help="Token vocabulary filename within the repo.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "scripts/all/cpu_infer_output_pretrained.wav"),
        help="Output wav path",
    )
    parser.add_argument(
        "--tokenizer",
        default="espeak",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type this checkpoint was trained with (master-branch "
        "architecture only -- use m02_infer_cpu.py for our own multilingual "
        "checkpoints).",
    )
    parser.add_argument(
        "--lang",
        default="vi",
        help="espeak-ng language code, used only when --tokenizer=espeak.",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=8,
        help="CPU threads for PyTorch (kept below full core count to leave "
        "headroom for other processes on this shared machine).",
    )
    args = parser.parse_args()

    model_dir = _prepare_local_model_dir(
        args.hf_repo, args.config_file, args.tokens_file, args.checkpoint_file
    )

    sys.argv = [
        "infer_zipvoice",
        "--model-name", "zipvoice",
        "--model-dir", str(model_dir),
        "--checkpoint-name", args.checkpoint_file,
        "--tokenizer", args.tokenizer,
        "--lang", args.lang,
        "--num-thread", str(args.num_thread),
        "--prompt-wav", str(REF_WAV),
        "--prompt-text", REF_TEXT_FILE.read_text().strip(),
        "--text", args.text,
        "--res-wav-path", args.out,
    ]

    from zipvoice.bin.infer_zipvoice import main as infer_main

    infer_main()
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
