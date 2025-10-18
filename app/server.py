import sys; sys.path.append(".") # fmt: skip
import os
import io
import abc
import json
import string
import logging
import tempfile
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from hydra.utils import get_class
from omegaconf import OmegaConf
from vocos import Vocos

from f5_tts.infer.utils_infer import load_model, infer_batch
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from data.ref_audio.metadata import REF_SPEAKERS_MAP
from app.text.normalizer.models.normalizer_forward import AutoTextNormalizer
from quick_utils.common.timer import Timer


logging.basicConfig(level=logging.INFO)


class BaseApp(abc.ABC):
    @abc.abstractmethod
    def generate_speech(self, texts: str | list[str], **params) -> tuple[list[np.ndarray], int]:
        pass


class F5TTSConfig(BaseModel):
    """Configuration for F5-TTS models."""

    model_type: str = "f5tts"
    f5_model_name: str
    ckpt_file: str
    vocab_file: str
    tokenizer_type: str


class F5TTSParameters(BaseModel):
    prompt_text: str | None = None
    prompt_audio_path: str | None = None

    speed: float = 1.0
    nfe_step: int = 32
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0


class F5TTSApp(BaseApp):
    def __init__(self, config: F5TTSConfig, device="cuda:0"):
        super().__init__()

        self.f5_model_name = config.f5_model_name
        self.ckpt_file = config.ckpt_file
        self.vocab_file = config.vocab_file
        self.config = config
        self.device = device

    def load_model(self):
        # Load F5-TTS model config
        model_cfg_path = str(files("f5_tts").joinpath(f"configs/{self.f5_model_name}.yaml"))
        model_cfg = OmegaConf.load(model_cfg_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arch = model_cfg.model.arch

        print(f"> Loading F5-TTS model: {self.f5_model_name}")
        model = load_model(
            model_cls,
            model_arch,
            self.ckpt_file,
            mel_spec_type="vocos",
            vocab_file=self.vocab_file,
            device=str(self.device),
        )

        # Load vocoder (Vocos for F5-TTS)
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoder = vocoder.to(self.device)
        vocoder.eval()

        self.model = model
        self.vocoder = vocoder

    def generate_speech(self, texts: str | list[str], **params) -> tuple[list[np.ndarray], int]:
        if isinstance(texts, str):
            texts = [texts]

        # Preprocess reference audio and text
        params = F5TTSParameters(**params)
        print(params.model_dump_json(indent=4))

        ref_audio = [params.prompt_audio_path for _ in range(len(texts))]
        ref_text = [params.prompt_text for _ in range(len(texts))]

        with torch.no_grad():

            # Use F5-TTS inference with infer_batch
            audio_list, sr = infer_batch(
                ref_audio,
                ref_text,
                texts,
                self.model,
                self.vocoder,
                mel_spec_type="vocos",
                target_rms=params.target_rms,
                cross_fade_duration=params.cross_fade_duration,
                nfe_step=params.nfe_step,
                cfg_strength=params.cfg_strength,
                sway_sampling_coef=params.sway_sampling_coef,
                speed=params.speed,
                device=str(self.device),
            )

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return audio_list, sr


def pad_2d(tensors: list[torch.Tensor], pad_value):
    max_len = max([t.size(1) for t in tensors])
    first = tensors[0]
    padded = torch.full((len(tensors), first.size(0), max_len), pad_value, dtype=first.dtype, device=first.device)
    for i, t in enumerate(tensors):
        padded[i, :, : t.size(1)] = t
    return padded


class ZipVoiceConfig(BaseModel):
    """Configuration for ZipVoice models."""

    model_type: str = "zipvoice"
    model_dir: str
    model_file: str
    tokenizer_type: str = "emilia"
    language: str | None = None
    audio_sample_rate: int = 24000


class ZipvoiceParameters(BaseModel):
    prompt_text: str | None = None
    prompt_audio_path: str | None = None

    speed: float = 1.0
    num_step: int = 32
    feat_scale: float = 0.1
    guidance_scale: float = 1.0
    t_shift: float = 0.5
    target_rms: float = 0.1


class ZipVoiceApp(BaseApp):
    def __init__(self, config: ZipVoiceConfig, device="cuda:0"):
        super().__init__()

        self.config = config
        self.model_dir = config.model_dir
        self.model_file = config.model_file
        self.tokenizer_type = config.tokenizer_type
        self.audio_sample_rate = config.audio_sample_rate
        self.tokenizer_type = config.tokenizer_type
        self.device = device
        self.feature_extractor = None
        self.tokenizer = None

    def load_model(self):
        """Load ZipVoice model and Vocos vocoder."""
        # Load model configuration
        model_config_path = Path(self.model_dir) / "model.json"
        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        # Load tokenizer (default to emilia)
        self.tokenizer = self._get_tokenizer(self.tokenizer_type, lang=self.config.language)

        tokenizer_config = {"vocab_size": self.tokenizer.vocab_size, "pad_id": self.tokenizer.pad_id}

        # Initialize model
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )

        # Load checkpoint
        model_ckpt = Path(self.model_dir) / self.model_file
        if self.model_file.endswith(".safetensors"):
            safetensors.torch.load_model(model, str(model_ckpt))
        elif self.model_file.endswith(".pt"):
            load_checkpoint(filename=str(model_ckpt), model=model, strict=True)
        else:
            raise NotImplementedError(f"Unsupported checkpoint format: {self.model_file}")

        print(f"> Loading ZipVoice checkpoint: {model_ckpt}")

        model = model.to(self.device)
        model.eval()

        # Load vocoder
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoder = vocoder.to(self.device)
        vocoder.eval()

        # Initialize feature extractor
        self.feature_extractor = VocosFbank()
        self.model = model
        self.vocoder = vocoder

    def _get_tokenizer(self, tokenizer_type: str, lang: str = "en-us"):
        """Get the appropriate tokenizer based on selection."""
        token_file = Path(self.model_dir) / "tokens.txt"

        if tokenizer_type == "emilia":
            return EmiliaTokenizer(token_file=str(token_file))
        elif tokenizer_type == "libritts":
            return LibriTTSTokenizer(token_file=str(token_file))
        elif tokenizer_type == "espeak":
            return EspeakTokenizer(token_file=str(token_file), lang=lang)
        else:  # simple
            return SimpleTokenizer(token_file=str(token_file))

    def generate_speech(self, texts: str | list[str], **params) -> tuple[list[np.ndarray], int]:
        """Generate speech using ZipVoice model."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        assert self.feature_extractor is not None

        params = ZipvoiceParameters(**params)
        print(params.model_dump_json(indent=4))

        if isinstance(texts, str):
            texts = [texts]

        # Convert text to tokens
        texts = [text + "." if text[-1] not in string.punctuation else text for text in texts]
        tokens = self.tokenizer.texts_to_token_ids(texts)
        prompt_tokens = self.tokenizer.texts_to_token_ids([params.prompt_text for _ in range(len(texts))])

        # Load and preprocess prompt wav
        prompt_wav, prompt_sampling_rate = torchaudio.load(params.prompt_audio_path)

        if prompt_sampling_rate != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=prompt_sampling_rate, new_freq=self.audio_sample_rate)
            prompt_wav: torch.Tensor = resampler(prompt_wav)

        # Normalize prompt wav
        prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
        target_rms = params.target_rms
        if prompt_rms < target_rms:
            prompt_wav = prompt_wav * target_rms / prompt_rms

        # Extract features
        prompt_features: torch.Tensor = self.feature_extractor.extract(prompt_wav, sampling_rate=self.audio_sample_rate)
        prompt_features = prompt_features.to(self.device)
        prompt_features = prompt_features.unsqueeze(0) * params.feat_scale
        prompt_features = prompt_features.expand((len(texts), -1, -1))
        prompt_features_lens = torch.tensor([prompt_features.size(1) for _ in range(len(texts))], device=self.device)

        # Generate features
        with torch.no_grad():
            pred_features, pred_features_lens, _, _ = self.model.sample(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed=params.speed,
                t_shift=params.t_shift,
                duration="predict",
                num_step=params.num_step,
                guidance_scale=params.guidance_scale,
            )

        # Convert features to audio
        pred_features = pred_features.permute(0, 2, 1) / params.feat_scale  # [B, n_mel, mel_length]
        pred_features = [pf[:, :l] for pf, l in zip(pred_features, pred_features_lens)]
        pred_features = pad_2d(pred_features, pad_value=0.0)

        with torch.no_grad():
            wav = self.vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

        # Adjust volume
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms

        # Move to CPU before cleanup
        audio_list = []
        hop_length = 256
        for audio, l in zip(wav, pred_features_lens):
            audio_list.append(audio.cpu().numpy()[: l.item() * hop_length])

        # Clean up GPU memory
        del tokens, prompt_tokens, prompt_features, prompt_features_lens, pred_features, wav
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_list, self.audio_sample_rate


# Global models storage

MODEL_APP_CONFIGS = {
    "F5TTS_vi": F5TTSConfig(
        f5_model_name="F5TTS_Base",
        ckpt_file="../F5-TTS/ckpts/f5_tts_vi/model_last.pt",
        vocab_file="../F5-TTS/ckpts/f5_tts_vi/vocab.txt",
        tokenizer_type="f5tts",
    ),
    "Zipvoice_vi": ZipVoiceConfig(
        model_dir="exp/version_0",
        model_file="best-valid-loss.pt",
        tokenizer_type="espeak",
        language="vi",
        audio_sample_rate=24000,
    ),
}


class Pack:
    def __init__(self, model_apps: dict[str, BaseApp], normalizer: AutoTextNormalizer) -> None:
        self.model_apps = model_apps
        self.normalizer = normalizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and cleanup at shutdown."""
    model_apps = {}

    print("=" * 70)
    print("Loading TTS models...")
    print("=" * 70)

    # Load models
    for app_name, config in MODEL_APP_CONFIGS.items():
        if config.model_type == "f5tts":
            model_app = F5TTSApp(config, device="cuda:0")
        elif config.model_type == "zipvoice":
            model_app = ZipVoiceApp(config, device="cuda:0")
        else:
            raise NotImplementedError(config.model_type)

        try:
            model_app.load_model()
            model_apps[app_name] = model_app
            print(f"✓ Model {app_name} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {app_name} model: {e}")
            model_apps[app_name] = None

    print("=" * 70)
    print(f"Models loaded: {list(model_apps.keys())}")
    print("API server ready!")
    print("=" * 70)

    # Load normalizer
    normalizer = AutoTextNormalizer(
        model_path="checkpoints/tagger/tagger_10_mMimiLM_v2_bi/checkpoint-4207",
        tokenizer_name="microsoft/Multilingual-MiniLM-L12-H384",
        label_matching="match_goal_tags",
    )

    app.state.pack = Pack(model_apps=model_apps, normalizer=normalizer)

    yield

    # Cleanup
    print("Shutting down...")
    model_apps.clear()


app = FastAPI(lifespan=lifespan)


def get_pack(request: Request) -> Pack:
    return request.app.state.pack


def convert_audio_to_bytes(audio: np.ndarray, sample_rate: int, format: str):
    with tempfile.NamedTemporaryFile(delete=False) as f_tmp:
        sf.write(file=f_tmp, data=audio, samplerate=sample_rate, format=format)
    with open(f_tmp.name, "rb") as f_bin:
        data = f_bin.read()
    os.remove(f_tmp.name)
    return data


class InputDTO(BaseModel):
    model: str
    text: str
    voice: str | None = None  # Voice name from REF_SPEAKERS_MAP
    params: F5TTSParameters | ZipvoiceParameters | None = None


def prepare_inputs(inp: InputDTO, pack: Pack):
    text = inp.text
    voice = inp.voice
    model_apps = pack.model_apps

    # normalize text
    text = pack.normalizer.normalize([text], norm_puncs=False)[0]

    # Get params from input
    params = inp.params.model_dump() if inp.params is not None else {}

    # Check if prompt_text and prompt_audio_path are already provided in params
    prompt_text = params.get("prompt_text")
    prompt_audio_path = params.get("prompt_audio_path")

    # If not provided, use voice lookup from REF_SPEAKERS_MAP
    if not prompt_text or not prompt_audio_path:
        if not voice:
            raise HTTPException(
                status_code=400, detail="Either provide 'voice' OR both 'prompt_text' and 'prompt_audio_path' in params"
            )

        if voice not in REF_SPEAKERS_MAP:
            raise HTTPException(
                status_code=400, detail=f"Unknown voice: {voice}. Available voices: {list(REF_SPEAKERS_MAP.keys())}"
            )

        ref_speaker = REF_SPEAKERS_MAP[voice]
        prompt_audio_path = ref_speaker.audio_path
        prompt_text = ref_speaker.text

        print(f"Using preset voice: {voice}")

    print(f"  Prompt text: {prompt_text}")
    print(f"  Prompt audio: {prompt_audio_path}")

    # Update params with prompt info
    params.update(
        {
            "prompt_text": prompt_text,
            "prompt_audio_path": prompt_audio_path,
        }
    )

    model_app = model_apps.get(inp.model)
    if model_app is None:
        raise HTTPException(500, detail=f"Model {inp.model} is not loaded yet")

    return model_app, text, params


@app.post("/api/synthesize")
async def synthesize(inp: InputDTO, pack: Pack = Depends(get_pack)):
    """
    Generate audio from text using specified TTS model.

    Args:
        inp: InputDTO with model name, text, and voice

    Returns:
        WAV audio as streaming response
    """
    audio_fmt = "wav"
    with Timer("Prepare inputs"):
        model_app, text, params = prepare_inputs(inp, pack)

    # Generate speech
    print(f"Generating audio with {inp.model} for text: {text[:50]}...")
    with Timer("Synthesize speech"):
        audio_list, sr = model_app.generate_speech(text, **params)

    wav_bytes = convert_audio_to_bytes(audio_list[0], sample_rate=sr, format=audio_fmt)

    return StreamingResponse(io.BytesIO(wav_bytes), media_type=f"audio/{audio_fmt}")


@app.post("/api/synthesize_prod")
async def synthesize_prod(inp: InputDTO, pack: Pack = Depends(get_pack)):
    audio_fmt = "wav"
    with Timer("Prepare inputs"):
        model_app, text, params = prepare_inputs(inp, pack)
    texts = [text, "đây là văn bản số hai. " + text, "đây là văn bản dài hơn của văn bản gốc, văn bản số 3. " + text]

    # Generate speech
    print(f"Generating audio with {inp.model} for text: {text[:50]}...")
    with Timer("Synthesize speech"):
        audio_list, sr = model_app.generate_speech(texts, **params)

    wav_bytes = convert_audio_to_bytes(audio_list[0], sample_rate=sr, format=audio_fmt)

    return StreamingResponse(io.BytesIO(wav_bytes), media_type=f"audio/{audio_fmt}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5555)
