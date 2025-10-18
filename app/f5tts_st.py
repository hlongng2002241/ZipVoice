#!/usr/bin/env python3
"""
F5-TTS Streamlit App
Inherits from BaseApp and implements F5-TTS-specific functionality
"""
import io
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import requests
import soundfile as sf
import streamlit as st

from app.base import VoiceCloneApp, VoiceCloneConfig

# F5-TTS imports
from f5_tts.infer.utils_infer import (
    load_model,
    infer_batch,
    preprocess_ref_audio_text,
)
from hydra.utils import get_class
from omegaconf import OmegaConf
from importlib.resources import files
from vocos import Vocos


@dataclass
class F5TTSConfig(VoiceCloneConfig):
    """Configuration for F5-TTS models."""

    f5_model_name: str
    ckpt_file: str
    vocab_file: str
    tokenizer_type: str

    def __post_init__(self):
        self.model_type = "f5tts"


class F5TTSApp(VoiceCloneApp):
    """F5-TTS-specific implementation of TTS app."""

    def __init__(self, config: F5TTSConfig):
        self.f5_model_name = config.f5_model_name
        self.ckpt_file = config.ckpt_file
        self.vocab_file = config.vocab_file

        super().__init__(config)

        self.config: F5TTSConfig  # type: ignore

    def load_model(self):
        """Load F5-TTS model and Vocos vocoder."""
        with st.spinner("Loading F5-TTS model... This may take a moment."):
            try:
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

            except Exception as e:
                st.error(f"Failed to load F5-TTS model: {str(e)}")
                st.stop()

    def render_parameters(self) -> Dict[str, Any]:
        with st.sidebar:
            st.subheader("🎛️ Generation Parameters")

            speed = st.slider(
                "Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.05, help="Speech generation speed multiplier"
            )

            nfe_step = st.slider(
                "NFE Steps", min_value=4, max_value=64, value=32, step=1, help="Number of function evaluations for generation"
            )

            target_rms = st.slider(
                "Target RMS", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Target RMS level for audio normalization"
            )

            cross_fade_duration = st.slider(
                "Cross Fade Duration",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                help="Duration for cross-fading between audio segments",
            )

            cfg_strength = st.slider(
                "CFG Strength", min_value=0.5, max_value=5.0, value=2.0, step=0.1, help="Classifier-free guidance strength"
            )

            sway_sampling_coef = st.slider(
                "Sway Sampling Coefficient",
                min_value=-2.0,
                max_value=2.0,
                value=-1.0,
                step=0.01,
                help="Sway sampling coefficient for generation",
            )

            return {
                "target_rms": target_rms,
                "cross_fade_duration": cross_fade_duration,
                "nfe_step": nfe_step,
                "cfg_strength": cfg_strength,
                "sway_sampling_coef": sway_sampling_coef,
                "speed": speed,
            }

    def generate_speech(self, text: str, **params) -> np.ndarray:
        """Generate speech using F5-TTS model."""
        import torch

        # Preprocess reference audio and text
        prompt_audio_path = params.pop("prompt_audio_path")
        prompt_text = params.pop("prompt_text")

        with torch.no_grad():
            ref_audio, ref_text = preprocess_ref_audio_text(prompt_audio_path, prompt_text)

            # Use F5-TTS inference with infer_batch
            audio_segments, _ = infer_batch(
                ref_audio,
                ref_text,
                text,
                self.model,
                self.vocoder,
                mel_spec_type="vocos",
                target_rms=params.pop("target_rms"),
                cross_fade_duration=params.pop("cross_fade_duration"),
                nfe_step=params.pop("nfe_step"),
                cfg_strength=params.pop("cfg_strength"),
                sway_sampling_coef=params.pop("sway_sampling_coef"),
                speed=params.pop("speed"),
                device=str(self.device),
            )

            wav = audio_segments[0].squeeze()

            # Clean up GPU memory
            del audio_segments, ref_audio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return wav


@dataclass
class F5TTSApiConfig(VoiceCloneConfig):
    """Configuration for F5-TTS API app."""

    api_base_url: str = "http://127.0.0.1:5555"
    api_path: str = "/api/synthesize"
    model_name: str = "F5TTS_vi"

    def __post_init__(self):
        self.model_type = "f5tts_api"


class F5TTSApiApp(VoiceCloneApp):
    """F5-TTS API implementation - calls FastAPI server."""

    def __init__(self, config: F5TTSApiConfig):
        self.api_base_url = config.api_base_url
        self.api_path = config.api_path
        self.model_name = config.model_name

        super().__init__(config)
        self.config: F5TTSApiConfig  # type: ignore

    def load_model(self):
        """No model loading needed - using API."""
        # Just verify the server is accessible
        try:
            response = requests.get(f"{self.api_base_url}/docs", timeout=2)
            if response.status_code == 200:
                print(f"✓ Connected to API server at {self.api_base_url}")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Warning: Could not connect to API server at {self.api_base_url}: {e}")
            print("  Make sure the server is running before generating speech")

    def render_parameters(self) -> Dict[str, Any]:
        with st.sidebar:
            st.subheader("🎛️ Generation Parameters")

            speed = st.slider(
                "Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.05, help="Speech generation speed multiplier"
            )

            nfe_step = st.slider(
                "NFE Steps", min_value=4, max_value=64, value=32, step=1, help="Number of function evaluations for generation"
            )

            target_rms = st.slider(
                "Target RMS", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Target RMS level for audio normalization"
            )

            cross_fade_duration = st.slider(
                "Cross Fade Duration",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                help="Duration for cross-fading between audio segments",
            )

            cfg_strength = st.slider(
                "CFG Strength", min_value=0.5, max_value=5.0, value=2.0, step=0.1, help="Classifier-free guidance strength"
            )

            sway_sampling_coef = st.slider(
                "Sway Sampling Coefficient",
                min_value=-2.0,
                max_value=2.0,
                value=-1.0,
                step=0.01,
                help="Sway sampling coefficient for generation",
            )

            return {
                "target_rms": target_rms,
                "cross_fade_duration": cross_fade_duration,
                "nfe_step": nfe_step,
                "cfg_strength": cfg_strength,
                "sway_sampling_coef": sway_sampling_coef,
                "speed": speed,
            }

    def generate_speech(self, text: str, **params) -> np.ndarray:
        """Generate speech by calling the API."""
        # Prepare request payload - send prompt_audio_path and prompt_text directly in params
        payload = {
            "model": self.model_name,
            "text": text,
            "params": params,
        }

        # Make API request
        try:
            response = requests.post(
                f"{self.api_base_url}{self.api_path}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,  # 1 minute timeout for generation
            )
            response.raise_for_status()

            # Parse audio from response
            audio_bytes = response.content
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

            # Ensure mono audio
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            return audio_data

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
