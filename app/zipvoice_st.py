#!/usr/bin/env python3
"""
ZipVoice Streamlit App
Inherits from BaseApp and implements ZipVoice-specific functionality
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import safetensors.torch
import streamlit as st
import torch
import torchaudio

from app.base import VoiceCloneApp, VoiceCloneConfig
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from vocos import Vocos


@dataclass
class ZipVoiceConfig(VoiceCloneConfig):
    """Configuration for ZipVoice models."""

    model_dir: str
    model_file: str
    default_tokenizer: str

    def __post_init__(self):
        self.model_type = "zipvoice"


class ZipVoiceApp(VoiceCloneApp):
    """ZipVoice-specific implementation of TTS app."""

    def __init__(self, config: ZipVoiceConfig):
        self.model_dir = config.model_dir
        self.model_file = config.model_file
        self.default_tokenizer = config.default_tokenizer
        self.feature_extractor = None
        self.tokenizer = None

        super().__init__(config)

        self.config: ZipVoiceConfig

    def load_model(self):
        """Load ZipVoice model and Vocos vocoder."""
        with st.spinner("Loading ZipVoice model... This may take a moment."):
            try:
                # Load model configuration
                model_config_path = Path(self.model_dir) / "model.json"
                with open(model_config_path, "r") as f:
                    model_config = json.load(f)

                # Load tokenizer (default to emilia)
                token_file = Path(self.model_dir) / "tokens.txt"
                tokenizer = EmiliaTokenizer(token_file=str(token_file))

                tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

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

            except Exception as e:
                st.error(f"Failed to load ZipVoice model: {str(e)}")
                st.stop()

    def render_tokenizer(self):
        """Render tokenizer selection UI for ZipVoice."""
        with st.sidebar:
            st.subheader("🔧 Model Settings")

            # Find index for default tokenizer
            tokenizer_options = ["emilia", "libritts", "espeak", "simple"]
            default_tokenizer_index = (
                tokenizer_options.index(self.default_tokenizer) if self.default_tokenizer in tokenizer_options else 0
            )

            tokenizer_type = st.selectbox(
                "Tokenizer",
                options=tokenizer_options,
                index=default_tokenizer_index,
                help="Select the tokenizer type for text processing",
            )

            default_lang = self.config.language
            lang = default_lang
            if tokenizer_type == "espeak":
                lang_options = ["en-us", "vi", "vi-vn-x-central", "vi-vn-x-south"]
                default_lang_index = lang_options.index(default_lang) if default_lang in lang_options else 0

                lang = st.selectbox(
                    "Language",
                    options=lang_options,
                    index=default_lang_index,
                    help="Language identifier for espeak tokenizer",
                )

            # Create and cache tokenizer
            self.tokenizer = self._get_tokenizer(tokenizer_type, lang)

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

    def render_parameters(self) -> Dict[str, Any]:
        self.render_tokenizer()

        with st.sidebar:
            st.subheader("🎛️ Generation Parameters")

            speed = st.slider(
                "Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.05, help="Speech generation speed multiplier"
            )

            num_step = st.slider(
                "Number of Steps", min_value=4, max_value=64, value=32, step=1, help="Number of diffusion steps for generation"
            )

            feat_scale = st.slider(
                "Feature Scale", min_value=0.1, max_value=2.0, value=1.0, step=0.05, help="Scale factor for audio features"
            )

            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=10.0,
                value=3.5,
                step=0.1,
                help="Controls the strength of guidance during generation",
            )

            t_shift = st.slider(
                "Time Shift", min_value=-1.0, max_value=1.0, value=0.0, step=0.05, help="Time shift parameter for generation"
            )

            target_rms = st.slider(
                "Target RMS", min_value=0.01, max_value=0.5, value=0.1, step=0.05, help="Target RMS level for audio normalization"
            )

            seed = st.number_input(
                "Seed",
                min_value=-1,
                max_value=2147483647,
                value=666,
                help="Random seed for reproducible generation (-1 for random)",
            )

            return {
                "guidance_scale": guidance_scale,
                "num_step": num_step,
                "feat_scale": feat_scale,
                "speed": speed,
                "t_shift": t_shift,
                "target_rms": target_rms,
                "seed": seed if seed != -1 else None,
            }

    def generate_speech(self, text: str, **params):
        """Generate speech using ZipVoice model."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Please select tokenizer first.")
        assert self.feature_extractor is not None

        prompt_text = params.pop("prompt_text")
        prompt_audio_path = params.pop("prompt_audio_path")

        # Convert text to tokens
        tokens = self.tokenizer.texts_to_token_ids([text])
        prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])

        # Load and preprocess prompt wav
        prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_audio_path)

        if prompt_sampling_rate != self.config.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=prompt_sampling_rate, new_freq=self.config.audio_sample_rate)
            prompt_wav: torch.Tensor = resampler(prompt_wav)

        # Normalize prompt wav
        prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
        target_rms = params.pop("target_rms")
        if prompt_rms < target_rms:
            prompt_wav = prompt_wav * target_rms / prompt_rms

        # Extract features
        prompt_features: torch.Tensor = self.feature_extractor.extract(prompt_wav, sampling_rate=self.config.audio_sample_rate)
        prompt_features = prompt_features.to(self.device)

        feat_scale = params.pop("feat_scale")
        prompt_features = prompt_features.unsqueeze(0) * feat_scale
        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=self.device)

        # Generate features
        pred_features, _, _, _ = self.model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            speed=params.pop("speed"),
            t_shift=params.pop("t_shift"),
            duration="predict",
            num_step=params.pop("num_step"),
            guidance_scale=params.pop("guidance_scale"),
        )

        # Convert features to audio
        pred_features = pred_features.permute(0, 2, 1) / feat_scale
        wav = self.vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

        # Adjust volume
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms

        wav = wav.cpu().numpy().squeeze()
        return wav
