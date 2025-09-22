#!/usr/bin/env python3
"""
Streamlit Demo for ZipVoice Text-to-Speech
A user-friendly interface for generating speech using ZipVoice model.
"""

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import uuid
from datetime import datetime

import numpy as np
import safetensors.torch
import streamlit as st
import torch
import torchaudio
from lhotse.utils import fix_random_seed
from vocos import Vocos

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank

# Import voice metadata
from data.ref_audio.metadata import ref_speakers_map, RefSpeaker


@dataclass
class HistoryEntry:
    """Data class for storing synthesis history."""

    id: str
    timestamp: str
    text: str
    voice_name: str
    prompt_text: str
    audio_path: str
    rtf: float
    parameters: Dict[str, Any]
    tokenizer_type: str
    model_name: str


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_CONFIGS = {
    # "ZipVoice Telesale (Best valid loss)": {
    #     "model_dir": "exp/zipvoice_tls",
    #     "model_name": "best-valid-loss.pt",
    #     "default_tokenizer": "espeak",
    #     "default_lang": "vi",
    # },
    # "ZipVoice Telesale (50 Epochs)": {
    #     "model_dir": "exp/zipvoice_tls",
    #     "model_name": "epoch-50.pt",
    #     "default_tokenizer": "espeak",
    #     "default_lang": "vi",
    # },
    # "ZipVoice Telesale V2 (39 Epochs)": {
    #     "model_dir": "exp/zipvoice_tls_train_v2",
    #     "model_name": "epoch-39.pt",
    #     "default_tokenizer": "espeak",
    #     "default_lang": "vi",
    # },
    # "ZipVoice Vietnamese": {
    #     "model_dir": "checkpoints/zipvoice_vi",
    #     "model_name": "model.pt",
    #     "default_tokenizer": "espeak",
    #     "default_lang": "vi",
    # },
    "ZipVoice Origin": {
        "model_dir": "checkpoints/zipvoice",
        "model_name": "model.pt",
        "default_tokenizer": "emilia",
        "default_lang": "en-us"
    },
}

# EXAMPLE_TEXTS = [
#     "Hiện tại thì chương trình ưu đãi này bên em chỉ còn vài ngày nữa là hết hạn rồi. nên anh cứ suy nghĩ và cân nhắc sớm để tham gia chương trình.",
#     "Em sẽ gọi lại sau để hỗ trợ đăng kí cho anh nhé. Chúc anh một ngày tốt lành.",
#     "Phân tích cách họ đang cạnh tranh và hợp tác với các ngân hàng truyền thống, từ đó thay đổi cách người dùng tiếp cận dịch vụ tài chính.",
# ]
EXAMPLE_TEXTS = [
    "The world is full of fascinating information and endless possibilities. I hope you have a wonderful and productive day.",
    "It's a beautiful Monday morning here in Hanoi as the city begins a new week. I hope your day is filled with purpose and success.",
    "Knowledge is a journey, not a destination. Let's learn something new together today.",
]

SAMPLING_RATE = 24000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Page configuration
st.set_page_config(page_title="ZipVoice TTS Demo", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Verdana', sans-serif;  /* Change your font family */
            font-size: 18px;  /* Change your font size */
        }
        
        .st-emotion-cache-1c7y2kd {
            background-color: #C0E5FF;  /* Change the background color for the user's messages */
            border-radius: 10px;  /* Optional: Add border-radius for rounded corners */
            padding: 10px;  /* Optional: Add padding for spacing */
            margin: 5px 0;  /* Optional: Add margin for spacing */
            position: relative;
        }
            
        [data-testid=stSidebar] {
            background: linear-gradient(155deg, #062F66  70%, #000000 100%);
            color: white;
        }
                
        /* Add these lines to target the text color */
        [data-testid=stSidebar] label {
            color: white !important;
        }
        [data-testid=stSidebar] title {
            color: white !important;
        }
        [data-testid=stAlert] {
            background-color: #71B2F0;
        }

        div[data-testid=stSelectbox]{
            color: white;
        }

        .stButton > button {
            display: block;
            margin: 0 auto;
        }     
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(checkpoint_name: str) -> Tuple[ZipVoice, Vocos, Dict[str, Any]]:
    """Load the ZipVoice model and vocoder."""
    try:
        with st.spinner("Loading ZipVoice model... This may take a moment."):
            # Get checkpoint config
            checkpoint_config = CHECKPOINT_CONFIGS[checkpoint_name]
            model_dir = checkpoint_config["model_dir"]
            model_name = checkpoint_config["model_name"]

            # Load model configuration
            model_config_path = Path(model_dir) / "model.json"
            with open(model_config_path, "r") as f:
                model_config = json.load(f)

            # Load tokenizer based on config (default to emilia)
            token_file = Path(model_dir) / "tokens.txt"
            tokenizer = EmiliaTokenizer(token_file=str(token_file))

            tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

            # Initialize model
            model = ZipVoice(
                **model_config["model"],
                **tokenizer_config,
            )

            # Load checkpoint
            model_ckpt = Path(model_dir) / model_name
            if model_name.endswith(".safetensors"):
                safetensors.torch.load_model(model, str(model_ckpt))
            elif model_name.endswith(".pt"):
                load_checkpoint(filename=str(model_ckpt), model=model, strict=True)
            else:
                raise NotImplementedError()
            print("> Loading checkpoint", model_ckpt)

            model = model.to(DEVICE)
            model.eval()

            # Load vocoder
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            vocoder = vocoder.to(DEVICE)
            vocoder.eval()

            return model, vocoder, model_config

    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()


@st.cache_resource
def get_tokenizer(tokenizer_type: str, model_dir: str, lang: str = "en-us"):
    """Get the appropriate tokenizer based on selection."""
    token_file = Path(model_dir) / "tokens.txt"

    if tokenizer_type == "emilia":
        return EmiliaTokenizer(token_file=str(token_file))
    elif tokenizer_type == "libritts":
        return LibriTTSTokenizer(token_file=str(token_file))
    elif tokenizer_type == "espeak":
        return EspeakTokenizer(token_file=str(token_file), lang=lang)
    else:  # simple
        return SimpleTokenizer(token_file=str(token_file))


@torch.inference_mode()
def generate_speech(
    text: str,
    prompt_wav_path: str,
    prompt_text: str,
    model: ZipVoice,
    vocoder: Vocos,
    tokenizer,
    feature_extractor: VocosFbank,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """Generate speech from text using the model."""
    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Load and preprocess prompt wav
    prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav_path)

    if prompt_sampling_rate != SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=prompt_sampling_rate, new_freq=SAMPLING_RATE)
        prompt_wav = resampler(prompt_wav)

    # Normalize prompt wav
    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < params["target_rms"]:
        prompt_wav = prompt_wav * params["target_rms"] / prompt_rms

    # Extract features
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=SAMPLING_RATE).to(DEVICE)

    prompt_features = prompt_features.unsqueeze(0) * params["feat_scale"]
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=DEVICE)

    # Generate features
    start_time = time.time()

    pred_features, pred_features_lens, _, _ = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=params["speed"],
        t_shift=params["t_shift"],
        duration="predict",
        num_step=params["num_step"],
        guidance_scale=params["guidance_scale"],
    )

    # Convert features to audio
    pred_features = pred_features.permute(0, 2, 1) / params["feat_scale"]
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Adjust volume
    if prompt_rms < params["target_rms"]:
        wav = wav * prompt_rms / params["target_rms"]

    generation_time = time.time() - start_time
    wav_seconds = wav.shape[-1] / SAMPLING_RATE
    rtf = generation_time / wav_seconds

    return wav.cpu().numpy().squeeze(), rtf


def initialize_history():
    """Initialize history in session state."""
    if "synthesis_history" not in st.session_state:
        st.session_state.synthesis_history = []


def add_to_history(entry: HistoryEntry):
    """Add a new entry to synthesis history."""
    if "synthesis_history" not in st.session_state:
        st.session_state.synthesis_history = []
    st.session_state.synthesis_history.append(entry)


def main():
    # Initialize history
    initialize_history()

    # Title and description
    st.title("🎙️ ZipVoice Text-to-Speech Demo")
    st.markdown(
        """
    <div class="info-box">
    <b>Welcome to ZipVoice TTS!</b> This demo allows you to generate high-quality speech
    from text using voice cloning. Select a reference voice or upload your own,
    enter text to synthesize, and adjust parameters to customize the output.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model Selection
        st.subheader("🤖 Model Selection")

        selected_checkpoint = st.selectbox(
            "Select Model",
            options=list(CHECKPOINT_CONFIGS.keys()),
            index=0,  # Default to ZipVoice TLS
            help="Choose which checkpoint to use for generation",
        )

        checkpoint_config = CHECKPOINT_CONFIGS[selected_checkpoint]

    # Load model based on selection
    model, vocoder, model_config = load_model(selected_checkpoint)
    feature_extractor = VocosFbank()

    with st.sidebar:
        # Model Settings
        st.subheader("🔧 Model Settings")

        # Set default tokenizer and language based on checkpoint
        default_tokenizer = checkpoint_config["default_tokenizer"]
        default_lang = checkpoint_config["default_lang"]

        # Find index for default tokenizer
        tokenizer_options = ["emilia", "libritts", "espeak", "simple"]
        default_tokenizer_index = tokenizer_options.index(default_tokenizer) if default_tokenizer in tokenizer_options else 0

        tokenizer_type = st.selectbox(
            "Tokenizer",
            options=tokenizer_options,
            index=default_tokenizer_index,
            help="Select the tokenizer type for text processing",
        )

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

        tokenizer = get_tokenizer(tokenizer_type, checkpoint_config["model_dir"], lang)

        # Voice Selection
        st.subheader("🎤 Voice Selection")

        voice_source = st.radio(
            "Voice Source", options=["Preset Voices", "Custom Upload"], help="Choose a preset voice or upload your own"
        )

        prompt_wav_path = None
        prompt_text = None

        if voice_source == "Preset Voices":
            selected_voice = st.selectbox(
                "Select Voice",
                options=list(ref_speakers_map.keys()),
                format_func=lambda x: x.replace("__", " - ").replace("_", " "),
            )

            if selected_voice:
                ref_speaker = ref_speakers_map[selected_voice]
                prompt_wav_path = ref_speaker.audio_file
                prompt_text = ref_speaker.text

                st.info(f"**Reference text:** {prompt_text}")

                # Display reference audio
                if os.path.exists(prompt_wav_path):
                    st.audio(prompt_wav_path, format="audio/wav")

        else:  # Custom Upload
            uploaded_file = st.file_uploader(
                "Upload Reference Audio", type=["wav", "mp3", "m4a"], help="Upload a reference audio file for voice cloning"
            )

            prompt_text = st.text_area(
                "Reference Text",
                placeholder="Enter the transcription of the reference audio...",
                help="Provide accurate transcription of the uploaded audio",
            )

            if uploaded_file and prompt_text:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    prompt_wav_path = tmp_file.name
                    st.audio(uploaded_file, format="audio/wav")

        # Generation Parameters
        st.subheader("🎛️ Generation Parameters")

        num_step = st.slider(
            "Sampling Steps",
            min_value=4,
            max_value=64,
            value=16,
            step=2,
            help="Number of diffusion steps (higher = better quality but slower)",
        )

        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Classifier-free guidance scale (higher = more adherence to prompt)",
        )

        speed = st.slider(
            "Speech Speed",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="Control speech speed (1.0 = normal, >1.0 = faster)",
        )

        t_shift = st.slider(
            "Time Shift",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Shift sampling timesteps (affects quality/speed trade-off)",
        )

        feat_scale = st.slider(
            "Feature Scale", min_value=0.05, max_value=0.5, value=0.1, step=0.01, help="Scale factor for acoustic features"
        )

        target_rms = st.slider(
            "Target RMS",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Target RMS for audio normalization (0 = no normalization)",
        )

        seed = st.number_input(
            "Random Seed", min_value=0, max_value=999999, value=666, step=1, help="Seed for reproducible generation"
        )

    # Main content area
    st.subheader("📝 Text to Synthesize")

    # Initialize persistent text in session state
    if "current_text" not in st.session_state:
        st.session_state.current_text = ""

    # Update current text if example was clicked
    if "example_text" in st.session_state:
        st.session_state.current_text = st.session_state.example_text
        del st.session_state.example_text

    text_to_synthesize = st.text_area(
        "Enter text",
        value=st.session_state.current_text,
        placeholder="Type or paste the text you want to convert to speech...",
        height=120,
        help="Enter the text you want to synthesize with the selected voice",
        key="text_input",
    )

    # Update session state when text area changes
    if text_to_synthesize != st.session_state.current_text:
        st.session_state.current_text = text_to_synthesize

    # Example texts below the text area
    with st.expander("💡 Example Texts", expanded=True):
        for i, text in enumerate(EXAMPLE_TEXTS, 1):
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"📄 Use Example {i}", key=f"use_example_{i}", use_container_width=True):
                    st.session_state.example_text = text
                    st.rerun()
            with col2:
                st.write(text)

    # Validation and Generate button
    can_generate = all(
        [text_to_synthesize, prompt_wav_path, prompt_text, os.path.exists(prompt_wav_path) if prompt_wav_path else False]
    )

    if not can_generate:
        st.warning("Please provide text and select/upload a reference voice")

    # Generate button (full width)
    if st.button("🚀 Generate Speech", disabled=not can_generate, type="primary", use_container_width=True):
        if can_generate:
            try:
                # Set random seed
                fix_random_seed(seed)
                torch.manual_seed(seed)

                # Prepare parameters
                params = {
                    "num_step": num_step,
                    "guidance_scale": guidance_scale,
                    "speed": speed,
                    "t_shift": t_shift,
                    "feat_scale": feat_scale,
                    "target_rms": target_rms,
                }

                # Print all information before inference
                print("\n" + "=" * 80)
                print("🎙️ ZIPVOICE TTS INFERENCE STARTING")
                print("=" * 80)
                print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"🤖 Model: {selected_checkpoint}")
                print(f"📁 Model Dir: {checkpoint_config['model_dir']}")
                print(f"📄 Model File: {checkpoint_config['model_name']}")
                print(f"🔤 Tokenizer: {tokenizer_type}")
                print(f"🌍 Language: {lang}")
                print(f"🎤 Voice: {selected_voice if voice_source == 'Preset Voices' else 'custom'}")
                print(f"🎵 Audio File: {prompt_wav_path}")
                print(f"💬 Prompt Text: {prompt_text}")
                print(f"📝 Target Text: {text_to_synthesize}")
                print(f"📏 Text Length: {len(text_to_synthesize)} characters")
                print(f"🎛️ Generation Parameters:")
                for param, value in params.items():
                    print(f"   • {param}: {value}")
                print(f"🎲 Random Seed: {seed}")
                print(f"💻 Device: {str(DEVICE).upper()}")
                print("=" * 80)

                # Generate speech with progress indicator
                with st.spinner("🎵 Generating speech... Please wait."):
                    audio_array, rtf = generate_speech(
                        text=text_to_synthesize,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        model=model,
                        vocoder=vocoder,
                        tokenizer=tokenizer,
                        feature_extractor=feature_extractor,
                        params=params,
                    )

                # Success message and metrics
                st.success(f"✅ Speech generated successfully! (RTF: {rtf:.3f})")

                # Display generated audio
                st.subheader("🔊 Generated Audio")
                st.audio(audio_array, sample_rate=SAMPLING_RATE)

                # Save audio to permanent location for history
                history_dir = Path("history")
                history_dir.mkdir(exist_ok=True)

                entry_id = str(uuid.uuid4())[:8]
                audio_filename = f"audio_{entry_id}_{int(time.time())}.wav"
                audio_path = history_dir / audio_filename

                torchaudio.save(str(audio_path), torch.from_numpy(audio_array).unsqueeze(0), SAMPLING_RATE)

                # Add to history
                history_entry = HistoryEntry(
                    id=entry_id,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    text=text_to_synthesize,
                    voice_name=selected_voice if voice_source == "Preset Voices" else "custom",
                    prompt_text=prompt_text,
                    audio_path=str(audio_path),
                    rtf=rtf,
                    parameters=params,
                    tokenizer_type=tokenizer_type,
                    model_name=selected_checkpoint,
                )
                add_to_history(history_entry)

                # Save audio option for download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    torchaudio.save(tmp_file.name, torch.from_numpy(audio_array).unsqueeze(0), SAMPLING_RATE)

                    with open(tmp_file.name, "rb") as f:
                        st.download_button(
                            label="💾 Download Audio",
                            data=f.read(),
                            file_name=f"generated_{int(time.time())}.wav",
                            mime="audio/wav",
                        )

                # Generation info
                st.success(f"✅ Speech generated successfully! (RTF: {rtf:.3f})")

                # Details in expander
                with st.expander("📊 Generation Details", expanded=False):
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.write("**Audio Info:**")
                        st.write(f"• Duration: {len(audio_array) / SAMPLING_RATE:.2f} seconds")
                        st.write(f"• Text length: {len(text_to_synthesize)} characters")
                        st.write(f"• RTF: {rtf:.3f}")
                        st.write(f"• Device: {str(DEVICE).upper()}")
                    with detail_col2:
                        st.write("**Generation Settings:**")
                        st.write(f"• Model: {selected_checkpoint}")
                        st.write(f"• Tokenizer: {tokenizer_type}")
                        st.write(f"• Voice: {selected_voice if voice_source == 'Preset Voices' else 'custom'}")
                        st.write(f"• Steps: {params['num_step']}")
                        st.write(f"• Guidance: {params['guidance_scale']}")
                        st.write(f"• Speed: {params['speed']}")
                        st.write(f"• Time shift: {params['t_shift']}")
                        st.write(f"• Feature scale: {params['feat_scale']}")
                        st.write(f"• Target RMS: {params['target_rms']}")

            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                logger.error(f"Generation error: {e}", exc_info=True)

    # History section
    if st.session_state.synthesis_history:
        st.markdown("---")

        # History header with controls
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header("📚 Synthesis History")
        with col2:
            if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                st.session_state.synthesis_history = []
                st.rerun()

        # Display history entries
        for entry in reversed(st.session_state.synthesis_history):
            with st.expander(f"🎵 {entry.timestamp} - {entry.voice_name} (Model: {entry.model_name})"):
                # Main content
                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    st.write(f"**Text:** {entry.text}")
                    st.write(f"**Model:** {entry.model_name}")
                    st.write(f"**Voice:** {entry.voice_name} | **Tokenizer:** {entry.tokenizer_type}")

                with col2:
                    if os.path.exists(entry.audio_path):
                        st.audio(entry.audio_path)
                    else:
                        st.error("Audio file not found")

                with col3:
                    # Compact controls
                    if os.path.exists(entry.audio_path):
                        with open(entry.audio_path, "rb") as f:
                            st.download_button(
                                label="Download",
                                data=f.read(),
                                file_name=f"history_{entry.id}.wav",
                                mime="audio/wav",
                                key=f"download_{entry.id}",
                                use_container_width=True,
                                type="secondary",
                            )

                    if st.button("Remove", key=f"delete_{entry.id}", use_container_width=True, type="primary"):
                        # Remove from history
                        st.session_state.synthesis_history = [h for h in st.session_state.synthesis_history if h.id != entry.id]
                        # Delete audio file
                        if os.path.exists(entry.audio_path):
                            os.remove(entry.audio_path)
                        st.rerun()

                # Details in history
                with st.expander("📊 Details", expanded=False):
                    hist_col1, hist_col2 = st.columns(2)
                    with hist_col1:
                        st.write("**Audio Info:**")
                        st.write(f"• Timestamp: {entry.timestamp}")
                        st.write(f"• RTF: {entry.rtf:.3f}")
                        st.write(f"• Text length: {len(entry.text)} characters")
                    with hist_col2:
                        st.write("**Generation Settings:**")
                        st.write(f"• Model: {entry.model_name}")
                        for param, value in entry.parameters.items():
                            st.write(f"• {param.replace('_', ' ').title()}: {value}")
                        st.write(f"• Tokenizer: {entry.tokenizer_type}")
                        st.write(f"• Voice: {entry.voice_name}")

                    st.write("**Prompt Text:**")
                    st.write(f"_{entry.prompt_text}_")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>Powered by ZipVoice | 🔬 Research Model | Running on {}</p>
    </div>
    """.format(
            str(DEVICE).upper()
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
