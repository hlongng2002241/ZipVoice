#!/usr/bin/env python3
"""
Main Streamlit App for Unified TTS Demo
Supports both ZipVoice and F5-TTS models with a clean, scalable architecture
"""

import sys; sys.path.append(".") # fmt: skip
import traceback
from typing import Dict, Any

import streamlit as st

from app.zipvoice_st import ZipVoiceApp, ZipVoiceConfig
from app.f5tts_st import F5TTSApp, F5TTSConfig
from app.base import BaseConfig


# Page configuration
st.set_page_config(page_title="Unified TTS Demo", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: 'Verdana', sans-serif;
            font-size: 18px;
        }

        .st-emotion-cache-1c7y2kd {
            background-color: #C0E5FF;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            position: relative;
        }

        [data-testid=stSidebar] {
            background: linear-gradient(155deg, #062F66  70%, #000000 100%);
            color: white;
            width: 400px !important;
            min-width: 400px !important;
        }

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


# Predefined model configurations
MODEL_CONFIGS: list[BaseConfig] = [
    # ZipVoice Models
    ZipVoiceConfig(
        app_name="ZipVoice Origin",
        model_type="zipvoice",
        audio_sample_rate=24000,
        language="en-us",
        model_dir="checkpoints/zipvoice",
        model_file="model.pt",
        default_tokenizer="emilia",
    ),
    # F5-TTS Models
    F5TTSConfig(
        app_name="F5-TTS Origin",
        model_type="f5tts",
        audio_sample_rate=24000,
        language="en-us",
        f5_model_name="F5TTS_Base",
        ckpt_file="/data2/longnh/projects/mine/F5-TTS/ckpts/original/model_1200000.pt",
        vocab_file="/data2/longnh/projects/mine/F5-TTS/ckpts/original/vocab.txt",
        default_tokenizer="f5tts",
    ),
    # Additional ZipVoice models (commented out for now)
    # "ZipVoice Telesale V3 (100 Epochs)": ZipVoiceConfig(
    #     model_name="ZipVoice Telesale V3 (100 Epochs)",
    #     model_dir="exp/zipvoice_tls_train_v3",
    #     model_file="epoch-100.pt",
    #     default_tokenizer="espeak",
    #     default_lang="vi",
    # ),
    # "ZipVoice Telesale V3 (Best valid loss)": ZipVoiceConfig(
    #     model_name="ZipVoice Telesale V3 (Best valid loss)",
    #     model_dir="exp/zipvoice_tls_train_v3",
    #     model_file="best-valid-loss.pt",
    #     default_tokenizer="espeak",
    #     default_lang="vi",
    # ),
]

MODEL_CONFIGS_DICT = {c.app_name: c for c in MODEL_CONFIGS}


def get_model_config(model_name: str) -> BaseConfig:
    """Get model configuration by name."""
    if model_name not in MODEL_CONFIGS_DICT:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CONFIGS_DICT[model_name]


def get_model_display_name(model_name: str) -> str:
    """Get the display name for a model (same as the key)."""
    return model_name


def list_available_models() -> list[str]:
    """List all available model names."""
    return list(MODEL_CONFIGS_DICT.keys())


@st.cache_resource
def create_app_instance(model_name: str) -> Any:
    """Create the appropriate app instance based on model type."""
    import torch

    # Clear CUDA cache before loading new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = get_model_config(model_name)

    if config.model_type == "zipvoice":
        return ZipVoiceApp(config)
    elif config.model_type == "f5tts":
        return F5TTSApp(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def main():
    """Main application entry point."""
    # Title and description
    st.title("🎙️ Unified TTS Demo")
    st.markdown(
        """
    <div class="info-box">
    <b>Welcome to the Unified TTS Demo!</b> This application supports multiple TTS models
    including ZipVoice and F5-TTS. Select a model below to get started with high-quality
    voice cloning and speech synthesis.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Model Selection in sidebar
    with st.sidebar:
        st.header("🤖 Model Selection")

        selected_model = st.selectbox(
            "Choose TTS Model", options=list_available_models(), index=0, help="Select which TTS model to use for generation"
        )

        config = get_model_config(selected_model)
        model_type = config.model_type

    # Update current model selection
    st.session_state.current_model = selected_model

    # Get cached app instance (automatically managed by st.cache_resource)
    try:
        with st.spinner(f"Loading {model_type.upper()} model... This may take a moment."):
            app = create_app_instance(selected_model)
    except Exception as e:
        st.error(f"Failed to initialize {model_type.upper()} model: {str(e)}")
        st.stop()

    # Display model-specific info in main area
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", model_type.upper())
    with col2:
        st.metric("Language", config.language)
    with col3:
        st.metric("Device", str(app.device).upper())

    # Run the app interface
    try:
        app.run()
    except Exception as e:
        traceback.print_exception(e)
        st.error(f"Application error: {str(e)}")
        st.write("Please refresh the page or select a different model.")

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
    <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>Unified TTS Demo | Current Model: {selected_model} | 🔬 Research Use</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
