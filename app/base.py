import os
import json
import uuid
import time
import tempfile
import subprocess
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import torch
import torchaudio
import streamlit as st
import numpy as np

from data.ref_audio.metadata import REF_SPEAKERS_MAP, RefSpeaker


def convert_audio_to_wav(input_path: str, output_path: str, sample_rate: int = 24000) -> bool:
    """
    Convert audio file to WAV format using ffmpeg.
    Returns True if successful, False otherwise.
    """
    try:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", "1",  # mono
            "-y",  # overwrite
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Audio conversion failed: {e}")
        return False


def save_additional_audio(audio_path: str, transcript: str, audio_dir: str = "data/ref_audio/additional/audio") -> tuple[str, str]:
    """
    Save audio file to additional audio directory with metadata.

    Args:
        audio_path: Path to the temporary audio file
        transcript: Transcription of the audio
        audio_dir: Directory to save audio files

    Returns:
        Tuple of (permanent_audio_path, voice_name)
    """
    # Create directory if it doesn't exist
    audio_dir_path = Path(audio_dir)
    audio_dir_path.mkdir(parents=True, exist_ok=True)

    # Calculate hash digest of audio file
    hash_md5 = hashlib.md5()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = hash_md5.hexdigest()

    # Generate filename based on hash
    voice_name = f"custom_{file_hash}"
    audio_filename = f"{file_hash}.wav"
    permanent_path = audio_dir_path / audio_filename

    # Copy audio file to permanent location (skip if already exists)
    if not permanent_path.exists():
        import shutil
        shutil.copy(audio_path, permanent_path)

    return str(permanent_path), voice_name


def append_audio_metadata(audio_path: str, transcript: str, metadata_file: str = "data/ref_audio/additional/metadata.jsonl"):
    """
    Append audio metadata to JSONL file.

    Args:
        audio_path: Path to the audio file
        transcript: Transcription of the audio
        metadata_file: Path to metadata JSONL file
    """
    metadata = {
        "audio_path": audio_path,
        "text": transcript
    }

    # Create parent directory if it doesn't exist
    metadata_path = Path(metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if this audio path already exists in metadata
    existing_paths = set()
    if metadata_path.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        existing_paths.add(entry.get("audio_path"))
                    except json.JSONDecodeError:
                        continue

    # Only append if this audio path doesn't already exist
    if audio_path not in existing_paths:
        with open(metadata_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")


@dataclass
class BaseConfig:
    app_name: str
    model_type: str
    audio_sample_rate: int
    language: str


@dataclass
class HistoryEntry:
    """Data class for storing synthesis history."""

    id: str
    app_name: str
    model_type: str
    text: str
    voice: str
    audio_path: str
    parameters: dict[str, Any]
    rtf: float


class BaseApp(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._initialize_history()
        self.load_model()

    @abstractmethod
    def load_model(self):
        """
        do something like:
            self.model = ...
        or
            self.engine = ...
        """

    @abstractmethod
    def render_voice_selection(self) -> tuple[str, dict[str, Any]]:
        """
        using streamlit to render voice selection.
        details will be described at TraditionalApp and VoiceCloneApp
        """

    @abstractmethod
    def render_parameters(self) -> dict[str, Any]:
        """
        using streamlit to render parameters adjustment
        using slider for number or dropdown for choices
        details will be described at concrete class

        return dictionary of parameters,
        """

    @abstractmethod
    def generate_speech(self, text: str, **params) -> np.ndarray:
        """
        Inputs are text from render_text_input() and params gathered from render_voice_selection() and render_parameters()

        return audio
        """

    def render_text_input(self) -> str:
        """Render text input area with examples."""
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
            example_texts = self.get_examples(self.config.language)
            for i, text in enumerate(example_texts, 1):
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(f"📄 Use Example {i}", key=f"use_example_{i}", use_container_width=True):
                        st.session_state.example_text = text
                        st.rerun()
                with col2:
                    st.write(text)

        return text_to_synthesize

    @staticmethod
    def get_examples(language: str):
        if language in ["en-us", "en"]:
            return [
                "The world is full of fascinating information and endless possibilities. I hope you have a wonderful and productive day.",
                "It's a beautiful Monday morning here in San Francisco as the city begins a new week. I hope your day is filled with purpose and success.",
                "Knowledge is a journey, not a destination. Let's learn something new together today and make progress towards our goals.",
            ]
        if language == "vi":
            return [
                "Hiện tại thì chương trình ưu đãi này bên em chỉ còn vài ngày nữa là hết hạn rồi. nên anh cứ suy nghĩ và cân nhắc sớm để tham gia chương trình.",
                "Em sẽ gọi lại sau để hỗ trợ đăng kí cho anh nhé. Chúc anh một ngày tốt lành.",
                "Phân tích cách họ đang cạnh tranh và hợp tác với các ngân hàng truyền thống, từ đó thay đổi cách người dùng tiếp cận dịch vụ tài chính.",
            ]
        return []

    def _initialize_history(self):
        """Initialize history in session state."""
        if "synthesis_history" not in st.session_state:
            st.session_state.synthesis_history = []

    def add_to_history(self, entry: HistoryEntry):
        """Add a new entry to synthesis history."""
        if "synthesis_history" not in st.session_state:
            st.session_state.synthesis_history = []
        st.session_state.synthesis_history.append(entry)

    def save_audio_and_create_history(
        self,
        text: str,
        voice: str,
        params: dict[str, Any],
        audio_array: np.ndarray,
        rtf: float,
    ) -> HistoryEntry:
        """Save audio to permanent location and create history entry."""
        # Save audio to permanent location for history
        history_dir = Path("history")
        history_dir.mkdir(exist_ok=True)

        entry_id = str(uuid.uuid4())[:8]
        audio_filename = f"audio_{entry_id}_{int(time.time())}.wav"
        audio_path = history_dir / audio_filename
        audio_path = str(audio_path)

        # Ensure audio is in the right shape for saving
        audio_tensor = torch.from_numpy(audio_array)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        torchaudio.save(audio_path, audio_tensor, self.config.audio_sample_rate)

        # Create history entry
        history_entry = HistoryEntry(
            id=entry_id,
            app_name=self.config.app_name,
            model_type=self.config.model_type,
            text=text,
            voice=voice,
            audio_path=audio_path,
            rtf=rtf,
            parameters=params,
        )

        return history_entry

    def render_audio_output(self, audio_array: np.ndarray, rtf: float):
        """Render the generated audio output."""
        # Success message and metrics
        st.success(f"✅ Speech generated successfully! (RTF: {rtf:.3f})")

        # Display generated audio
        st.subheader("🔊 Generated Audio")
        st.audio(audio_array, sample_rate=self.config.audio_sample_rate)

        # Save audio option for download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            torchaudio.save(tmp_file.name, torch.from_numpy(audio_array).unsqueeze(0), self.config.audio_sample_rate)

            with open(tmp_file.name, "rb") as f:
                st.download_button(
                    label="💾 Download Audio",
                    data=f.read(),
                    file_name=f"generated_{int(time.time())}.wav",
                    mime="audio/wav",
                )

    def render_generation_details(
        self, text: str | None, voice: str | None, params: dict[str, Any], rtf: float | None, show_basic=True
    ):
        """Render generation details in an expander."""
        with st.expander("📊 Generation Details", expanded=False):

            if show_basic:
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"• Model: {self.config.app_name}")
                    st.write(f"• Model Type: {self.config.model_type}")
                    if text is not None:
                        st.write(f"• Text: {text}")
                    if voice is not None:
                        st.write(f"• Voice: {voice}")
                    if rtf is not None:
                        st.write(f"• RTF: {rtf}")

            else:
                col2 = st.container()

            with col2:
                st.write("**Generation Settings:**")
                st.write(
                    "```json\n"
                    + json.dumps({k: v for k, v in params.items() if k != "seed"}, indent=4, ensure_ascii=False)
                    + "\n```"
                )
                # for param, value in params.items():
                #     if param not in ["seed"]:  # Exclude some params for cleaner display
                #         st.write(f"• {param.replace('_', ' ').title()}: {value}")

    def render_history(self):
        """Render synthesis history section."""
        # Initialize history if not exists
        if "synthesis_history" not in st.session_state:
            st.session_state.synthesis_history = []

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
                entry: HistoryEntry
                with st.expander(f"🎵 {entry.app_name} ({entry.model_type}) - {entry.voice}"):
                    # Main content
                    col1, col2 = st.columns([5, 1])

                    with col1:
                        st.write(f"**Text:** {entry.text}")

                        if os.path.exists(entry.audio_path):
                            try:
                                # Load audio using torchaudio (same as saved)
                                audio_tensor, sample_rate = torchaudio.load(entry.audio_path)

                                # Convert to numpy and squeeze to 1D
                                audio_numpy = audio_tensor.squeeze().numpy()

                                # Use the same method as render_audio_output
                                st.audio(audio_numpy, sample_rate=sample_rate)

                            except Exception as e:
                                st.error(f"Error loading audio: {str(e)}")
                        else:
                            st.error(f"Audio file not found: {entry.audio_path}")

                    with col2:
                        # Compact controls
                        if os.path.exists(entry.audio_path):
                            with open(entry.audio_path, "rb") as f:
                                st.download_button(
                                    label="Download",
                                    data=f.read(),
                                    file_name=os.path.basename(entry.audio_path),
                                    mime="audio/wav",
                                    key=f"download_{entry.id}",
                                    use_container_width=True,
                                    type="secondary",
                                )

                        if st.button("Remove", key=f"delete_{entry.id}", use_container_width=True, type="primary"):
                            # Remove from history
                            st.session_state.synthesis_history = [
                                h for h in st.session_state.synthesis_history if h.id != entry.id
                            ]
                            # Delete audio file
                            if os.path.exists(entry.audio_path):
                                os.remove(entry.audio_path)
                            st.rerun()

                    # Details in history
                    self.render_generation_details(
                        text=None, voice=None, params=entry.parameters, rtf=entry.rtf, show_basic=False
                    )

    def log_inference_info(self, text: str, voice: str, params: dict[str, Any]):
        """Log inference information to console."""
        print("\n" + "=" * 80)
        print(f"🎙️ INFERENCE STARTING")
        print("=" * 80)
        print(f"🤖 Model: {self.config.app_name}")
        print(f"🤖 Model Type: {self.config.model_type}")
        print(f"📝 Target Text: {text}")
        print(f"📝 Voice: {voice}")
        print(f"🎛️ Generation Parameters:")
        for param, value in params.items():
            print(f"   • {param}: {value}")
        print(f"💻 Device: {str(self.device).upper()}")
        print("=" * 80)

    def run(self):
        """Main application loop. Should be called by main.py."""
        # Render voice selection
        voice_name, voice_params = self.render_voice_selection()

        model_params = self.render_parameters()

        # Combine all parameters
        params = {**voice_params, **model_params}

        # Render text input
        text_to_synthesize = self.render_text_input()

        # Validation and Generate button
        can_generate = all([text_to_synthesize] + list(voice_params.values()))

        if not can_generate:
            st.warning("Please provide text and select/upload a reference voice")

        # Generate button (full width)
        if st.button("🚀 Generate Speech", disabled=not can_generate, type="primary", use_container_width=True):
            if can_generate:
                try:

                    # Log inference information
                    self.log_inference_info(text_to_synthesize, voice_name, params)

                    # Generate speech with progress indicator
                    with st.spinner("🎵 Generating speech... Please wait."):
                        start_time = time.time()
                        audio_array = self.generate_speech(text=text_to_synthesize, **params)
                        generation_time = time.time() - start_time
                        wav_seconds = len(audio_array) / self.config.audio_sample_rate
                        rtf = generation_time / wav_seconds

                    # Clean up after generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Render audio output
                    self.render_audio_output(audio_array, rtf)

                    # Save to history
                    history_entry = self.save_audio_and_create_history(
                        text_to_synthesize, voice_name, params, audio_array, rtf=rtf
                    )
                    self.add_to_history(history_entry)

                    # Render generation details
                    self.render_generation_details(None, None, params, rtf)

                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                    print(f"Generation error: {e}")

        # Render history
        self.render_history()


class VoiceCloneConfig(BaseConfig):
    voices_map: dict[str, RefSpeaker] = REF_SPEAKERS_MAP


class VoiceCloneApp(BaseApp):
    def __init__(self, config: VoiceCloneConfig):
        super().__init__(config)

        self.config: VoiceCloneConfig

    def render_voice_selection(self):
        """Render voice selection UI and return selected voice info."""
        with st.sidebar:
            st.subheader("🎤 Voice Selection")

            voice_source = st.radio(
                "Voice Source",
                options=["Preset Voices", "Custom Upload", "Record Audio"],
                help="Choose a preset voice, upload your own, or record directly"
            )

            prompt_audio_path = None
            prompt_text = None
            selected_voice = "custom"

            if voice_source == "Preset Voices":
                selected_voice = st.selectbox(
                    "Select Voice",
                    options=list(self.config.voices_map.keys()),
                    format_func=lambda x: x.replace("__", " - ").replace("_", " "),
                )

                if selected_voice:
                    ref_speaker = self.config.voices_map[selected_voice]
                    prompt_audio_path = ref_speaker.audio_path
                    prompt_text = ref_speaker.text

                    st.info(f"**Reference text:** {prompt_text}")

                    # Display reference audio
                    if os.path.exists(prompt_audio_path):
                        st.audio(prompt_audio_path, format="audio/wav")

            elif voice_source == "Custom Upload":
                uploaded_file = st.file_uploader(
                    "Upload Reference Audio", type=["wav", "mp3", "m4a"], help="Upload a reference audio file for voice cloning"
                )

                prompt_text = st.text_area(
                    "Reference Text",
                    placeholder="Enter the transcription of the reference audio...",
                    help="Provide accurate transcription of the uploaded audio",
                )

                if uploaded_file and prompt_text:
                    # Save uploaded file temporarily with original extension
                    file_extension = Path(uploaded_file.name).suffix.lower()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_path = tmp_file.name

                    # Convert m4a to wav if needed (torchaudio doesn't support m4a by default)
                    if file_extension == ".m4a":
                        wav_path = temp_path.replace(".m4a", ".wav")
                        if convert_audio_to_wav(temp_path, wav_path, self.config.audio_sample_rate):
                            converted_path = wav_path
                            os.remove(temp_path)  # Remove original m4a file
                            st.success("✅ M4A file converted to WAV successfully!")
                        else:
                            st.error("❌ Failed to convert M4A file. Please ensure ffmpeg is installed.")
                            converted_path = None
                    else:
                        converted_path = temp_path

                    if converted_path:
                        # Save to permanent location and append metadata
                        prompt_audio_path, selected_voice = save_additional_audio(converted_path, prompt_text)
                        append_audio_metadata(prompt_audio_path, prompt_text)
                        # Extract just the hash from voice_name for display
                        file_hash = selected_voice.replace("custom_", "")
                        st.success(f"✅ Audio saved with hash: `{file_hash[:16]}...`")
                        st.audio(uploaded_file)

            else:  # Record Audio
                st.info("🎙️ Click the microphone button below to start recording")

                recorded_audio = st.audio_input("Record your voice")

                prompt_text = st.text_area(
                    "Reference Text",
                    placeholder="Enter the transcription of what you just recorded...",
                    help="Provide accurate transcription of your recorded audio",
                )

                if recorded_audio and prompt_text:
                    # Save recorded audio to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(recorded_audio.read())
                        temp_path = tmp_file.name

                    # Save to permanent location and append metadata
                    prompt_audio_path, selected_voice = save_additional_audio(temp_path, prompt_text)
                    append_audio_metadata(prompt_audio_path, prompt_text)
                    # Extract just the hash from voice_name for display
                    file_hash = selected_voice.replace("custom_", "")
                    st.success(f"✅ Recording saved with hash: `{file_hash[:16]}...`")
                    st.audio(recorded_audio)

            return selected_voice, dict(prompt_audio_path=prompt_audio_path, prompt_text=prompt_text)


class TraditionalConfig(BaseConfig):
    voices_map: dict[str, int]


class TraditionalApp(BaseApp):
    def __init__(self, config: TraditionalConfig):
        super().__init__(config)

        self.config: TraditionalConfig

    def render_voice_selection(self) -> tuple[str, dict[str, Any]]:
        """
        Return voice, dict(speaker_id=self.config.voices_map[voice])
        """
        with st.sidebar:
            st.subheader("🎤 Voice Selection")

            selected_voice = st.selectbox(
                "Select Voice",
                options=list(self.config.voices_map.keys()),
                format_func=lambda x: x.replace("_", " ").title(),
                help="Choose a voice from the available speakers",
            )

            speaker_id = self.config.voices_map[selected_voice]

            return selected_voice, {"speaker_id": speaker_id}
