"""
Service for transcribing audio using WhisperX.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import gc
from huggingface_hub import snapshot_download

# Quick fix to make `import load_model` load faster
#   Source: https://github.com/m-bain/whisperX/issues/656#issuecomment-1877955404
from huggingface_hub.utils import _runtime

_runtime._is_google_colab = False

from surah_splitter_new.models.transcription import Transcription, RecognizedWordSegment
from surah_splitter_new.utils.app_logger import logger, LoggerTimingContext


class TranscriptionService:
    """Service for transcribing audio using WhisperX."""

    def __init__(self):
        self.device = "cpu"
        self.compute_type = "int8"
        self.wx_trans_model = None
        self.wx_align_model = None
        self.wx_load_audio = None
        self.wx_align = None
        self.torch_cuda = None

    def initialize(self, model_name: str = "OdyAsh/faster-whisper-base-ar-quran", device: Optional[str] = None):
        """Initialize WhisperX models.

        Args:
            model_name: Name of the model to use
            device: Device to use (cuda/cpu)
        """
        logger.info(f"Initializing transcription service with model: {model_name}")

        from torch import cuda

        self.torch_cuda = cuda

        # Set device (cuda if available, otherwise cpu)
        if device is None:
            self.device = "cuda" if self.torch_cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.debug(f"Using device: {self.device}")

        # Initialize WhisperX transcription model
        try:
            with LoggerTimingContext("Initializing WhisperX models"):
                from whisperx.asr import load_model

            # If it's a HuggingFace model, first download it
            # Otherwise assume it's a whisperx model size
            if "/" in model_name:
                with LoggerTimingContext(f'Downloading "{model_name}" from HuggingFace'):
                    model_path = snapshot_download(repo_id=model_name)
                    model_name = model_path  # Use the local path after download

            with LoggerTimingContext(f'Loading "{model_name}" WhisperX model'):
                self.wx_trans_model = load_model(model_name, self.device, compute_type=self.compute_type)

            # Initialize audio loading function
            from whisperx.audio import load_audio

            self.wx_load_audio = load_audio

            # Initialize alignment model
            with LoggerTimingContext("Importing WhisperX alignment model"):
                from whisperx.alignment import load_align_model, align

                self.wx_align_model, self.align_metadata = load_align_model(language_code="ar", device=self.device)
                self.wx_align = align

        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise

    def transcribe(self, audio_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Transcribe audio file and return word-level timestamps.

        Args:
            audio_path: Path to audio file
            output_dir: Optional directory to save intermediate files

        Returns:
            Dict containing transcription results with word-level timestamps
        """
        logger.info(f"Transcribing audio file: {audio_path}")

        # Load audio
        with LoggerTimingContext("Loading audio file"):
            audio = self.wx_load_audio(audio_path)

        # Ensure models are initialized
        if self.wx_trans_model is None or self.wx_align_model is None:
            logger.debug("Models not initialized, initializing now")
            self.initialize()

        # Perform transcription
        with LoggerTimingContext("Transcribing audio"):
            trans_result = self.wx_trans_model.transcribe(audio, batch_size=16)

        # Perform word alignment
        with LoggerTimingContext("Aligning to word-level timestamps"):
            align_result = self.wx_align(
                trans_result["segments"], self.wx_align_model, self.align_metadata, audio, self.device
            )

        # Create result dictionary
        result = {
            "transcription": align_result.get("text", ""),
            "word_segments": align_result.get("word_segments", []),
            "language": align_result.get("language", "ar"),
        }

        # Save intermediates if output_dir provided
        if output_dir:
            logger.debug(f"Saving intermediate files to: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save transcription result
            with open(output_dir / "01_transcription.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        # Clean up GPU memory if needed
        if self.device == "cuda":
            logger.debug("Cleaning up GPU memory")
            gc.collect()
            self.torch_cuda.empty_cache()

        logger.success(f"Transcription of {audio_path.name} complete")
        return result

    def __del__(self):
        """Clean up resources when the service is destroyed."""
        # Clean up GPU memory if needed
        if self.device == "cuda":
            logger.debug("Cleaning up GPU memory on service destruction")
            gc.collect()
            if self.torch_cuda:
                self.torch_cuda.empty_cache()
