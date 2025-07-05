"""
Service for orchestrating the complete processing pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from surah_splitter_new.services.transcription_service import TranscriptionService
from surah_splitter_new.services.ayah_matching_service import AyahMatchingService
from surah_splitter_new.services.segmentation_service import SegmentationService
from surah_splitter_new.services.quran_metadata_service import QuranMetadataService
from surah_splitter_new.utils.app_logger import logger


class PipelineService:
    """Service for orchestrating the complete processing pipeline."""

    def __init__(self):
        self.transcription_service = TranscriptionService()
        self.ayah_matching_service = AyahMatchingService()
        self.segmentation_service = SegmentationService()
        self.quran_service = QuranMetadataService()

    def process_surah(
        self,
        audio_path: Path,
        surah_number: int,
        reciter_name: str,
        output_dir: Path,
        ayah_numbers: Optional[list[int]] = None,
        model_name: str = "OdyAsh/faster-whisper-base-ar-quran",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        save_intermediates: bool = False,
        save_incoming_surah_audio: bool = False,
    ) -> Dict[str, Any]:
        """Process a surah audio file through the complete pipeline.

        Args:
            audio_path: Path to surah audio file
            surah_number: Surah number (1-114)
            reciter_name: Name of the reciter
            output_dir: Base output directory
            model_name: WhisperX model name
            device: Device to use (cuda/cpu)
            save_intermediates: Whether to save intermediate files
            save_incoming_surah_audio: Whether to save original surah audio

        Returns:
            Dict with processing results and paths
        """
        logger.info(f"Starting processing pipeline for surah {surah_number} by {reciter_name}")

        # Create timestamps directory if needed
        timestamps_dir = None
        if save_intermediates:
            timestamps_dir = output_dir / reciter_name / "timestamps" / f"{surah_number:03d}"
            timestamps_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created timestamps directory: {timestamps_dir}")

        # Step 1: Initialize transcription service
        logger.info("Initializing transcription service")
        self.transcription_service.initialize(model_name, device, compute_type)

        # Step 2: Transcribe audio
        logger.info(f"Transcribing audio: {audio_path}")
        transcription_result = self.transcription_service.transcribe(audio_path, timestamps_dir)
        logger.success("Transcription completed")

        # Step 3: Match ayahs to transcription
        logger.info("Matching ayahs to transcription")
        ayah_matching_result = self.ayah_matching_service.match_ayahs(
            transcription_result, surah_number, ayah_numbers, timestamps_dir, save_intermediates
        )
        logger.success("Ayah matching completed")

        # Step 4: Split audio by ayahs
        logger.info("Splitting audio by ayahs")
        segmentation_result = self.segmentation_service.split_audio(
            audio_path,
            ayah_matching_result["ayah_timestamps"],
            surah_number,
            reciter_name,
            output_dir,
            save_incoming_surah_audio,
        )
        logger.success("Audio segmentation completed")

        logger.success(f"Pipeline processing completed for surah {surah_number} by {reciter_name}")
        return {
            "transcription": transcription_result,
            "ayah_matching": ayah_matching_result,
            "segmentation": {ayah: str(path) for ayah, path in segmentation_result.items()},
            "reciter_name": reciter_name,
            "surah_number": surah_number,
        }
