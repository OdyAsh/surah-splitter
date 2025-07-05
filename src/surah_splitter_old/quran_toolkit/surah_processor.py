"""
Main module for Surah Splitter.

This module implements functions to split Quranic audio files into individual ayahs
based on the alignment between transcribed text and ground truth ayah text.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
from huggingface_hub import snapshot_download
from whisperx.audio import load_audio
from whisperx.asr import load_model
from whisperx.alignment import load_align_model, align
import torch
import gc
from pydub import AudioSegment
from dataclasses import asdict
import shutil

from surah_splitter_old.quran_toolkit.ayah_matcher import match_ayahs_to_transcription
from surah_splitter_old.utils.app_logger import logger


def _process_audio_file(
    audio_path: Union[str, Path],
    ayahs: List[str],
    output_dir: Union[str, Path],
    model_name: str = "small",
    device: str = None,
    save_intermediates: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process a Quran audio file, transcribe it using WhisperX, and match ayahs.

    Args:
        audio_path: Path to the audio file
        ayahs: List of ground truth ayah texts
        output_dir: Directory to save timestamp outputs (e.g., outputs/reciter/timestamps/001/)
        model_name: WhisperX model name or size
        device: Device to use for processing (cuda or cpu)
        save_intermediates: Whether to save intermediate files

    Returns:
        List of dictionaries with ayah timing information
    """
    # Set up paths
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure batch size and compute type based on device
    batch_size = 16 if device == "cuda" else 4
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(f"Processing audio file: {audio_path}")
    logger.debug(f"Using device: {device}, model name/size: {model_name}")
    # Load audio
    logger.debug("Loading audio...")
    audio = load_audio(str(audio_path))
    audio_duration = len(audio) / 16000  # in seconds
    logger.debug(f"Audio duration: {audio_duration:.2f} seconds")

    # Load model and transcribe
    logger.info(f"Loading WhisperX {model_name} model...")
    try:
        model_transcribe = load_model(model_name, device, compute_type=compute_type, language="ar")
    except RuntimeError as e:
        if "Unable to open file 'model.bin' in model" not in str(e):
            raise e
        logger.warning(f"Model {model_name} couldn't be used from local cache. Downloading from Hugging Face Hub...")
        model_path = snapshot_download(model_name)
        model_transcribe = load_model(model_path, device, compute_type=compute_type, language="ar")

    logger.info("Transcribing audio...")
    initial_transcribed_result = model_transcribe.transcribe(
        audio,
        batch_size=batch_size,
        language="ar",
        print_progress=True,
        combined_progress=True,
        verbose=True,
    )
    logger.info("Transcription completed!")
    logger.debug(f"Detected language: {initial_transcribed_result['language']}")
    logger.debug(f"Number of segments: {len(initial_transcribed_result['segments'])}")

    if save_intermediates:
        transcription_file = output_dir / "01_transcription.json"
        with open(transcription_file, "w", encoding="utf-8") as f:
            json.dump(initial_transcribed_result, f, ensure_ascii=False, indent=2)
        logger.info(f"01 Transcription saved to: {transcription_file}")

    # Clean up model to free memory
    del model_transcribe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load alignment model
    try:
        logger.info("Loading alignment model for Arabic...")
        model_align, metadata = load_align_model(language_code="ar", device=device)
        logger.info("Arabic alignment model loaded successfully!")
    except Exception as e:
        logger.exception(f"Arabic alignment model not available: {e}")
        logger.error("Proceeding without word-level alignment")
        model_align = None
        metadata = None

    # Perform alignment if model is available
    if model_align is not None:
        logger.info("Performing forced alignment...")
        aligned_result = align(
            initial_transcribed_result["segments"],
            model_align,
            metadata,
            audio,
            device,
            return_char_alignments=False,
            print_progress=True,
            combined_progress=True,
        )

        logger.info("Alignment completed!")
        if "segments" in aligned_result:
            logger.debug(f"Number of aligned segments: {len(aligned_result['segments'])}")

        if save_intermediates:
            alignment_file = output_dir / "02_alignment.json"
            with open(alignment_file, "w", encoding="utf-8") as f:
                json.dump(aligned_result, f, ensure_ascii=False, indent=2)
            logger.info(f"02 Alignment saved to: {alignment_file}")

        # Clean up alignment model
        del model_align
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        # Use transcription result if alignment unavailable
        aligned_result = initial_transcribed_result

    # Match ayahs to transcription
    logger.info("Matching ayahs to transcription...")
    ayah_timestamps = match_ayahs_to_transcription(
        ayahs=ayahs,
        whisperx_result=aligned_result,
        audio_data=audio,
        save_intermediates=save_intermediates,
        output_dir=output_dir if save_intermediates else None,
        # audio_file_stem removed, assuming ayah_matcher.py is adapted
    )

    # Convert to dictionary format
    ayah_timestamps_dict = [asdict(ts) for ts in ayah_timestamps]

    # Save ayah timestamps
    possible_prefix = "final_" if save_intermediates else ""
    ayah_file = output_dir / f"{possible_prefix}ayah_timestamps.json"
    with open(ayah_file, "w", encoding="utf-8") as f:
        json.dump(ayah_timestamps_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"(Final output) Ayah timestamps saved to: {ayah_file}")

    # Display results
    logger.info("Ayah Timestamps of first and last Ayahs:")
    if ayah_timestamps:
        logger.info(
            f"Ayah {ayah_timestamps[0].ayah_number}: {ayah_timestamps[0].start_time:.2f}s - {ayah_timestamps[0].end_time:.2f}s ({ayah_timestamps[0].duration:.2f}s)"
        )
        logger.info(
            f"Ayah {ayah_timestamps[-1].ayah_number}: {ayah_timestamps[-1].start_time:.2f}s - {ayah_timestamps[-1].end_time:.2f}s ({ayah_timestamps[-1].duration:.2f}s)"
        )

    return ayah_timestamps_dict


def _split_audio_by_ayahs(
    audio_path: Union[str, Path],
    ayah_timestamps: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    file_prefix: str = "",
    audio_format: str = "mp3",
) -> List[Path]:
    """
    Split an audio file into individual ayahs based on timestamps.

    Args:
        audio_path: Path to the audio file
        ayah_timestamps: List of ayah timestamp dictionaries
        output_dir: Directory to save split audio files (e.g., outputs/reciter/ayah_audios/001/)
        file_prefix: Prefix for output files (e.g., "001_" for Surah 001)
        audio_format: Format for output audio files (mp3, wav, etc.)

    Returns:
        List of paths to the created audio files
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    # Load the audio file
    logger.info(f"Loading audio for splitting: {audio_path}")
    audio = AudioSegment.from_file(str(audio_path))

    # Sort timestamps by ayah number
    ayah_timestamps = sorted(ayah_timestamps, key=lambda x: x["ayah_number"])

    output_files = []

    for ts in ayah_timestamps:
        ayah_number = ts["ayah_number"]
        start_ms = int(ts["start_time"] * 1000)  # Convert seconds to milliseconds
        end_ms = int(ts["end_time"] * 1000)

        # Extract segment
        segment = audio[start_ms:end_ms]

        # Format ayah number with leading zeros
        ayah_str = str(ayah_number).zfill(3)

        # Create output filename
        # file_suffix is removed from here
        filename = f"{file_prefix}{ayah_str}.{audio_format}"  # e.g., 001_001.mp3
        output_path = output_dir / filename

        # Export segment
        segment.export(str(output_path), format=audio_format)
        output_files.append(output_path)

    logger.info(f"Split (and exported) {len(output_files)} ayahs to {output_dir}")
    return output_files


def process_surah(
    audio_path: Union[str, Path],
    ayahs: List[str],
    output_dir: Union[str, Path],
    surah_number: int,
    reciter_name: str,
    model_name: str = "small",
    device: str = None,
    save_intermediates: bool = False,
    save_incoming_surah_audio: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """
    Process a complete surah audio file, transcribe it, match ayahs, and split into individual ayahs.

    Args:
        audio_path: Path to the surah audio file
        ayahs: List of ground truth ayah texts
        output_dir: Base directory to save outputs (e.g., data/outputs)
        surah_number: The number of the surah
        reciter_name: Name of the reciter
        model_name: WhisperX model name (or size) to use
        device: Device to use for processing
        save_intermediates: Whether to save intermediate files

    Returns:
        Tuple of (ayah_timestamps, output_files)
    """
    logger.info(f"Processing Surah {surah_number}")
    logger.info(f"Audio file: {audio_path}")
    logger.debug(f"Number of ayahs: {len(ayahs)}")

    # Normalize reciter name and create base output directory for the reciter
    normalized_reciter_name = reciter_name.lower().replace(" ", "_").replace("-", "_")
    base_reciter_output_dir = Path(output_dir) / normalized_reciter_name

    # Define specific output directories based on the new structure
    surah_audio_output_dir = base_reciter_output_dir / "surah_audios"
    timestamps_output_dir_for_surah = base_reciter_output_dir / "timestamps" / f"{surah_number:03d}"
    ayah_audio_output_dir_for_surah = base_reciter_output_dir / "ayah_audios" / f"{surah_number:03d}"

    # Create all necessary output directories
    if save_incoming_surah_audio:
        surah_audio_output_dir.mkdir(parents=True, exist_ok=True)
    timestamps_output_dir_for_surah.mkdir(parents=True, exist_ok=True)
    ayah_audio_output_dir_for_surah.mkdir(parents=True, exist_ok=True)

    if save_incoming_surah_audio:
        # Copy original surah audio to the new location
        target_surah_audio_path = surah_audio_output_dir / f"{surah_number:03d}{Path(audio_path).suffix}"
        shutil.copy(Path(audio_path), target_surah_audio_path)
        logger.info(f"Copied original surah audio to: {target_surah_audio_path}")

    # Process audio to get ayah timestamps
    ayah_timestamps = _process_audio_file(
        audio_path=audio_path,
        ayahs=ayahs,
        output_dir=timestamps_output_dir_for_surah,
        model_name=model_name,
        device=device,
        save_intermediates=save_intermediates,
    )

    # Format the file prefix for split audio files
    file_prefix = f"{surah_number:03d}_"

    # Split the audio file into individual ayahs
    output_files = _split_audio_by_ayahs(
        audio_path=audio_path,
        ayah_timestamps=ayah_timestamps,
        output_dir=ayah_audio_output_dir_for_surah,
        file_prefix=file_prefix,
    )

    return ayah_timestamps, output_files
