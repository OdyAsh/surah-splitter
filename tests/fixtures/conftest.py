"""Test fixtures for the Surah Splitter tests."""

import json
import os
import pytest
from pathlib import Path
import tempfile

import numpy as np


@pytest.fixture
def test_audio_path():
    """Return a path to sample test audio file."""
    # Ideally, this would be a small test audio file in the fixtures directory
    # For now, we'll use an existing file from the data directory
    file_path = Path(__file__).parent.parent / "data" / "input_surahs_to_split" / "adel_ryyan" / "076 Al-Insaan.mp3"

    if not file_path.exists():
        pytest.skip(f"Test audio file not found at {file_path}")

    return file_path


@pytest.fixture
def mock_transcription_result():
    """Return a mock transcription result."""
    return {
        "transcription": "بسم الله الرحمن الرحيم هل أتى على الإنسان حين من الدهر",
        "language": "ar",
        "word_segments": [
            {"word": "بسم", "start": 0.5, "end": 0.9, "score": 0.98},
            {"word": "الله", "start": 1.0, "end": 1.3, "score": 0.99},
            {"word": "الرحمن", "start": 1.4, "end": 1.8, "score": 0.97},
            {"word": "الرحيم", "start": 1.9, "end": 2.3, "score": 0.96},
            {"word": "هل", "start": 2.5, "end": 2.7, "score": 0.95},
            {"word": "أتى", "start": 2.8, "end": 3.1, "score": 0.94},
            {"word": "على", "start": 3.2, "end": 3.5, "score": 0.93},
            {"word": "الإنسان", "start": 3.6, "end": 4.0, "score": 0.92},
            {"word": "حين", "start": 4.1, "end": 4.4, "score": 0.91},
            {"word": "من", "start": 4.5, "end": 4.7, "score": 0.90},
            {"word": "الدهر", "start": 4.8, "end": 5.2, "score": 0.89},
        ],
    }


@pytest.fixture
def mock_ayah_timestamps():
    """Return mock ayah timestamps for testing."""
    return {
        "ayah_timestamps": [
            {
                "ayah_number": 1,
                "start_time": 0.5,
                "end_time": 5.2,
                "text": "هل أتى على الإنسان حين من الدهر لم يكن شيئا مذكورا",
            }
        ],
        "word_spans": [
            {
                "reference_index_start": 0,
                "reference_index_end": 10,
                "reference_words_segment": "هل أتى على الإنسان حين من الدهر",
                "input_words_segment": "هل أتى على الإنسان حين من الدهر",
                "start": 0.5,
                "end": 5.2,
                "flags": 4,
            }
        ],
    }


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
