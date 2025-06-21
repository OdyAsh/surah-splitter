"""Unit tests for the TranscriptionService."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from surah_splitter_new.services.transcription_service import TranscriptionService


@pytest.mark.unit
class TestTranscriptionService:
    """Test the TranscriptionService."""

    @patch("surah_splitter_new.services.transcription_service.load_model")
    @patch("surah_splitter_new.services.transcription_service.load_align_model")
    def test_initialize(self, mock_load_align_model, mock_load_model):
        """Test initializing the transcription service."""
        # Setup mocks
        mock_trans_model = MagicMock()
        mock_align_model = MagicMock()
        mock_align_metadata = MagicMock()

        mock_load_model.return_value = mock_trans_model
        mock_load_align_model.return_value = (mock_align_model, mock_align_metadata)

        # Create service
        service = TranscriptionService()

        # Initialize with default params
        service.initialize()

        # Assert models were loaded correctly
        mock_load_model.assert_called_once()
        mock_load_align_model.assert_called_once_with(language_code="ar", device=service.device)

        assert service.trans_model is mock_trans_model
        assert service.align_model is mock_align_model
        assert service.align_metadata is mock_align_metadata

    @patch("surah_splitter_new.services.transcription_service.load_audio")
    @patch("surah_splitter_new.services.transcription_service.load_model")
    @patch("surah_splitter_new.services.transcription_service.load_align_model")
    @patch("surah_splitter_new.services.transcription_service.align")
    def test_transcribe(self, mock_align, mock_load_align_model, mock_load_model, mock_load_audio):
        """Test the transcribe method with mocked dependencies."""
        # Setup mocks
        mock_audio = MagicMock()
        mock_trans_model = MagicMock()
        mock_align_model = MagicMock()
        mock_align_metadata = MagicMock()

        # Configure return values
        mock_load_audio.return_value = mock_audio
        mock_load_model.return_value = mock_trans_model
        mock_load_align_model.return_value = (mock_align_model, mock_align_metadata)

        # Configure transcription result
        trans_result = {"segments": [{"text": "Test transcription"}], "language": "ar"}
        mock_trans_model.transcribe.return_value = trans_result

        # Configure alignment result
        align_result = {
            "text": "Test transcription",
            "word_segments": [
                {"word": "Test", "start": 0.0, "end": 0.5, "score": 0.9},
                {"word": "transcription", "start": 0.5, "end": 1.0, "score": 0.8},
            ],
            "language": "ar",
        }
        mock_align.return_value = align_result

        # Create service
        service = TranscriptionService()
        service.initialize()  # This will use the mocked functions

        # Call transcribe
        audio_path = Path("dummy/path.mp3")
        result = service.transcribe(audio_path)

        # Assertions
        mock_load_audio.assert_called_once_with(audio_path)
        mock_trans_model.transcribe.assert_called_once()
        mock_align.assert_called_once()

        # Check result structure
        assert result["transcription"] == "Test transcription"
        assert len(result["word_segments"]) == 2
        assert result["word_segments"][0]["word"] == "Test"
        assert result["language"] == "ar"
