"""Unit tests for the AyahMatchingService."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from surah_splitter_new.services.ayah_matching_service import AyahMatchingService


@pytest.mark.unit
class TestAyahMatchingService:
    """Test the AyahMatchingService."""

    @patch("surah_splitter_new.services.ayah_matching_service.QuranMetadataService")
    def test_match_ayahs(self, mock_quran_service_class, mock_transcription_result, temp_output_dir):
        """Test matching recognized words to reference ayahs."""
        # Setup mock
        mock_quran_service = MagicMock()
        mock_quran_service_class.return_value = mock_quran_service

        # Configure mock return values
        mock_quran_service.get_ayahs.return_value = ["هل أتى على الإنسان حين من الدهر لم يكن شيئا مذكورا"]

        # Create service instance
        service = AyahMatchingService()

        # Call match_ayahs
        result = service.match_ayahs(mock_transcription_result, 76, temp_output_dir, True)

        # Assertions
        mock_quran_service.get_ayahs.assert_called_once_with(76)

        # Check result structure
        assert "ayah_timestamps" in result
        assert isinstance(result["ayah_timestamps"], list)
        assert len(result["ayah_timestamps"]) > 0

        # Check first ayah timestamp
        first_ayah = result["ayah_timestamps"][0]
        assert "ayah_number" in first_ayah
        assert "start_time" in first_ayah
        assert "end_time" in first_ayah
        assert "text" in first_ayah

        # Verify that intermediates were saved if requested
        if temp_output_dir.exists():
            # There should be multiple intermediate files
            saved_files = list(temp_output_dir.glob("*.json"))
            assert len(saved_files) > 0
