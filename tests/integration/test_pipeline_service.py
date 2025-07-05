"""Integration tests for the PipelineService."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from surah_splitter.services.pipeline_service import PipelineService


@pytest.mark.integration
class TestPipelineService:
    """Integration tests for the PipelineService."""

    @patch("surah_splitter_new.services.transcription_service.TranscriptionService")
    @patch("surah_splitter_new.services.ayah_matching_service.AyahMatchingService")
    @patch("surah_splitter_new.services.segmentation_service.SegmentationService")
    @patch("surah_splitter_new.services.quran_metadata_service.QuranMetadataService")
    def test_process_surah(
        self,
        mock_quran_service_class,
        mock_segment_service_class,
        mock_matching_service_class,
        mock_trans_service_class,
        temp_output_dir,
        mock_transcription_result,
        mock_ayah_timestamps,
    ):
        """Test the complete pipeline process with mocked services."""
        # Create mock service instances
        mock_trans_service = MagicMock()
        mock_matching_service = MagicMock()
        mock_segment_service = MagicMock()
        mock_quran_service = MagicMock()

        # Configure class mocks to return instance mocks
        mock_trans_service_class.return_value = mock_trans_service
        mock_matching_service_class.return_value = mock_matching_service
        mock_segment_service_class.return_value = mock_segment_service
        mock_quran_service_class.return_value = mock_quran_service

        # Configure mock return values
        mock_trans_service.transcribe.return_value = mock_transcription_result
        mock_matching_service.match_ayahs.return_value = mock_ayah_timestamps
        mock_segment_service.split_audio.return_value = {1: Path(temp_output_dir / "076_001.mp3")}

        # Create pipeline service (this will use the mocked services)
        pipeline_service = PipelineService()

        # Test parameters
        audio_path = Path("dummy/path.mp3")
        surah_number = 76
        reciter_name = "test_reciter"
        output_dir = temp_output_dir

        # Call process_surah
        result = pipeline_service.process_surah(
            audio_path=audio_path,
            surah_number=surah_number,
            reciter_name=reciter_name,
            output_dir=output_dir,
            save_intermediates=True,
        )

        # Verify service initialization and method calls
        mock_trans_service.initialize.assert_called_once()
        mock_trans_service.transcribe.assert_called_once_with(
            audio_path, output_dir / reciter_name / "timestamps" / f"{surah_number:03d}"
        )

        mock_matching_service.match_ayahs.assert_called_once_with(
            mock_transcription_result, surah_number, output_dir / reciter_name / "timestamps" / f"{surah_number:03d}", True
        )

        mock_segment_service.split_audio.assert_called_once_with(
            audio_path, mock_ayah_timestamps["ayah_timestamps"], surah_number, reciter_name, output_dir, False
        )

        # Verify result structure
        assert "transcription" in result
        assert "ayah_matching" in result
        assert "segmentation" in result
        assert result["reciter_name"] == reciter_name
        assert result["surah_number"] == surah_number
