"""Unit tests for the QuranMetadataService."""

import pytest
from pathlib import Path
import json
import shutil

from surah_splitter_new.services.quran_metadata_service import QuranMetadataService


@pytest.fixture
def mock_metadata_dir(temp_output_dir):
    """Create mock metadata files for testing."""
    mock_dir = temp_output_dir / "quran_metadata"
    mock_dir.mkdir(parents=True, exist_ok=True)

    # Create mock ayahs file
    mock_ayahs = {
        "1": {"1": "بسم الله الرحمن الرحيم", "2": "الحمد لله رب العالمين"},
        "76": {"1": "هل أتى على الإنسان حين من الدهر لم يكن شيئا مذكورا"},
    }
    with open(mock_dir / "surah_to_simple_ayahs.json", "w", encoding="utf-8") as f:
        json.dump(mock_ayahs, f, ensure_ascii=False)

    # Create mock surah names file
    mock_names = {"1": "الفاتحة", "76": "الإنسان"}
    with open(mock_dir / "quran-metadata-surah-name.json", "w", encoding="utf-8") as f:
        json.dump(mock_names, f, ensure_ascii=False)

    return mock_dir


@pytest.mark.unit
class TestQuranMetadataService:
    """Test the QuranMetadataService."""

    def test_get_ayahs(self, mock_metadata_dir, monkeypatch):
        """Test retrieving ayahs for a surah."""
        # Mock the QURAN_METADATA_PATH
        monkeypatch.setattr("surah_splitter_new.services.quran_metadata_service.QURAN_METADATA_PATH", mock_metadata_dir)

        # Create service
        service = QuranMetadataService()

        # Test getting ayahs for surah 1
        ayahs = service.get_ayahs(1)
        assert len(ayahs) == 2
        assert ayahs[0] == "بسم الله الرحمن الرحيم"
        assert ayahs[1] == "الحمد لله رب العالمين"

        # Test getting ayahs for surah 76
        ayahs = service.get_ayahs(76)
        assert len(ayahs) == 1
        assert ayahs[0] == "هل أتى على الإنسان حين من الدهر لم يكن شيئا مذكورا"

        # Test caching - calls should return the same objects
        ayahs2 = service.get_ayahs(1)
        assert ayahs2 is service.metadata_cache["ayahs_1"]

    def test_get_surah_name(self, mock_metadata_dir, monkeypatch):
        """Test retrieving the name of a surah."""
        # Mock the QURAN_METADATA_PATH
        monkeypatch.setattr("surah_splitter_new.services.quran_metadata_service.QURAN_METADATA_PATH", mock_metadata_dir)

        # Create service
        service = QuranMetadataService()

        # Test getting surah names
        assert service.get_surah_name(1) == "الفاتحة"
        assert service.get_surah_name(76) == "الإنسان"

        # Test caching
        assert "surah_name_1" in service.metadata_cache
        assert service.metadata_cache["surah_name_1"] == "الفاتحة"

    def test_file_not_found_error(self, temp_output_dir, monkeypatch):
        """Test error handling when metadata files are not found."""
        # Mock the QURAN_METADATA_PATH to a directory without the required files
        monkeypatch.setattr("surah_splitter_new.services.quran_metadata_service.QURAN_METADATA_PATH", temp_output_dir)

        # Create service
        service = QuranMetadataService()

        # Test that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            service.get_ayahs(1)

        with pytest.raises(FileNotFoundError):
            service.get_surah_name(1)
