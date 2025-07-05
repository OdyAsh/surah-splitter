"""
Service for accessing Quranic metadata.
"""

import json
from typing import List, Optional

from surah_splitter.utils.paths import QURAN_METADATA_PATH
from surah_splitter.utils.file_utils import load_json


class QuranMetadataService:
    """Service for accessing Quranic metadata."""

    def get_ayahs(self, surah_number: int, ayah_numbers: Optional[list[int]] = None) -> List[str]:
        """Get cleaned ayahs for a given surah.

        Args:
            surah_number: Surah number (1-114)
            ayah_numbers: Optional list of specific ayahs to match

        Returns:
            List of cleaned ayah texts

        Raises:
            FileNotFoundError: If metadata file not found
            ValueError: If surah not found in metadata
        """

        # Load from file
        surah_to_simple_ayahs_path = QURAN_METADATA_PATH / "surah_to_simple_ayahs.json"
        if not surah_to_simple_ayahs_path.exists():
            raise FileNotFoundError(
                f"Ayah data file not found at {surah_to_simple_ayahs_path}. "
                "Please run simple_ayahs_extractor.py script first."
            )

        surah_to_simple_ayahs_dict = load_json(surah_to_simple_ayahs_path)

        surah_number_str = str(surah_number)
        if surah_number_str not in surah_to_simple_ayahs_dict:
            raise ValueError(f"Surah {surah_number} not found in the ayah data. " f"Check {surah_to_simple_ayahs_path}")

        ayahs_dict = surah_to_simple_ayahs_dict[surah_number_str]
        ayahs = [
            ayahs_dict[v_id] for v_id in sorted(ayahs_dict.keys(), key=int) if not ayah_numbers or int(v_id) in ayah_numbers
        ]

        return ayahs

    def get_surah_name(self, surah_number: int) -> str:
        """Get the name of a surah by number.

        Args:
            surah_number: Surah number (1-114)

        Returns:
            Surah name

        Raises:
            FileNotFoundError: If metadata file not found
            ValueError: If surah not found in metadata
        """

        # Load from file
        surah_metadata_path = QURAN_METADATA_PATH / "quran-metadata-surah-name.json"
        if not surah_metadata_path.exists():
            raise FileNotFoundError(f"Surah metadata file not found at {surah_metadata_path}")

        with open(surah_metadata_path, "r", encoding="utf-8") as f:
            surah_metadata = json.load(f)

        surah_number_str = str(surah_number)
        if surah_number_str not in surah_metadata:
            raise ValueError(f"Surah {surah_number} not found in the metadata")

        surah_name = surah_metadata[surah_number_str]

        return surah_name
