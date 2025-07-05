"""
Test for the updated word spans functionality in the ayah matching service.
"""

import unittest
import json
from pathlib import Path

import numpy as np

from surah_splitter.models.alignment import ReferenceWord, SegmentedWordSpan
from surah_splitter.services.ayah_matching_service import AyahMatchingService


class TestAyahMatchingService(unittest.TestCase):
    """Test the AyahMatchingService's word span functionality."""

    def setUp(self):
        self.service = AyahMatchingService()

    def test_convert_to_word_spans(self):
        """Test that word spans are created correctly from alignment indices."""
        # Create test data
        alignment_indices = [(0, 0), (1, 1), (2, 2), (3, 3)]

        recognized_words = [("word1", 0.0, 1.0, 1.0), ("word2", 1.0, 2.0, 0.9), ("word3", 2.0, 3.0, 0.8)]

        reference_words = [
            ReferenceWord(word="word1", ayah_number=1, word_location_wrt_ayah=1, word_location_wrt_surah=1),
            ReferenceWord(word="word2", ayah_number=1, word_location_wrt_ayah=2, word_location_wrt_surah=2),
            ReferenceWord(word="different", ayah_number=1, word_location_wrt_ayah=3, word_location_wrt_surah=3),
        ]

        # Call the function
        result = self.service._convert_to_word_spans(alignment_indices, recognized_words, reference_words)

        # Verify results
        self.assertEqual(len(result), 3)

        # First span - exact match
        self.assertEqual(result[0].reference_index_start, 0)
        self.assertEqual(result[0].reference_index_end, 1)
        self.assertEqual(result[0].reference_words_segment, "word1")
        self.assertEqual(result[0].input_words_segment, "word1")
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 1.0)
        self.assertTrue(result[0].flags & SegmentedWordSpan.EXACT)

        # Second span - exact match
        self.assertEqual(result[1].reference_index_start, 1)
        self.assertEqual(result[1].reference_index_end, 2)
        self.assertEqual(result[1].reference_words_segment, "word2")
        self.assertEqual(result[1].input_words_segment, "word2")
        self.assertEqual(result[1].start, 1.0)
        self.assertEqual(result[1].end, 2.0)
        self.assertTrue(result[1].flags & SegmentedWordSpan.EXACT)

        # Third span - inexact match
        self.assertEqual(result[2].reference_index_start, 2)
        self.assertEqual(result[2].reference_index_end, 3)
        self.assertEqual(result[2].reference_words_segment, "different")
        self.assertEqual(result[2].input_words_segment, "word3")
        self.assertEqual(result[2].start, 2.0)
        self.assertEqual(result[2].end, 3.0)
        self.assertTrue(result[2].flags & SegmentedWordSpan.INEXACT)


if __name__ == "__main__":
    unittest.main()
