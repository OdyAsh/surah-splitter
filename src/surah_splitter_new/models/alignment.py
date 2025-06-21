"""
Data models for alignment and matching.
"""

from dataclasses import dataclass


@dataclass
class ReferenceWord:
    """Represents a word from the ground truth text with position information."""

    word: str
    ayah_number: int
    word_location_wrt_ayah: int
    word_location_wrt_surah: int


@dataclass
class SegmentedWordSpan:
    """
    Represents a span of words with timing information, matching the structure
    used in the quran-align C++ implementation but adapted for Python.
    """

    reference_index_start: int  # Start index within reference (i.e., ground truth) words
    reference_index_end: int  # End index (exclusive) within reference words
    reference_words_segment: str  # Segment of reference words (just for tracing purposes)

    input_words_segment: str  # Segment of input (i.e., WhisperX recognized) words (just for tracing purposes)
    start: float  # Start time in seconds
    end: float  # End time in seconds

    flags: int = 0  # Flags indicating match quality

    # Flag values (matching the C++ impl)
    CLEAR = 0
    MATCHED_INPUT = 1
    MATCHED_REFERENCE = 2
    EXACT = 4
    INEXACT = 8


@dataclass
class SegmentationStats:
    """Track statistics about the matching process."""

    insertions: int = 0
    deletions: int = 0
    transpositions: int = 0


@dataclass
class AyahTimestamp:
    """The final output format for ayah timing information."""

    ayah_number: int
    start_time: float
    end_time: float
    text: str
