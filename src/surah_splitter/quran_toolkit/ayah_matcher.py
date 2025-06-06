"""
Ayah-Words Matching Algorithm.

This module implements algorithms for matching transcribed words to ground truth ayahs,
inspired by the quran-align C++ implementation but reimplemented in Python to solve
the specific problems outlined in the ayah_words_matching_problem.md document.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from dataclasses import dataclass

# TODO soon: Add params for "save_intermediates" which will save intermediate results
#   and "save_stats" to save statistics about the matching process.
#   (don't forget to add "_0x_" numbering to the filenames to understand the order of execution)


@dataclass
class RecognizedWord:
    """Represents a word recognized by WhisperX with timing information."""

    word: str
    start: float
    end: float
    score: float = 1.0


@dataclass
class SegmentedWordSpan:
    """
    Represents a span of words with timing information, matching the structure
    used in the quran-align C++ implementation but adapted for Python.
    """

    index_start: int  # Start index within ground truth words
    index_end: int  # End index (exclusive) within ground truth words
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
    text: str
    start_time: float
    end_time: float
    duration: float


def clean_text(text: str) -> str:
    """Clean Arabic text by removing diacritics and non-Arabic characters."""
    # Remove diacritics (tashkeel)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Keep only Arabic letters and spaces
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def align_words(
    input_words: List[RecognizedWord],
    reference_words: List[str],
    stats: SegmentationStats,
) -> List[SegmentedWordSpan]:
    """
    Align transcribed words with reference words using dynamic programming.
    This is a Python implementation of the algorithm in match.cc.

    Args:
        input_words: List of words recognized by WhisperX with timing info
        reference_words: List of ground truth words
        stats: Object to track alignment statistics

    Returns:
        List of word spans with timing and alignment information
    """
    # Initialize matrices for dynamic programming
    rows = len(input_words) + 1
    cols = len(reference_words) + 1

    # Cost matrix and backtrace matrix
    cost_matrix = np.zeros((rows, cols), dtype=np.int16)
    back_matrix = np.zeros((rows, cols), dtype=np.str_)

    # Constants for path tracing
    I = "I"  # Move in input direction  # noqa: E741
    J = "J"  # Move in reference direction
    B = "B"  # Move diagonally (both)

    # Initialize matrices for base cases
    for i in range(rows):
        cost_matrix[i, 0] = i
        back_matrix[i, 0] = I

    for j in range(cols):
        cost_matrix[0, j] = j
        back_matrix[0, j] = J

    # Constants for scoring
    mismatch_penalty = 1
    gap_penalty = 1

    # Fill the matrices using dynamic programming
    for i in range(1, rows):
        for j in range(1, cols):
            # Check if words match
            if clean_text(input_words[i - 1].word) == clean_text(reference_words[j - 1]):
                match_cost = 0
            else:
                match_cost = mismatch_penalty

            # Calculate costs for different paths
            cost_both = cost_matrix[i - 1, j - 1] + match_cost
            cost_i = cost_matrix[i - 1, j] + gap_penalty
            cost_j = cost_matrix[i, j - 1] + gap_penalty

            # Choose the minimum cost path
            if cost_j <= cost_both and cost_j <= cost_i:
                back_matrix[i, j] = J
                cost_matrix[i, j] = cost_j
            elif cost_i <= cost_both and cost_i <= cost_j:
                back_matrix[i, j] = I
                cost_matrix[i, j] = cost_i
            else:
                back_matrix[i, j] = B
                cost_matrix[i, j] = cost_both

    # Backtrace to build the alignment
    i, j = rows - 1, cols - 1
    alignment = []

    NO_MATCH = -1  # Constant for no match

    while i > 0 or j > 0:
        if i > 0 and j > 0 and back_matrix[i, j] == B:
            # Both reference and input words matched
            i -= 1
            j -= 1
            alignment.insert(0, (i, j))
        elif i > 0 and back_matrix[i, j] == I:
            # Input word but no reference word (insertion)
            i -= 1
            alignment.insert(0, (i, NO_MATCH))
            stats.insertions += 1
        elif j > 0 and back_matrix[i, j] == J:
            # Reference word but no input word (deletion)
            j -= 1
            alignment.insert(0, (NO_MATCH, j))
            stats.deletions += 1
        else:
            break

    # Process the alignment to create SegmentedWordSpan objects
    result = []
    current_span = None
    in_run = False

    for idx, (input_idx, ref_idx) in enumerate(alignment):
        if input_idx != NO_MATCH and ref_idx != NO_MATCH:
            input_word = input_words[input_idx]
            ref_word = reference_words[ref_idx]

            # Check if exact match
            if clean_text(input_word.word) == clean_text(ref_word):
                # Exact match - close any existing span
                if in_run:
                    if current_span.index_end > current_span.index_start:
                        if not current_span.end:
                            current_span.end = input_word.start
                        result.append(current_span)
                    in_run = False

                # Create a new span just for this word
                result.append(
                    SegmentedWordSpan(
                        index_start=ref_idx,
                        index_end=ref_idx + 1,
                        start=input_word.start,
                        end=input_word.end,
                        flags=SegmentedWordSpan.MATCHED_INPUT | SegmentedWordSpan.MATCHED_REFERENCE | SegmentedWordSpan.EXACT,
                    )
                )
            else:
                # Inexact match
                if not in_run:
                    in_run = True
                    current_span = SegmentedWordSpan(
                        index_start=ref_idx,
                        index_end=ref_idx + 1,
                        start=input_word.start,
                        end=input_word.end,
                        flags=SegmentedWordSpan.MATCHED_INPUT
                        | SegmentedWordSpan.MATCHED_REFERENCE
                        | SegmentedWordSpan.INEXACT,
                    )
                else:
                    # Extend the current span
                    current_span.index_end = ref_idx + 1
                    current_span.end = input_word.end
                    current_span.flags |= (
                        SegmentedWordSpan.MATCHED_INPUT | SegmentedWordSpan.MATCHED_REFERENCE | SegmentedWordSpan.INEXACT
                    )

                stats.transpositions += 1

        elif input_idx != NO_MATCH:
            # Input word but no reference word (insertion)
            input_word = input_words[input_idx]

            # Start a new span for tracking timing
            if not in_run:
                in_run = True
                current_span = SegmentedWordSpan(
                    index_start=NO_MATCH,
                    index_end=NO_MATCH,
                    start=input_word.start,
                    end=input_word.end,
                    flags=SegmentedWordSpan.MATCHED_INPUT,
                )
            else:
                # Extend the current span
                current_span.end = input_word.end
                current_span.flags |= SegmentedWordSpan.MATCHED_INPUT

        elif ref_idx != NO_MATCH:
            # Reference word but no input word (deletion)
            # Start a new span if needed
            if not in_run:
                in_run = True
                prev_end = 0
                if result:
                    prev_end = result[-1].end

                current_span = SegmentedWordSpan(
                    index_start=ref_idx,
                    index_end=ref_idx + 1,
                    start=prev_end,
                    end=0,
                    flags=SegmentedWordSpan.MATCHED_REFERENCE,
                )
            else:
                # Extend the current span
                if current_span.index_start == NO_MATCH:
                    current_span.index_start = ref_idx
                current_span.index_end = ref_idx + 1
                current_span.flags |= SegmentedWordSpan.MATCHED_REFERENCE

    # Close any open span
    if in_run and current_span and current_span.index_end > current_span.index_start:
        result.append(current_span)

    return result


def detect_silence_periods(audio_data: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float]]:
    """
    Detect silence periods in audio, similar to the C++ function discriminate_silence_periods.

    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate of the audio (default: 16000 Hz)

    Returns:
        List of (start_time, end_time) tuples for silence periods
    """
    # Parameters (similar to the C++ implementation)
    window_size = int(0.05 * sample_rate)  # 50ms window
    step_size = window_size
    silence_threshold_start = -100  # dB
    silence_threshold_end = -75  # dB

    silences = []
    in_silence = False
    silence_start = 0

    for i in range(window_size, len(audio_data), step_size):
        # Calculate RMS power in dB
        window = audio_data[i - window_size : i]
        rms = np.sqrt(np.mean(window**2))
        if rms == 0:
            power_db = -np.inf
        else:
            power_db = 20 * np.log10(rms)

        # Apply hysteresis
        if not in_silence and power_db < silence_threshold_start:
            in_silence = True
            silence_start = (i - window_size) / sample_rate
        elif in_silence and power_db > silence_threshold_end:
            in_silence = False
            silence_end = i / sample_rate
            silences.append((silence_start, silence_end))

    # Close any open silence at the end
    if in_silence:
        silence_end = len(audio_data) / sample_rate
        silences.append((silence_start, silence_end))

    return silences


def transform_ayahs_to_words(ayahs: List[str]) -> List[Dict[str, Any]]:
    """
    Transform list of ayahs to word-level data with position information.

    Args:
        ayahs: List of ayah texts

    Returns:
        List of dicts with word information
    """
    all_surah_words = []
    word_location_wrt_surah = 1

    for ayah_number, ayah in enumerate(ayahs, 1):
        # Split ayah into words
        words = ayah.strip().split()

        for word_location_wrt_ayah, word in enumerate(words, 1):
            all_surah_words.append(
                {
                    "word": word,
                    "ayah_number": ayah_number,
                    "word_location_wrt_ayah": word_location_wrt_ayah,
                    "word_location_wrt_surah": word_location_wrt_surah,
                }
            )
            word_location_wrt_surah += 1

    return all_surah_words


def match_ayahs_to_transcription(
    ayahs: List[str],
    whisperx_result: Dict[str, Any],
    audio_data: Optional[np.ndarray] = None,
) -> List[AyahTimestamp]:
    """
    Match ayahs to transcription using word alignment.

    Args:
        ayahs: List of ayah texts
        whisperx_result: Result from WhisperX with word timestamps
        audio_data: Optional audio data for silence detection

    Returns:
        List of ayah timestamps
    """
    # Extract all words with timestamps from WhisperX result
    # TODO later: remove all these conditions, and assume word_segments/word/start/end keys exists (i.e., first `if`)
    all_recognized_words = []
    if "word_segments" in whisperx_result:
        for word in whisperx_result["word_segments"]:
            if "word" in word and "start" in word and "end" in word:
                all_recognized_words.append(
                    RecognizedWord(
                        word=word["word"],
                        start=word["start"],
                        end=word["end"],
                        score=word.get("score", 1.0),
                    )
                )
    else:
        # Extract from segments if word_segments not available
        for segment in whisperx_result.get("segments", []):
            if "words" in segment:
                for word in segment["words"]:
                    if "word" in word and "start" in word and "end" in word:
                        all_recognized_words.append(
                            RecognizedWord(
                                word=word["word"],
                                start=word["start"],
                                end=word["end"],
                                score=word.get("score", 1.0),
                            )
                        )

    # Transform ayahs into word-level ground truth
    all_surah_words = transform_ayahs_to_words(ayahs)
    reference_words = [word_data["word"] for word_data in all_surah_words]

    # Perform alignment
    stats = SegmentationStats()
    aligned_spans = align_words(all_recognized_words, reference_words, stats)

    # If audio data is available, use it to improve alignment
    if audio_data is not None:
        silence_periods = detect_silence_periods(audio_data)
        # Adjust spans based on silence
        for i, span in enumerate(aligned_spans):
            # Find next silence after the end of this span
            for silence_start, silence_end in silence_periods:
                if silence_start > span.start and i + 1 < len(aligned_spans):
                    if silence_start < aligned_spans[i + 1].start:
                        # Adjust the end of this span to the start of the silence
                        span.end = silence_start
                        break

    # Group spans by ayah to get ayah-level timestamps
    ayah_timestamps = []
    current_ayah = 0
    start_time = None
    end_time = None

    # Helper to find the corresponding ayah for a word index
    def find_ayah_for_word_index(word_index):
        if word_index < 0:
            return 0
        for i, word_data in enumerate(all_surah_words):
            if i == word_index:
                return word_data["ayah_number"]
        return len(ayahs)  # Beyond last ayah

    # Process each span
    for span in aligned_spans:
        if span.index_start >= 0:
            # Find ayah for this span
            span_ayah = find_ayah_for_word_index(span.index_start)

            if span_ayah != current_ayah:
                # Save previous ayah if we have one
                if current_ayah > 0 and start_time is not None and end_time is not None:
                    ayah_timestamps.append(
                        AyahTimestamp(
                            ayah_number=current_ayah,
                            text=ayahs[current_ayah - 1],
                            start_time=round(start_time, 3),
                            end_time=round(end_time, 3),
                            duration=round(end_time - start_time, 3),
                        )
                    )

                # Start new ayah
                current_ayah = span_ayah
                start_time = span.start
                end_time = span.end
            else:
                # Update end time for current ayah
                end_time = span.end

    # Add last ayah if needed
    if current_ayah > 0 and start_time is not None and end_time is not None:
        ayah_timestamps.append(
            AyahTimestamp(
                ayah_number=current_ayah,
                text=ayahs[current_ayah - 1],
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
            )
        )

    return ayah_timestamps
