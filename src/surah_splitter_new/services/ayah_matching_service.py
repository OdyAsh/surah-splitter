"""
Service for matching transcribed words to reference ayahs.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from surah_splitter_new.utils.app_logger import logger
from surah_splitter_new.services.quran_metadata_service import QuranMetadataService
from surah_splitter_new.models.alignment import ReferenceWord, SegmentedWordSpan, SegmentationStats, AyahTimestamp


class AyahMatchingService:
    """Service for matching transcribed words to reference ayahs."""

    def __init__(self):
        self.quran_service = QuranMetadataService()

    def match_ayahs(
        self,
        transcription_result: Dict[str, Any],
        surah_number: int,
        output_dir: Optional[Path] = None,
        save_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Match transcribed words to reference ayahs.

        Args:
            transcription_result: Result from TranscriptionService.transcribe()
            surah_number: Surah number to match against
            output_dir: Directory to save intermediate files
            save_intermediates: Whether to save intermediate files

        Returns:
            Dict containing ayah timestamps and other alignment info
        """
        logger.info(f"Matching transcribed words to ayahs for surah {surah_number}")

        # Get reference ayahs
        reference_ayahs = self.quran_service.get_ayahs(surah_number)
        logger.debug(f"Loaded {len(reference_ayahs)} reference ayahs for surah {surah_number}")

        # Extract recognized words from transcription
        recognized_words = self._extract_recognized_words(transcription_result)
        logger.debug(f"Extracted {len(recognized_words)} recognized words from transcription")

        # Extract reference words
        reference_words = self._extract_reference_words(reference_ayahs)
        logger.debug(f"Extracted {len(reference_words)} reference words")

        # Save intermediates if requested
        if save_intermediates and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save recognized words
            with open(output_dir / "03_recognized_words.json", "w", encoding="utf-8") as f:
                json.dump(
                    [{"word": w[0], "start": w[1], "end": w[2], "score": w[3]} for w in recognized_words],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Save reference words
            with open(output_dir / "04_reference_words.json", "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "word": w.word,
                            "ayah_number": w.ayah_number,
                            "word_location_wrt_ayah": w.word_location_wrt_ayah,
                            "word_location_wrt_surah": w.word_location_wrt_surah,
                        }
                        for w in reference_words
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # Align words (dynamic programming)
        cost_matrix, back_matrix = self._compute_alignment_matrices(recognized_words, reference_words)

        if save_intermediates and output_dir:
            # Save cost and back matrices
            np.save(output_dir / "05_cost_matrix.npy", cost_matrix)
            np.save(output_dir / "06_back_matrix.npy", back_matrix)

            # Also save as JSON for human readability
            with open(output_dir / "05_cost_matrix.json", "w", encoding="utf-8") as f:
                json.dump(cost_matrix.tolist(), f)
            with open(output_dir / "06_back_matrix.json", "w", encoding="utf-8") as f:
                json.dump(back_matrix.tolist(), f)

        # Trace back to get alignment
        alignment_ij_indices = self._traceback_alignment(cost_matrix, back_matrix)

        if save_intermediates and output_dir:
            # Save alignment indices
            with open(output_dir / "07_alignment_ij_indices.json", "w", encoding="utf-8") as f:
                json.dump(alignment_ij_indices, f, ensure_ascii=False, indent=2)

        # Convert alignment indices to word spans
        word_spans = self._convert_to_word_spans(alignment_ij_indices, recognized_words, reference_words)

        if save_intermediates and output_dir:
            # Save word spans
            with open(output_dir / "08_word_spans.json", "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "reference_index_start": span.reference_index_start,
                            "reference_index_end": span.reference_index_end,
                            "reference_words_segment": span.reference_words_segment,
                            "input_words_segment": span.input_words_segment,
                            "start": span.start,
                            "end": span.end,
                            "flags": span.flags,
                        }
                        for span in word_spans
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # Extract ayah timestamps
        ayah_timestamps = self._extract_ayah_timestamps(word_spans, reference_words, reference_ayahs)

        if save_intermediates and output_dir:
            # Save ayah timestamps
            with open(output_dir / "09_ayah_timestamps.json", "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"ayah_number": ts.ayah_number, "start_time": ts.start_time, "end_time": ts.end_time, "text": ts.text}
                        for ts in ayah_timestamps
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        logger.success(f"Successfully matched {len(ayah_timestamps)} ayahs for surah {surah_number}")

        return {
            "ayah_timestamps": [
                {"ayah_number": ts.ayah_number, "start_time": ts.start_time, "end_time": ts.end_time, "text": ts.text}
                for ts in ayah_timestamps
            ],
            "word_spans": [
                {
                    "reference_index_start": span.reference_index_start,
                    "reference_index_end": span.reference_index_end,
                    "reference_words_segment": span.reference_words_segment,
                    "input_words_segment": span.input_words_segment,
                    "start": span.start,
                    "end": span.end,
                    "flags": span.flags,
                }
                for span in word_spans
            ],
        }

    def _extract_recognized_words(self, transcription_result: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
        """Extract recognized words from transcription result.

        Args:
            transcription_result: Result from TranscriptionService.transcribe()

        Returns:
            List of tuples (word, start_time, end_time, score)
        """
        word_segments = transcription_result.get("word_segments", [])

        # Clean and prepare words
        recognized_words = []
        for segment in word_segments:
            # Normalize word text
            word = self._clean_word(segment["word"])

            # Skip empty words
            if not word:
                continue

            recognized_words.append(
                (
                    word,
                    segment["start"],  # Start time
                    segment["end"],  # End time
                    segment.get("score", 1.0),  # Confidence score
                )
            )

        return recognized_words

    def _clean_word(self, word: str) -> str:
        """Clean a word by removing diacritics and special characters.

        Args:
            word: Input word

        Returns:
            Cleaned word
        """
        # Remove diacritics (tashkeel) - Arabic-specific
        word = re.sub(r"[\u064B-\u065F\u0670]", "", word)

        # Remove special characters
        word = re.sub(r"[^\w\s]", "", word)

        # Trim whitespace
        word = word.strip()

        return word

    def _extract_reference_words(self, reference_ayahs: List[str]) -> List[ReferenceWord]:
        """Extract words from reference ayahs.

        Args:
            reference_ayahs: List of ayah texts

        Returns:
            List of ReferenceWord objects
        """
        reference_words = []

        for ayah_idx, ayah_text in enumerate(reference_ayahs, start=1):
            # Split into words and clean
            words = [self._clean_word(w) for w in ayah_text.split()]

            # Filter out empty words
            words = [w for w in words if w]

            # Create ReferenceWord objects
            for word_idx, word in enumerate(words, start=1):
                reference_words.append(
                    ReferenceWord(
                        word=word,
                        ayah_number=ayah_idx,
                        word_location_wrt_ayah=word_idx,
                        word_location_wrt_surah=len(reference_words) + 1,
                    )
                )

        return reference_words

    def _compute_alignment_matrices(
        self, recognized_words: List[Tuple[str, float, float, float]], reference_words: List[ReferenceWord]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the cost and back matrices for alignment using dynamic programming.

        Args:
            recognized_words: List of recognized words
            reference_words: List of reference words

        Returns:
            Tuple of (cost_matrix, back_matrix)
        """
        # Define matrix dimensions
        n = len(recognized_words) + 1  # +1 for the empty string case
        m = len(reference_words) + 1  # +1 for the empty string case

        # Initialize matrices
        cost_matrix = np.zeros((n, m), dtype=float)
        back_matrix = np.zeros((n, m), dtype=int)

        # Constants for costs
        INSERTION_COST = 1.0
        DELETION_COST = 1.0
        EXACT_MATCH_COST = 0.0
        INEXACT_MATCH_COST = 0.5

        # Initialize first row and column
        for i in range(n):
            cost_matrix[i, 0] = i * INSERTION_COST
            back_matrix[i, 0] = 1  # Insertion (1)

        for j in range(m):
            cost_matrix[0, j] = j * DELETION_COST
            back_matrix[0, j] = 2  # Deletion (2)

        back_matrix[0, 0] = 0  # No operation for (0,0)

        # Fill the matrices
        for i in range(1, n):
            for j in range(1, m):
                rec_word = recognized_words[i - 1][0]
                ref_word = reference_words[j - 1].word

                # Calculate costs
                if rec_word == ref_word:
                    # Exact match
                    substitution_cost = EXACT_MATCH_COST
                    operation_type = 3  # Match (3)
                else:
                    # Inexact match - we could refine this with edit distance/similarity
                    substitution_cost = INEXACT_MATCH_COST
                    operation_type = 4  # Substitution (4)

                # Find minimum cost operation
                insertion = cost_matrix[i - 1, j] + INSERTION_COST
                deletion = cost_matrix[i, j - 1] + DELETION_COST
                substitution = cost_matrix[i - 1, j - 1] + substitution_cost

                # Choose the minimum cost operation
                if insertion <= deletion and insertion <= substitution:
                    cost_matrix[i, j] = insertion
                    back_matrix[i, j] = 1  # Insertion
                elif deletion <= insertion and deletion <= substitution:
                    cost_matrix[i, j] = deletion
                    back_matrix[i, j] = 2  # Deletion
                else:
                    cost_matrix[i, j] = substitution
                    back_matrix[i, j] = operation_type  # Match or Substitution

        return cost_matrix, back_matrix

    def _traceback_alignment(self, cost_matrix: np.ndarray, back_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Trace back through the alignment matrices to get the alignment.

        Args:
            cost_matrix: The cost matrix
            back_matrix: The back matrix with operation codes

        Returns:
            List of (i,j) indices representing the alignment
        """
        i = cost_matrix.shape[0] - 1
        j = cost_matrix.shape[1] - 1

        alignment = []

        # Trace back from bottom-right to top-left
        while i > 0 or j > 0:
            alignment.append((i, j))

            operation = back_matrix[i, j]

            if operation == 1:  # Insertion
                i -= 1
            elif operation == 2:  # Deletion
                j -= 1
            else:  # Match or Substitution
                i -= 1
                j -= 1

        # Reverse to get the alignment in the correct order
        alignment.reverse()

        return alignment

    def _convert_to_word_spans(
        self,
        alignment_ij_indices: List[Tuple[int, int]],
        recognized_words: List[Tuple[str, float, float, float]],
        reference_words: List[ReferenceWord],
    ) -> List[SegmentedWordSpan]:
        """Convert alignment indices to word spans.

        Args:
            alignment_ij_indices: List of (i,j) alignment indices
            recognized_words: List of recognized words
            reference_words: List of reference words

        Returns:
            List of SegmentedWordSpan objects
        """
        word_spans = []

        # Process alignment to create word spans
        prev_j = 0
        span_start_j = 0
        span_start_time = 0

        for idx, (i, j) in enumerate(alignment_ij_indices):
            # Skip the (0,0) case
            if i == 0 and j == 0:
                continue

            # Check if there's a gap in reference indices (j)
            if j > 0 and prev_j > 0 and j - prev_j > 1:
                # We have a gap, so create a span for the previous segment
                if span_start_j < prev_j:
                    # Create segment from span_start_j to prev_j
                    start_time = span_start_time
                    # Find the end time from the last recognized word in this span
                    end_time = recognized_words[i - 1][2] if i > 0 else 0

                    # Get the reference words segment
                    ref_segment = " ".join([rw.word for rw in reference_words[span_start_j:prev_j]])

                    # Get the recognized words segment
                    input_segment = " ".join([rw[0] for rw in recognized_words[:i]])

                    word_spans.append(
                        SegmentedWordSpan(
                            reference_index_start=span_start_j,
                            reference_index_end=prev_j,
                            reference_words_segment=ref_segment,
                            input_words_segment=input_segment,
                            start=start_time,
                            end=end_time,
                            flags=SegmentedWordSpan.MATCHED_REFERENCE,
                        )
                    )

                    # Start a new span
                    span_start_j = prev_j
                    span_start_time = end_time

            prev_j = j

        # Add the final span
        if span_start_j < len(reference_words):
            # Get the last alignment indices
            last_i, last_j = alignment_ij_indices[-1]

            # Get the reference words segment
            ref_segment = " ".join([rw.word for rw in reference_words[span_start_j:]])

            # Get the recognized words segment
            input_segment = " ".join([rw[0] for rw in recognized_words[:last_i]])

            # Find the end time from the last recognized word
            end_time = recognized_words[last_i - 1][2] if last_i > 0 else 0

            word_spans.append(
                SegmentedWordSpan(
                    reference_index_start=span_start_j,
                    reference_index_end=len(reference_words),
                    reference_words_segment=ref_segment,
                    input_words_segment=input_segment,
                    start=span_start_time,
                    end=end_time,
                    flags=SegmentedWordSpan.MATCHED_REFERENCE,
                )
            )

        return word_spans

    def _extract_ayah_timestamps(
        self, word_spans: List[SegmentedWordSpan], reference_words: List[ReferenceWord], reference_ayahs: List[str]
    ) -> List[AyahTimestamp]:
        """Extract ayah timestamps from word spans.

        Args:
            word_spans: List of word spans
            reference_words: List of reference words
            reference_ayahs: List of reference ayah texts

        Returns:
            List of AyahTimestamp objects
        """
        ayah_timestamps = []

        # Group reference words by ayah number
        ayah_to_ref_word_indices = {}
        for i, ref_word in enumerate(reference_words):
            if ref_word.ayah_number not in ayah_to_ref_word_indices:
                ayah_to_ref_word_indices[ref_word.ayah_number] = []
            ayah_to_ref_word_indices[ref_word.ayah_number].append(i)

        # For each ayah, find the span that contains its words
        for ayah_number in sorted(ayah_to_ref_word_indices.keys()):
            ref_word_indices = ayah_to_ref_word_indices[ayah_number]
            ayah_start_idx = min(ref_word_indices)
            ayah_end_idx = max(ref_word_indices) + 1  # +1 because end is exclusive

            # Find spans that overlap with this ayah
            ayah_spans = []
            for span in word_spans:
                # Check if this span overlaps with the ayah
                if span.reference_index_end > ayah_start_idx and span.reference_index_start < ayah_end_idx:
                    ayah_spans.append(span)

            if ayah_spans:
                # Get start and end times for this ayah
                start_time = min(span.start for span in ayah_spans)
                end_time = max(span.end for span in ayah_spans)

                # Get the ayah text
                ayah_text = reference_ayahs[ayah_number - 1]

                ayah_timestamps.append(
                    AyahTimestamp(ayah_number=ayah_number, start_time=start_time, end_time=end_time, text=ayah_text)
                )

        return ayah_timestamps
