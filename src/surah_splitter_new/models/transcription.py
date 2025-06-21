"""
Data models for transcription results.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RecognizedWordSegment:
    """Word segment with timing information."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    score: Optional[float] = None

    @classmethod
    def from_whisperx(cls, word_data: Dict[str, Any]) -> "RecognizedWordSegment":
        """Create from WhisperX word segment data."""
        return cls(text=word_data["word"], start=word_data["start"], end=word_data["end"], score=word_data.get("score"))


@dataclass
class Transcription:
    """Audio transcription with word-level timing."""

    text: str
    word_segments: List[RecognizedWordSegment]
    language: str = "ar"

    @classmethod
    def from_whisperx(cls, result: Dict[str, Any]) -> "Transcription":
        """Create from WhisperX result."""
        return cls(
            text=result.get("text", ""),
            word_segments=[RecognizedWordSegment.from_whisperx(w) for w in result.get("word_segments", [])],
            language=result.get("language", "ar"),
        )
