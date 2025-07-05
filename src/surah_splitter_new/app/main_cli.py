"""
Command line interface for Surah Splitter.

This module provides command-line functionality for processing and splitting Quran audio files.

Usage example (terminal):
```bash
    # Full pipeline processing
    python main_cli.py -au "./data/input_surahs_to_split/adel_ryyan/076 Al-Insaan.mp3" -su 76 -re "adel_rayyan" -si -ssu
    or:
    python ./src/surah_splitter_new/app/main_cli.py -au "./data/input_surahs_to_split/omar_bin_diaa_al_din/002_al-baqarah_partial.mp3" -su 2 -re "omar_bin_diaa_al_din" -si -ssu -ay 155,156,157

    # Just transcribe
    python main_cli.py transcribe_audio -au "./data/input_surahs_to_split/adel_ryyan/076 Al-Insaan.mp3" -o "./data/outputs/transcription.json"

    # Match pre-transcribed audio
    python main_cli.py match_ayahs -tf "./data/outputs/transcription.json" -su 76 -o "./data/outputs/timestamps.json"

    # Segment with existing timestamps
    python main_cli.py segment_audio -au "./data/input_surahs_to_split/adel_ryyan/076 Al-Insaan.mp3" -tf "./data/outputs/timestamps.json" -su 76 -re "adel_rayyan"
```
"""

from surah_splitter_new.utils.app_logger import logger
import sys
import json
from pathlib import Path
from typing import Literal, Annotated, Optional

from cyclopts import App, Parameter, validators
from rich.console import Console

from surah_splitter_new.services.transcription_service import TranscriptionService
from surah_splitter_new.services.ayah_matching_service import AyahMatchingService
from surah_splitter_new.services.segmentation_service import SegmentationService
from surah_splitter_new.services.pipeline_service import PipelineService
from surah_splitter_new.utils.paths import OUTPUTS_PATH
from surah_splitter_new.utils.file_utils import save_intermediate_json

# Create cyclopts app and rich console
app = App(help="Process and split Quran audio files into individual ayahs.")
console = Console()


@app.default()
def process_surah(
    ######### Required args #########
    audio_file: Annotated[Path, Parameter(name=["audio_file", "-au"])],
    surah: Annotated[int, Parameter(name=["--surah", "-su"], validator=validators.Number(gte=1, lte=114))],
    reciter: Annotated[str, Parameter(name=["--reciter", "-re"])],
    ######### Optional args #########
    ayahs: Annotated[Optional[str], Parameter(name=["--ayahs", "-ay"])] = None,
    model_name: Annotated[str, Parameter(name=["--model-name", "-mn"])] = "OdyAsh/faster-whisper-base-ar-quran",
    model_size: Annotated[Literal["tiny", "small", "medium", "large"], Parameter(name=["--model-size", "-ms"])] = "small",
    device: Annotated[Optional[Literal["cuda", "cpu"]], Parameter(name=["--device", "-d"])] = None,
    compute_type: Annotated[Optional[str], Parameter(name=["--compute-type", "-ct"])] = None,
    output_dir: Annotated[Path, Parameter(name=["--output-dir", "-o"])] = OUTPUTS_PATH,
    save_intermediates: Annotated[bool, Parameter(name=["--save-intermediates", "-si"])] = False,
    save_incoming_surah_audio: Annotated[bool, Parameter(name=["--save-incoming-surah-audio", "-ssu"])] = False,
):
    """Process and split a Quran audio file into individual ayahs.

    This command runs the complete pipeline of transcription, matching, and segmentation.
    """
    try:
        # Validate ayah number(s) if provided
        ayahs = [int(x.strip()) for x in ayahs.split(",")] if ayahs else None
        if ayahs:
            for ayah in ayahs:
                # TODO later: create a surah_num->max_ayah_num mapping (json file) to validate ayah numbers
                if ayah < 1:
                    raise ValueError(f"Invalid ayah number: {ayah}. Must be a positive integer.")

        # Create pipeline service
        pipeline_service = PipelineService()

        # Process the surah
        model_name = model_name or model_size
        result = pipeline_service.process_surah(
            audio_path=audio_file,
            surah_number=surah,
            reciter_name=reciter,
            output_dir=output_dir,
            ayah_numbers=ayahs,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            save_intermediates=save_intermediates,
            save_incoming_surah_audio=save_incoming_surah_audio,
        )

        logger.success(
            f"Processing completed successfully! Generated {len(result['ayah_matching']['ayah_timestamps'])} ayah segments."
        )
        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


@app.command(name="transcribe_audio")
def transcribe_audio(
    audio_file: Annotated[Path, Parameter(name=["audio_file", "-au"])],
    output_file: Annotated[Optional[Path], Parameter(name=["--output-file", "-o"])] = None,
    model_name: Annotated[str, Parameter(name=["--model-name", "-mn"])] = "OdyAsh/faster-whisper-base-ar-quran",
    device: Annotated[Optional[Literal["cuda", "cpu"]], Parameter(name=["--device", "-d"])] = None,
    compute_type: Annotated[Optional[str], Parameter(name=["--compute-type", "-ct"])] = None,
):
    """Transcribe an audio file and get word-level timestamps.

    This command only performs the transcription step without matching or segmentation.
    """
    try:
        # Create transcription service
        transcription_service = TranscriptionService()
        transcription_service.initialize(model_name, device, compute_type)

        # Transcribe audio
        result = transcription_service.transcribe(audio_file, output_file.parent if output_file else None)

        # Save result if output specified
        if output_file:
            save_intermediate_json(
                data=result,
                output_dir=output_file.parent,
                filename=output_file.name,
                log_message=f"Transcription result saved to {output_file.name}",
            )
        else:
            # Print to console
            console.print_json(json.dumps(result, ensure_ascii=False))

        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


@app.command(name="match_ayahs")
def match_ayahs(
    transcription_file: Annotated[Path, Parameter(name=["transcription_file", "-tf"])],
    surah: Annotated[int, Parameter(name=["--surah", "-su"], validator=validators.Number(gte=1, lte=114))],
    ayahs: Annotated[Optional[str], Parameter(name=["--ayahs", "-ay"])] = None,
    output_file: Annotated[Optional[Path], Parameter(name=["--output-file", "-o"])] = None,
):
    """Match transcribed words to reference ayahs.

    This command takes a transcription result file and matches it to reference ayahs.
    """
    try:
        # Validate ayah number(s) if provided
        ayahs = [int(x.strip()) for x in ayahs.split(",")] if ayahs else None
        if ayahs:
            for ayah in ayahs:
                # TODO later: create a surah_num->max_ayah_num mapping (json file) to validate ayah numbers
                if ayah < 1:
                    raise ValueError(f"Invalid ayah number: {ayah}. Must be a positive integer.")

        # Load transcription file
        with open(transcription_file, "r", encoding="utf-8") as f:
            transcription_result = json.load(f)

        # Create ayah matching service
        matching_service = AyahMatchingService()

        # Match ayahs
        result = matching_service.match_ayahs(
            transcription_result,
            surah,
            ayahs,
            output_file.parent if output_file else None,
            save_intermediates=bool(output_file),
        )

        # Save result if output specified
        if output_file:
            save_intermediate_json(
                data=result,
                output_dir=output_file.parent,
                filename=output_file.name,
                log_message=f"Matching results saved to {output_file.name}",
            )
        else:
            # Print to console
            console.print_json(json.dumps(result, ensure_ascii=False))

        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


@app.command(name="segment_audio")
def segment_audio(
    audio_file: Annotated[Path, Parameter(name=["audio_file", "-au"])],
    timestamps_file: Annotated[Path, Parameter(name=["timestamps_file", "-tf"])],
    surah: Annotated[int, Parameter(name=["--surah", "-su"], validator=validators.Number(gte=1, lte=114))],
    reciter: Annotated[str, Parameter(name=["--reciter", "-re"])],
    output_dir: Annotated[Path, Parameter(name=["--output-dir", "-o"])] = OUTPUTS_PATH,
    save_incoming_surah_audio: Annotated[bool, Parameter(name=["--save-incoming-surah-audio", "-ssu"])] = False,
):
    """Segment audio file based on ayah timestamps.

    This command takes an audio file and timestamps and creates individual ayah segments.
    """
    try:
        # Load timestamps file
        with open(timestamps_file, "r", encoding="utf-8") as f:
            timestamps_data = json.load(f)

        # Get ayah timestamps from the file
        ayah_timestamps = timestamps_data["ayah_timestamps"]

        # Create segmentation service
        segmentation_service = SegmentationService()

        # Segment audio
        result = segmentation_service.split_audio(
            audio_file, ayah_timestamps, surah, reciter, output_dir, save_incoming_surah_audio
        )

        # Print results
        logger.info(f"Created {len(result)} ayah audio files")
        for ayah_num, path in result.items():
            logger.info(f"Ayah {ayah_num}: {path}")

        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def main():
    """Run the Surah Splitter CLI application."""
    # Run the app
    return app()


if __name__ == "__main__":
    sys.exit(main())
