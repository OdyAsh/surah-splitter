"""
Command line interface for Surah Splitter.

This module provides command-line functionality for processing and splitting Quran audio files.

Usage example (terminal):
```bash
    # e.g.1
    python ./src/surah_splitter/app/main_cli.py -au "./data/input_surahs_to_split/adel_ryyan/076 Al-Insaan.mp3" -su "76" -re "adel_rayyan" -si -ssu

    # e.g.2
    python ./src/surah_splitter/app/main_cli.py -au "./data/input_surahs_to_split/omar_bin_diaa_al_din/002_al-baqarah_partial.mp3" -su "2" -re "omar_bin_diaa_al_din" -si -ssu
```

TODO soon 1: restructure repo to be OOP + server based (e.g., trans_model and align_model are defined in the init() of the respective class, etc.)
TODO soon 2: understand DP algo. and generate docs to explain it
TODO soon 3: based on (2), adjust that DP algo. to support going backward 2 ayahs in case next word isn't an exact match (or ask ai how to adjust :])
TODO soon 4: try to run trans_model on float32 instead of int8 to see if results are better or not
TODO soon 5: add README.md
"""

import json
import sys
from pathlib import Path
from typing import List, Literal, Annotated

from cyclopts import App, Parameter, validators
from rich.console import Console

from surah_splitter_old.quran_toolkit.surah_processor import process_surah
from surah_splitter_old.utils.app_logger import logger
from surah_splitter_old.utils.paths import OUTPUTS_PATH, QURAN_METADATA_PATH

# Create cyclopts app and rich console
app = App(help="Process and split Quran audio files into individual ayahs.")
console = Console()


def get_ayahs_of_surah(surah_number: int) -> List[str]:
    """
    Get cleaned ayahs for a given surah number from the generated JSON file.

    Args:
        surah_number: Number of the surah (1-114).

    Returns:
        List of cleaned ayah texts for the surah.

    Raises:
        FileNotFoundError: If the surah_to_simple_ayahs.json file is not found.
        ValueError: If the surah number is not found in the JSON data.
    """
    # Path to the ayah data JSON file
    surah_to_simple_ayahs_path = QURAN_METADATA_PATH / "surah_to_simple_ayahs.json"
    if not surah_to_simple_ayahs_path.exists():
        raise FileNotFoundError(
            f"Ayah data file not found at {surah_to_simple_ayahs_path}. Please run simple_ayahs_extractor.py script first."
        )

    with open(surah_to_simple_ayahs_path, "r", encoding="utf-8") as f:
        surah_to_simple_ayahs_dict = json.load(f)

    surah_number_str = str(surah_number)
    if surah_number_str not in surah_to_simple_ayahs_dict:
        raise ValueError(f"Surah {surah_number} not found in the ayah data. Check {surah_to_simple_ayahs_path}")

    ayahs_dict = surah_to_simple_ayahs_dict[surah_number_str]
    ayahs = [ayahs_dict[v_id] for v_id in sorted(ayahs_dict.keys(), key=int)]
    return ayahs


@app.default()
def process(
    ######### Required args #########
    audio_file: Annotated[Path, Parameter(name=["audio_file", "-au"])],
    surah: Annotated[int, Parameter(name=["--surah", "-su"], validator=validators.Number(gte=1, lte=114))],
    reciter: Annotated[str, Parameter(name=["--reciter", "-re"])],  # Made reciter a required argument
    ######### Optional args #########
    model_name: Annotated[str, Parameter(name=["--model-name", "-mn"])] = "OdyAsh/faster-whisper-base-ar-quran",
    model_size: Annotated[Literal["tiny", "small", "medium", "large"], Parameter(name=["--model-size", "-ms"])] = "small",
    device: Annotated[Literal["cuda", "cpu"], Parameter(name=["--device", "-d"])] = None,
    output_dir: Annotated[Path, Parameter(name=["--output-dir", "-o"])] = OUTPUTS_PATH,
    save_intermediates: Annotated[bool, Parameter(name=["--save-intermediates", "-si"])] = False,
    save_incoming_surah_audio: Annotated[bool, Parameter(name=["--save-incoming-surah-audio", "-ssu"])] = False,
):
    """Process and split a Quran audio file into individual ayahs.

    Args:
        audio_file: Path to the surah audio file.
        surah: Surah number (1-114).
        reciter: Name of the reciter (e.g., "Adel Rayyan"). This is now required.
        output_dir: Base output directory for all files (e.g., data/outputs).
        model_name: Name of the WhisperX model (if not provided, `model_size` will be used).
        model_size: WhisperX model size (tiny, small, medium, large).
        device: Device to use (cuda/cpu).
        save_intermediates: Whether to save intermediate files.
        save_incoming_surah_audio: Whether to save the incoming surah audio file to the output directory (e.g., output_dir/reciter/surah_audios/001.mp3)

    Returns:
        int: 0 for success, 1 for failure.
    """
    try:
        # Get ayahs for the surah using the new function
        ayahs = get_ayahs_of_surah(surah)

        logger.debug(f"Loaded {len(ayahs)} ayahs for Surah {surah}")

        model_name = model_name or model_size

        # Process the surah
        process_surah(
            audio_path=audio_file,
            ayahs=ayahs,
            output_dir=output_dir,
            surah_number=surah,
            reciter_name=reciter,
            model_name=model_name,
            device=device,
            save_intermediates=save_intermediates,
            save_incoming_surah_audio=save_incoming_surah_audio,
        )

        logger.success("Processing completed successfully!")
        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def main():
    """Run the Surah Splitter CLI application.

    Returns exit code (0 for success, 1 for failure).
    """
    # This will run `process()` function
    return app()


if __name__ == "__main__":
    sys.exit(main())
