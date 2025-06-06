"""
Command line interface for Surah Splitter.

This module provides command-line functionality for processing and splitting Quran audio files.

Usage example (terminal):
```bash
    python ./src/surah_splitter/app/cli.py -a "./data/input_surahs_to_split/adel_ryyan/076 Al-Insaan.mp3" -s "76" -r "adel_rayyan"
```
"""

import json
import sys
from pathlib import Path
from typing import List, Literal, Optional, Annotated

from cyclopts import App, Parameter, validators
from rich.console import Console

from surah_splitter.quran_toolkit.surah_processor import process_surah
from surah_splitter.utils.app_logger import logger
from surah_splitter.utils.paths import OUTPUTS_PATH

# Create cyclopts app and rich console
app = App(help="Process and split Quran audio files into individual ayahs.")
console = Console()


def read_from_json(file_path: str) -> List[str]:
    """
    Load data from a JSON file.

    Parameters
    ----------
    file_path: str
        Path to the JSON file containing ayahs.

    Returns
    -------
    List[str]
        List of ayah texts extracted from the JSON file.

    Raises
    ------
    ValueError
        If the JSON format couldn't be recognized.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "ayahs" in data:
        return data["ayahs"]
    elif isinstance(data, dict):
        # Try to extract from the structure
        ayahs = []
        for key in sorted([int(k) for k in data.keys() if k.isdigit()]):
            ayah_text = data[str(key)]
            if isinstance(ayah_text, str):
                ayahs.append(ayah_text)
            elif isinstance(ayah_text, dict) and "text" in ayah_text:
                ayahs.append(ayah_text["text"])
        return ayahs

    raise ValueError("Could not extract ayahs from JSON file. Format not recognized.")


def read_from_txt(file_path: str) -> List[str]:
    """
    Load ayahs from a text file with one ayah per line.

    Parameters
    ----------
    file_path: str
        Path to the text file containing ayahs, one per line.

    Returns
    -------
    List[str]
        List of ayah texts extracted from the text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_ayahs_for_surah(surah_number: int, quran_data_path: Optional[str] = None) -> List[str]:
    """
    Get ayahs for a given surah number from built-in data or external source.

    Args:
        surah_number: Number of the surah (1-114)
        quran_data_path: Optional path to a Quran data file (JSON or TXT)

    Returns:
        List of ayah texts for the surah
    """
    if quran_data_path:
        path = Path(quran_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Quran data file not found: {quran_data_path}")

        if path.suffix.lower() == ".json":
            all_data = json.loads(path.read_text(encoding="utf-8"))

            # Try to extract the specific surah
            if isinstance(all_data, dict):
                if str(surah_number) in all_data:
                    surah_data = all_data[str(surah_number)]
                    if isinstance(surah_data, list):
                        return surah_data
                    elif isinstance(surah_data, dict) and "ayahs" in surah_data:
                        return [ayah["text"] for ayah in surah_data["ayahs"]]
                    elif isinstance(surah_data, dict):
                        # Try to extract from the structure
                        ayahs = []
                        for key in sorted([int(k) for k in surah_data.keys() if k.isdigit()]):
                            ayah_text = surah_data[str(key)]
                            if isinstance(ayah_text, str):
                                ayahs.append(ayah_text)
                            elif isinstance(ayah_text, dict) and "text" in ayah_text:
                                ayahs.append(ayah_text["text"])
                        return ayahs

            raise ValueError(f"Could not extract ayahs for surah {surah_number} from JSON file")

        elif path.suffix.lower() == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            # Try to extract the specific surah based on simple formatting (surah number followed by ayahs)
            surah_start = -1
            next_surah_start = -1

            for i, line in enumerate(all_lines):
                if line.strip().startswith(f"{surah_number}.") or line.strip() == str(surah_number):
                    surah_start = i
                elif surah_start >= 0 and (
                    line.strip().startswith(f"{surah_number+1}.") or line.strip() == str(surah_number + 1)
                ):
                    next_surah_start = i
                    break

            if surah_start >= 0:
                if next_surah_start > surah_start:
                    return [line.strip() for line in all_lines[surah_start + 1 : next_surah_start] if line.strip()]
                else:
                    return [line.strip() for line in all_lines[surah_start + 1 :] if line.strip()]

            raise ValueError(f"Could not find surah {surah_number} in the text file")

    # Fallback for a few built-in surahs
    if surah_number == 1:  # Al-Fatiha
        return [
            "بسم الله الرحمن الرحيم",
            "الحمد لله رب العالمين",
            "الرحمن الرحيم",
            "مالك يوم الدين",
            "إياك نعبد وإياك نستعين",
            "اهدنا الصراط المستقيم",
            "صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين",
        ]
    elif surah_number == 76:  # Al-Insaan
        return [
            "هل أتى على الإنسان حين من الدهر لم يكن شيئا مذكورا",
            "إنا خلقنا الإنسان من نطفة أمشاج نبتليه فجعلناه سميعا بصيرا",
            "إنا هديناه السبيل إما شاكرا وإما كفورا",
            "إنا أعتدنا للكافرين سلاسل وأغلالا وسعيرا",
            "إن الأبرار يشربون من كأس كان مزاجها كافورا",
            "عينًا يشرب بها عباد الله يفجرونها تفجيرا",
            "يوفون بالنذر ويخافون يوما كان شره مستطيرا",
            "ويطعمون الطعام على حبه مسكينا ويتيما وأسيرا",
            "إنما نطعمكم لوجه الله لا نريد منكم جزاء ولا شكورا",
            "إنا نخاف من ربنا يوما عبوسا قمطريرا",
            "فوقاهم الله شر ذلك اليوم ولقاهم نضرة وسرورا",
            "وجزاهم بما صبروا جنة وحريرا",
            "متكئين فيها على الأرائك لا يرون فيها شمسا ولا زمهريرا",
            "ودانية عليهم ظلالها وذللت قطوفها تذليلا",
            "ويطاف عليهم بآنية من فضة وأكواب كانت قواريرا",
            "قوارير من فضة قدروها تقديرا",
            "ويسقون فيها كأسا كان مزاجها زنجبيلا",
            "عينًا فيها تسمى سلسبيلا",
            "ويطوف عليهم ولدان مخلدون إذا رأيتهم حسبتهم لؤلؤا منثورا",
            "وإذا رأيت ثم رأيت نعيما وملكا كبيرا",
            "عاليهم ثياب سندس خضر وإستبرق وحلوا أساور من فضة وسقاهم ربهم شرابا طهورا",
            "إن هذا كان لكم جزاء وكان سعيكم مشكورا",
            "إنا نحن نزلنا عليك القرآن تنزيلا",
            "فاصبر لحكم ربك ولا تطع منهم آثما أو كفورا",
            "واذكر اسم ربك بكرة وأصيلا",
            "ومن الليل فاسجد له وسبحه ليلا طويلا",
            "إن هؤلاء يحبون العاجلة ويذرون وراءهم يوما ثقيلا",
            "نحن خلقناهم وشددنا أسرهم وإذا شئنا بدلنا أمثالهم تبديلا",
            "إن هذه تذكرة فمن شاء اتخذ إلى ربه سبيلا",
            "وما تشاءون إلا أن يشاء الله إن الله كان عليما حكيما",
            "يدخل من يشاء في رحمته والظالمين أعد لهم عذابا أليما",
        ]
    elif surah_number == 112:  # Al-Ikhlas
        return [
            "بسم الله الرحمن الرحيم",
            "قل هو الله أحد",
            "الله الصمد",
            "لم يلد ولم يولد",
            "ولم يكن له كفوا أحد",
        ]
    elif surah_number == 113:  # Al-Falaq
        return [
            "بسم الله الرحمن الرحيم",
            "قل أعوذ برب الفلق",
            "من شر ما خلق",
            "ومن شر غاسق إذا وقب",
            "ومن شر النفاثات في العقد",
            "ومن شر حاسد إذا حسد",
        ]
    elif surah_number == 114:  # An-Nas
        return [
            "بسم الله الرحمن الرحيم",
            "قل أعوذ برب الناس",
            "ملك الناس",
            "إله الناس",
            "من شر الوسواس الخناس",
            "الذي يوسوس في صدور الناس",
            "من الجنة و الناس",
        ]

    raise ValueError(f"No built-in data for surah {surah_number}. Please provide a Quran data file.")


@app.default()
def process(
    ######### Required args #########
    audio_file: Annotated[Path, Parameter(name=["audio_file", "-a"])],
    surah: Annotated[int, Parameter(name=["--surah", "-s"], validator=validators.Number(gte=1, lte=114))],
    ######### Optional args #########
    reciter: Annotated[str, Parameter(name=["--reciter", "-r"])] = "",
    model_name: Annotated[str, Parameter(name=["--model-name", "-mn"])] = "",
    model_size: Annotated[Literal["tiny", "small", "medium", "large"], Parameter(name=["--model-size", "-ms"])] = "small",
    device: Annotated[Literal["cuda", "cpu"], Parameter(name=["--device", "-d"])] = None,
    quran_data: Annotated[Optional[Path], Parameter(name=["--quran-data", "-q"])] = None,
    ayahs_file: Annotated[Optional[Path], Parameter(name=["--ayahs-file", "-f"])] = None,
    output_dir: Annotated[Path, Parameter(name=["--output-dir", "-o"])] = OUTPUTS_PATH,
    save_intermediates: Annotated[bool, Parameter(name=["--save-intermediates", "-i"])] = True,
):
    """Process and split a Quran audio file into individual ayahs.

    Args:
        audio_file: Path to the surah audio file.
        surah: Surah number (1-114).
        output_dir: Output directory for all files.
        reciter: Name of the reciter.
        model_name: Name of the WhisperX model (if not provided, `model_size` will be used).
        model_size: WhisperX model size (tiny, small, medium, large).
        device: Device to use (cuda/cpu).
        quran_data: Path to Quran data file (JSON or TXT).
        ayahs_file: Path to file with ayahs for this surah only (JSON or TXT).
        save_intermediates: Whether to save intermediate files.

    Returns:
        int: 0 for success, 1 for failure.
    """
    try:
        # Get ayahs for the surah
        if ayahs_file:
            path = Path(ayahs_file)
            if path.suffix.lower() == ".json":
                ayahs = read_from_json(str(ayahs_file))
            else:
                ayahs = read_from_txt(str(ayahs_file))
        else:
            ayahs = get_ayahs_for_surah(surah, quran_data)

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
        )

        logger.info("Processing completed successfully!")
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
