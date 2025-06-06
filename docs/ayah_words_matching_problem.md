This file is an attempt for me to brainstorm and document the different scenarios that can occur when matching ayahs to transcriptions. It is not meant to be exhaustive, but rather a starting point for further discussion and refinement.

TOC:

- [Structure of the Given Inputs](#structure-of-the-given-inputs)
  - [Structure of the Transcriptions](#structure-of-the-transcriptions)
  - [Structure of the Ayahs](#structure-of-the-ayahs)
- [Structure of the Required Outputs](#structure-of-the-required-outputs)
- [The Matching Problem Statement](#the-matching-problem-statement)
- [Matching Scenarios](#matching-scenarios)
  - [Case 1 - Perfect Match](#case-1---perfect-match)
  - [Case 2 - Filler and Wrong Words](#case-2---filler-and-wrong-words)
  - [Case 3 - Repeated Words](#case-3---repeated-words)
  - [Case 4 - Audio Starts From a Middle Ayah](#case-4---audio-starts-from-a-middle-ayah)

---

# Structure of the Given Inputs

## Structure of the Transcriptions

The structure of the transcription segments returned by `transcribed_text = whisperx.load_model(...).transcribe(...)` is as follows:

```json
{
    "segments": [
        {
            "text": " أعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم",
            "start": 0.031,
            "end": 5.633
        },
        // {
        //     and so on for each segment...
        // }
    ]
}
```

Let's refer to the above format as `transcribed_segments`.

However, we're interested in the structure of whisperx's alignment results returned by `aligned_text = whisperx.align(result["segments"], ...)`, and is as follows:

```json
{
    "segments": [
        {
            "text": " أعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم",
            "start": 0.031,
            "end": 5.633,
            "words": [
                {
                    "word": "أعوذ",
                    "start": 0.031,
                    "end": 0.631,
                    "score": 0.898
                },
                {
                    "word": "بالله",
                    "start": 0.474,
                    "end": 0.938,
                    "score": 0.839
                },
                // and so on for each word...
            ]
        },
        // and so on for each segment...
    ],
    "word_segments": [
        {
            "word": "أعوذ",
            "start": 0.031,
            "end": 0.631,
            "score": 0.898
        },
        {
            "word": "بالله",
            "start": 0.474,
            "end": 0.938,
            "score": 0.839
        },
        // and so on for each word (i.e., all words across all segments)...
    ]
}
```

Let's refer to the above format as `aligned_words`.

## Structure of the Ayahs

The structure of the ayahs is as follows:

```json
[
    "بسم الله الرحمن الرحيم",
    "الحمد لله رب العالمين",
    "الرحمن الرحيم",
    "مالك يوم الدين",
    "إياك نعبد وإياك نستعين",
    "اهدنا الصراط المستقيم",
    "صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين",
]
```

Let's refer to the above format as `ayahs_ground_truth`.

# Structure of the Required Outputs

Let's 

```json
    {
        "ayah_number": 1,
        "text": "بسم الله الرحمن الرحيم",
        "start_time": 3.437,
        "end_time": 5.653,
        "duration": 2.216
    },
    // and so on for each ayah...
```

Let's refer to the above format as `transcribed_ayahs`.

# The Matching Problem Statement


We need to match the `ayahs_ground_truth` to the `aligned_words` and produce a list of `transcribed_ayahs` that contains the following information for each ayah:
- `ayah_number`: The index of the ayah in the `ayahs_ground_truth` (starting from 1).
- `text`: The text of the ayah.
- `start_time`: The start time of the ayah in the audio.
- `end_time`: The end time of the ayah in the audio.
- `duration`: The duration of the ayah in the audio.


# Matching Scenarios

Cases 1 through 4 are found in the [assets](../docs/assets/) folder and are named the following:

## Case 1 - Perfect Match



## Case 2 - Filler and Wrong Words



## Case 3 - Repeated Words



## Case 4 - Audio Starts From a Middle Ayah





