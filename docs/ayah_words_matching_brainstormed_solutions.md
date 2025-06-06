This file is an attempt for me to brainstorm and document the different solutions for the scenarios of the matching problem statement discussed in [the relevant docs](./ayah_words_matching_problem.md). It is not meant to be exhaustive, but rather a starting point for further discussion and refinement.

TOC:

- [Structure of the Required Outputs](#structure-of-the-required-outputs)
- [Brainstorming Matching Algorithms - INCOMPLETE](#brainstorming-matching-algorithms---incomplete)
  - [Algorithm 1 - Linear Matching](#algorithm-1---linear-matching)
  - [Assumptions](#assumptions)
  - [Transformed Inputs](#transformed-inputs)
  - [INCOMPLETE LOGIC - Algorithm Steps - Looping Over `all_surah_words_ground_truth`](#incomplete-logic---algorithm-steps---looping-over-all_surah_words_ground_truth)
  - [Algorithm Steps - Looping Over `all_surah_words_aligned`](#algorithm-steps---looping-over-all_surah_words_aligned)

---

# Structure of the Required Outputs

For this algorithm, we will first get this output structure, and then transform it to the final output structure as described in [the problem statement docs](./ayah_words_matching_problem.md#structure-of-the-required-outputs).

```json
[
    {
        "word": "أعوذ",
        "start": 0.031,
        "end": 0.631,
        "score": 0.898,
        "ayah_number": 1,
        "word_location_wrt_ayah": 1,
        "word_location_wrt_surah": 1
    },
    {
        "word": "بالله",
        "start": 0.474,
        "end": 0.938,
        "score": 0.839,
        "ayah_number": 1,
        "word_location_wrt_ayah": 2,
        "word_location_wrt_surah": 2
    },
    // and so on for each word (i.e., all words across all segments)...
]
```

# Brainstorming Matching Algorithms - INCOMPLETE

NOTE: All of the below section will be changed soon once we're done with the QuranAlign (C++ to Python) porting ISA.

## Algorithm 1 - Linear Matching

## Assumptions

1. The audio contains the first ayah (even if it's preceeded by filler words).

## Transformed Inputs

Transform `ayahs_ground_truth`, mentioned in [the problem statement docs](./ayah_words_matching_problem.md#structure-of-the-ayahs) to the following format:

```json
[
    {
        "word": "بسم",
        "ayah_number": 1,
        "word_location_wrt_ayah": 1,
        "word_location_wrt_surah": 1
    },
    {
        "word": "الله",
        "ayah_number": 1,
        "word_location_wrt_ayah": 2,
        "word_location_wrt_surah": 2
    },
    
    // ...
    
    {
        "word": "الحمد",
        "ayah_number": 2,
        "word_location_wrt_ayah": 1,
        "word_location_wrt_surah": 5
    }
    // and so on for each word in all of the ayahs of the current surah...
]
```

Let's refer to the above format as `all_surah_words_ground_truth`.


We'll also need `aligned_words` as described in [the problem statement docs](./ayah_words_matching_problem.md#structure-of-the-given-inputs), but just use the list value of the `word_segments` key. I.e.:

```json
[
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
```

Let's refer to the above format as `all_surah_words_aligned`.

## INCOMPLETE LOGIC - Algorithm Steps - Looping Over `all_surah_words_ground_truth`

1. For each word in `all_surah_words_ground_truth`:
   1. Create a new list of dicts called `all_surah_words_algo_result` which starts as an empty list.
   2. Find the first corresponding word in the `all_surah_words_aligned[last_match_idx:]` search space based on the word text.
      1. Where "corresponding word" means fuzzy matching with 90% confidence.
      2. Where `last_match_idx` is the ...
      3. MENTAL DEADLOCK: the algo. proposed here due to the following scenario:
         1. if for example the 1st word of an ayah isn't correctly transcribed, but a later ayah contains that same word, then the algo. will match the first word of the ayah to the later ayah's word, which is not what we want.
         2. Don't know how to proceed, so will ditch this algorithm for now...
2. If a match is found, create the following dict:
   - `word`: the word text from `all_surah_words_ground_truth`,
   - `start`: the start time of the matched word from `all_surah_words_aligned`,
   - `end`: the end time of the matched word from `all_surah_words_aligned`,
   - `score`: the confidence score of the fuzzy match between the 2 words.
   - `ayah_number`: the ayah number from `all_surah_words_ground_truth`,
   - `word_location_wrt_ayah`: the word's location within the ayah from `all_surah_words_ground_truth`,
   - `word_location_wrt_surah`: the word's location within the surah from `all_surah_words_ground_truth`.
3. Else, if no match is found, create the following dict:
   - `word`: the word text from `all_surah_words_ground_truth`,
   - `start`: `null`,
   - `end`: `null`,
   - `score`: `0.0`,
   - `ayah_number`: the ayah number,
   - `word_location_wrt_ayah`: the word's location within the ayah,
   - `word_location_wrt_surah`: the word's location within the surah.
4. Add the created dict to `all_surah_words_algo_result`.
5. If (3.) is true, then add the index of the current iteration to a `matched_indices` list.
6. If (4.) is true, then add the index of the current iteration to a `unmatched_indices` list.
7. Then, as a data quality check, loop over each object in `all_surah_words_aligned` where its index is in `matched_indices` , then:
   1. Extract the value of its `start` key
   2. Get all the values of the `start` keys in the subsequent objects in `all_surah_words_algo_result` where their indices are in `matched_indices`.
   3. Now, check the following logic: if the value of the `start` key in the current object is more than the minimum value of the `start` keys in the subsequent objects, then move the index of the current object to `unmatched_indices` and remove it from `matched_indices`, and change its corresponding object in `all_surah_words_algo_result` to have key values as in (4.).
      1. TODO: This is a bit tricky, so we need to think about it more. Re-imagine "CASE 3" from excalidraw to see if this logic should apply for repeated words or no. we want it to block out words that came into view too early, NOT the words of a previous ayah that got repeated later in the audio file.


## Algorithm Steps - Looping Over `all_surah_words_aligned`


1. For each word in `all_surah_words_aligned`:
   1. Create a new list of dicts called `all_surah_words_algo_result` which starts as an empty list.
   2. Find the first corresponding word in the `all_surah_words_ground_truth` search space based on the word text.
      1. Where "corresponding word" means fuzzy matching with 90% confidence.
      2. Where `last_match_idx` is the index of the last matched word in `all_surah_words_ground_truth`.
   3. If a match is found, create the following dict:
      - `word`: the word text from `all_surah_words_aligned`,
      - `start`: the start time of the matched word from `all_surah_words_aligned`,
      - `end`: the end time of the matched word from `all_surah_words_aligned`,
      - `score`: the confidence score of the fuzzy match between the 2 words,
      - `ayah_number`: the ayah number from `all_surah_words_ground_truth`,
      - `word_location_wrt_ayah`: the word's location within the ayah from `all_surah_words_ground_truth`,
      - `word_location_wrt_surah`: the word's location within the surah from `all_surah_words_ground_truth`.
   4. Else, if no match is found, create the following dict:
      - `word`: the word text from `all_surah_words_aligned`,
      - `start`: null,
      - `end`: null,
      - `score`: 0.0,
      - `ayah_number`: null,
      - `word_location_wrt_ayah`: null,
      - `word_location_wrt_surah`: null.
   5. Add the created dict to `all_surah_words_algo_result`.