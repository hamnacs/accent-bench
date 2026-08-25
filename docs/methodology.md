# Methodology

## Overview

AccentBench evaluates the robustness of OpenAI Whisper automatic speech recognition (ASR) models under different language-conditioning settings. The experiment compares Whisper Base and Whisper Small on Urdu and Korean speech, measuring whether explicitly specifying the target language improves transcription accuracy compared with Whisper's automatic language detection.

The primary evaluation metric is Word Error Rate (WER), where lower values indicate better transcription performance.

## Models

Two Whisper model sizes were evaluated:

* **Whisper Base**
* **Whisper Small**

Both models were evaluated under two language-conditioning conditions:

1. **Automatic language detection** — Whisper determines the spoken language itself.
2. **Forced language** — the expected language is explicitly provided to Whisper during transcription.

This produces four experimental conditions per language:

* Base + automatic language detection
* Base + forced language
* Small + automatic language detection
* Small + forced language

## Urdu Evaluation

The Urdu experiments used a manually selected 100-clip benchmark from the Urdu speech data used in AccentBench.

Each clip contained a reference transcription. The same clips were processed using Whisper Base and Whisper Small under both automatic and forced-Urdu conditions.

Some clips failed during individual inference runs. To ensure a fair comparison between experimental conditions, the final Urdu comparison was restricted to clips for which all four experiments produced valid WER values.

This resulted in **80 matched Urdu clips** for the final comparison.

## Korean Evaluation

For Korean, the **Zeroth Korean** dataset was used. The official test split contains 457 examples from 10 speakers.

Rather than using the entire test set, AccentBench selected **100 clips**, with **10 clips per speaker**, providing a balanced small evaluation set across the available speakers.

The same 100 clips were evaluated using Whisper Base and Whisper Small under both automatic Korean language detection and forced-Korean conditions.

All 100 Korean clips produced successful results in all four experimental conditions, so the final Korean comparison contains **100 matched clips**.

## Evaluation Metric

Transcription quality was measured using **Word Error Rate (WER)**:

[
WER = \frac{S + D + I}{N}
]

where:

* (S) = substitutions
* (D) = deletions
* (I) = insertions
* (N) = number of words in the reference transcription

Lower WER represents better transcription accuracy.

Mean and median WER were reported for each language, model, and language-conditioning condition.

## Matched-Clip Evaluation

A matched-clip approach was used for the final comparison.

For each language, only clips that successfully produced WER values in **all four experimental conditions** were included. This prevents missing or failed inference runs from changing the comparison between automatic and forced-language transcription.

The final evaluation therefore compares the same audio clips across model sizes and language-conditioning settings.

## Analysis

The analysis focused on three questions:

1. Does explicitly forcing the language reduce WER?
2. Does the effect differ between Whisper Base and Whisper Small?
3. Does the effect differ between Urdu and Korean?

Three visualizations were generated:

* Overall WER comparison across languages, model sizes, and conditions.
* Automatic versus forced-language WER.
* Relative WER improvement produced by language forcing.

Relative improvement was calculated as:

[
Improvement =
\frac{WER_{automatic} - WER_{forced}}
{WER_{automatic}}
\times 100
]

All experiments were performed locally using the project Python environment. Whisper inference was performed on CPU.
