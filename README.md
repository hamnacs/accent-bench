# AccentBench

**Measuring accent, language-identification, and code-switching robustness in OpenAI's Whisper ASR model.**

## Motivation

While co-developing an AI-Enabled Hospital Information Management System (deployed at two hospitals in Pakistan) featuring an autonomous voice-booking agent, I noticed the underlying speech recognition pipeline seemed to degrade noticeably for accented and code-switched speech — a pattern that was never formally measured, just anecdotally observed while debugging.

AccentBench is a small, reproducible study built to actually measure that observation: how does a widely-used ASR model (Whisper) perform across different English accents, how does it handle Hindi-English code-switching, and does explicitly specifying the language improve transcription accuracy?

The code-switching analysis is particularly relevant to the Urdu-English speech my FYP's voice agent needed to handle. Because a suitable public Urdu-English speech corpus was not identified for the initial benchmark, Hindi-English was used as a linguistically related proxy.

---

## Summary of Findings

The project now contains four experimental tracks examining different aspects of multilingual ASR robustness:

1. **Accent robustness** — How does Whisper perform across different English accent groups?
2. **Language identification** — How often does Whisper incorrectly identify the language of accented English speech?
3. **Code-switching** — How does Whisper handle Hindi-English speech containing multiple languages within the same utterance?
4. **Language conditioning** — Does explicitly forcing the correct language improve transcription accuracy?

---

## Key Takeaways

* Whisper is generally robust across the evaluated English accent groups, with WER generally improving as model size increases.
* Singaporean English exposes a persistent language-identification failure that is largely corrected by forcing English decoding.
* Hindi-English code-switching remains substantially harder than accented English, even for the larger model.
* Many code-switch errors arise from script conversion and phonetic transliteration of English terms rather than complete hallucination.
* Language forcing produces a **large improvement for Urdu** but essentially **no improvement for Korean** in the current evaluation.
* The usefulness of explicit language conditioning therefore appears to be **language-dependent rather than universal**.

---

# Track 1 — Accent Robustness

## 1. Accent-driven Word Error Rate

AccentBench evaluated **600 utterances**, with 100 samples from each of six accent groups in the [DTU54DL/common-accent](https://huggingface.co/datasets/DTU54DL/common-accent) dataset.

| Accent                                 | WER (Whisper-base) | WER (Whisper-small) |
| -------------------------------------- | -----------------: | ------------------: |
| German (non-native)                    |              0.169 |               0.130 |
| Hong Kong English                      |              0.194 |               0.178 |
| Southern African                       |              0.211 |               0.192 |
| Filipino                               |              0.224 |               0.188 |
| South Asian (India/Pakistan/Sri Lanka) |              0.252 |               0.187 |
| Singaporean English                    |              0.410 |               0.410 |

Scaling from Whisper-base to Whisper-small improved WER for every accent group **except Singaporean English**, which remained essentially flat.

This suggests that the errors observed for Singaporean English are not fully explained by model capacity alone.

> **Note:** The "South Asian" label bundles Indian, Pakistani, and Sri Lankan English together. The dataset does not provide a clean Pakistani-only subset, so this measures South Asian English robustness broadly rather than Pakistani English specifically.

---

# Track 2 — Language Identification

## 2. Language misidentification is a distinct failure mode

Beyond ordinary transcription errors, a subset of clips triggered a more severe failure: Whisper misidentified the spoken language entirely, producing non-English output for English audio.

| Accent              | Language misidentification rate |
| ------------------- | ------------------------------: |
| Singaporean English |                      13.1–13.3% |
| Filipino            |                            1.0% |
| South Asian         |                          0–1.0% |
| German              |                              0% |
| Hong Kong English   |                              0% |
| Southern African    |                              0% |

This rate remained essentially constant across Whisper-base and Whisper-small, suggesting that simply increasing model size does not necessarily eliminate this failure mode.

### Forced-English recovery

For the 14 Singaporean clips that triggered language misidentification, forcing English decoding reduced mean WER from:

**1.044 → 0.266**

This represents approximately a **75% relative improvement**.

Two of the 14 clips went from complete failure to perfect transcription.

This indicates that at least some of the observed errors originate in **language detection rather than a fundamental inability to transcribe the accent**.

---

# Track 3 — Code-Switching

## 3. Hindi-English code-switching

AccentBench also evaluates Hindi-English code-switched speech using the [MUCS 2021](https://www.openslr.org/104/) dataset.

A suitable publicly available Urdu-English speech corpus was not identified for the initial study, so Hindi-English was used as a linguistically related proxy. Hindi and Urdu are closely related at the spoken-language level, although the two languages are not interchangeable.

|            | Whisper-base | Whisper-small |
| ---------- | -----------: | ------------: |
| Mean WER   |        1.150 |         1.058 |
| Median WER |        1.000 |         0.804 |

Even Whisper-small performs substantially worse on this code-switched speech than on any of the individual accent groups evaluated in Track 1.

This suggests that code-switching is not simply another form of accent variation; it presents a qualitatively different challenge.

### Transliteration pattern

Manual and automated inspection of Whisper-small outputs revealed a recurring pattern in utterances containing embedded English technical terms.

In approximately **88% of utterances containing embedded English technical terms**, those terms were rendered as approximate phonetic Devanagari spellings rather than preserved in their original Latin script.

Examples included terms such as:

* "operating system"
* "shortcut"
* "ctrl"

being rendered in approximate Devanagari transliteration.

A separate hallucination pattern, where generated content was not supported by the source audio, appeared in approximately **8% of utterances**.

These observations suggest that Whisper can often follow the semantic content of code-switched speech while struggling to identify the precise language boundary within a single sentence.

---

# Track 4 — Language Conditioning: Urdu vs Korean

The fourth experiment investigates whether explicitly specifying the expected language improves Whisper's transcription accuracy.

This extends the language-identification analysis from Track 2: rather than only measuring when Whisper identifies a language incorrectly, this experiment tests whether **forcing the correct language can recover transcription accuracy**.

Two Whisper model sizes were evaluated:

* Whisper-base
* Whisper-small

Each was tested under two conditions:

1. **Automatic** — Whisper detects the language automatically.
2. **Forced** — the correct language is explicitly provided during decoding.

## Urdu Results

The final Urdu comparison contains **80 matched clips** for which all four experimental conditions produced valid WER values.

| Model         | Automatic WER | Forced Urdu WER | Relative Improvement |
| ------------- | ------------: | --------------: | -------------------: |
| Whisper-base  |        0.6855 |          0.5821 |            **15.1%** |
| Whisper-small |        0.8473 |          0.3984 |            **53.0%** |

Forcing Urdu substantially improved transcription accuracy, with the largest effect occurring for Whisper-small.

The result suggests that automatic language identification can be a significant source of transcription errors for Urdu speech.

## Korean Results

The Korean evaluation used **100 clips from the Zeroth Korean test set**, with **10 clips selected from each of 10 speakers**.

All 100 clips successfully completed all four experimental conditions.

| Model         | Automatic WER | Forced Korean WER | Relative Improvement |
| ------------- | ------------: | ----------------: | -------------------: |
| Whisper-base  |        0.4737 |            0.4726 |            **0.23%** |
| Whisper-small |        0.3791 |            0.3791 |            **0.00%** |

Whisper automatically detected Korean on **100/100 clips**.

For Whisper-small, the automatic and forced transcriptions were identical for all 100 clips:

* Identical predictions: **100/100**
* Different predictions: **0/100**
* WER difference: **0.0**

Therefore, forcing Korean provided essentially no additional benefit on this evaluation set.

## Cross-Language Finding

The contrast between Urdu and Korean is the central result of this track.

Language forcing produced a substantial improvement for Urdu but almost no improvement for Korean:

| Language | Base Improvement | Small Improvement |
| -------- | ---------------: | ----------------: |
| Urdu     |        **15.1%** |         **53.0%** |
| Korean   |        **0.23%** |         **0.00%** |

This indicates that the usefulness of explicit language conditioning is **language-dependent** rather than a universal improvement to Whisper transcription.

The result also demonstrates why multilingual ASR evaluation should separate **language identification failures** from ordinary transcription errors. A model may perform well when the language is correctly identified but degrade substantially when its automatic language decision is wrong.

## Korean Dataset Scope

The Korean experiment should not be interpreted as a comprehensive test of Korean accent or dialect robustness.

The Zeroth Korean evaluation provides speech from multiple speakers, but the experiment was not designed as a controlled regional dialect benchmark. The purpose of this track is to compare the effect of **automatic versus forced language conditioning**, not to establish Whisper's performance across all Korean varieties.

---

# Dashboard

An interactive Streamlit dashboard presents the original AccentBench evaluation tracks with live charts, expandable raw-data tables, and a live upload-your-own-clip demo.

### Accent Robustness

The Accent Robustness tab presents WER by accent, language misidentification rate, and forced-English recovery.

![Accent Robustness tab](screenshots/accent_tab.png)

### Code-Switching

The Code-Switching tab presents WER comparisons, transliteration failure rates, and detected-language distributions.

![Code-Switching tab](screenshots/codeswitch_tab.png)

### Try It Yourself

The Try It Yourself tab allows users to upload a WAV/MP3 clip and see Whisper's transcription, with an option to force English detection.

![Try It Yourself tab](screenshots/try_it_yourself_tab.png)

Run the dashboard locally with:

```bash
streamlit run src/dashboard.py
```

---

# Methodology

## Models

* OpenAI Whisper `base`
* OpenAI Whisper `small`
* Inference performed locally using the `openai-whisper` Python package.
* Experiments were run on CPU.

## Accent Data

The accent evaluation uses the [DTU54DL/common-accent](https://huggingface.co/datasets/DTU54DL/common-accent) dataset.

* 6 accent groups
* 100 samples per group
* 600 utterances total

## Code-Switching Data

The code-switching evaluation uses the [MUCS 2021 Hindi-English test set](https://www.openslr.org/104/).

* 100 randomly sampled utterances
* Fixed sampling seed for reproducibility
* Hindi-English used as a proxy for the Urdu-English use case

## Language Conditioning Data

Track 4 evaluates Urdu and Korean speech under automatic and forced language decoding.

### Urdu

The final four-condition comparison uses:

* 80 matched clips
* Whisper-base
* Whisper-small
* Automatic language detection
* Forced Urdu decoding

Only clips with valid WER values under all four conditions were included in the final comparison.

### Korean

The Korean evaluation uses the **Zeroth Korean** test split.

The original test split contains **457 examples from 10 speakers**.

AccentBench selected:

* 100 clips
* 10 clips per speaker
* 10 speakers

All 100 clips successfully completed all four experimental conditions.

## Metric

Word Error Rate (WER) was calculated using `jiwer`.

Lower WER indicates better transcription accuracy.

For the language-conditioning experiments, relative improvement is calculated as:

```text
(Automatic WER - Forced WER)
-------------------------------- × 100
       Automatic WER
```

---

# Results

The final Track 4 comparison is available in:

[`results/final/final_comparison.csv`](results/final/final_comparison.csv)

The language-forcing comparison is available in:

[`results/final/language_forcing_effect.csv`](results/final/language_forcing_effect.csv)

Generated visualizations:

* [WER comparison](results/plots/wer_comparison.png)
* [Language forcing](results/plots/language_forcing.png)
* [Forcing improvement](results/plots/forcing_improvement.png)

Detailed methodology:

[`docs/methodology.md`](docs/methodology.md)

Detailed findings and limitations:

[`docs/findings_limitations.md`](docs/findings_limitations.md)

---

# Limitations

The benchmark is intentionally small and exploratory, so the results should not be interpreted as definitive measurements of Whisper's global multilingual robustness.

Important limitations include:

* Sample sizes are relatively small, particularly for the Track 4 Urdu evaluation.
* The "South Asian" accent label bundles Indian, Pakistani, and Sri Lankan English and cannot isolate Pakistani English specifically.
* No suitable public Urdu-English speech corpus was identified for the initial code-switching study; Hindi-English is used as a linguistically related proxy rather than a substitute.
* The transliteration-failure heuristic is word-overlap based and may over- or under-count edge cases.
* The Track 4 Urdu comparison uses only the 80 clips that successfully produced valid results under all four conditions.
* The Korean experiment is not a controlled regional accent or dialect benchmark.
* The Korean result therefore should not be interpreted as evidence that Whisper performs equally well across all Korean dialects.
* Only Whisper-base and Whisper-small were evaluated.
* Larger models such as medium and large may behave differently.
* Experiments were performed using CPU-only inference.

---

# Project Structure

```text
AccentBench/
│
├── data/                              # datasets (not committed)
│
├── src/
│   ├── explore_data.py
│   ├── run_inference.py
│   ├── analyze_errors.py
│   ├── analyze_errors_small.py
│   ├── forced_language_test.py
│   ├── load_code_switch_data.py
│   ├── run_inference_codeswitch.py
│   ├── run_inference_codeswitch_small.py
│   ├── tally_codeswitch_patterns.py
│   ├── dashboard.py
│   │
│   ├── final_analysis.py
│   ├── forced_urdu_test.py
│   ├── inspect_korean.py
│   ├── korean_base_auto.py
│   ├── prepare_korean.py
│   ├── run_remaining_korean.py
│   ├── run_urdu_base.py
│   ├── run_urdu_base_forced.py
│   ├── run_urdu_benchmark.py
│   ├── run_urdu_small_matched.py
│   └── test_urdu.py
│
├── results/
│   ├── final/
│   │   ├── final_comparison.csv
│   │   └── language_forcing_effect.csv
│   │
│   └── plots/
│       ├── wer_comparison.png
│       ├── language_forcing.png
│       └── forcing_improvement.png
│
├── docs/
│   ├── methodology.md
│   └── findings_limitations.md
│
├── screenshots/
│   ├── accent_tab.png
│   ├── codeswitch_tab.png
│   └── try_it_yourself_tab.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Running It Yourself

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Accent Benchmark

```bash
python src/explore_data.py
python src/run_inference.py
python src/analyze_errors.py
```

### Dashboard

```bash
streamlit run src/dashboard.py
```

### Track 4 Analysis

After generating the experiment CSVs:

```bash
python src/final_analysis.py
```

This generates:

```text
results/final/final_comparison.csv
results/final/language_forcing_effect.csv
results/plots/wer_comparison.png
results/plots/language_forcing.png
results/plots/forcing_improvement.png
```

## Dataset Downloads

Accent data loads automatically through Hugging Face `datasets`.

For the MUCS code-switching experiment, download the Hindi-English test data from [OpenSLR 104](https://www.openslr.org/104/) and extract it under:

```text
data/mucs_hindi_english/
```

The Korean Track 4 evaluation uses the [Zeroth Korean dataset](https://huggingface.co/datasets/kresnik/zeroth_korean).

---

# Future Work

Potential extensions include:

* Extend the benchmark to Urdu-English code-switching if a suitable public speech corpus becomes available.
* Supplement Urdu-English evaluation with a carefully documented self-recorded dataset.
* Test larger Whisper models such as medium and large.
* Investigate why language forcing produces a much larger improvement for Urdu than Korean.
* Examine whether the Urdu improvement is concentrated in clips where automatic language identification fails.
* Expand the number of Urdu speakers and recordings.
* Add controlled accent and dialect categories.
* Evaluate additional multilingual ASR systems.
* Separate language-identification accuracy from transcription accuracy more explicitly.

---

# License

This project uses datasets and resources subject to their respective licenses.

The Zeroth Korean dataset is available under **CC BY 4.0**.
