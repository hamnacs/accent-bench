# AccentBench: A Failure-Mode-Oriented Evaluation of Accent, Language Identification, and Code-Switching in Whisper ASR

**Hamna Masood**

---

## Abstract

Automatic speech recognition (ASR) systems are increasingly deployed in multilingual and code-switched settings, yet standard evaluation practice still relies heavily on aggregate Word Error Rate (WER) over monolingual, native-accented benchmarks. This masks failure modes that matter in practice: likely language misidentification, loss of embedded-language tokens under code-switching, and inconsistent behavior across accents. This paper presents AccentBench, a four-track empirical study of OpenAI's Whisper ASR system (base and small checkpoints) across accent-stratified performance, likely language misidentification, Hindi-English and Korean-English code-switching, and language-conditioning effects for Urdu and Korean. Across 600 accented-English utterances, 100 Hindi-English code-switched utterances, 100 Korean-English code-switched utterances evaluated under four experimental conditions, and 180 language-conditioning clips (80 Urdu, 100 Korean), we find that (1) automatic language detection outperforms forced-language decoding on the Korean-English code-switching benchmark by 32.3% relative WER for Whisper-small, (2) a specific accent group (Singaporean English) exhibits output patterns consistent with language misidentification at a substantially higher rate (13-14%) than five other accent groups (0-1%), and forcing English recovers most of the resulting error, (3) language conditioning has an opposite and language-dependent effect for Urdu (up to 53.0% relative WER reduction) versus Korean (0.0% to 0.23%), and (4) code-switching WER varies systematically by switching granularity, with phrase-level switching the hardest category. We release the benchmark scripts, raw predictions, and analysis pipeline for reproducibility, and discuss the limitations of a small-scale, exploratory study of this kind.

---

## 1. Introduction

Whisper [1] and similar large-scale multilingual ASR systems are trained on hundreds of thousands of hours of weakly supervised audio, and their ability to recognize many languages makes it easy to assume they will also handle a wide range of accents well. In practice, though, some of Whisper's errors trace back to the model failing to identify the spoken language correctly in the first place, rather than to the accent itself.

This question grew out of my work on an AI-enabled hospital information management system, where I built a voice-driven booking agent. Speech in that setting rarely stayed in one language: Urdu and English were routinely mixed within the same sentence, a pattern common in everyday Pakistani speech. During this mixing, the system would sometimes fail to recognize the language at all, lock onto one language and transcribe the rest as if it were that language, or miss parts of the utterance entirely. These looked at first like isolated errors, but they raised a question worth testing directly rather than dismissing as noise from microphone quality or background conditions.

AccentBench evaluates four research questions:

1. **Accent robustness.** How does Whisper's transcription performance vary across English accent groups?
2. **Language identification.** How accurately does Whisper identify the spoken language, and can misidentification explain some transcription errors?
3. **Code-switching.** How does Whisper handle speech where speakers switch between languages mid-utterance, and does the difficulty vary by switching granularity (word, phrase, or sentence level)?
4. **Language conditioning.** Does explicitly specifying the expected language improve transcription accuracy, and is that effect consistent across languages?

A single WER score cannot distinguish between these causes. The same poor score can mean the model identified the language correctly but struggled with the accent, misidentified the language outright, dropped words when the speaker switched into a second language, or simply lacked the correct language hint that would have fixed the error, as with the Korean case in this study, where supplying that hint made results worse instead of better (Section 5.3). Knowing that a WER number is bad is not enough to fix a real deployed system; knowing which of these it actually is, is. The central thesis of this paper is that aggregate WER hides qualitatively different multilingual ASR failure modes, and the correct mitigation depends on which failure mode is actually present.

Concretely, this paper makes four contributions:

1. A controlled evaluation that separates accent-driven acoustic difficulty from language-identification failure, showing that one accent group's elevated error rate in this benchmark is substantially attributable to misidentification rather than acoustic difficulty alone, and that forcing the correct language recovers most of the gap (Section 5.1).
2. Evidence that language conditioning does not have a uniform effect: forcing the expected language helps substantially for Urdu and for the misidentified Singaporean-English subset, yet is neutral for Korean and actively harmful on the Korean-English code-switching benchmark (Sections 5.3-5.4).
3. Evidence that code-switching difficulty is not a single number: it varies systematically with switching granularity and topical domain, and an English-token-preservation metric reveals a script-conversion failure mode that overall WER understates (Section 5.3).
4. A reproducible collection of benchmark scripts, raw per-clip predictions, and analysis code across all four tracks, released for independent verification.

The HiKE dataset itself, including its audio, transcripts, and word/phrase/sentence switching-level labels, is an existing resource [6]; this paper's contribution with respect to Track 3 is the Whisper evaluation built on top of it, specifically the automatic-versus-forced-language comparison, the granularity- and domain-stratified analysis, and the English-token-preservation metric, none of which are part of the original HiKE release.

---

## 2. Related Work

**Multilingual and robust ASR.** Whisper [1] is trained with large-scale weak supervision on 680,000 hours of multilingual and multitask audio-transcript pairs and demonstrates strong zero-shot generalization across many existing benchmarks. Because Whisper performs joint language identification and transcription in a single decoder, errors in the former can propagate into the latter, a coupling that motivates Research Question 2 above.

**Accented speech recognition.** Accent robustness has been studied both as a data problem and a modeling problem. The Accented English Speech Recognition Challenge (AESRC2020) [2] released a labeled multi-accent English corpus and established accent recognition and accented transcription as a shared task, spurring subsequent architectural work on accent-invariant and accent-aware ASR. The CommonAccent recipe [3] built a large accent-labeled resource from Common Voice, covering sixteen English accent categories with an accent-classification model; a derived sample of this resource (six accent groups: German, Hong Kong, Southern African, Filipino, South Asian, and Singaporean English) forms Track 1 of AccentBench.

**Code-switching ASR.** Code-switching, the alternation between two or more languages within a single utterance or conversation, has long been studied as a linguistic phenomenon [4] and more recently as an ASR benchmarking problem. The MUCS 2021 shared task [5] released Hindi-English and Bengali-English code-switched speech alongside six monolingual low-resource Indian languages, with the explicit goal of measuring ASR performance under code-switching rather than assuming it degrades gracefully from monolingual performance. HiKE [6] is a more recent, hierarchical Korean-English code-switching benchmark that labels each utterance's switching granularity (word, phrase, or sentence level) in addition to providing loanword annotations, enabling exactly the kind of granularity-stratified analysis used in Track 3 of this paper. Both resources motivate treating code-switching as a first-class evaluation axis rather than an edge case of standard WER evaluation.

**Language identification in multilingual and code-switched speech.** Because many multilingual ASR systems, including Whisper, perform implicit language identification before or during decoding, identification errors are an underexplored source of downstream transcription error. Recent work has begun evaluating language identification specifically under domain shift and code-switching conditions [7], reinforcing that identification accuracy should be measured separately from transcription accuracy. AccentBench's forced-language experiments (Tracks 1 and 4) are a direct, low-cost way to isolate this effect: comparing automatic detection against a forced ground-truth language reveals how much of a system's error is attributable to identification failure versus acoustic-transcription failure.

**Evaluation metric.** Word Error Rate, computed as the minimum edit distance between reference and hypothesis transcripts normalized by reference length, remains the standard ASR evaluation metric [8] despite known limitations discussed further in Section 8. All WER values in this paper were computed using the `jiwer` Python package [9].

---

## 3. AccentBench Benchmark

AccentBench consists of four tracks, summarized in Table 1.

**Table 1: Benchmark tracks**

| Track | Research question | Dataset | Size |
|---|---|---|---|
| 1 | Accent-stratified performance and likely language misidentification | CommonAccent-derived sample [3] (via DTU54DL/common-accent) | 600 utterances, 6 accent groups |
| 2 | Hindi-English code-switching | MUCS 2021 [5] | 100 utterances |
| 3 | Korean-English code-switching | HiKE [6] | 100 clips, 4 conditions, 400 transcriptions |
| 4 | Language conditioning (Urdu, Korean) | UrduSpeech US-benchmark [10]; Zeroth Korean [11] | 80 Urdu clips, 100 Korean clips |

**Track 1 (Accent).** Six accent groups, each with 100 sampled utterances: German (non-native), Hong Kong English, Southern African, Filipino, South Asian (combining Indian, Pakistani, and Sri Lankan English), and Singaporean English. Each utterance was transcribed with Whisper-base and Whisper-small under automatic language detection, and WER was computed against the reference transcript.

**Track 2 (Hindi-English code-switching).** 100 utterances sampled from the MUCS 2021 Hindi-English code-switching subtask [5], transcribed with Whisper-base and Whisper-small under automatic language detection. This track also serves as a linguistically related proxy for the Urdu-English case, since a suitable public Urdu-English code-switching corpus was not identified at the time the original benchmark was built (see Section 3.1 and Section 8 for the important caveat that Hindi and Urdu, while closely related at the spoken level, are not interchangeable).

**Track 3 (Korean-English code-switching).** 100 clips from the HiKE benchmark [6], which provides word-, phrase-, and sentence-level code-switching annotations and eight topical domains. Each clip was transcribed under four conditions: {Whisper-base, Whisper-small} times {automatic language detection, forced Korean}, producing 400 total transcriptions.

**Track 4 (Language conditioning).** For Urdu, 80 matched clips (i.e., clips for which all four experimental conditions produced a valid WER value) drawn from the UrduSpeech US-benchmark set [10], which was independently identified after Track 2 was built and is now the primary Urdu-specific resource used in AccentBench. For Korean, 100 clips sampled 10 per speaker from the 10-speaker Zeroth Korean test set [11]. Both were evaluated with Whisper-base and Whisper-small under automatic and forced-language conditions.

### 3.1 A Note on the Urdu-English Proxy

An earlier design decision in this project used Hindi-English code-switched speech (Track 2, MUCS 2021) as a proxy for Urdu-English, since no suitable public Urdu-English code-switching corpus was identified at the time. This choice is linguistically motivated: Hindi and Urdu are mutually intelligible at the spoken level and share substantial phonological and syntactic structure, differing mainly in script and register. It is, however, an approximation, not a substitute; Section 8 discusses this limitation and Section 9 discusses the subsequent identification of a dedicated Urdu resource [10] used in Track 4.

---

## 4. Experimental Setup

**Models.** Two OpenAI Whisper checkpoints were evaluated: `base` and `small`, using the open-source `openai-whisper` Python package. All inference was performed locally on CPU.

**Metric.** Word Error Rate (WER), computed with `jiwer` [9] as (S + D + I) / N, where S, D, and I are substitutions, deletions, and insertions, and N is the reference word count. Lower is better. Both mean and median WER are reported, since WER distributions in small samples can be skewed by a small number of near-total-failure transcriptions. A WER above 1.0 does not indicate an invalid metric; it occurs when the combined number of substitutions, deletions, and insertions exceeds the number of reference words, which is possible whenever the hypothesis is substantially longer or more disordered than the reference.

**Uncertainty estimation.** For the three comparisons in this paper with the smallest sample sizes or the most consequential conclusions (the Track 1 Singaporean forced-English recovery, the Track 3 automatic-versus-forced comparison, and the Track 4 Urdu and Korean language-conditioning comparisons), paired non-parametric bootstrap 95% confidence intervals (10,000 resamples, seed 42) were computed over the per-clip WER differences between conditions. This directly addresses whether an observed difference in mean WER is well supported by the underlying per-clip distribution rather than driven by a small number of clips.

**Language conditioning.** For Tracks 1 and 4, two decoding conditions were compared: automatic language detection, in which Whisper infers the spoken language from the audio itself, and forced-language decoding, in which the expected language is supplied explicitly as a decoding parameter. Relative improvement from forcing is computed as (WER_automatic − WER_forced) / WER_automatic times 100.

**Language misidentification heuristic (Track 1).** A transcription was flagged as a likely language-misidentification failure if the Whisper output contained non-Latin script characters (Cyrillic, CJK, Hangul, or similar Unicode ranges) or matched a small set of common Malay/Indonesian function words, under the assumption that correctly transcribed accented English should not contain either. This is a heuristic, not a ground-truth language label, and undercounts misidentification into other Latin-script languages; see Section 8.

**Matched-clip comparison (Track 4).** Some clips failed during individual inference runs (e.g., empty audio, decoding errors). To keep the automatic-versus-forced comparison fair, the final Urdu and Korean comparisons in Track 4 include only clips for which all four experimental conditions (two models times two language settings) produced a valid WER value, yielding 80 matched Urdu clips and 100 matched Korean clips.

**Sampling methodology (Track 3).** The 100 HiKE clips evaluated were drawn as a simple, unstratified random sample from the benchmark's full 1,121-clip test split, using `pandas.Series.sample(n=100, random_state=42)` over the full index range before any model was run. The sample was not stratified by code-switching level or topical domain, and the resulting distribution across levels (41 word, 53 phrase, 6 sentence) and domains (7 to 16 clips each) reflects the natural composition of the underlying test split rather than a designed balance. This is disclosed explicitly here because Section 5.3.1 and 5.3.2 report WER broken down by these same categories, and the reader should weigh the smaller categories (in particular the six sentence-level clips) accordingly. The sample was fixed once, before any transcription was produced, so no clip selection was informed by model outputs.

### 4.1 Reproducibility

All inference used the open-source `openai-whisper` Python package with CPU-only execution and no GPU acceleration (`fp16=False` throughout). Decoding used the package's default temperature-fallback schedule and default beam/greedy search settings; no custom beam size, temperature, or best-of value was set in any script. Where a specific dataset required random sampling (selecting 100 clips from a larger pool for Tracks 2, 3, and portions of Track 4), a fixed random seed of 42 was used consistently across sampling scripts (`random.seed(42)` or `random_state=42`, depending on whether the Python `random` module or `pandas` sampling was used), so the specific subset of clips evaluated is deterministic given the same source data. The `openai-whisper`, `jiwer`, `datasets`, and `pandas` package versions were not pinned in the project's `requirements.txt` at the time these experiments were run, which is itself noted here as a reproducibility gap rather than omitted; anyone attempting to reproduce these exact numbers should pin package versions, since Whisper's decoding behavior has changed across releases in the past.

---

## 5. Results

### 5.1 Accent-Stratified Performance (Track 1)

Table 2 reports mean WER per accent group for both model sizes. These six groups are a sample, not a representative census of global English accents, so the results below are reported as accent-stratified performance on this specific set rather than as a general claim about Whisper's robustness to accented English as a whole; the broader robustness question is returned to in Section 7.

**Table 2: Accent-driven Word Error Rate**

| Accent | WER (Whisper-base) | WER (Whisper-small) |
|---|---:|---:|
| German (non-native) | 0.169 | 0.130 |
| Hong Kong English | 0.194 | 0.178 |
| Southern African | 0.211 | 0.192 |
| Filipino | 0.224 | 0.188 |
| South Asian | 0.252 | 0.187 |
| Singaporean English | 0.410 | 0.410 |

Scaling from Whisper-base to Whisper-small reduced WER for every accent group except Singaporean English, where performance was essentially flat (0.410 to 0.410). Note that the South Asian category here combines Indian, Pakistani, and Sri Lankan English; it is not a Pakistani-English-specific result. Singaporean English stands out as an outlier: its WER is roughly double that of every other accent group under both model sizes, and it is the only group that does not improve with the larger model. Section 5.1.1 investigates why.

![Mean WER by accent group, Whisper-base versus Whisper-small. Singaporean English is the only group that does not improve with the larger model.](figures/fig5_accent_wer.png)

#### 5.1.1 Likely Language Misidentification

A subset of accented-English clips triggered a more severe failure than ordinary mistranscription: Whisper produced output consistent with a non-English decoding decision, rather than an ordinary transcription error. This is inferred from output characteristics (Section 4), not observed directly from Whisper's internal language-identification decision, which the package does not expose in a form this study captured; the rate reported below should be read as a heuristic proxy rather than a ground-truth misidentification rate. Table 3 reports this proxy rate per accent group.

**Table 3: Language-misidentification heuristic rate**

| Accent | Whisper-base | Whisper-small |
|---|---:|---:|
| Singaporean English | 14.3% | 13.1% |
| Filipino | 1.0% | 1.0% |
| South Asian | 1.0% | 0.0% |
| German | 0.0% | 0.0% |
| Hong Kong English | 0.0% | 1.0% |
| Southern African | 0.0% | 1.0% |

![Language-misidentification heuristic rate by accent group. Singaporean English is 13 to 14 times higher than any other group in this sample.](figures/fig6_misid_rate.png)

Singaporean English triggers our language-misidentification heuristic at 13-14 times the rate of every other accent group, and the rate barely moves between model sizes. That points toward an explanation for the anomaly in Table 2: Singaporean English's elevated, scale-resistant WER may be driven mainly by a language-selection failure rather than by acoustic difficulty alone, since a bigger model does nothing to fix it. We do not treat this as proven; the forced-English experiment below tests it more directly, though it too works from the same flagged subset rather than an independent confirmation.

#### 5.1.2 Forced-English Recovery

To test this explanation, the 14 Singaporean-English clips flagged by the heuristic under Whisper-base were re-transcribed with English explicitly forced as the decoding language. Table 4 reports the result.

**Table 4: Forced-English recovery for flagged Singaporean clips (n = 14)**

| Condition | Mean WER |
|---|---:|
| Original (automatic detection) | 1.091 |
| Forced English | 0.242 |

Forcing English reduced mean WER by 77.8% relative to the original automatic-detection transcriptions. The paired per-clip reduction was 0.849 WER (95% bootstrap CI [0.752, 0.943], n = 14), a confidence interval that excludes zero by a wide margin despite the small sample, indicating the recovery effect is consistent across nearly all fourteen clips rather than driven by one or two outliers. Two of the fourteen clips went from a WER at or above 1.0 (effectively a failed transcription) to a perfect transcription (WER = 0.0) once English was forced. On this small sample (n = 14), the result supports the view that a substantial share of Whisper's difficulty with Singaporean-accented English in this dataset is a language-identification failure rather than an inherent inability to transcribe the accent acoustically: once the model is told the correct language, its acoustic decoding recovers most of the lost accuracy. A larger, independently sampled set of Singaporean-English clips would be needed before treating this as a general property of Whisper rather than a pattern observed in this specific benchmark.

### 5.2 Hindi-English Code-Switching (Track 2)

**Table 5: Hindi-English code-switching WER**

| Model | Mean WER | Median WER |
|---|---:|---:|
| Whisper-base | 1.150 | 1.000 |
| Whisper-small | 1.058 | 0.804 |

Both models show mean WER above 1.0, meaning the number of transcription errors on average exceeds the number of words in the reference. This is substantially worse than the WER observed on any single accent group in Track 1 (all of which stayed below 0.5), indicating that code-switched speech is considerably more difficult for Whisper than monolingual accented speech, even before controlling for the specific language pair. Whisper-small modestly outperforms Whisper-base on this track (mean WER 1.058 versus 1.150), but the improvement is smaller in relative terms than the improvements observed for accent-stratified performance in Track 1. Relative to the other three tracks, this one is best read as an exploratory baseline: it establishes that Hindi-English code-switching is difficult for Whisper, but does not probe why in the same depth as Tracks 1, 3, and 4 do for their respective failure modes.

### 5.3 Korean-English Code-Switching (Track 3)

All 100 HiKE clips successfully completed all four experimental conditions, yielding 400 transcription results.

**Table 6: Overall Korean-English code-switching WER**

| Configuration | Mean WER | Median WER |
|---|---:|---:|
| Whisper-small + automatic | 0.4209 | 0.3333 |
| Whisper-base + automatic | 0.5648 | 0.5714 |
| Whisper-small + forced Korean | 0.6216 | 0.6250 |
| Whisper-base + forced Korean | 0.6729 | 0.6340 |

Two effects are visible in Table 6. Scaling from Whisper-base to Whisper-small under automatic detection reduces mean WER by 25.5% relative, which is expected. What is more surprising is that forcing Korean explicitly *increases* WER relative to automatic detection: by 32.3% relative for Whisper-small (0.4209 to 0.6216) and 16.1% relative for Whisper-base (0.5648 to 0.6729). For Whisper-small, the paired per-clip increase from forcing Korean was 0.201 WER (95% bootstrap CI [0.131, 0.276], n = 100), well clear of zero, so this isn't a fluke driven by a handful of clips. This runs opposite to the Track 1 forced-English recovery result and the Track 4 Urdu result (Section 5.4). We don't read it as evidence that automatic detection beats forcing in general; it looks specific to this Korean-English code-switched dataset, where forcing a single language may work against a model trying to represent content that legitimately contains both languages. One plausible mechanism, not tested directly here: Whisper's automatic detection may be responding to whichever language dominates a given clip, so forcing Korean on a clip that's decodably English-heavy could push the decoder away from the correctly-cased English output it would otherwise produce. We leave testing that mechanism for future work (Section 9) rather than claiming it here.

![Overall WER across the four experimental conditions in Track 3 (n = 100 clips per bar). Automatic detection outperforms forced Korean for both model sizes.](figures/fig1_hike_overall_wer.png)

#### 5.3.1 Effect of Code-Switching Granularity

The HiKE benchmark labels each clip's code-switching level. Table 7 reports WER by level under the best-performing configuration (Whisper-small, automatic detection).

**Table 7: WER by code-switching level (Whisper-small, automatic)**

| Level | Clips | Mean WER |
|---|---:|---:|
| Word | 41 | 0.3458 |
| Sentence | 6 | 0.4107 |
| Phrase | 53 | 0.4802 |

In this sample, phrase-level code-switching, where a contiguous multi-word phrase switches language within the utterance, produced the highest WER, followed by sentence-level and then word-level switching. This ordering is a sample-level finding, not a claim about code-switching difficulty in general; the sentence-level result in particular is based on only six clips and is included for completeness rather than as a robust estimate.

![Mean WER by code-switching level (word, sentence, phrase), Whisper-small under automatic detection. Clip counts per level: word n = 41, sentence n = 6, phrase n = 53.](figures/fig2_hike_cs_level.png)

#### 5.3.2 Domain Effects

**Table 8: WER by topical domain (Whisper-small, automatic)**

| Domain | Mean WER |
|---|---:|
| Travel and culture | 0.2335 |
| Entertainment | 0.3562 |
| Academic | 0.3862 |
| Language education | 0.4128 |
| Software development | 0.4533 |
| Medical | 0.4707 |
| Everyday conversation | 0.4929 |
| Business | 0.4975 |

Travel and culture produced the lowest WER by a wide margin, while business and everyday-conversation domains were among the hardest. With only 7 to 16 clips per domain, though, we'd stop short of claiming this says something general about Korean ASR and domain difficulty; it's a pattern in this specific sample.

#### 5.3.3 English Token Preservation

Overall WER treats a code-switched utterance as a single sequence and does not distinguish whether an error involved losing an embedded English term specifically. To capture this, English-token preservation was measured directly: the proportion of reference English tokens embedded in Korean speech that survive as recognizable English tokens in the hypothesis, rather than being converted into Korean-script phonetic approximations.

**Table 9: English token preservation**

| Configuration | Preservation rate |
|---|---:|
| Whisper-small + automatic | 72.6% |
| Whisper-base + automatic | 56.9% |
| Whisper-base + forced Korean | 36.4% |
| Whisper-small + forced Korean | 27.7% |

The pattern mirrors the overall WER result in Table 6: automatic detection substantially outperforms forced Korean on this metric as well, and the gap is larger for token preservation than for WER (Whisper-small automatic preserves 72.6% of English tokens versus only 27.7% under forced Korean, a 44.9 percentage-point gap). Qualitatively, the dominant failure mode under forced Korean is that embedded English terms are rendered as Korean-script phonetic approximations of their pronunciation rather than being transcribed in Latin script. For example, a reference containing the English terms "pull request," "test case," and "check" was transcribed under forced Korean with all three terms converted to Korean-script phonetic renderings, while the surrounding Korean content remained largely intact. This illustrates why WER alone, which does not distinguish a script-conversion error from an equivalently-scored acoustic mistranscription, understates the practical severity of this particular failure mode: a phonetically converted English term is often unusable downstream (e.g., in a transcript intended for search or command extraction) even though it may contribute a similar WER penalty to an ordinary substitution error.

![Proportion of reference English tokens surviving as recognizable English tokens in the Whisper hypothesis, by model size and language-detection condition.](figures/fig3_token_preservation.png)

### 5.4 Language Conditioning: Urdu versus Korean (Track 4)

**Table 10: Language-conditioning effect**

| Language | Model | Automatic WER | Forced WER | Relative improvement |
|---|---|---:|---:|---:|
| Urdu | Whisper-base | 0.6855 | 0.5821 | 15.1% |
| Urdu | Whisper-small | 0.8473 | 0.3984 | 53.0% |
| Korean | Whisper-base | 0.4737 | 0.4726 | 0.23% |
| Korean | Whisper-small | 0.3791 | 0.3791 | 0.00% |

The contrast is stark. Forcing Urdu produces a substantial improvement for both model sizes, and the effect is far larger for Whisper-small (53.0% relative reduction, paired difference 0.449 WER, 95% bootstrap CI [0.366, 0.534], n = 80) than for Whisper-base (15.1% relative reduction, paired difference 0.103 WER, 95% bootstrap CI [0.036, 0.167], n = 80); both intervals exclude zero. Forcing Korean, by contrast, produces almost no change for Whisper-base (0.23%) and produces exactly zero change for Whisper-small: automatic and forced decoding produced identical predictions on all 100 Korean clips (verified string-for-string, not merely equal WER, by directly comparing the two prediction columns row by row), indicating Whisper already identified Korean correctly on every clip in this sample before any language was forced. Because every prediction was identical, the paired difference for Whisper-small Korean is exactly 0.0000 with a degenerate (zero-width) bootstrap interval, which is the expected result of a null effect rather than a modeling artifact.

![Mean WER under automatic versus forced-language decoding, Urdu and Korean, both model sizes. Urdu shows a large, model-size-dependent improvement from forcing; Korean shows essentially none.](figures/fig4_language_forcing_urdu_korean.png)

An additional model-size interaction is notable within Urdu: under automatic detection, Whisper-base (0.6855) actually outperforms Whisper-small (0.8473), the opposite of the ordering seen in Tracks 1 and 3. Once Urdu is forced, this reverses and Whisper-small (0.3984) substantially outperforms Whisper-base (0.5821). This suggests Whisper-small may be more prone to *misidentifying* Urdu specifically under automatic detection (plausibly toward a closely related language such as Hindi, given their shared spoken-level similarity discussed in Section 3.1), even though it has the stronger underlying acoustic model once given the correct language.

The Urdu and Korean language-conditioning experiments used different datasets, speaker populations, and recording conditions (Section 3), so the Urdu-versus-Korean contrast in Table 10 should be read as evidence that language-conditioning effects are dataset- and language-dependent, not as a controlled, causal comparison isolating language identity as the sole variable.

---

## 6. Error Analysis

This section distinguishes two kinds of findings. WER, the language-misidentification heuristic rate, the forced-decoding comparisons, and the English-token-preservation metric (Sections 5.1-5.4) are measured findings, computed the same way across the full sample in each track. The transliteration and hallucination patterns below are exploratory observations: they were noticed during manual inspection of examples rather than measured with a validated automated detector across the full dataset, and should be weighted accordingly.

**Transliteration of embedded terms.** As in the Korean-English example in Section 5.3.3, English technical terms embedded in Hindi-English code-switched speech (Track 2) were frequently rendered as approximate Devanagari transliterations rather than preserved in Latin script. This preserves rough phonetic content while breaking the language boundary of the original utterance, a failure mode WER scores similarly to an unrelated substitution error despite being qualitatively different and arguably more disruptive for downstream use.

**Hallucination under code-switching.** A distinct pattern, separate from transliteration, was content in the hypothesis that was not supported by the source audio at all. Because both the transliteration and hallucination measurements in this study relied partly on heuristic, manual, and semi-automated inspection of examples rather than a fully automated, validated detector, these findings should be treated as exploratory error-analysis observations rather than definitive population-level estimates of their frequency.

**Model size does not uniformly help.** Across all four tracks, moving from Whisper-base to Whisper-small improved mean WER in most conditions but not all: Singaporean-English misidentification (Table 3) was essentially unaffected by model size, and Urdu automatic-detection WER (Table 10) was actually worse for the small model than the base model. This indicates that "use a larger model" is not a reliable universal mitigation for the specific failure modes studied here; some failures appear to be more closely tied to language-identification behavior than to acoustic modeling capacity, and scaling the acoustic model does not necessarily fix an identification problem.

---

## 7. Discussion

The results across all four tracks support the thesis stated in Section 1: aggregate WER hides qualitatively different failure modes, and the correct mitigation depends on which one is present. The most surprising result in this study was Singaporean English: under automatic detection its WER was unexpectedly high, well above every other accent group. Investigating why showed the model was not reliably recognizing the speech as English at all, triggering the language-misidentification heuristic on a meaningful share of clips; forcing English explicitly recovered most of this, cutting mean WER from 1.091 to 0.242 (77.8% relative) on the flagged subset. What looked at first like a purely accent-driven acoustic problem appears, on this evidence, to be substantially a language-selection problem instead, though the forced-decoding result demonstrates recoverability rather than directly confirming what Whisper's internal detector predicted for those clips. For anyone building multilingual or accent-inclusive speech interfaces with Whisper-class models, three things stand out from the results as a whole.

Aggregate WER can mask a language-identification failure that has a cheap, targeted fix. When a system knows, or can reasonably constrain, the expected language, forcing it (the Track 1 Singaporean-English case, the Track 4 Urdu case) can substantially beat trusting automatic detection.

But that fix is not a default you can apply blindly. The Track 3 Korean-English code-switching result and the Track 4 Korean result both show automatic detection matching or beating forced decoding. Korean in particular contradicted my own expectations going into this study: I assumed forcing the correct language would improve or at minimum preserve performance, and instead it made WER worse by roughly 32% relative for the small model. A system builder cannot assume "always force the expected language" without testing it on their own language pair and use case first.

And code-switching difficulty is not one number. Switching granularity (Table 7) and topical domain (Table 8) both move WER substantially within a single language pair, which means a single aggregate WER for "a Korean-English code-switching system" hides a wide range of real performance depending on how heavily, and where, the code-switching happens in a given utterance.

The main practical lesson I take from this project is that automatic language detection should not be assumed reliable across all users. A real system should evaluate language identification separately from transcription accuracy, and should be tested against the specific accents and code-switching patterns its actual users will produce, rather than judged on a single aggregate WER figure.

Taken together, these results support evaluating multilingual ASR along multiple separated axes, rather than a single overall accuracy number, particularly for systems intended to serve accented, multilingual, or code-switching-heavy user populations such as the hospital voice-booking context that originally motivated this project.

---

## 8. Limitations

AccentBench is intentionally small and exploratory, and its results should not be read as definitive measurements of Whisper's global multilingual or accent robustness. Specific limitations include:

- **Sample sizes are small** relative to what would be needed for tight statistical confidence, particularly for the sentence-level code-switching category (6 clips, Table 7) and the per-domain breakdown (7-16 clips per domain, Table 8).
- **The South Asian accent category is a composite** of Indian, Pakistani, and Sri Lankan English and should not be interpreted as a Pakistani-English-specific result.
- **The Hindi-English proxy for Urdu-English (Track 2) is an approximation.** Hindi and Urdu are closely related at the spoken level but are not interchangeable languages; conclusions about code-switching behavior from Track 2 should not be assumed to transfer directly to genuine Urdu-English code-switching. Track 4 uses a dedicated Urdu resource [10] but evaluates language conditioning rather than code-switching specifically, so the code-switching gap for genuine Urdu-English speech remains only partially addressed by this study.
- **The transliteration and hallucination analyses (Section 6) rely partly on heuristic and manual inspection** rather than a fully automated, independently validated detector, and should be treated as exploratory rather than definitive frequency estimates.
- **The language-misidentification heuristic (Section 4) is a proxy, not a ground-truth label.** It flags non-Latin script and a small set of Malay/Indonesian markers, and will undercount misidentification into other Latin-script languages that would not trigger either signal. It also does not observe Whisper's internal language-identification decision directly; it infers a likely misidentification from output characteristics, and the automatic-versus-forced comparisons in Tracks 1 and 4 should be read as automatic language selection versus explicitly forced decoding, not as a direct correction of an observed wrong language-ID prediction.
- **Speaker independence within Track 1 is not established.** The accent-track results (Table 2, Table 3) are reported per utterance; the released metadata does not retain speaker identifiers, so it is not possible to state from this study's own artifacts how many distinct speakers contributed the 100 utterances per accent group, or whether utterances from the same speaker appear more than once within a group. If speakers are unevenly represented, the effective sample size for inference is smaller than 100 per group, and the bootstrap confidence intervals reported for other tracks (which are clip-level, not speaker-level, throughout this study) should be read with that caveat in mind.
- **The Korean language-conditioning experiment (Track 4) and the Korean-English code-switching experiment (Track 3) use different datasets** (Zeroth Korean versus HiKE) and different speaker populations, so they should not be treated as a controlled comparison of the same underlying speech under two different tasks.
- **Only Whisper-base and Whisper-small were evaluated.** Larger checkpoints (medium, large) may exhibit different accent, code-switching, and language-identification behavior, and the model-size effects reported here (Section 6) should not be extrapolated beyond the two sizes actually tested.
- **All experiments used CPU-only inference.** This affected runtime but was not intended as an experimental variable and should not have influenced the accuracy results.
- **WER alone does not capture every dimension of transcription quality** relevant to multilingual and code-switched speech, as illustrated directly by the token-preservation metric in Section 5.3.3, which reveals a substantially larger gap between conditions than overall WER does for the same comparison.

---

## 9. Future Work

Potential extensions of this study include: identifying or constructing a dedicated Urdu-English code-switching corpus to replace the Hindi-English proxy used in Track 2, and applying the same code-switching-granularity and domain-stratified analysis used in Track 3 to that corpus once available; supplementing this with a carefully documented, consent-based self-recorded dataset if no suitable public corpus is identified; evaluating Whisper-medium and Whisper-large across all four tracks to test whether the model-size effects reported here persist at larger scale; expanding the Korean-English and Urdu-English tracks with additional speakers to improve the reliability of the smaller subcategories (e.g., the six-clip sentence-level category in Track 3); replacing the heuristic language-misidentification detector (Section 4) with a validated language-identification model to produce more precise misidentification estimates; investigating directly why automatic language detection outperforms forced decoding specifically on the HiKE code-switched benchmark, since this study establishes the effect but does not fully explain its mechanism; and extending the benchmark to additional multilingual ASR systems beyond Whisper, to test whether the failure modes documented here are Whisper-specific or shared more broadly across large-scale multilingual ASR architectures.

---

## 10. Conclusion

This paper presented AccentBench, a four-track empirical evaluation of Whisper ASR across accent-stratified performance, likely language misidentification, Hindi-English and Korean-English code-switching, and language conditioning for Urdu and Korean. By separating these failure modes rather than relying on a single aggregate WER, the study found that a specific accent group's elevated error rate is substantially explained by output consistent with language misidentification rather than pure acoustic difficulty, that forcing the expected language helps substantially in some settings (Urdu, forced-English recovery for the flagged Singaporean subset) and actively hurts in others (Korean-English code-switching), and that code-switching difficulty varies systematically with switching granularity and topical domain. These findings support disaggregated, failure-mode-specific evaluation as a complement to aggregate WER when assessing multilingual ASR systems for deployment in accented, multilingual, and code-switching-heavy contexts.

---

## References

[1] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *arXiv preprint arXiv:2212.04356*.

[2] Shi, X., Yu, F., Lu, Y., Liang, Y., Feng, Q., Wang, D., Qian, Y., & Xie, L. (2021). The Accented English Speech Recognition Challenge 2020: Open Datasets, Tracks, Baselines, Results and Methods. In *ICASSP 2021 - IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 6918-6922).

[3] Zuluaga-Gomez, J., Ahmed, S., Visockas, D., & Subakan, C. (2023). CommonAccent: Exploring Large Acoustic Pretrained Models for Accent Classification Based on Common Voice. In *Proc. Interspeech 2023* (pp. 5291-5295).

[4] Myers-Scotton, C. (1993). *Duelling Languages: Grammatical Structure in Codeswitching*. Oxford University Press.

[5] Diwan, A., Vaideeswaran, R., Shah, S., Singh, A., Raghavan, S., Khare, S., Unni, V., Vyas, S., Rajpuria, A., Yarra, C., Mittal, A., Ghosh, P. K., Jyothi, P., Bali, K., Seshadri, V., Sitaram, S., Bharadwaj, S., Nanavati, J., Nanavati, R., & Sankaranarayanan, K. (2021). MUCS 2021: Multilingual and Code-Switching ASR Challenges for Low Resource Indian Languages. In *Proc. Interspeech 2021* (pp. 2446-2450).

[6] Paik, G., Kim, Y., Lee, S., Ahn, S., & Kim, C. (2026). HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition. In *Findings of the Association for Computational Linguistics: EACL 2026* (pp. 673-681).

[7] Ojo, J., Kamel, Z., & Adelani, D. I. (2025). DIVERS-Bench: Evaluating Language Identification Across Domain Shifts and Code-Switching. *arXiv preprint arXiv:2509.17768*.

[8] Morris, A., Maier, V., & Green, P. (2004). From WER and RIL to MER and WIL: Improved Evaluation Measures for Connected Speech Recognition. In *Proc. Interspeech 2004*.

[9] Jitsi. jiwer: Similarity Measures for Automatic Speech Recognition Evaluation. https://github.com/jitsi/jiwer

[10] Haq, A. N., Zhu, Z., Hu, J., He, C., & Xie, L. (2026). UrduSpeech: A 156-Hour Urdu Speech Corpus with 12-Dimension Paralinguistic Annotations. *arXiv preprint arXiv:2605.17846*.

[11] Jo, L., & Lee, W. (2018). Zeroth-Korean: An Open-Source Korean Speech Corpus. Available via OpenSLR, resource 40. http://www.openslr.org/40/
