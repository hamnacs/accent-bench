# AccentBench: A Failure-Mode-Oriented Evaluation of Accent, Language Identification, and Code-Switching in Whisper ASR

**Hamna Masood**

---

## Abstract

Automatic speech recognition (ASR) systems are increasingly deployed in multilingual and code-switched settings, yet standard evaluation practice still relies heavily on aggregate Word Error Rate (WER) over monolingual, native-accented benchmarks. This masks failure modes that matter in practice: likely language misidentification, loss of embedded-language tokens under code-switching, and inconsistent behavior across accents. This paper presents AccentBench, a four-track exploratory study of OpenAI's Whisper ASR system (base and small checkpoints) across accent-stratified performance, likely language misidentification, Hindi-English and Korean-English code-switching, and language-conditioning effects for Urdu and Korean. The study uses 600 sampled accented-English utterances, 100 sampled Hindi-English code-switched utterances (a proxy for, not a measurement of, Urdu-English speech), 100 Korean-English code-switched clips evaluated under four experimental conditions, and 180 language-conditioning clips (80 matched Urdu, 100 Korean). On these small samples, we find that (1) on the HiKE Korean-English benchmark, Whisper-small's WER under automatic language detection was 32.3% lower than under forced Korean (0.4209 versus 0.6216), which is equivalent to a 47.7% relative increase when Korean is forced, and the difference was concentrated in clips where automatic decoding selected English; (2) Singaporean English shows output patterns consistent with language misidentification at a higher rate (13-14%) than five other accent groups (0-1%), and forcing English on the 14 flagged clips reduced mean WER from 1.091 to 0.242, which is consistent with a substantial language-selection component but does not account for the whole accent gap; (3) language conditioning has a language- and dataset-dependent effect, with relative WER reductions of up to 53.0% for Urdu versus 0.0% to 0.23% for Korean (Zeroth Korean); and (4) within the HiKE sample, mean WER differed descriptively by switching granularity (phrase level highest) and by topical domain, although these subgroups are small and unbalanced. Paired bootstrap confidence intervals are reported for six comparisons, including a Hindi-English Base-versus-Small comparison whose interval includes zero. We release the available benchmark scripts, raw per-clip predictions, and analysis code, document where exact reproducibility is not possible, and discuss the limitations of a small-scale, exploratory study of this kind.

---

## 1. Introduction

Whisper [1] and similar large-scale multilingual ASR systems are trained on hundreds of thousands of hours of weakly supervised audio, and their ability to recognize many languages makes it easy to assume they will also handle a wide range of accents well. In practice, though, some of Whisper's errors appear to trace back to the model failing to identify the spoken language correctly in the first place, rather than to the accent itself.

This question grew out of my work on an AI-enabled hospital information management system, where I built a voice-driven booking agent. Speech in that setting rarely stayed in one language: Urdu and English were routinely mixed within the same sentence, a pattern common in everyday Pakistani speech. During this mixing, the system would sometimes fail to recognize the language at all, lock onto one language and transcribe the rest as if it were that language, or miss parts of the utterance entirely. These looked at first like isolated errors, but they raised a question worth testing directly rather than dismissing as noise from microphone quality or background conditions.

AccentBench evaluates four research questions:

1. **Accent robustness.** How does Whisper's transcription performance vary across English accent groups?
2. **Language identification.** How accurately does Whisper identify the spoken language, and can misidentification explain some transcription errors?
3. **Code-switching.** How does Whisper handle speech where speakers switch between languages mid-utterance, and does the difficulty vary by switching granularity (word, phrase, or sentence level)?
4. **Language conditioning.** Does explicitly specifying the expected language improve transcription accuracy, and is that effect consistent across languages?

A single WER score cannot distinguish between these causes. The same poor score can mean the model identified the language correctly but struggled with the accent, misidentified the language outright, dropped words when the speaker switched into a second language, or simply lacked the correct language hint that would have fixed the error, as with the Urdu case in this study. The Korean-English case shows the opposite: there, supplying the hint made results worse instead of better (Section 5.3). Knowing that a WER number is bad is not enough to fix a real deployed system; knowing which of these it actually is, is. The central thesis of this paper is that aggregate WER can hide qualitatively different multilingual ASR failure modes, and that the appropriate mitigation may depend on which failure mode is actually present.

Concretely, this paper makes four contributions:

1. A targeted evaluation that separates accent-driven acoustic difficulty from output patterns consistent with language-selection failure. In this benchmark, forcing English substantially reduces WER on the flagged Singaporean-English subset, which is consistent with a language-selection component to that accent group's elevated error rate, although it does not explain the whole gap (Section 5.1).
2. Evidence that language conditioning does not have a uniform effect: forcing the expected language helps substantially for Urdu and for the flagged Singaporean-English subset, yet is neutral for Zeroth Korean and harmful on the HiKE Korean-English code-switching benchmark (Sections 5.3-5.4).
3. Descriptive evidence that, within the HiKE sample, WER differs across switching granularity and topical domain, and that an English-token-preservation metric reveals a script-conversion failure mode that overall WER understates (Section 5.3). These subgroup differences are small-sample observations, not established effects.
4. A collection of benchmark scripts, raw per-clip predictions, and analysis code across all four tracks, released together with an explicit account of the reproducibility gaps that remain (Section 4.1).

The HiKE dataset itself, including its audio, transcripts, and word/phrase/sentence switching-level labels, is an existing resource [6]; this paper's contribution with respect to Track 3 is the Whisper evaluation built on top of it, specifically the automatic-versus-forced-language comparison, the granularity- and domain-stratified analysis, and the English-token-preservation metric, none of which are part of the original HiKE release.

---

## 2. Related Work

**Multilingual and robust ASR.** Whisper [1] is trained with large-scale weak supervision on 680,000 hours of multilingual and multitask audio-transcript pairs and demonstrates strong zero-shot generalization across many existing benchmarks. Because Whisper performs joint language identification and transcription in a single decoder, errors in the former can propagate into the latter, a coupling that motivates Research Question 2 above.

**Accented speech recognition.** Accent robustness has been studied both as a data problem and a modeling problem. The Accented English Speech Recognition Challenge (AESRC2020) [2] released a labeled multi-accent English corpus and established accent recognition and accented transcription as a shared task, spurring subsequent architectural work on accent-invariant and accent-aware ASR. The CommonAccent recipe [3] built a large accent-labeled resource from Common Voice, covering sixteen English accent categories with an accent-classification model; a derived sample of this resource (six accent groups: German, Hong Kong, Southern African, Filipino, South Asian, and Singaporean English) forms Track 1 of AccentBench.

**Code-switching ASR.** Code-switching, the alternation between two or more languages within a single utterance or conversation, has long been studied as a linguistic phenomenon [4] and more recently as an ASR benchmarking problem. The MUCS 2021 shared task [5] released Hindi-English and Bengali-English code-switched speech alongside six monolingual low-resource Indian languages, with the explicit goal of measuring ASR performance under code-switching rather than assuming it degrades gracefully from monolingual performance. HiKE [6] is a more recent, hierarchical Korean-English code-switching benchmark that labels each utterance's switching granularity (word, phrase, or sentence level) in addition to providing loanword annotations, enabling exactly the kind of granularity-stratified analysis used in Track 3 of this paper. Both resources motivate treating code-switching as a first-class evaluation axis rather than an edge case of standard WER evaluation.

**Language identification in multilingual and code-switched speech.** Because many multilingual ASR systems, including Whisper, perform implicit language identification before or during decoding, identification errors are an underexplored source of downstream transcription error. Recent work has begun evaluating language identification specifically under domain shift and code-switching conditions [7], reinforcing that identification accuracy should be measured separately from transcription accuracy. AccentBench's forced-language experiments (Tracks 1, 3, and 4) are a direct, low-cost way to probe this effect: comparing automatic detection against an explicitly forced expected language indicates how much of a system's error can be changed by specifying the language, although it does not by itself show what the internal language decision was.

**Evaluation metric.** Word Error Rate, computed as the minimum edit distance between reference and hypothesis transcripts normalized by reference length, remains the standard ASR evaluation metric [8] despite known limitations discussed further in Section 8. WER values in this paper were computed with the `jiwer` Python package [9] for Tracks 1 and 2 and for the Urdu part of Track 4, and with an equivalent whitespace-token edit-distance implementation for Track 3 and the Korean part of Track 4 (Section 4).

---

## 3. AccentBench Benchmark

AccentBench consists of four tracks, summarized in Table 1.

**Table 1: Benchmark tracks**

| Track | Research question | Dataset | Size |
|---|---|---|---|
| 1 | Accent-stratified performance and likely language misidentification | CommonAccent-derived sample [3] (via DTU54DL/common-accent) | 600 utterances, 6 accent groups (591 Base / 589 Small valid) |
| 2 | Hindi-English code-switching (proxy for Urdu-English) | MUCS 2021 [5] | 100 utterances (99 Base / 98 Small valid; 97 matched) |
| 3 | Korean-English code-switching | HiKE [6] | 100 clips, 4 conditions, 400 transcriptions |
| 4 | Language conditioning (Urdu, Korean) | UrduSpeech US-benchmark [10]; Zeroth Korean [11] | 80 matched Urdu clips, 100 Korean clips |

**Track 1 (Accent).** Six accent groups, each with 100 sampled utterances: German (non-native), Hong Kong English, Southern African, Filipino, South Asian (combining Indian, Pakistani, and Sri Lankan English), and Singaporean English. Each utterance was transcribed with Whisper-base and Whisper-small under automatic language detection, and WER was computed against the reference transcript. Some predictions failed or were empty and have no recorded WER, so 591 Whisper-base and 589 Whisper-small predictions contribute to the means reported below.

**Track 2 (Hindi-English code-switching).** 100 utterances sampled from the MUCS 2021 Hindi-English code-switching subtask [5], transcribed with Whisper-base and Whisper-small under automatic language detection. After excluding predictions without a valid WER, 99 Whisper-base and 98 Whisper-small utterances remain, and 97 utterances are valid for both models. This track is a linguistically related **proxy** for the Urdu-English case, because a suitable public Urdu-English code-switching corpus was not identified when the original benchmark was built. It is not a measurement of Urdu-English speech (see Section 3.1 and Section 8 for the caveat that Hindi and Urdu, while closely related at the spoken level, are not interchangeable).

**Track 3 (Korean-English code-switching).** 100 clips from the HiKE benchmark [6], which provides word-, phrase-, and sentence-level code-switching annotations and eight topical domains. Each clip was transcribed under four conditions: {Whisper-base, Whisper-small} times {automatic language detection, forced Korean}, producing 400 total transcriptions.

**Track 4 (Language conditioning).** For Urdu, 100 Standard Urdu clips were randomly sampled (seed 42) from the UrduSpeech US-benchmark set [10], which was independently identified after Track 2 was built and is now the primary Urdu-specific resource used in AccentBench. The final analysis uses the 80 matched clips for which all four experimental conditions produced a valid WER value. For Korean, 100 clips were sampled 10 per speaker from the 10-speaker Zeroth Korean test set [11]. Both were evaluated with Whisper-base and Whisper-small under automatic and forced-language conditions.

### 3.1 A Note on the Urdu-English Proxy

An earlier design decision in this project used Hindi-English code-switched speech (Track 2, MUCS 2021) as a proxy for Urdu-English, since no suitable public Urdu-English code-switching corpus was identified at the time. This choice is linguistically motivated: Hindi and Urdu are mutually intelligible at the spoken level and share substantial phonological and syntactic structure, differing mainly in script and register. It is, however, an approximation, not a substitute. Track 2 is not genuine Urdu-English code-switching, and its results should not be presented as direct evidence about Urdu-English ASR. The dedicated Urdu resource [10] used in Track 4 does not replace it, because Track 4 tests language conditioning on Urdu speech rather than Urdu-English code-switching. Genuine Urdu-English code-switching therefore remains incompletely evaluated (Sections 8 and 9).

---

## 4. Experimental Setup

**Models.** Two OpenAI Whisper checkpoints were evaluated: `base` and `small`, using the open-source `openai-whisper` Python package. All inference was performed locally on CPU.

**Metric.** Word Error Rate (WER) is computed as (S + D + I) / N, where S, D, and I are substitutions, deletions, and insertions, and N is the reference word count. Lower is better. The implementation differs by track:

- Tracks 1 and 2, and the Urdu part of Track 4, use `jiwer` [9]. In Track 1 (and in its forced-English re-run) both reference and hypothesis are lowercased first. The Track 2 and Urdu scripts apply no case or punctuation normalization.
- Track 3 (HiKE) and the Korean part of Track 4 (Zeroth Korean) use a whitespace-token Levenshtein implementation in the inference scripts, again with no case or punctuation normalization. On the HiKE predictions this implementation reproduced `jiwer` values to floating-point precision.

Because punctuation, numerals, and spacing are not normalized, absolute WER values are sensitive to orthographic conventions, particularly for Korean, where whitespace-delimited tokens carry attached particles and punctuation. Comparisons between conditions within a track use the same procedure, but absolute values should not be compared across tracks as if they were measured identically. Both mean and median WER are reported, since WER distributions in small samples can be skewed by a small number of near-total-failure transcriptions. A WER above 1.0 does not indicate an invalid metric; it occurs when the combined number of substitutions, deletions, and insertions exceeds the number of reference words, which is possible whenever the hypothesis is substantially longer or more disordered than the reference.

**Uncertainty estimation.** Six paired comparisons are reported with bootstrap confidence intervals:

1. Track 1: Singaporean original minus forced English (n = 14).
2. Track 2: Hindi-English proxy, Whisper-base minus Whisper-small (n = 97 matched utterances).
3. Track 3: HiKE Whisper-small, forced Korean minus automatic (n = 100).
4. Track 4: Urdu Whisper-base, automatic minus forced (n = 80).
5. Track 4: Urdu Whisper-small, automatic minus forced (n = 80).
6. Track 4: Korean (Zeroth) Whisper-small, automatic minus forced (n = 100).

For each comparison, a paired percentile bootstrap 95% confidence interval was computed over the per-clip WER differences between conditions. Each interval used 10,000 bootstrap resamples with replacement, `np.random.RandomState(42)`, and the 2.5th and 97.5th percentiles of the resampled means. This quantifies uncertainty in the observed mean per-clip difference without assuming a parametric distribution. The intervals are clip-level, not speaker-level: they do not model clustering of clips within speakers. The implementation is `src/bootstrap_ci.py`, which writes `results/bootstrap_ci_results.csv`; all six intervals are tabulated in Appendix A.

**Language conditioning.** For Tracks 1, 3, and 4, two decoding conditions were compared: automatic language detection, in which Whisper infers the spoken language from the audio itself, and forced-language decoding, in which the expected language is supplied explicitly as a decoding parameter. Where forcing the language reduced WER (Tracks 1 and 4 Urdu), relative improvement is computed as (WER_automatic - WER_forced) / WER_automatic times 100. Where forcing increased WER (Track 3), the direction of the comparison and the denominator are stated explicitly: the relative increase from forcing is (WER_forced - WER_automatic) / WER_automatic, and the equivalent relative reduction obtained by using automatic detection is (WER_forced - WER_automatic) / WER_forced. These are the same difference expressed against different baselines and are not interchangeable.

**Language misidentification heuristic (Track 1).** A transcription was flagged as a likely language-misidentification failure if the Whisper output contained non-Latin script characters (Cyrillic, CJK, Hangul, or similar Unicode ranges) or matched a small set of common Malay/Indonesian function words, under the assumption that correctly transcribed accented English should not contain either. This is an output-based heuristic, not a ground-truth language label, and undercounts misidentification into other Latin-script languages; see Section 8. The Track 1 inference scripts did not record Whisper's own detected-language label. (The Track 2, Track 3, and Track 4 automatic runs did record it, and those labels are used where noted.)

**Matched-clip comparison (Track 4).** Some clips failed during individual inference runs; in the Urdu result files every row without a recorded WER has an empty prediction. To keep the automatic-versus-forced comparison fair, the final Urdu and Korean comparisons in Track 4 include only clips for which all four experimental conditions (two models times two language settings) produced a valid WER value, yielding 80 matched Urdu clips and 100 matched Korean clips. The Track 2 paired comparison similarly uses only utterances with a valid WER for both models (97).

**Sampling methodology (Track 3).** The 100 HiKE clips evaluated were drawn as a simple, unstratified random sample from the benchmark's full 1,121-clip test split, using `pandas.Series.sample(n=100, random_state=42)` over the full index range before any model was run. The sample was not stratified by code-switching level or topical domain, and the resulting distribution across levels (41 word, 53 phrase, 6 sentence) and domains (7 to 16 clips each) reflects the natural composition of the underlying test split rather than a designed balance. This is disclosed explicitly here because Section 5.3.1 and 5.3.2 report WER broken down by these same categories, and the reader should weigh the smaller categories (in particular the six sentence-level clips) accordingly. The sample was fixed once, before any transcription was produced, so no clip selection was informed by model outputs. The 1,121-clip size is the split length implied by the sampling procedure: it is the only split size consistent with the recorded sample indices, but the exact HiKE dataset revision downloaded was not recorded.

### 4.1 Reproducibility

All inference used the open-source `openai-whisper` Python package with CPU-only execution and no GPU acceleration. `fp16=False` was passed explicitly in the Track 1, Track 2, and Urdu inference scripts; the HiKE and Zeroth Korean scripts did not pass it, but on CPU the package uses fp32 regardless. Decoding used the package's default temperature-fallback schedule and default beam/greedy search settings; no custom beam size, temperature, or best-of value was set in any script, and no PyTorch random seed was fixed. Whisper decoding with temperature fallback can therefore be nondeterministic. This was observed directly: the Whisper-small automatic Urdu condition was run twice, and on the 92 clips with a valid WER in both runs, 8 predictions differed. The mean WER over the 80 matched clips was approximately 0.824 in the original run and 0.8473 in the rerun used for the results in this paper. The reported Urdu numbers therefore correspond to a specific run and should not be treated as exactly repeatable.

Where a dataset required random sampling, a fixed seed of 42 was used (`random.seed(42)` for the Track 1, Track 2, and Urdu sampling scripts; `random_state=42` for the HiKE sample and the per-speaker Zeroth Korean sample), so the specific subsets evaluated are deterministic given the same source data. Source-data versions matter here: the exact HiKE dataset revision was not recorded, so the HiKE subset cannot be guaranteed to be identical if the dataset has since changed.

The package versions used in the original experiments were not recorded or pinned, and `requirements.txt` lists the required packages without version pins. The author's recollection is that the `openai-whisper` release in use was approximately 20250625; this was not logged by the experiment scripts and cannot be verified from the repository. Anyone attempting to reproduce these exact numbers should pin package versions, since Whisper's decoding behavior has changed across releases in the past.

The original Track 1 Whisper-base inference script is not preserved in the current repository history. The released Whisper-base result file (`results/wer_results_full.csv`) is retained, as is the clip sample it was computed on, which is traceable to `src/explore_data.py` (seed 42), and its WER values are consistent with the scoring in the current scripts when recomputed. However, the historical Base run cannot be regenerated from the repository alone, and no claim of script-level reproducibility is made for it. The current `src/run_inference.py` is the later Whisper-small run.

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

Scaling from Whisper-base to Whisper-small reduced WER for every accent group except Singaporean English, where performance was essentially flat (0.410 to 0.410). Note that the South Asian category here combines Indian, Pakistani, and Sri Lankan English; it is not a Pakistani-English-specific result. Singaporean English stands out as an outlier: its WER is well above that of every other accent group under both model sizes (0.410, against 0.130 to 0.252 for the others), and it is the only group that does not improve with the larger model. Section 5.1.1 investigates why.

![Mean WER by accent group, Whisper-base versus Whisper-small. Singaporean English is the only group that does not improve with the larger model.](figures/fig5_accent_wer.png)

#### 5.1.1 Likely Language Misidentification

A subset of accented-English clips triggered a more severe failure than ordinary mistranscription: Whisper produced output consistent with a non-English decoding decision, rather than an ordinary transcription error. This is inferred from output characteristics (Section 4), not observed directly from Whisper's internal language-identification decision, which was not captured in the Track 1 scripts; the rate reported below should be read as a heuristic proxy rather than a ground-truth misidentification rate. Table 3 reports this proxy rate per accent group, using valid predictions as the denominator.

**Table 3: Language-misidentification heuristic rate**

| Accent | Whisper-base | Whisper-small |
|---|---:|---:|
| Singaporean English | 14.3% | 13.1% |
| Filipino | 1.0% | 1.0% |
| South Asian | 1.0% | 0.0% |
| German | 0.0% | 0.0% |
| Hong Kong English | 0.0% | 1.0% |
| Southern African | 0.0% | 1.0% |

![Language-misidentification heuristic rate by accent group. Singaporean English is at 13 to 14%, against 0 to 1% for every other group in this sample.](figures/fig6_misid_rate.png)

Singaporean English triggers the heuristic at 13 to 14%, against 0 to 1% for the other groups, and the rate barely moves between model sizes (14 of 98 valid clips for Whisper-base, 13 of 99 for Whisper-small). The flagged clips are not the same set, however: only 5 clips are flagged under both models, so similar rates should not be read as the same clips failing under both. That pattern points toward a possible explanation for the anomaly in Table 2: Singaporean English's elevated, scale-resistant WER may be driven in part by a language-selection failure rather than by acoustic difficulty alone. We do not treat this as proven; the forced-English experiment below tests it more directly, though it too works from the same flagged subset rather than an independent confirmation.

#### 5.1.2 Forced-English Recovery

To test this explanation, the 14 Singaporean-English clips flagged by the heuristic under Whisper-base were re-transcribed with English explicitly forced as the decoding language. Table 4 reports the result.

**Table 4: Forced-English recovery for flagged Singaporean clips (n = 14)**

| Condition | Mean WER |
|---|---:|
| Original (automatic detection) | 1.091 |
| Forced English | 0.242 |

Forcing English reduced mean WER by 77.8% relative to the original automatic-detection transcriptions. The paired per-clip reduction was 0.8489 WER (95% bootstrap CI [0.7524, 0.9430], n = 14), a confidence interval that excludes zero by a wide margin despite the small sample, with all fourteen paired differences non-negative, indicating that the observed recovery was not produced by a single unusually large improvement. Two of the fourteen clips went from a WER at or above 1.0 (effectively a failed transcription) to a perfect transcription (WER = 0.0) once English was forced. Flagged clips were selected because their output looked non-English, so a high original WER on this subset is expected by construction; the informative result is how much of it is recoverable.

On this small sample (n = 14), the result is consistent with a substantial language-selection component to Whisper's difficulty with Singaporean-accented English in this dataset: specifying English substantially improves transcription on the flagged clips. It does not account for the whole gap between Singaporean English and the other groups. Excluding the 14 flagged clips, the group's Whisper-base mean WER is 0.297; replacing each flagged clip's WER with its forced-English WER gives about 0.289, which is lower than 0.410 but still above the 0.169 to 0.252 range of the other groups. Because the language-misidentification measure is heuristic and Whisper's internal language decision was not recorded, this experiment demonstrates recoverability under forced English rather than directly identifying the model's internal failure mechanism. A larger, independently sampled set of Singaporean-English clips would be needed before treating this as a general property of Whisper rather than a pattern observed in this specific benchmark.

### 5.2 Hindi-English Code-Switching (Track 2)

This track uses Hindi-English speech as a proxy; it is not Urdu-English code-switching (Section 3.1). Table 5 reports the descriptive means and medians over the valid predictions for each model (99 for Whisper-base, 98 for Whisper-small).

**Table 5: Hindi-English code-switching WER**

| Model | Valid clips | Mean WER | Median WER |
|---|---:|---:|---:|
| Whisper-base | 99 | 1.150 | 1.000 |
| Whisper-small | 98 | 1.058 | 0.804 |

Both models show mean WER above 1.0, meaning the number of transcription errors on average exceeds the number of words in the reference. A small number of extreme outputs can dominate such means, which is why medians are reported alongside. These values are substantially higher than the WER observed on any single accent group in Track 1 (all of which stayed below 0.5), but the two tracks use different datasets, speech conditions, and tasks, so this is a descriptive contrast and not a controlled comparison.

The two valid sets are not identical (the Whisper-base file has two utterances with no valid WER that Whisper-small does not, and vice versa for one), so the model comparison uses the 97 utterances that are valid for both. On these 97 matched utterances, the paired mean difference (Whisper-base WER minus Whisper-small WER) was 0.0821, with a 95% bootstrap CI of [-0.0270, 0.1771]. Because the interval includes zero, the paired bootstrap does not clearly distinguish the observed mean difference from zero, and a model-size difference on this track should not be treated as established. This track is best read as an exploratory baseline: it indicates that Hindi-English code-switching is difficult for Whisper in this sample, but does not probe why in the same depth as Tracks 1, 3, and 4 do for their respective failure modes.

### 5.3 Korean-English Code-Switching (Track 3)

All 100 HiKE clips successfully completed all four experimental conditions, yielding 400 transcription results.

**Table 6: Overall Korean-English code-switching WER**

| Configuration | Mean WER | Median WER |
|---|---:|---:|
| Whisper-small + automatic | 0.4209 | 0.3333 |
| Whisper-base + automatic | 0.5648 | 0.5714 |
| Whisper-small + forced Korean | 0.6216 | 0.6250 |
| Whisper-base + forced Korean | 0.6729 | 0.6340 |

Two effects are visible in Table 6. Scaling from Whisper-base to Whisper-small under automatic detection reduces mean WER by 25.5% relative to the Whisper-base result. Forcing Korean explicitly *increases* WER relative to automatic detection. Using automatic-detection WER as the denominator, forcing Korean raises WER by 47.7% for Whisper-small (0.4209 to 0.6216) and by 19.1% for Whisper-base (0.5648 to 0.6729). Equivalently, using forced-Korean WER as the denominator, automatic detection lowers WER by 32.3% for Whisper-small and by 16.1% for Whisper-base. For Whisper-small, the paired per-clip increase from forcing Korean was 0.2007 WER (95% bootstrap CI [0.1294, 0.2736], n = 100), with the interval entirely above zero. This runs opposite to the Track 1 forced-English recovery result and the Track 4 Urdu result (Section 5.4). We do not read it as evidence that automatic detection beats forcing in general; it is specific to this Korean-English code-switched sample.

**Where the difference comes from.** The per-clip results show that the Whisper-small difference is concentrated in a subset of clips. Under automatic decoding, Whisper-small labelled 70 of the 100 clips as Korean, 29 as English, and 1 as Portuguese. On the 29 English-labelled clips, mean WER was 0.22 under automatic decoding and 0.86 when Korean was forced, and forcing Korean was worse on 27 of them. These 29 clips account for 0.1858 of the 0.2007 mean paired difference. On the 70 Korean-labelled clips, the means were 0.4913 (automatic) and 0.5078 (forced Korean), the WER was exactly equal on 67 of the 70, and the predicted text was identical on 65 of them. In other words, forcing Korean made little difference where Whisper had already chosen Korean, and a large difference where it had chosen English.

**Table 6b: Whisper-small on HiKE by automatic language label**

| Automatic label | Clips | Mean WER (automatic) | Mean WER (forced Korean) |
|---|---:|---:|---:|
| Korean | 70 | 0.4913 | 0.5078 |
| English | 29 | 0.2196 | 0.8603 |
| Portuguese | 1 | 1.3333 | 1.6667 |

This is consistent with the Korean-language Track 4 result, where forcing Korean changed nothing on clips Whisper had already labelled Korean. It supports a plausible interpretation, which we present as a hypothesis rather than a tested mechanism: Whisper's automatic detection may be responding to whichever language dominates a given clip, so forcing Korean on a clip whose speech is largely English could push the decoder away from the English output it would otherwise produce. The analysis shows where the difference occurs; it does not test why, and the automatic language label is a model output, not an independently verified ground-truth language.

![Overall WER across the four experimental conditions in Track 3 (n = 100 clips per bar). Automatic detection has lower mean WER than forced Korean for both model sizes.](figures/fig1_hike_overall_wer.png)

#### 5.3.1 Effect of Code-Switching Granularity

The HiKE benchmark labels each clip's code-switching level. Table 7 reports WER by level under the best-performing configuration (Whisper-small, automatic detection).

**Table 7: WER by code-switching level (Whisper-small, automatic)**

| Level | Clips | Mean WER |
|---|---:|---:|
| Word | 41 | 0.3458 |
| Sentence | 6 | 0.4107 |
| Phrase | 53 | 0.4802 |

In this sample, phrase-level code-switching, where a contiguous multi-word phrase switches language within the utterance, had the highest mean WER, followed by sentence-level and then word-level switching. This ordering is a sample-level finding, not a claim about code-switching difficulty in general. The sentence-level result is based on only six clips and is included for completeness rather than as a robust estimate. In addition, the single level label is a coarse grouping: 23 of the 53 phrase-level clips carry more than one switching label in the HiKE metadata, so the phrase group is not a pure category.

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

In this sample, travel and culture had the lowest mean WER and business and everyday conversation the highest. With only 7 to 16 clips per domain, and with the domain ranking read off after the fact among eight categories, we stop short of claiming this says something general about Korean ASR and domain difficulty. These are descriptive patterns in this specific, unbalanced sample, not controlled domain effects.

#### 5.3.3 English Token Preservation

Overall WER treats a code-switched utterance as a single sequence and does not distinguish whether an error involved losing an embedded English term specifically. To capture this, English-token preservation was measured directly. For each clip whose reference contains Latin-script (English) tokens, each reference English token is counted as preserved if its letters appear, case-insensitively, within the Whisper hypothesis. The rate is pooled over all reference English tokens (715 tokens in 91 of the 100 clips; nine clips contain no English). The metric detects Latin-script presence, not correct position or meaning, and because it is a substring match it is lenient for very short tokens. A stricter whole-word match gives the same ordering of the four conditions.

**Table 9: English token preservation**

| Configuration | Preservation rate |
|---|---:|
| Whisper-small + automatic | 72.6% |
| Whisper-base + automatic | 56.9% |
| Whisper-base + forced Korean | 36.4% |
| Whisper-small + forced Korean | 27.7% |

The pattern mirrors the overall WER result in Table 6: automatic detection yields higher token preservation than forced Korean, and the gap is larger for token preservation than for WER (Whisper-small automatic preserves 72.6% of English tokens versus 27.7% under forced Korean, a 44.9 percentage-point gap). Qualitatively, a common failure mode is that embedded English terms are rendered as Korean-script phonetic approximations of their pronunciation rather than being transcribed in Latin script. As an illustration of the failure mode, in one clip a Korean sentence contains the embedded English terms "pull request" and "test case" (the third term in the sentence, a word for "check", is already written in Korean script in the reference). Both models rendered the two English terms in Korean script, while the surrounding Korean content remained largely intact. In this particular clip the conversion occurred under automatic decoding as well as forced Korean, so it illustrates the kind of error, not the contrast between conditions. This illustrates why WER alone, which does not distinguish a script-conversion error from an equivalently-scored acoustic mistranscription, can understate the practical severity of this failure mode: a phonetically converted English term is often unusable downstream (e.g., in a transcript intended for search or command extraction) even though it may contribute a similar WER penalty to an ordinary substitution error.

![Proportion of reference English tokens surviving as recognizable English tokens in the Whisper hypothesis, by model size and language-detection condition.](figures/fig3_token_preservation.png)

### 5.4 Language Conditioning: Urdu versus Korean (Track 4)

**Table 10: Language-conditioning effect**

| Language | Model | Automatic WER | Forced WER | Relative improvement |
|---|---|---:|---:|---:|
| Urdu | Whisper-base | 0.6855 | 0.5821 | 15.1% |
| Urdu | Whisper-small | 0.8473 | 0.3984 | 53.0% |
| Korean | Whisper-base | 0.4737 | 0.4726 | 0.23% |
| Korean | Whisper-small | 0.3791 | 0.3791 | 0.00% |

Relative improvement is (automatic - forced) / automatic. Forcing Urdu produces a substantial improvement for both model sizes, and the effect is far larger for Whisper-small (53.0% relative reduction, paired difference 0.4489 WER, 95% bootstrap CI [0.3653, 0.5361], n = 80) than for Whisper-base (15.1% relative reduction, paired difference 0.1034 WER, 95% bootstrap CI [0.0366, 0.1663], n = 80); both intervals exclude zero. Forcing Korean, by contrast, produces almost no change for Whisper-base (0.23%, driven entirely by one clip: 99 of 100 predictions were identical) and produces exactly zero change for Whisper-small: automatic and forced decoding produced identical predictions on all 100 Korean clips (verified string-for-string, not merely equal WER, by directly comparing the two prediction columns row by row). Whisper's automatic language label was Korean on all 100 clips, consistent with no clip being affected by forcing. Because every prediction was identical, the paired difference for Whisper-small Korean is exactly 0.0000 with a degenerate (zero-width) bootstrap interval [0.0000, 0.0000], as expected when every paired difference is zero.

![Mean WER under automatic versus forced-language decoding, Urdu and Korean, both model sizes. Urdu shows a large, model-size-dependent improvement from forcing; Korean shows essentially none.](figures/fig4_language_forcing_urdu_korean.png)

**Automatic language labels for Urdu.** The automatic runs recorded Whisper's detected-language label for each clip. For Whisper-small, the automatic label was Hindi on 54 of the 80 matched clips (67.5%) and Urdu on 25, with one clip labelled English. All 54 Hindi-labelled outputs were written in Devanagari script, and their mean WER was 1.046. Because the references are in Urdu script, a Devanagari output scores close to 1.0 by script mismatch alone, so this is not evidence about acoustic recognition quality. On the 25 Urdu-labelled clips, automatic and forced predictions were identical (25 of 25). The benefit of forcing Urdu for Whisper-small was therefore concentrated in clips where automatic decoding chose a different language. The label is a model output and not an independently verified language classification.

Whisper-base behaved differently. It labelled 43 of the 80 clips as Hindi, but, unlike Whisper-small, it wrote most of those outputs in Urdu script (26 of the 43; 13 were Latin script and 3 Devanagari), and its Urdu-labelled clips (30) were predicted identically with and without forcing on 28 of 30. The label alone therefore does not determine the output script.

An additional model-size interaction is notable within Urdu: under automatic detection, Whisper-base (0.6855) outperforms Whisper-small (0.8473), the opposite of the ordering seen in Tracks 1 and 3. Once Urdu is forced, this reverses and Whisper-small (0.3984) substantially outperforms Whisper-base (0.5821). The language labels are consistent with this pattern: Whisper-small more often committed to Hindi labels with Devanagari output under automatic decoding (54 of 80 clips), even though it has the stronger acoustic model once given the expected language. This is an interpretation of output labels and scripts; it does not show that Whisper-small misidentifies Urdu acoustically, and the overall rate of non-Urdu labels is similar for the two models (55 of 80 for small, 50 of 80 for base), so the difference lies mainly in the output script.

The Urdu and Korean language-conditioning experiments used different datasets, speaker populations, and recording conditions (Section 3), so the Urdu-versus-Korean contrast in Table 10 should be read as evidence that language-conditioning effects are dataset- and language-dependent, not as a controlled, causal comparison isolating language identity as the sole variable. The 80 matched Urdu clips come from 23 speakers, with one speaker contributing 29 of them, so the clip-level confidence intervals should not be read as speaker-independent estimates.

---

## 6. Error Analysis

This section distinguishes two kinds of findings. WER, the language-misidentification heuristic rate, the forced-decoding comparisons, and the English-token-preservation metric (Sections 5.1-5.4) are measured findings, computed the same way across the full sample in each track, although the heuristic and the token metric are output-based proxies with the limitations described above. The transliteration and hallucination patterns below are exploratory observations: they were noticed during manual inspection of examples and a simple script-level check rather than measured with a validated automated detector across the full dataset, and should be weighted accordingly.

**Transliteration of embedded terms.** As in the Korean-English example in Section 5.3.3, English technical terms embedded in Hindi-English code-switched speech (Track 2) were frequently rendered as approximate Devanagari transliterations rather than preserved in Latin script. This preserves rough phonetic content while breaking the language boundary of the original utterance, a failure mode WER scores similarly to an unrelated substitution error despite being qualitatively different and arguably more disruptive for downstream use. As a rough, exploratory indication for Whisper-small: of the 66 evaluable utterances whose reference contains English words, 58 had no reference English word reproduced in Latin script in the prediction. Of those 58, 52 contained no Latin text at all, 6 contained some Latin text, and 56 contained Devanagari. This count is not a transliteration failure rate: it is an overlap check that treats any single matching Latin word as success, it does not verify that the missing words were transliterated rather than dropped or garbled, and it is not a validated population-level statistic.

**Hallucination under code-switching.** A distinct pattern, separate from transliteration, was content in the hypothesis that was not supported by the source audio at all, flagged heuristically when the prediction was substantially longer than the reference. Because both the transliteration and hallucination observations in this study relied partly on heuristic, manual, and semi-automated inspection of examples rather than a fully automated, validated detector, these findings should be treated as exploratory error-analysis observations rather than definitive population-level estimates of their frequency.

**Model size does not uniformly help.** Across all four tracks, moving from Whisper-base to Whisper-small improved mean WER in most conditions but not all: Singaporean-English misidentification (Table 3) was essentially unaffected by model size, the Hindi-English paired difference (Section 5.2) has a confidence interval that includes zero, and Urdu automatic-detection WER (Table 10) was actually worse for the small model than the base model. This indicates that "use a larger model" is not a reliable universal mitigation for the specific failure modes studied here; some failures appear to be more closely tied to language-selection behavior than to acoustic modeling capacity, and scaling the acoustic model does not necessarily fix a language-selection problem.

---

## 7. Discussion

The results across the four tracks are consistent with the thesis stated in Section 1: aggregate WER can hide qualitatively different failure modes, and the appropriate mitigation may depend on which one is present. The most surprising result in this study was Singaporean English: under automatic detection its WER was unexpectedly high, well above every other accent group. Investigating why showed that a share of its clips produced output consistent with a non-English decoding decision, triggering the language-misidentification heuristic; forcing English explicitly reduced mean WER from 1.091 to 0.242 (77.8% relative) on the flagged subset. What looked at first like a purely accent-driven acoustic problem appears, on this evidence, to include a substantial language-selection component, though the forced-decoding result demonstrates recoverability rather than directly confirming what Whisper's internal detector predicted for those clips, and the group remains the highest-WER accent even after accounting for the flagged clips. For anyone building multilingual or accent-inclusive speech interfaces with Whisper-class models, three things stand out from the results as a whole.

Aggregate WER can mask a language-selection failure that has a cheap, targeted fix. When a system knows, or can reasonably constrain, the expected language, forcing it (the Track 1 Singaporean-English case, the Track 4 Urdu case) can substantially beat trusting automatic detection.

But that fix is not a default you can apply blindly. The Track 3 Korean-English code-switching result shows automatic detection beating forced decoding, and the Track 4 Korean result shows no difference where Whisper already selected Korean. Korean-English code-switching contradicted my own expectations going into this study: I assumed forcing the correct language would improve or at minimum preserve performance, and instead on HiKE it raised WER by 47.7% relative for the small model, with the increase concentrated in clips where automatic decoding had selected English. A system builder cannot assume "always force the expected language" without testing it on their own language pair and use case first.

And in this sample, code-switching difficulty was not a single number. Switching granularity (Table 7) and topical domain (Table 8) differ descriptively within a single language pair, which suggests that a single aggregate WER for "a Korean-English code-switching system" can hide a range of performance depending on how heavily, and where, the code-switching happens in a given utterance. The sample is small and unbalanced, so these differences are hypotheses for larger studies, not established effects.

The main practical lesson I take from this project is that automatic language detection should not be assumed reliable for every user population without testing. A real system should evaluate language identification separately from transcription accuracy, and should be tested against the specific accents and code-switching patterns its actual users will produce, rather than judged on a single aggregate WER figure.

Taken together, these results support evaluating multilingual ASR along multiple separated axes, rather than a single overall accuracy number, particularly for systems intended to serve accented, multilingual, or code-switching-heavy user populations such as the hospital voice-booking context that originally motivated this project.

---

## 8. Limitations

AccentBench is intentionally small and exploratory, and its results should not be read as definitive measurements of Whisper's global multilingual or accent robustness, nor as results about multilingual ASR systems in general. Specific limitations include:

- **Sample sizes are small** relative to what would be needed for tight statistical confidence, particularly for the sentence-level code-switching category (6 clips, Table 7) and the per-domain breakdown (7-16 clips per domain, Table 8). Some valid-clip counts are below the sampled counts (Track 1: 591 and 589 of 600; Track 2: 99 and 98 of 100; Track 4 Urdu: 80 of 100).
- **The South Asian accent category is a composite** of Indian, Pakistani, and Sri Lankan English and should not be interpreted as a Pakistani-English-specific result.
- **The Hindi-English proxy for Urdu-English (Track 2) is an approximation.** Hindi and Urdu are closely related at the spoken level but are not interchangeable languages; Track 2 is not genuine Urdu-English code-switching, and its conclusions should not be assumed to transfer to it. Track 4 uses a dedicated Urdu resource [10] but evaluates language conditioning rather than code-switching specifically, so genuine Urdu-English code-switching remains incompletely evaluated by this study. The Track 2 Base-versus-Small paired confidence interval includes zero.
- **The transliteration and hallucination analyses (Section 6) rely partly on heuristic and manual inspection** rather than a fully automated, independently validated detector, and should be treated as exploratory rather than definitive frequency estimates.
- **The language-misidentification heuristic (Section 4) is a proxy, not a ground-truth label.** It flags non-Latin script and a small set of Malay/Indonesian markers, and will undercount misidentification into other Latin-script languages that would not trigger either signal. It also does not observe Whisper's internal language-identification decision directly in Track 1; it infers a likely misidentification from output characteristics, and the automatic-versus-forced comparisons should be read as automatic language selection versus explicitly forced decoding, not as a direct correction of an observed wrong language-ID prediction. Where automatic language labels were recorded (Tracks 2 to 4), they are model outputs, not verified ground truth.
- **Speaker independence is not established.** In Track 1, the released metadata does not retain speaker identifiers, so it is not possible to state how many distinct speakers contributed the 100 utterances per accent group. In Track 4 Urdu, the 80 matched clips come from 23 speakers, one of whom contributes 29 clips. The bootstrap confidence intervals in this paper are clip-level, not speaker-level, so they likely understate uncertainty where speakers are unevenly represented.
- **The Korean language-conditioning experiment (Track 4) and the Korean-English code-switching experiment (Track 3) use different datasets** (Zeroth Korean versus HiKE) and different speaker populations, so they should not be treated as a controlled comparison of the same underlying speech under two different tasks.
- **Only Whisper-base and Whisper-small were evaluated.** Larger checkpoints (medium, large) may exhibit different accent, code-switching, and language-identification behavior, and the model-size effects reported here (Section 6) should not be extrapolated beyond the two sizes actually tested.
- **WER is computed without text normalization beyond lowercasing in Track 1**, and by two different (numerically equivalent) implementations across tracks (Section 4). Absolute WER values, especially for Korean and Urdu, are sensitive to orthography, spacing, punctuation, and numerals.
- **The English-token-preservation metric is a lenient substring-based diagnostic** that detects Latin-script presence, not correct placement or meaning.
- **Inference was not deterministic and the environment was not pinned.** No PyTorch seed was set, Whisper's temperature fallback can alter outputs between runs (observed for the Urdu Whisper-small automatic condition, 8 of 92 clips differing between two runs), and the package versions used were not recorded (the author's recollection is approximately `openai-whisper` 20250625).
- **Provenance gaps.** The original Track 1 Whisper-base inference script is not preserved, and the exact HiKE dataset revision was not recorded. The released result files are retained, but those parts cannot be regenerated exactly from the repository.
- **All experiments used CPU-only inference.** This affected runtime but was not intended as an experimental variable and should not have influenced the accuracy results.
- **WER alone does not capture every dimension of transcription quality** relevant to multilingual and code-switched speech, as illustrated directly by the token-preservation metric in Section 5.3.3, which shows a substantially larger gap between conditions than overall WER does for the same comparison.

---

## 9. Future Work

Potential extensions of this study include: identifying or constructing a dedicated Urdu-English code-switching corpus to replace the Hindi-English proxy used in Track 2, and applying the same code-switching-granularity and domain-stratified analysis used in Track 3 to that corpus once available; supplementing this with a carefully documented, consent-based self-recorded dataset if no suitable public corpus is identified; evaluating Whisper-medium and Whisper-large across all four tracks to test whether the model-size effects reported here persist at larger scale; expanding the Korean-English and Urdu-English tracks with additional speakers to improve the reliability of the smaller subcategories (e.g., the six-clip sentence-level category in Track 3); replacing the heuristic language-misidentification detector (Section 4) with a validated language-identification model, and recording Whisper's detected-language output in every track, to produce more precise misidentification estimates; testing directly why automatic language detection outperforms forced decoding on the HiKE code-switched benchmark, since this study shows where the difference occurs (clips labelled English under automatic decoding) but does not establish its mechanism; adding speaker-level resampling to the confidence intervals; fixing and logging PyTorch seeds, package versions, and dataset revisions; and extending the benchmark to additional multilingual ASR systems beyond Whisper, to test whether the failure modes documented here are Whisper-specific or shared more broadly across large-scale multilingual ASR architectures.

---

## 10. Conclusion

This paper presented AccentBench, a four-track exploratory evaluation of Whisper ASR (base and small) across accent-stratified performance, likely language misidentification, Hindi-English and Korean-English code-switching, and language conditioning for Urdu and Korean. By separating these failure modes rather than relying on a single aggregate WER, the study found that, in these small samples, a specific accent group's elevated error rate has a substantial component consistent with language misidentification rather than purely acoustic difficulty (without explaining the whole gap), that forcing the expected language helps substantially in some settings (Urdu, forced-English recovery for the flagged Singaporean subset) and hurts in others (Korean-English code-switching on HiKE, where the increase was concentrated in clips automatically labelled English), and that, within the HiKE sample, WER differed descriptively by switching granularity and topical domain. The Hindi-English track, a proxy for Urdu-English, did not distinguish the two model sizes at the paired level. These findings support disaggregated, failure-mode-specific evaluation as a complement to aggregate WER when assessing multilingual ASR systems for deployment in accented, multilingual, and code-switching-heavy contexts, while leaving genuine Urdu-English code-switching and larger-scale replication as open work.

---

## Appendix A: Paired Bootstrap Confidence Intervals

All intervals are paired percentile bootstrap 95% CIs over per-clip WER differences (10,000 resamples with replacement, `np.random.RandomState(42)`), produced by `src/bootstrap_ci.py` and saved to `results/bootstrap_ci_results.csv`. Intervals are clip-level.

| Comparison | Difference | n | Mean difference | 95% CI |
|---|---|---:|---:|---|
| Track 1: Singaporean, original minus forced English | original - forced_en | 14 | 0.8489 | [0.7524, 0.9430] |
| Track 2: Hindi-English proxy, Base minus Small | base - small | 97 | 0.0821 | [-0.0270, 0.1771] |
| Track 3: HiKE Small, forced Korean minus automatic | forced_ko - auto | 100 | 0.2007 | [0.1294, 0.2736] |
| Track 4: Urdu Base, automatic minus forced | auto - forced | 80 | 0.1034 | [0.0366, 0.1663] |
| Track 4: Urdu Small, automatic minus forced | auto - forced | 80 | 0.4489 | [0.3653, 0.5361] |
| Track 4: Korean (Zeroth) Small, automatic minus forced | auto - forced | 100 | 0.0000 | [0.0000, 0.0000] |

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
