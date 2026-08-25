# Findings and Limitations

## Findings

The experiments show that explicitly specifying the spoken language has a substantially different effect depending on the language being transcribed.

### Urdu

Language forcing produced a clear improvement for Urdu transcription.

For Whisper Base, mean WER decreased from **0.6855** with automatic language detection to **0.5821** when Urdu was forced, corresponding to a **15.1% relative improvement**.

The effect was considerably larger for Whisper Small. Mean WER decreased from **0.8473** to **0.3984**, corresponding to a **53.0% relative improvement**.

This suggests that automatic language identification can be an important source of transcription errors for Urdu speech. The effect is particularly pronounced for the Small model, which produced substantially better Urdu transcriptions when the expected language was explicitly provided.

### Korean

Korean showed a very different pattern.

Whisper detected Korean automatically for **all 100 Korean test clips**. For Whisper Base, mean WER changed only from **0.4737** to **0.4726**, representing a relative improvement of approximately **0.23%**.

For Whisper Small, automatic and forced-language transcription produced exactly the same results:

* Mean WER: **0.3791**
* Median WER: **0.3333**
* Identical predictions: **100/100**
* Different predictions: **0/100**

Therefore, explicitly forcing Korean provided essentially no benefit on this evaluation set.

### Model Comparison

The results also show that model size alone does not guarantee better performance.

For Urdu under automatic language detection, Whisper Base achieved a lower mean WER (**0.6855**) than Whisper Small (**0.8473**).

However, after forcing Urdu, Whisper Small substantially outperformed Base:

* Base + forced Urdu: **0.5821**
* Small + forced Urdu: **0.3984**

For Korean, Whisper Small performed better than Base under both conditions.

These results suggest that language conditioning and model size can interact differently depending on the language.

## Overall Interpretation

The main finding of AccentBench is that **language conditioning can have a large effect on multilingual ASR performance, but the effect is language-dependent**.

Urdu benefited substantially from explicit language specification, particularly with Whisper Small. Korean, in contrast, was already reliably identified by Whisper, so forcing the language provided virtually no additional benefit.

This demonstrates why evaluating multilingual ASR only through overall accuracy can hide important language-specific behavior.

## Limitations

### Limited Evaluation Size

The final comparison used **80 matched Urdu clips** and **100 Korean clips**. These subsets are useful for a controlled experiment but are not large enough to represent all possible speech patterns within either language.

### Urdu Benchmark Selection

The Urdu benchmark was manually constructed and therefore may contain selection bias. The selected clips should not be treated as a statistically representative sample of all Urdu speech.

### Korean Dataset Scope

The Korean evaluation used 100 clips from the Zeroth Korean test set, distributed across 10 speakers. Although this provides speaker diversity, it was not designed as a controlled Korean dialect or accent benchmark.

Consequently, the Korean results should not be interpreted as evidence that Whisper performs equally well across all Korean regional varieties.

### Failed Inference Runs

Some Urdu clips failed during individual inference runs. Rather than rerunning every failed experiment, the final comparison used only clips for which all four experimental conditions produced valid WER values.

This produced a consistent **80-clip matched subset**, but it also means the final Urdu results may differ from statistics calculated over each individual CSV independently.

### CPU Inference

The experiments were performed locally on CPU. This affected runtime but was not intended as an experimental variable. GPU-based inference could substantially reduce execution time without changing the underlying evaluation design.

### WER Limitations

WER is useful for measuring transcription errors, but it does not capture every aspect of transcription quality. For languages such as Urdu and Korean, tokenization, word segmentation, spelling conventions, and script-specific characteristics can influence the metric.

### Language Forcing Is Not Accent Conditioning

The experiments test **language conditioning**, not accent recognition or accent adaptation.

A forced language tells Whisper which language to expect. It does not tell the model which regional accent or dialect is being spoken.

Therefore, the results should be described as evidence about **language identification and language conditioning**, rather than direct evidence of accent robustness.
