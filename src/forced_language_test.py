from datasets import load_from_disk, Audio
import whisper
from jiwer import wer
import pandas as pd
import soundfile as sf
import tempfile
import os
import io
import time

# Load the flagged examples (ground truth + ids) to know which clips to re-test
flagged = pd.read_csv("results/language_switch_examples.csv")

# Load the full accent sample again so we can find the matching audio
ds = load_from_disk("data/accent_sample")
ds = ds.cast_column("audio", Audio(decode=False))

model = whisper.load_model("base")

results = []

for i, example in enumerate(ds):
    ground_truth = example["sentence"]
    accent = example["accent"]

    # Only process clips that match a flagged ground_truth (simple matching by text)
    if ground_truth not in flagged["ground_truth"].values:
        continue

    audio_bytes = example["audio"]["bytes"]
    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        continue

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sr)
        tmp_path = tmp.name

    try:
        # Force English this time
        result = model.transcribe(tmp_path, fp16=False, language="en")
        prediction = result["text"].strip()
        error_rate = wer(ground_truth.lower(), prediction.lower())
    except Exception as e:
        prediction = ""
        error_rate = None
    finally:
        for attempt in range(5):
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.2)

    results.append({
        "accent": accent,
        "ground_truth": ground_truth,
        "prediction_forced_en": prediction,
        "wer_forced_en": error_rate
    })
    print(f"{accent[:25]:25s} | forced-EN WER: {error_rate:.2f}" if error_rate is not None else "failed")

df = pd.DataFrame(results)
df.to_csv("results/forced_english_results.csv", index=False)

# Compare against original WER for the same clips
original = flagged[["ground_truth", "wer"]].rename(columns={"wer": "wer_original"})
comparison = df.merge(original, on="ground_truth", how="left")
comparison.to_csv("results/before_after_comparison.csv", index=False)

print("\n=== Before vs After (forcing English) ===")
print(comparison[["accent", "wer_original", "wer_forced_en"]])
print(f"\nMean WER original (language-switch cases): {comparison['wer_original'].mean():.3f}")
print(f"Mean WER forced English:                    {comparison['wer_forced_en'].mean():.3f}")