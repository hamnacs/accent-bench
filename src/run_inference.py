from datasets import load_from_disk, Audio
import whisper
from jiwer import wer
import pandas as pd
import numpy as np
import soundfile as sf
import tempfile
import os
import io
import time

# Load your full saved sample (600 examples, 100 per accent)
ds = load_from_disk("data/accent_sample")

# Avoid torchcodec entirely — decode audio manually
ds = ds.cast_column("audio", Audio(decode=False))

print(f"Full run: {len(ds)} examples")

model = whisper.load_model("small")

results = []

for i, example in enumerate(ds):
    accent = example["accent"]
    ground_truth = example["sentence"]

    audio_bytes = example["audio"]["bytes"]

    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        print(f"[{i+1}/{len(ds)}] Failed to decode audio: {e}")
        results.append({
            "accent": accent,
            "ground_truth": ground_truth,
            "prediction": "",
            "wer": None
        })
        continue

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sr)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, fp16=False)
        prediction = result["text"].strip()
        error_rate = wer(ground_truth.lower(), prediction.lower())
    except Exception as e:
        print(f"[{i+1}/{len(ds)}] Error: {e}")
        prediction = ""
        error_rate = None
    finally:
        # Retry delete a few times in case Windows still has the file locked
        for attempt in range(5):
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.2)

    results.append({
        "accent": accent,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "wer": error_rate
    })

    if error_rate is not None:
        print(f"[{i+1}/{len(ds)}] {accent[:30]:30s} | WER: {error_rate:.2f}")
    else:
        print(f"[{i+1}/{len(ds)}] {accent[:30]:30s} | failed")

    # Save progress every 50 examples in case of a crash
    if (i + 1) % 50 == 0:
        pd.DataFrame(results).to_csv("results/wer_results_progress_small.csv", index=False)

# Final save
df = pd.DataFrame(results)
df.to_csv("results/wer_results_small.csv", index=False)

print("\nDone. Results saved to results/wer_results_full.csv")
print(f"\nSuccessful: {df['wer'].notna().sum()} / {len(df)}")
print("\nMean WER by accent (full run, n=100 per group):")
print(df.groupby("accent")["wer"].mean().sort_values())
print("\nMedian WER by accent:")
print(df.groupby("accent")["wer"].median().sort_values())