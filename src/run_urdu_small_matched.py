import json
import whisper
import pandas as pd
from pathlib import Path
from jiwer import wer

BASE = Path("data/urdu_benchmark/benchmark/US-benchmark-Std/short")
REFERENCE = Path("results/forced_urdu_results.csv")
OUTPUT = Path("results/urdu_small_matched.csv")

print("Loading reference clips...")

reference = pd.read_csv(REFERENCE)
audio_names = reference["audio"].dropna().tolist()

print(f"Reference clips: {len(audio_names)}")

# Find transcripts and audio
entries = {}

for jsonl in BASE.rglob("clean_transcription.jsonl"):
    audio_dir = jsonl.parent / "audio"

    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            audio_name = x.get("Audio_Clip", "").strip()
            ground_truth = x.get("ground_truth", "").strip()

            if not audio_name or not ground_truth:
                continue

            audio_path = audio_dir / audio_name

            if audio_path.exists():
                entries[audio_name] = {
                    "audio": audio_name,
                    "audio_path": str(audio_path),
                    "ground_truth": ground_truth
                }

sample = [
    entries[audio]
    for audio in audio_names
    if audio in entries
]

print(f"Matched clips: {len(sample)}")

print("\nLoading Whisper-small...")
model = whisper.load_model("small")
print("Model loaded!\n")

results = []

for i, item in enumerate(sample, 1):

    try:
        result = model.transcribe(
            item["audio_path"],
            fp16=False
        )

        prediction = result["text"].strip()
        detected_language = result.get("language", "unknown")

        error_rate = wer(
            item["ground_truth"],
            prediction
        )

        print(
            f"[{i}/{len(sample)}] "
            f"{item['audio']} | "
            f"lang={detected_language} | "
            f"WER={error_rate:.3f}"
        )

        results.append({
            "audio": item["audio"],
            "ground_truth": item["ground_truth"],
            "prediction": prediction,
            "detected_language": detected_language,
            "wer": error_rate
        })

    except Exception as e:

        print(f"[{i}/{len(sample)}] ERROR: {e}")

        results.append({
            "audio": item["audio"],
            "ground_truth": item["ground_truth"],
            "prediction": "",
            "detected_language": "error",
            "wer": None
        })

df = pd.DataFrame(results)

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Successful: {df['wer'].notna().sum()} / {len(df)}")
print(f"Mean WER: {df['wer'].mean():.3f}")
print(f"Median WER: {df['wer'].median():.3f}")

print("\nDetected languages:")
print(df["detected_language"].value_counts())

print(f"\nSaved to: {OUTPUT}")