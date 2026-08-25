import pandas as pd
import whisper
from jiwer import wer
from pathlib import Path

INPUT = "results/urdu_benchmark_small.csv"
OUTPUT = "results/forced_urdu_results.csv"

AUDIO_BASE = Path(
    "data/urdu_benchmark/benchmark/US-benchmark-Std/short"
)

print("Loading previous Urdu benchmark...")

df = pd.read_csv(INPUT)

# Same Standard Urdu sample used in the previous experiment
df = df[df["category"] == "US-benchmark-Std"].copy()

print(f"Found {len(df)} Standard Urdu clips")

print("\nFinding audio files...")

audio_lookup = {}

for wav in AUDIO_BASE.rglob("*.WAV"):
    audio_lookup[wav.name] = wav

print(f"Found {len(audio_lookup)} WAV files")

print("\nLoading Whisper-small...")
model = whisper.load_model("small")
print("Model loaded!\n")

results = []

for i, row in enumerate(df.itertuples(index=False), start=1):

    audio_path = audio_lookup.get(row.audio)

    if audio_path is None:
        print(f"[{i}/{len(df)}] MISSING: {row.audio}")
        continue

    print(f"[{i}/{len(df)}] {row.audio}")

    try:
        result = model.transcribe(
            str(audio_path),
            language="ur",
            fp16=False
        )

        prediction = result["text"].strip()

        error_rate = wer(
            row.ground_truth,
            prediction
        )

        print(
            f"    forced language=ur | "
            f"WER={error_rate:.3f}"
        )

    except Exception as e:
        print(f"    ERROR: {e}")
        prediction = ""
        error_rate = None

    results.append({
        "audio": row.audio,
        "ground_truth": row.ground_truth,
        "prediction": prediction,
        "detected_language": "ur_forced",
        "wer": error_rate
    })

out = pd.DataFrame(results)

out.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Successful: "
    f"{out['wer'].notna().sum()} / {len(out)}"
)

print(f"Mean WER: {out['wer'].mean():.3f}")
print(f"Median WER: {out['wer'].median():.3f}")

print(f"\nSaved to: {OUTPUT}")