import json
import random
from pathlib import Path

import pandas as pd
import whisper
from jiwer import wer


BASE = Path("data/urdu_benchmark/benchmark")

# Number of samples per category
SAMPLES_PER_CATEGORY = 100

random.seed(42)


def load_category(category_name):
    category_dir = BASE / category_name / "short"

    entries = []

    for transcript_file in category_dir.rglob("clean_transcription.jsonl"):
        audio_dir = transcript_file.parent

        with open(transcript_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                audio_name = item.get("Audio_Clip")
                ground_truth = item.get("ground_truth", "").strip()

                if not audio_name or not ground_truth:
                    continue

                audio_path = audio_dir / "audio" / audio_name

                if not audio_path.exists():
                    continue

                entries.append({
                    "category": category_name,
                    "audio": str(audio_path),
                    "audio_name": audio_name,
                    "ground_truth": ground_truth,
                    "speaker": item.get("Speaker_id", ""),
                    "source": item.get("Audio_category", ""),
                })

    return entries


categories = [
    "US-benchmark-Std",
    "US-benchmark-EngPk",
    "US-benchmark-CS",
]


print("Loading UrduSpeech metadata...\n")

all_entries = []

for category in categories:
    entries = load_category(category)

    print(f"{category}: {len(entries)} valid clips")

    random.shuffle(entries)

    entries = entries[:SAMPLES_PER_CATEGORY]

    all_entries.extend(entries)

print(f"\nTotal sampled clips: {len(all_entries)}")

print("\nLoading Whisper-small...")
model = whisper.load_model("small")
print("Model loaded!\n")


results = []

for i, item in enumerate(all_entries, start=1):

    print(
        f"[{i}/{len(all_entries)}] "
        f"{item['category']} | "
        f"{item['audio_name']}"
    )

    try:
        result = model.transcribe(
            item["audio"],
            fp16=False
        )

        prediction = result["text"].strip()
        detected_language = result.get("language", "unknown")

        error_rate = wer(
            item["ground_truth"],
            prediction
        )

        print(
            f"    language={detected_language} "
            f"| WER={error_rate:.3f}"
        )

    except Exception as e:

        print(f"    ERROR: {e}")

        prediction = ""
        detected_language = "error"
        error_rate = None

    results.append({
        "category": item["category"],
        "audio": item["audio_name"],
        "speaker": item["speaker"],
        "source": item["source"],
        "ground_truth": item["ground_truth"],
        "prediction": prediction,
        "detected_language": detected_language,
        "wer": error_rate,
    })

    if i % 25 == 0:
        pd.DataFrame(results).to_csv(
            "results/urdu_benchmark_progress.csv",
            index=False
        )


df = pd.DataFrame(results)

df.to_csv(
    "results/urdu_benchmark_small.csv",
    index=False
)


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"\nSuccessful: {df['wer'].notna().sum()} / {len(df)}")

print("\nWER by category:")

print(
    df.groupby("category")["wer"]
    .agg(["count", "mean", "median"])
)


print("\nDetected language by category:")

print(
    pd.crosstab(
        df["category"],
        df["detected_language"]
    )
)

print(
    "\nResults saved to "
    "results/urdu_benchmark_small.csv"
)