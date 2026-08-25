import json
import whisper
import pandas as pd
from pathlib import Path
from jiwer import wer

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE = Path("data/urdu_benchmark/benchmark/US-benchmark-Std/short")

# Use the existing forced-Urdu results to identify
# the exact 100 clips used in that experiment.
REFERENCE_RESULTS = Path("results/forced_urdu_results.csv")

OUTPUT = Path("results/urdu_base_results.csv")


# --------------------------------------------------
# Load the exact same 100 audio clips
# --------------------------------------------------

print("Loading reference clip list...")

reference = pd.read_csv(REFERENCE_RESULTS)

audio_names = reference["audio"].dropna().tolist()

print(f"Found {len(audio_names)} reference clips")


# --------------------------------------------------
# Find transcripts/audio
# --------------------------------------------------

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


# --------------------------------------------------
# Verify all clips exist
# --------------------------------------------------

sample = []

missing = []

for audio_name in audio_names:

    if audio_name in entries:
        sample.append(entries[audio_name])
    else:
        missing.append(audio_name)


print(f"Matched clips: {len(sample)}")

if missing:
    print(f"Missing clips: {len(missing)}")
    for x in missing:
        print("Missing:", x)


# --------------------------------------------------
# Load Whisper-base
# --------------------------------------------------

print("\nLoading Whisper-base...")

model = whisper.load_model("base")

print("Model loaded!\n")


# --------------------------------------------------
# Run inference
# --------------------------------------------------

results = []

for i, item in enumerate(sample, 1):

    try:

        result = model.transcribe(
            item["audio_path"],
            fp16=False
        )

        prediction = result["text"].strip()

        detected_language = result.get(
            "language",
            "unknown"
        )

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

        print(
            f"[{i}/{len(sample)}] "
            f"ERROR: {e}"
        )

        results.append({
            "audio": item["audio"],
            "ground_truth": item["ground_truth"],
            "prediction": "",
            "detected_language": "error",
            "wer": None
        })


# --------------------------------------------------
# Save
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(OUTPUT, index=False)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Successful: "
    f"{df['wer'].notna().sum()} / {len(df)}"
)

print(
    f"Mean WER: "
    f"{df['wer'].mean():.3f}"
)

print(
    f"Median WER: "
    f"{df['wer'].median():.3f}"
)

print("\nDetected languages:")

print(
    df["detected_language"]
    .value_counts()
)

print(f"\nSaved to: {OUTPUT}")