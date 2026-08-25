import json
from pathlib import Path

import whisper
from jiwer import wer


BASE = Path("data/urdu_benchmark/benchmark/US-benchmark-Std/short")
CATEGORY = "INTERVIEWS"

jsonl_path = BASE / CATEGORY / "clean_transcription.jsonl"
audio_dir = BASE / CATEGORY / "audio"

print("Loading transcripts...")
with open(jsonl_path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print(f"Found {len(rows)} transcript entries")

print("Loading Whisper-small...")
model = whisper.load_model("small")
print("Model loaded!")


for i, row in enumerate(rows[:5]):

    audio_file = audio_dir / row["Audio_Clip"]
    ground_truth = row["ground_truth"].strip()

    print("\n" + "=" * 60)
    print(f"Example {i + 1}")
    print("Audio:", audio_file.name)
    print("Ground truth:", ground_truth)

    if not audio_file.exists():
        print("ERROR: audio file not found")
        continue

    try:
        result = model.transcribe(
            str(audio_file),
            language="ur",
            fp16=False
        )

        prediction = result["text"].strip()
        error_rate = wer(ground_truth, prediction)

        print("Prediction:", prediction)
        print("WER:", round(error_rate, 3))
        print("Detected language:", result.get("language"))

    except Exception as e:
        print("ERROR:", e)

print("\nDone.")