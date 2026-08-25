import pandas as pd
import whisper
from pathlib import Path
import io
import soundfile as sf

PARQUET = Path("data/zeroth_korean/data/test-00000-of-00001.parquet")
META = Path("results/korean_test_100.csv")
OUTPUT = Path("results/korean_base_auto.csv")

print("Loading metadata...")
meta = pd.read_csv(META)

print("Loading audio dataset...")
df = pd.read_parquet(PARQUET)

print("Loading Whisper Base...")
model = whisper.load_model("base")
print("Model loaded!")

results = []

for i, row in meta.iterrows():

    audio_row = df.iloc[int(row["audio_index"])]
    audio_bytes = audio_row["audio"]["bytes"]

    try:
        # Read FLAC audio from the parquet bytes
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        # Whisper expects float32 mono audio
        audio = audio.astype("float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        result = model.transcribe(
            audio,
            task="transcribe"
        )

        prediction = result["text"].strip()
        detected_language = result.get("language", "unknown")

        # Simple WER calculation
        reference_words = row["text"].strip().split()
        prediction_words = prediction.split()

        # Levenshtein distance
        dp = [[0] * (len(prediction_words) + 1)
              for _ in range(len(reference_words) + 1)]

        for a in range(len(reference_words) + 1):
            dp[a][0] = a

        for b in range(len(prediction_words) + 1):
            dp[0][b] = b

        for a in range(1, len(reference_words) + 1):
            for b in range(1, len(prediction_words) + 1):
                cost = 0 if reference_words[a - 1] == prediction_words[b - 1] else 1

                dp[a][b] = min(
                    dp[a - 1][b] + 1,
                    dp[a][b - 1] + 1,
                    dp[a - 1][b - 1] + cost
                )

        wer = dp[-1][-1] / max(len(reference_words), 1)

        print(
            f"[{i + 1:3d}/100] "
            f"{row['id']} | "
            f"language={detected_language} | "
            f"WER={wer:.3f}"
        )

        results.append({
            "id": row["id"],
            "speaker_id": row["speaker_id"],
            "ground_truth": row["text"],
            "prediction": prediction,
            "detected_language": detected_language,
            "wer": wer
        })

    except Exception as e:

        print(
            f"[{i + 1:3d}/100] "
            f"{row['id']} | ERROR: {e}"
        )

        results.append({
            "id": row["id"],
            "speaker_id": row["speaker_id"],
            "ground_truth": row["text"],
            "prediction": None,
            "detected_language": "error",
            "wer": None
        })

out = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print("Successful:", out["wer"].notna().sum(), "/", len(out))
print("Mean WER:", out["wer"].mean())
print("Median WER:", out["wer"].median())
print()
print("Detected languages:")
print(out["detected_language"].value_counts())
print()
print("Saved to:", OUTPUT)