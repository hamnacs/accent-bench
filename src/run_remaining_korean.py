import pandas as pd
import whisper
from pathlib import Path
import io
import soundfile as sf
import time

PARQUET = Path("data/zeroth_korean/data/test-00000-of-00001.parquet")
META = Path("results/korean_test_100.csv")

df = pd.read_parquet(PARQUET)
meta = pd.read_csv(META)


def calculate_wer(reference, hypothesis):
    reference_words = reference.strip().split()
    hypothesis_words = hypothesis.strip().split()

    dp = [[0] * (len(hypothesis_words) + 1)
          for _ in range(len(reference_words) + 1)]

    for i in range(len(reference_words) + 1):
        dp[i][0] = i

    for j in range(len(hypothesis_words) + 1):
        dp[0][j] = j

    for i in range(1, len(reference_words) + 1):
        for j in range(1, len(hypothesis_words) + 1):
            cost = (
                0
                if reference_words[i - 1] == hypothesis_words[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1] / max(len(reference_words), 1)


def run_experiment(model_name, forced_language, output_file):

    print()
    print("=" * 60)
    print(f"MODEL: {model_name}")
    print(
        "LANGUAGE:",
        "Korean (forced)" if forced_language else "Automatic"
    )
    print("=" * 60)

    model = whisper.load_model(model_name)

    results = []

    for i, row in meta.iterrows():

        try:
            audio_row = df.iloc[int(row["audio_index"])]
            audio_bytes = audio_row["audio"]["bytes"]

            audio, sr = sf.read(io.BytesIO(audio_bytes))

            audio = audio.astype("float32")

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            kwargs = {
                "task": "transcribe"
            }

            if forced_language:
                kwargs["language"] = "ko"

            result = model.transcribe(audio, **kwargs)

            prediction = result["text"].strip()

            detected_language = (
                "ko_forced"
                if forced_language
                else result.get("language", "unknown")
            )

            error_rate = calculate_wer(
                row["text"],
                prediction
            )

            print(
                f"[{i + 1:3d}/100] "
                f"{row['id']} | "
                f"language={detected_language} | "
                f"WER={error_rate:.3f}"
            )

            results.append({
                "id": row["id"],
                "speaker_id": row["speaker_id"],
                "ground_truth": row["text"],
                "prediction": prediction,
                "detected_language": detected_language,
                "wer": error_rate
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

    out.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Successful:", out["wer"].notna().sum(), "/ 100")
    print("Mean WER:", out["wer"].mean())
    print("Median WER:", out["wer"].median())
    print("Saved to:", output_file)

    del model

    return out


# ============================================================
# 1. BASE — FORCED KOREAN
# ============================================================

run_experiment(
    "base",
    True,
    "results/korean_base_forced.csv"
)


# ============================================================
# 2. SMALL — AUTOMATIC
# ============================================================

run_experiment(
    "small",
    False,
    "results/korean_small_auto.csv"
)


# ============================================================
# 3. SMALL — FORCED KOREAN
# ============================================================

run_experiment(
    "small",
    True,
    "results/korean_small_forced.csv"
)


print()
print("=" * 60)
print("ALL THREE REMAINING KOREAN EXPERIMENTS COMPLETE")
print("=" * 60)