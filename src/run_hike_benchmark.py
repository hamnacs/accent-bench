
import io
import json
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import whisper
from datasets import load_dataset, Audio


# ============================================================
# CONFIG
# ============================================================

NUM_CLIPS = 100
RANDOM_SEED = 42

DATASET = "thetaone-ai/HiKE"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = RESULTS_DIR / "hike_whisper_predictions.csv"
SUMMARY_FILE = RESULTS_DIR / "hike_whisper_summary.csv"


# ============================================================
# WER
# ============================================================

def calculate_wer(reference, hypothesis):
    reference_words = reference.strip().split()
    hypothesis_words = hypothesis.strip().split()

    dp = [
        [0] * (len(hypothesis_words) + 1)
        for _ in range(len(reference_words) + 1)
    ]

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


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("HIKE KOREAN-ENGLISH WHISPER BENCHMARK")
print("=" * 70)

print("\nLoading HiKE...")

dataset = load_dataset(DATASET)

# IMPORTANT:
# Prevent Hugging Face from automatically decoding audio
# through TorchCodec.
test = dataset["test"].cast_column(
    "audio",
    Audio(decode=False)
)

print("Total HiKE test clips:", len(test))


# ============================================================
# SELECT FIXED SAMPLE
# ============================================================

print(f"\nSelecting {NUM_CLIPS} clips...")

if NUM_CLIPS > len(test):
    raise ValueError(
        f"NUM_CLIPS={NUM_CLIPS} is larger than dataset size "
        f"{len(test)}"
    )

sample_indices = (
    pd.Series(range(len(test)))
    .sample(
        n=NUM_CLIPS,
        random_state=RANDOM_SEED
    )
    .tolist()
)

print("Random seed:", RANDOM_SEED)
print("Selected clips:", len(sample_indices))


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = [
    {
        "model": "base",
        "forced_language": False,
        "name": "base_auto"
    },
    {
        "model": "base",
        "forced_language": True,
        "name": "base_forced_ko"
    },
    {
        "model": "small",
        "forced_language": False,
        "name": "small_auto"
    },
    {
        "model": "small",
        "forced_language": True,
        "name": "small_forced_ko"
    }
]


# ============================================================
# RUN
# ============================================================

all_results = []

for experiment_number, experiment in enumerate(experiments, start=1):

    model_name = experiment["model"]
    forced_language = experiment["forced_language"]
    experiment_name = experiment["name"]

    print()
    print("=" * 70)
    print(
        f"EXPERIMENT {experiment_number}/4: "
        f"{experiment_name}"
    )
    print("=" * 70)

    if forced_language:
        print("Language: Korean (forced)")
    else:
        print("Language: Automatic")

    print("Model:", model_name)

    print("\nLoading Whisper", model_name, "...")

    model = whisper.load_model(model_name)

    print("Model loaded!")

    experiment_results = []

    start_time = time.time()

    for position, dataset_index in enumerate(
        sample_indices,
        start=1
    ):

        row = test[dataset_index]

        try:

            # ------------------------------------------------
            # Get raw audio bytes
            # ------------------------------------------------

            audio_info = row["audio"]

            audio_bytes = audio_info["bytes"]

            if audio_bytes is None:
                raise ValueError(
                    "Audio bytes are missing"
                )

            # ------------------------------------------------
            # Decode audio using SoundFile
            # ------------------------------------------------

            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes)
            )

            audio = audio.astype("float32")

            # Convert stereo to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # ------------------------------------------------
            # Whisper settings
            # ------------------------------------------------

            kwargs = {
                "task": "transcribe"
            }

            if forced_language:
                kwargs["language"] = "ko"

            # ------------------------------------------------
            # Transcribe
            # ------------------------------------------------

            result = model.transcribe(
                audio,
                **kwargs
            )

            prediction = result["text"].strip()

            if forced_language:
                detected_language = "ko_forced"
            else:
                detected_language = result.get(
                    "language",
                    "unknown"
                )

            # ------------------------------------------------
            # WER
            # ------------------------------------------------

            reference = row["text"]

            wer = calculate_wer(
                reference,
                prediction
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            result_row = {
                "experiment": experiment_name,
                "model": model_name,
                "language_mode": (
                    "forced_ko"
                    if forced_language
                    else "auto"
                ),
                "dataset_index": dataset_index,
                "sample_id": row["sample_id"],
                "ground_truth": reference,
                "prediction": prediction,
                "detected_language": detected_language,
                "wer": wer,
                "cs_level": row["cs_level"],
                "cs_levels_all": row["cs_levels_all"],
                "category": row["category"],
                "loanwords": row["loanwords"],
                "sample_rate": sample_rate,
                "audio_seconds": len(audio) / sample_rate
            }

            experiment_results.append(result_row)
            all_results.append(result_row)

            print(
                f"[{position:3d}/{NUM_CLIPS}] "
                f"{row['sample_id']} | "
                f"WER={wer:.3f} | "
                f"lang={detected_language}"
            )

        except Exception as e:

            print(
                f"[{position:3d}/{NUM_CLIPS}] "
                f"{row['sample_id']} | "
                f"ERROR: {e}"
            )

            result_row = {
                "experiment": experiment_name,
                "model": model_name,
                "language_mode": (
                    "forced_ko"
                    if forced_language
                    else "auto"
                ),
                "dataset_index": dataset_index,
                "sample_id": row["sample_id"],
                "ground_truth": row["text"],
                "prediction": None,
                "detected_language": "error",
                "wer": None,
                "cs_level": row["cs_level"],
                "cs_levels_all": row["cs_levels_all"],
                "category": row["category"],
                "loanwords": row["loanwords"],
                "sample_rate": None,
                "audio_seconds": None
            }

            experiment_results.append(result_row)
            all_results.append(result_row)

    # --------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------

    experiment_df = pd.DataFrame(
        experiment_results
    )

    successful = experiment_df[
        experiment_df["wer"].notna()
    ]

    elapsed = time.time() - start_time

    print()
    print("-" * 70)
    print("EXPERIMENT COMPLETE")
    print("-" * 70)

    print(
        "Successful:",
        len(successful),
        "/",
        NUM_CLIPS
    )

    if len(successful) > 0:

        print(
            "Mean WER:",
            f"{successful['wer'].mean():.4f}"
        )

        print(
            "Median WER:",
            f"{successful['wer'].median():.4f}"
        )

    print(
        "Time:",
        f"{elapsed / 60:.2f} minutes"
    )

    # Free model before next experiment
    del model


# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

predictions_df = pd.DataFrame(
    all_results
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Predictions saved to:",
    PREDICTIONS_FILE
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for experiment_name in predictions_df[
    "experiment"
].unique():

    subset = predictions_df[
        predictions_df["experiment"] == experiment_name
    ]

    successful = subset[
        subset["wer"].notna()
    ]

    if len(successful) == 0:
        continue

    summary_rows.append({
        "experiment": experiment_name,
        "model": successful["model"].iloc[0],
        "language_mode": successful[
            "language_mode"
        ].iloc[0],
        "clips": len(successful),
        "mean_wer": successful["wer"].mean(),
        "median_wer": successful["wer"].median(),
        "min_wer": successful["wer"].min(),
        "max_wer": successful["wer"].max()
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

if len(summary_df) > 0:

    print(
        summary_df.to_string(
            index=False
        )
    )

else:

    print("No successful experiments.")


print()
print("Summary saved to:")
print(SUMMARY_FILE)

print()
print("=" * 70)
print("HIKE BENCHMARK COMPLETE")
print("=" * 70)

