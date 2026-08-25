import ast
import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("results/hike_whisper_predictions.csv")
OUTPUT_DIR = Path("results/hike_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("HIKE KOREAN-ENGLISH RESULTS ANALYSIS")
print("=" * 70)

print("\nLoading predictions...")

df = pd.read_csv(INPUT)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nExperiments:")
print(df["experiment"].value_counts().to_string())


# ============================================================
# 1. OVERALL WER
# ============================================================

print()
print("=" * 70)
print("1. OVERALL WER")
print("=" * 70)

overall = (
    df.groupby(
        ["experiment", "model", "language_mode"],
        as_index=False
    )
    .agg(
        clips=("wer", "count"),
        mean_wer=("wer", "mean"),
        median_wer=("wer", "median"),
        min_wer=("wer", "min"),
        max_wer=("wer", "max")
    )
    .sort_values("mean_wer")
)

print(
    overall.to_string(index=False)
)

overall.to_csv(
    OUTPUT_DIR / "overall_wer.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 2. WER BY CODE-SWITCHING LEVEL
# ============================================================

print()
print("=" * 70)
print("2. WER BY CODE-SWITCHING LEVEL")
print("=" * 70)

cs_level = (
    df.groupby(
        ["experiment", "cs_level"],
        as_index=False
    )
    .agg(
        clips=("wer", "count"),
        mean_wer=("wer", "mean"),
        median_wer=("wer", "median")
    )
    .sort_values(
        ["cs_level", "mean_wer"]
    )
)

print(
    cs_level.to_string(index=False)
)

cs_level.to_csv(
    OUTPUT_DIR / "wer_by_cs_level.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 3. WER BY CATEGORY
# ============================================================

print()
print("=" * 70)
print("3. WER BY CATEGORY")
print("=" * 70)

category = (
    df.groupby(
        ["experiment", "category"],
        as_index=False
    )
    .agg(
        clips=("wer", "count"),
        mean_wer=("wer", "mean"),
        median_wer=("wer", "median")
    )
    .sort_values(
        ["experiment", "mean_wer"]
    )
)

print(
    category.to_string(index=False)
)

category.to_csv(
    OUTPUT_DIR / "wer_by_category.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. AUTO VS FORCED LANGUAGE
# ============================================================

print()
print("=" * 70)
print("4. AUTO VS FORCED KOREAN")
print("=" * 70)

auto_forced = (
    df.groupby(
        ["model", "language_mode"],
        as_index=False
    )
    .agg(
        clips=("wer", "count"),
        mean_wer=("wer", "mean"),
        median_wer=("wer", "median")
    )
)

print(
    auto_forced.to_string(index=False)
)

auto_forced.to_csv(
    OUTPUT_DIR / "auto_vs_forced.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. RELATIVE WER IMPROVEMENT
# ============================================================

print()
print("=" * 70)
print("5. AUTO VS FORCED RELATIVE DIFFERENCE")
print("=" * 70)

comparison_rows = []

for model in ["base", "small"]:

    auto = df[
        (df["model"] == model)
        & (df["language_mode"] == "auto")
    ]["wer"].mean()

    forced = df[
        (df["model"] == model)
        & (df["language_mode"] == "forced_ko")
    ]["wer"].mean()

    absolute_change = forced - auto

    relative_change = (
        absolute_change / forced * 100
        if forced != 0
        else 0
    )

    comparison_rows.append({
        "model": model,
        "auto_wer": auto,
        "forced_ko_wer": forced,
        "absolute_change": absolute_change,
        "relative_reduction_when_auto_percent": relative_change
    })

comparison = pd.DataFrame(
    comparison_rows
)

print(
    comparison.to_string(index=False)
)

comparison.to_csv(
    OUTPUT_DIR / "auto_forced_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# TOKEN HELPERS
# ============================================================

def english_tokens(text):
    """
    Extract English-looking tokens.
    """
    if not isinstance(text, str):
        return []

    return re.findall(
        r"\b[A-Za-z][A-Za-z'-]*\b",
        text
    )


def korean_tokens(text):
    """
    Extract Korean syllable sequences.
    """
    if not isinstance(text, str):
        return []

    return re.findall(
        r"[가-힣]+",
        text
    )


def normalize_token(token):
    return re.sub(
        r"[^a-zA-Z]",
        "",
        token
    ).lower()


# ============================================================
# 6. ENGLISH TOKEN ANALYSIS
# ============================================================

print()
print("=" * 70)
print("6. ENGLISH TOKEN PRESERVATION")
print("=" * 70)

english_results = []

for _, row in df.iterrows():

    reference = row["ground_truth"]
    prediction = row["prediction"]

    if not isinstance(reference, str):
        continue

    if not isinstance(prediction, str):
        prediction = ""

    ref_tokens = english_tokens(reference)

    pred_lower = prediction.lower()

    if len(ref_tokens) == 0:
        continue

    preserved = 0

    for token in ref_tokens:

        normalized = normalize_token(token)

        if normalized and normalized in pred_lower:
            preserved += 1

    preservation_rate = (
        preserved / len(ref_tokens)
    )

    english_results.append({
        "experiment": row["experiment"],
        "sample_id": row["sample_id"],
        "english_tokens": len(ref_tokens),
        "english_tokens_preserved": preserved,
        "english_preservation_rate": preservation_rate,
        "ground_truth": reference,
        "prediction": prediction
    })


english_df = pd.DataFrame(
    english_results
)

english_summary = (
    english_df.groupby(
        "experiment",
        as_index=False
    )
    .agg(
        samples=("sample_id", "count"),
        english_tokens=("english_tokens", "sum"),
        preserved_tokens=(
            "english_tokens_preserved",
            "sum"
        ),
        mean_preservation=(
            "english_preservation_rate",
            "mean"
        )
    )
)

english_summary[
    "token_preservation_rate"
] = (
    english_summary["preserved_tokens"]
    / english_summary["english_tokens"]
)

print(
    english_summary.to_string(index=False)
)

english_summary.to_csv(
    OUTPUT_DIR / "english_token_preservation.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. WORST EXAMPLES
# ============================================================

print()
print("=" * 70)
print("7. WORST TRANSCRIPTION EXAMPLES")
print("=" * 70)

worst = (
    df.sort_values(
        "wer",
        ascending=False
    )
    [
        [
            "experiment",
            "wer",
            "cs_level",
            "category",
            "ground_truth",
            "prediction"
        ]
    ]
    .head(30)
)

print(
    worst.to_string(index=False)
)

worst.to_csv(
    OUTPUT_DIR / "worst_examples.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. BEST EXAMPLES
# ============================================================

print()
print("=" * 70)
print("8. BEST EXAMPLES")
print("=" * 70)

best = (
    df.sort_values(
        "wer",
        ascending=True
    )
    [
        [
            "experiment",
            "wer",
            "cs_level",
            "category",
            "ground_truth",
            "prediction"
        ]
    ]
    .head(30)
)

print(
    best.to_string(index=False)
)

best.to_csv(
    OUTPUT_DIR / "best_examples.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. LIKELY ENGLISH → KOREAN TRANSLITERATION
# ============================================================

print()
print("=" * 70)
print("9. LIKELY ENGLISH → KOREAN TRANSLITERATION")
print("=" * 70)

transliteration_rows = []

for _, row in df.iterrows():

    reference = row["ground_truth"]
    prediction = row["prediction"]

    if not isinstance(reference, str):
        continue

    if not isinstance(prediction, str):
        continue

    ref_english = english_tokens(reference)

    if not ref_english:
        continue

    pred_korean = korean_tokens(prediction)

    if not pred_korean:
        continue

    # If the reference contains English but the prediction
    # contains Korean syllables, flag it for manual review.
    transliteration_rows.append({
        "experiment": row["experiment"],
        "sample_id": row["sample_id"],
        "wer": row["wer"],
        "ground_truth": reference,
        "prediction": prediction,
        "english_reference_tokens": ", ".join(
            ref_english
        ),
        "korean_prediction_tokens": ", ".join(
            pred_korean
        )
    })


transliteration_df = pd.DataFrame(
    transliteration_rows
)

transliteration_df = (
    transliteration_df
    .sort_values(
        ["experiment", "wer"],
        ascending=[True, False]
    )
)

print(
    transliteration_df.head(50).to_string(
        index=False
    )
)

transliteration_df.to_csv(
    OUTPUT_DIR / "possible_transliterations.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. ERROR RATE BY CATEGORY — BEST MODEL
# ============================================================

print()
print("=" * 70)
print("10. BEST MODEL BY CATEGORY")
print("=" * 70)

best_model = (
    overall.sort_values(
        "mean_wer"
    )
    .iloc[0]["experiment"]
)

best_model_df = df[
    df["experiment"] == best_model
]

best_category = (
    best_model_df.groupby(
        "category",
        as_index=False
    )
    .agg(
        clips=("wer", "count"),
        mean_wer=("wer", "mean"),
        median_wer=("wer", "median")
    )
    .sort_values(
        "mean_wer"
    )
)

print(
    "Best overall experiment:",
    best_model
)

print()

print(
    best_category.to_string(index=False)
)

best_category.to_csv(
    OUTPUT_DIR / "best_model_by_category.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. ERROR RATE BY CS LEVEL — BEST MODEL
# ============================================================

print()
print("=" * 70)
print("11. BEST MODEL BY CODE-SWITCHING LEVEL")
print("=" * 70)

best_cs = (
    best_model_df.groupby(
        "cs_level",
        as_index=False
    )
    .agg(
        clips=("wer", "count"),
        mean_wer=("wer", "mean"),
        median_wer=("wer", "median")
    )
    .sort_values(
        "mean_wer"
    )
)

print(
    best_cs.to_string(index=False)
)

best_cs.to_csv(
    OUTPUT_DIR / "best_model_by_cs_level.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL RESEARCH SUMMARY")
print("=" * 70)

best_row = overall.iloc[0]

print(
    f"\nBest experiment: {best_row['experiment']}"
)

print(
    f"Mean WER: {best_row['mean_wer']:.4f}"
)

print(
    f"Median WER: {best_row['median_wer']:.4f}"
)

print(
    "\nAnalysis files saved to:"
)

print(
    OUTPUT_DIR
)

print()
print("Files:")

for file in sorted(
    OUTPUT_DIR.glob("*.csv")
):
    print(" -", file.name)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
